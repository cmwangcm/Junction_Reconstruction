import argparse
import copy
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.methods.connection_parser import arm_heading_midpoint, enrich_channelized_features, normalize_driving_side
from scripts.methods.image_processing import load_processed_image
from scripts.methods.llm_kg.arm_level.prompts import (
    FULL_DIAGRAM_USER_TEXT,
    arm_crop_user_text,
    arm_level_prompt,
    arm_refine_prompt,
)
from scripts.methods.llm_kg.arm_level.schemas import ARM_LEVEL_SCHEMA, ARM_REFINE_SCHEMA
from scripts.methods.method_utils import (
    arm_level_dir,
    load_case_driving_side,
    prepare_output_dir,
    resolve_case_inputs,
    resolve_run_dir,
    set_verbose,
    unpack_model_config,
)
from scripts.methods.llm_utils import BaseLLM, CostTracker, build_llm, dump_json, to_jsonable


def _normalize_heading_deg(heading_deg):
    return float(heading_deg) % 360.0


def _heading_range_span(heading_deg_min, heading_deg_max):
    """CCW span in degrees from heading_deg_min to heading_deg_max."""
    hmin = _normalize_heading_deg(heading_deg_min)
    hmax = _normalize_heading_deg(heading_deg_max)
    return (hmax - hmin) % 360.0


def _arm_heading(arm):
    """Outward heading: heading_deg, else midpoint of a legacy heading_deg_min/max range."""
    heading = arm_heading_midpoint(arm)
    if heading is None:
        raise ValueError("arm %s has no heading_deg" % arm.get("id"))
    return heading


HEADING_FAN_SPAN_CAP_DEG = 60.0


def _heading_fan_extra_u(v_extent_px, arm):
    """Lateral pad from a legacy heading range; 0 when the arm has only heading_deg."""
    if arm.get("heading_deg_min") is None or arm.get("heading_deg_max") is None:
        return 0.0
    span = min(_heading_range_span(arm["heading_deg_min"], arm["heading_deg_max"]), HEADING_FAN_SPAN_CAP_DEG)
    return float(v_extent_px) * math.tan(math.radians(span / 2.0))


def _shift_heading(arm, delta_deg):
    """Rotate heading_deg (and a legacy min/max range, if present) by the same delta."""
    arm["heading_deg"] = _normalize_heading_deg(_arm_heading(arm) + delta_deg)
    if arm.get("heading_deg_min") is not None and arm.get("heading_deg_max") is not None:
        arm["heading_deg_min"] = _normalize_heading_deg(arm["heading_deg_min"] + delta_deg)
        arm["heading_deg_max"] = _normalize_heading_deg(arm["heading_deg_max"] + delta_deg)


def _heading_of_image_vec(dx, dy):
    """Heading of an image-space vector (x right, y down; 0°=right, 90°=up)."""
    return math.degrees(math.atan2(-float(dy), float(dx))) % 360.0


def _signed_heading_delta(from_deg, to_deg):
    return (_normalize_heading_deg(to_deg) - _normalize_heading_deg(from_deg) + 180.0) % 360.0 - 180.0


def _through_road_headings(heading_a, heading_b):
    """Return two outward headings on the same axis, exactly 180° apart."""
    a0 = math.radians(_normalize_heading_deg(heading_a))
    a1 = math.radians(_normalize_heading_deg(heading_b) + 180.0)
    axis = math.degrees(math.atan2(math.sin(a0) + math.sin(a1), math.cos(a0) + math.cos(a1))) % 360.0
    return axis, (axis + 180.0) % 360.0


OFFSET_SIGN_DISAGREE_MIN_DEG = 2.0
MOUTH_GEOMETRY_RETRY_LIMIT = 1
THROUGH_ROAD_HEADING_RETRY_DEG = 15.0  # deviation from 180° that triggers a through-road retry
THROUGH_ROAD_WIDTH_RETRY_RATIO = 1.5  # max/min mouth width that triggers a through-road retry
THROUGH_ROAD_RETRY_LIMIT = 1


def _arm_heading_offset(arm):
    if arm.get("heading_offset_deg") is None:
        return None
    return _signed_heading_delta(0.0, arm["heading_offset_deg"])


def _undo_heading_offset(arm):
    """Restore heading_deg to the pre-refine value so a new offset can be applied."""
    offset = _arm_heading_offset(arm)
    if offset:
        _shift_heading(arm, -offset)
    arm["heading_offset_deg"] = 0.0
    return arm


def _arms_by_id(result):
    return {arm["id"]: arm for arm in result.get("arms") or []}


def _iter_through_road_pairs(result):
    """Yield (arm_a, arm_b) for each two-arm through-road group."""
    arms_by_id = _arms_by_id(result)
    for through_road in result.get("through_roads") or []:
        arm_ids = through_road.get("arm_ids") or []
        if len(arm_ids) != 2:
            continue
        arm_a = arms_by_id.get(arm_ids[0])
        arm_b = arms_by_id.get(arm_ids[1])
        if arm_a is None or arm_b is None:
            continue
        yield arm_a, arm_b


def drop_disagreeing_through_road_offsets(result):
    """Undo stage-2 residuals when a through-road pair tilts opposite ways."""
    for arm_a, arm_b in _iter_through_road_pairs(result):
        da, db = _arm_heading_offset(arm_a), _arm_heading_offset(arm_b)
        if da is None or db is None:
            continue
        if da * db >= 0.0:
            continue
        if abs(da) < OFFSET_SIGN_DISAGREE_MIN_DEG or abs(db) < OFFSET_SIGN_DISAGREE_MIN_DEG:
            continue
        _shift_heading(arm_a, -da)
        _shift_heading(arm_b, -db)
        arm_a["heading_offset_deg"] = 0.0
        arm_b["heading_offset_deg"] = 0.0
    return result


def align_through_road_headings(result):
    for arm_a, arm_b in _iter_through_road_pairs(result):
        old_mid_a = _arm_heading(arm_a)
        old_mid_b = _arm_heading(arm_b)
        new_mid_a, new_mid_b = _through_road_headings(old_mid_a, old_mid_b)
        _shift_heading(arm_a, _signed_heading_delta(old_mid_a, new_mid_a))
        _shift_heading(arm_b, _signed_heading_delta(old_mid_b, new_mid_b))
    return result


def unify_through_road_widths(result):
    """Set both ends of each through-road pair to the wider reported width."""
    for arm_a, arm_b in _iter_through_road_pairs(result):
        wa, wb = arm_a.get("width"), arm_b.get("width")
        if wa is None or wb is None:
            continue
        width = max(float(wa), float(wb))
        arm_a["width"] = width
        arm_b["width"] = width
    return result


def align_through_road_centerlines(result):
    """Shift through-road crop-origin points laterally onto one shared centerline.

    Uses the mouth midpoint when present, otherwise the stage-1 junction center. Headings
    stay 180° apart; each point keeps its station along the heading.
    """
    for arm_a, arm_b in _iter_through_road_pairs(result):
        e_u = _heading_unit(_arm_heading(arm_a) + 90.0)
        ka, kb = _arm_origin_key(arm_a), _arm_origin_key(arm_b)
        pa, pb = _arm_origin(arm_a), _arm_origin(arm_b)
        mx, my = (pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0
        da = (pa[0] - mx) * e_u[0] + (pa[1] - my) * e_u[1]
        db = (pb[0] - mx) * e_u[0] + (pb[1] - my) * e_u[1]
        arm_a[ka] = [pa[0] - da * e_u[0], pa[1] - da * e_u[1]]
        arm_b[kb] = [pb[0] - db * e_u[0], pb[1] - db * e_u[1]]
    return result


def finalize_through_road_geometry(result, *, drop_offsets=False):
    """Align through-road heading and mouth edges after stage-2 LLM output."""
    if drop_offsets:
        drop_disagreeing_through_road_offsets(result)
    align_through_road_headings(result)
    if any(_has_mouth_edges(arm) for arm in result.get("arms") or []):
        align_through_road_mouth_edges(result)
        apply_mouth_geometry(result)
    else:
        unify_through_road_widths(result)
        align_through_road_centerlines(result)
    return result


def _has_mouth_edges(arm):
    edges = arm.get("mouth_edges") or []
    if len(edges) != 2 or edges[0] is None or edges[1] is None:
        return False
    return len(edges[0]) >= 2 and len(edges[1]) >= 2


def _mouth_edge_pair(arm):
    if not _has_mouth_edges(arm):
        raise ValueError("arm %s needs mouth_edges with exactly two [x, y] points" % arm.get("id"))
    p0, p1 = arm["mouth_edges"][0], arm["mouth_edges"][1]
    return (float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))


def _xy_field(arm, key):
    pt = arm.get(key)
    if pt is None or len(pt) < 2:
        raise ValueError("arm %s has no %s" % (arm.get("id"), key))
    return float(pt[0]), float(pt[1])


def _arm_origin_key(arm):
    """Mouth center after refine, else the stage-1 junction center used to place the crop."""
    if _has_mouth_edges(arm) or (
        arm.get("arm_mouth_center") is not None and len(arm.get("arm_mouth_center") or []) >= 2
    ):
        return "arm_mouth_center"
    for key in ("junction_center", "seed_point", "anchor_point"):
        pt = arm.get(key)
        if pt is not None and len(pt) >= 2:
            return key
    raise ValueError("arm %s has no mouth, junction_center, seed_point, or anchor_point" % arm.get("id"))


def _arm_origin(arm):
    if _has_mouth_edges(arm):
        return mouth_center_of(arm)
    return _xy_field(arm, _arm_origin_key(arm))


def _mouth_crop_left_right(arm):
    """Return ((left_idx, left_xy), (right_idx, right_xy)) in heading-aligned crop axes.

    Crop +x is heading+90. Visual left of that crop is −e_u; visual right is +e_u.
    """
    p0, p1 = _mouth_edge_pair(arm)
    e_u = _heading_unit(_arm_heading(arm) + 90.0)
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    t0 = (p0[0] - mx) * e_u[0] + (p0[1] - my) * e_u[1]
    if t0 >= 0.0:
        return (1, p1), (0, p0)
    return (0, p0), (1, p1)


def _stamp_mouth_fields(arm):
    """Write arm_mouth_center and width from the current mouth_edges."""
    p0, p1 = _mouth_edge_pair(arm)
    arm["arm_mouth_center"] = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0]
    arm["width"] = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    return arm


def _mouth_left_right(arm):
    """Return ((left_idx, left_xy), (right_idx, right_xy)) in outward-heading left/right."""
    p0, p1 = _mouth_edge_pair(arm)
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    mid = _arm_heading(arm)
    left_ux, left_uy = _heading_unit(mid + 90.0)
    a_is_left = (p0[0] - mx) * left_ux + (p0[1] - my) * left_uy >= 0.0
    if a_is_left:
        return (0, p0), (1, p1)
    return (1, p1), (0, p0)


def _project_st(pt, origin, axis_u, lat_u):
    dx, dy = pt[0] - origin[0], pt[1] - origin[1]
    return dx * axis_u[0] + dy * axis_u[1], dx * lat_u[0] + dy * lat_u[1]


def _from_st(s, t, origin, axis_u, lat_u):
    return [
        origin[0] + s * axis_u[0] + t * lat_u[0],
        origin[1] + s * axis_u[1] + t * lat_u[1],
    ]


def _align_shared_curb(pa, pb, origin, axis_u, lat_u, *, pick_max):
    sa, ta = _project_st(pa, origin, axis_u, lat_u)
    sb, tb = _project_st(pb, origin, axis_u, lat_u)
    t = max(ta, tb) if pick_max else min(ta, tb)
    return _from_st(sa, t, origin, axis_u, lat_u), _from_st(sb, t, origin, axis_u, lat_u)


def align_through_road_mouth_edges(result):
    """Align corresponding curb coordinates of each through-road pair to the outer extent.

    Opposite arms share two physical outer curbs: this arm's left continues as the
    peer's right. For each pair, use the more extreme lateral coordinate (max for the
    left curb, min for the right curb) instead of averaging, so the aligned mouth spans
    the widest consistent outer boundaries. Each point keeps its station along the road.
    """
    for arm_a, arm_b in _iter_through_road_pairs(result):
        try:
            (i_al, p_al), (i_ar, p_ar) = _mouth_left_right(arm_a)
            (i_bl, p_bl), (i_br, p_br) = _mouth_left_right(arm_b)
        except ValueError:
            continue
        mid_a = _arm_heading(arm_a)
        axis_u = _heading_unit(mid_a)
        lat_u = _heading_unit(mid_a + 90.0)
        ca, cb = mouth_center_of(arm_a), mouth_center_of(arm_b)
        origin = ((ca[0] + cb[0]) / 2.0, (ca[1] + cb[1]) / 2.0)

        new_al, new_br = _align_shared_curb(p_al, p_br, origin, axis_u, lat_u, pick_max=True)
        new_ar, new_bl = _align_shared_curb(p_ar, p_bl, origin, axis_u, lat_u, pick_max=False)
        edges_a = [list(p) for p in _mouth_edge_pair(arm_a)]
        edges_b = [list(p) for p in _mouth_edge_pair(arm_b)]
        edges_a[i_al], edges_a[i_ar] = new_al, new_ar
        edges_b[i_bl], edges_b[i_br] = new_bl, new_br
        arm_a["mouth_edges"] = edges_a
        arm_b["mouth_edges"] = edges_b
    return result


def mouth_center_of(arm):
    p0, p1 = _mouth_edge_pair(arm)
    return (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0


def _junction_center(arms):
    if not arms:
        return None
    n = float(len(arms))
    return (
        sum(mouth_center_of(arm)[0] for arm in arms) / n,
        sum(mouth_center_of(arm)[1] for arm in arms) / n,
    )


def validate_refine_box(parsed, arm_id, *, min_width=0.05, min_length=0.08):
    """Validate stage-2 mouth_edges and length in crop space."""
    issues = []
    edges = parsed.get("mouth_edges") or []
    if len(edges) != 2:
        issues.append("arm %s: mouth_edges must be two [x, y] points in this crop" % arm_id)
    else:
        pts = []
        for i, pt in enumerate(edges):
            if pt is None or len(pt) < 2:
                issues.append("arm %s: mouth_edges[%s] must be [x, y] in this crop" % (arm_id, i))
                continue
            x, y = float(pt[0]), float(pt[1])
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                issues.append(
                    "arm %s: mouth_edges[%s] (%.3f, %.3f) is outside the crop [0, 1]×[0, 1]"
                    % (arm_id, i, x, y)
                )
            pts.append((x, y))
        if len(pts) == 2:
            lateral = abs(pts[1][0] - pts[0][0])
            if lateral < min_width:
                issues.append(
                    "arm %s: mouth_edges lateral span %.3f is below minimum %.3f of crop width"
                    % (arm_id, lateral, min_width)
                )
    try:
        length = float(parsed["length"])
    except (KeyError, TypeError, ValueError):
        issues.append("arm %s: length is required" % arm_id)
    else:
        if length < min_length:
            issues.append(
                "arm %s: length %.3f is below minimum %.3f of crop height" % (arm_id, length, min_length)
            )
    return issues


def _format_retry_text(title, issues, *, instructions, schema_hint):
    bullet = "\n".join(f"- {issue}" for issue in (issues or []) if issue)
    return (
        f"{title}\n\n"
        "Violations:\n"
        f"{bullet or '- (unspecified)'}\n\n"
        f"{instructions}\n\n"
        f"{schema_hint}"
    )


_BOX_REFINE_INSTRUCTIONS = (
    "Report geometry only for the arm in the LOWER half of this crop "
    "(not the opposite through-road arm that may appear in the upper half). "
    "mouth_edges: two points on the outer motor-vehicle boundaries of the complete "
    "junction-entry mouth, including a channelized slip lane of this arm "
    "(near the crop top). heading_offset_deg from the downward crop axis, "
    "and length as a fraction of crop height from the mouth midpoint down the arm. "
    "Do not report an anchor or width."
)


def format_refine_box_retry_text(issues):
    return _format_retry_text(
        "ARM MOUTH VALIDATION FAILED — resubmit corrected mouth_edges/heading/length for this arm.",
        issues,
        instructions=_BOX_REFINE_INSTRUCTIONS,
        schema_hint="Return only the corrected schema object for this same request.",
    )


def _arm_width(arm):
    if arm.get("width") is not None:
        return float(arm["width"])
    if not _has_mouth_edges(arm):
        return None
    p0, p1 = _mouth_edge_pair(arm)
    return math.hypot(p1[0] - p0[0], p1[1] - p0[1])


def _through_road_heading_gap_deg(arm_a, arm_b):
    """Degrees away from a perfect through-road (opposite headings). 0 = 180° apart."""
    opposite = _normalize_heading_deg(_arm_heading(arm_a) + 180.0)
    return abs(_signed_heading_delta(_arm_heading(arm_b), opposite))


def _through_road_width_ratio(arm_a, arm_b):
    wa, wb = _arm_width(arm_a), _arm_width(arm_b)
    if wa is None or wb is None:
        return None
    lo, hi = min(wa, wb), max(wa, wb)
    if lo <= 1e-6:
        return float("inf") if hi > 1e-6 else 1.0
    return hi / lo


def through_road_pair_issues(
    arm_a, arm_b,
    *, heading_tol_deg=THROUGH_ROAD_HEADING_RETRY_DEG,
    width_ratio=THROUGH_ROAD_WIDTH_RETRY_RATIO,
):
    """Return human-readable mismatches for a through-road pair, or []."""
    issues = []
    gap = _through_road_heading_gap_deg(arm_a, arm_b)
    if gap > float(heading_tol_deg):
        issues.append(
            "arms %s/%s headings %.1f° and %.1f° are %.1f° off opposite (limit %.1f°)"
            % (arm_a.get("id"), arm_b.get("id"), _arm_heading(arm_a), _arm_heading(arm_b),
               gap, heading_tol_deg)
        )
    if not (_has_mouth_edges(arm_a) and _has_mouth_edges(arm_b)):
        return issues
    ratio = _through_road_width_ratio(arm_a, arm_b)
    if ratio is not None and ratio > float(width_ratio):
        issues.append(
            "arms %s/%s widths %.3f and %.3f differ by ratio %.2f (limit %.2f)"
            % (arm_a.get("id"), arm_b.get("id"), _arm_width(arm_a), _arm_width(arm_b),
               ratio, width_ratio)
        )
    return issues


def _snapshot_arm(arm):
    return copy.deepcopy(arm)


def _restore_arm(arm, backup):
    arm.clear()
    arm.update(copy.deepcopy(backup))


def apply_mouth_geometry(result):
    """Stamp arm_mouth_center and width from unordered mouth_edges (e2e / legacy)."""
    for arm in result.get("arms") or []:
        if not _has_mouth_edges(arm):
            continue
        _stamp_mouth_fields(arm)
    return result


def finalize_arm_geometry(result, *, snap_heading=True, align_headings=True):
    """Post-process mouth-based arm geometry (e2e / legacy)."""
    result = align_through_road_mouth_edges(result)
    result = apply_mouth_geometry(result)
    if snap_heading:
        result = snap_heading_to_mouth(result)
    if align_headings:
        result = align_through_road_headings(result)
    return result


_ARM_OVERLAY_COLORS = (
    (255, 64, 64),
    (32, 140, 255),
    (32, 190, 90),
    (255, 170, 24),
    (190, 80, 220),
    (24, 190, 190),
)


def _mouth_outward_heading(p0, p1, junction_xy):
    """Outward heading = mouth-chord perpendicular pointing away from the junction."""
    chord = _heading_of_image_vec(p1[0] - p0[0], p1[1] - p0[1])
    n1 = (chord + 90.0) % 360.0
    n2 = (chord - 90.0) % 360.0
    mx, my = (p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0
    away = _heading_of_image_vec(mx - junction_xy[0], my - junction_xy[1])
    if abs(_signed_heading_delta(away, n1)) <= abs(_signed_heading_delta(away, n2)):
        return n1
    return n2


def snap_heading_to_mouth(result):
    """Set each arm's heading_deg to the mouth outward normal."""
    arms = result.get("arms") or []
    center = _junction_center(arms)
    if center is None:
        return result
    for arm in arms:
        if not _has_mouth_edges(arm):
            continue
        p0, p1 = _mouth_edge_pair(arm)
        target = _mouth_outward_heading(p0, p1, center)
        _shift_heading(arm, _signed_heading_delta(_arm_heading(arm), target))
    return result


def _heading_unit(heading_deg):
    """Image-space unit vector for an outward heading (0°=right, 90°=up)."""
    theta = math.radians(_normalize_heading_deg(heading_deg))
    return math.cos(theta), -math.sin(theta)


DEFAULT_WIDTH_EXT_FRAC = 0.6
DEFAULT_STAGE1_WIDTH_FRAC = 0.3
DEFAULT_LOOKBACK_FRAC = 0.10  # opposite-heading pad from origin, fraction of image side
DEFAULT_MIN_WIDTH_EXT_CM = 0.5
DEFAULT_IMAGE_DPI = 300.0
CROP_LOOKBACK_FRAC = 0.2
MIN_ARM_LENGTH_PX = 48.0
_CM_PER_INCH = 2.54


def paper_cm_to_px(min_cm=DEFAULT_MIN_WIDTH_EXT_CM, dpi=DEFAULT_IMAGE_DPI):
    """Convert paper centimeters into processed-image pixels (1:1 with the scan)."""
    if min_cm is None or float(min_cm) <= 0:
        return 0.0
    return float(min_cm) / _CM_PER_INCH * float(dpi)


def _ensure_norm_xy(pt, width, height, *, independent=False):
    """Coerce a point into [0, 1]. Pixel coords (max abs > 1.5) are divided by size.

    Full-image points use the square-side ``max(width, height)``. Crop points pass
    ``independent=True`` so x is divided by width and y by height.
    """
    x, y = float(pt[0]), float(pt[1])
    if max(abs(x), abs(y)) <= 1.5:
        return [x, y]
    if independent:
        return [x / float(width), y / float(height)]
    side = float(max(width, height))
    return [x / side, y / side]


def _ensure_norm_arm_xy(arm, width, height):
    """Rewrite junction / mouth / center into [0, 1]; restamp width from mouth_edges if present."""
    for key in ("junction_center", "seed_point", "anchor_point", "arm_mouth_center"):
        pt = arm.get(key)
        if pt is not None and len(pt) >= 2:
            arm[key] = _ensure_norm_xy(pt, width, height)
    if _has_mouth_edges(arm):
        arm["mouth_edges"] = [_ensure_norm_xy(p, width, height) for p in arm["mouth_edges"]]
        _stamp_mouth_fields(arm)
    elif arm.get("width") is not None and float(arm["width"]) > 1.5:
        arm["width"] = float(arm["width"]) / float(max(width, height))
    return arm


def _norm_to_px(point, width, height):
    side = float(max(width, height))
    x, y = _ensure_norm_xy(point, width, height)
    return x * side, y * side


def _heading_frame_image_extents(origin_px, heading_deg, width, height):
    """Min/max (v along heading, u lateral) of the image rectangle about origin_px.

    The four corners bound any linear projection, so this covers all image content
    along the heading — not just the first time the centerline hits an edge.
    """
    e_v = _heading_unit(heading_deg)
    e_u = _heading_unit(heading_deg + 90.0)
    ox, oy = origin_px
    vs, us = [], []
    for x, y in (
        (0.0, 0.0),
        (float(width), 0.0),
        (float(width), float(height)),
        (0.0, float(height)),
    ):
        dx, dy = x - ox, y - oy
        vs.append(dx * e_v[0] + dy * e_v[1])
        us.append(dx * e_u[0] + dy * e_u[1])
    return min(vs), max(vs), min(us), max(us)


def _image_span_cap_px(width, height):
    """Longest crop side needed to cover the image (diagonal + 5%)."""
    return math.hypot(float(width), float(height)) * 1.05


def _clamp_heading_span(lookback_px, length_px, width_px, cap_px):
    """Keep a heading-aligned crop from exceeding the image diagonal."""
    cap = max(1.0, float(cap_px))
    width_px = min(max(1.0, float(width_px)), cap)
    lookback_px = max(0.0, float(lookback_px))
    length_px = max(MIN_ARM_LENGTH_PX, float(length_px))
    span_v = lookback_px + length_px
    if span_v > cap:
        scale = cap / span_v
        lookback_px *= scale
        length_px *= scale
    return lookback_px, length_px, width_px


def arm_stage1_crop_region(
    arm, width, height,
    *, width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
    lookback_frac=DEFAULT_LOOKBACK_FRAC,
):
    """Heading-aligned strip from the junction center along the arm.

    Junction center is the crop origin (vertex). Length runs from the center to
    the far image edge along heading. A short lookback extends opposite the heading.
    """
    side = float(max(width, height))
    center = _norm_to_px(_arm_origin(arm), width, height)
    hmid = _arm_heading(arm)
    e_v = _heading_unit(hmid)
    e_u = _heading_unit(hmid + 90.0)
    _v_min, v_max, _u_min, _u_max = _heading_frame_image_extents(center, hmid, width, height)
    wfrac = float(arm["width"]) if arm.get("width") is not None else float(width_frac)
    width_px = max(1.0, wfrac * side)
    lookback_px = max(0.0, float(lookback_frac) * side)
    length_px = max(MIN_ARM_LENGTH_PX, float(v_max))
    lookback_px, length_px, width_px = _clamp_heading_span(
        lookback_px, length_px, width_px, _image_span_cap_px(width, height),
    )
    half_u = width_px / 2.0
    return {
        "heading_deg": hmid,
        "center": center,
        "e_u": e_u,
        "e_v": e_v,
        "u0": -half_u,
        "v0": -lookback_px,
        "span_u": width_px,
        "span_v": lookback_px + length_px,
        "length_px": length_px,
    }


def apply_stage1_localization(
    result, image_size,
    *, width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
):
    """Stamp derived width/length/lookback from junction_center + heading."""
    width, height = image_size
    side = float(max(width, height))
    center = result.get("junction_center")
    if center is None or len(center) < 2:
        for arm in result.get("arms") or []:
            pt = arm.get("junction_center") or arm.get("seed_point")
            if pt is not None and len(pt) >= 2:
                center = pt
                break
    if center is None or len(center) < 2:
        raise ValueError("stage 1 needs junction_center")
    result["junction_center"] = _ensure_norm_xy(center, width, height)
    for arm in result.get("arms") or []:
        arm["junction_center"] = list(result["junction_center"])
        arm.pop("seed_point", None)
        _ensure_norm_arm_xy(arm, width, height)
        arm.pop("width", None)
        region = arm_stage1_crop_region(arm, width, height, width_frac=width_frac)
        arm["width"] = float(width_frac)
        arm["length"] = region["length_px"] / side
        arm["lookback"] = DEFAULT_LOOKBACK_FRAC
    return result


def arm_crop_region(
    arm, width, height,
    *, width_ext_frac=0.0, lookback_frac=None, min_width_ext_px=None,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
):
    """Heading-aligned rectangle from heading / junction-or-mouth / width / length.

    A stage-1 arm (no refined ``mouth_edges``) is cropped from the shared
    ``junction_center`` along the heading, with a short lookback opposite the heading.
    After stage 2, the origin is the mouth midpoint, ``length`` is from that mouth
    along the arm, and a lookback extends opposite the heading into the junction.
    ``lookback_frac`` (lane-level) is a fraction of that length;
    otherwise stored ``lookback`` (or ``DEFAULT_LOOKBACK_FRAC``) is a fraction
    of the image side. ``width_ext_frac`` expands the stored mouth width for
    crop/overlay only (0.3 = 30% wider). It is not written back onto the arm.
    """
    side = float(max(width, height))
    center = _norm_to_px(_arm_origin(arm), width, height)
    hmid = _arm_heading(arm)
    e_v = _heading_unit(hmid)
    e_u = _heading_unit(hmid + 90.0)
    _v_min, v_max, _u_min, _u_max = _heading_frame_image_extents(center, hmid, width, height)

    if arm.get("width") is not None:
        wfrac = float(arm["width"])
    else:
        wfrac = float(stage1_width_frac)

    if (
        (_has_mouth_edges(arm) or arm.get("arm_mouth_center") is not None or arm.get("anchor_point") is not None)
        and arm.get("length") is not None
    ):
        length_px = max(MIN_ARM_LENGTH_PX, float(arm["length"]) * side)
    else:
        length_px = max(MIN_ARM_LENGTH_PX, float(v_max))
    if lookback_frac is not None:
        lookback_px = max(0.0, float(lookback_frac) * length_px)
    elif arm.get("lookback") is not None:
        lookback_px = max(0.0, float(arm["lookback"]) * side)
    else:
        lookback_px = max(0.0, DEFAULT_LOOKBACK_FRAC * side)

    true_width_px = max(1.0, wfrac * side)
    extra = true_width_px * float(width_ext_frac)
    if float(width_ext_frac) > 0.0:
        floor = paper_cm_to_px() if min_width_ext_px is None else float(min_width_ext_px or 0.0)
        extra = max(extra, 2.0 * floor)
    width_px = max(1.0, true_width_px + extra)
    extra_u = _heading_fan_extra_u(max(lookback_px, length_px), arm)
    width_px += 2.0 * extra_u
    lookback_px, length_px, width_px = _clamp_heading_span(
        lookback_px, length_px, width_px, _image_span_cap_px(width, height),
    )
    half_u = width_px / 2.0
    return {
        "heading_deg": hmid,
        "center": center,
        "e_u": e_u,
        "e_v": e_v,
        "u0": -half_u,
        "v0": -lookback_px,
        "span_u": width_px,
        "true_span_u": true_width_px,
        "span_v": lookback_px + length_px,
        "length_px": length_px,
    }


def _region_point(region, u, v):
    e_u, e_v = region["e_u"], region["e_v"]
    cx, cy = region["center"]
    return cx + u * e_u[0] + v * e_v[0], cy + u * e_u[1] + v * e_v[1]


def _region_corners(region, span_u=None):
    v0, sv = region["v0"], region["span_v"]
    if span_u is None:
        u0, su = region["u0"], region["span_u"]
    else:
        su = float(span_u)
        u0 = -su / 2.0
    return [
        _region_point(region, u0, v0),
        _region_point(region, u0 + su, v0),
        _region_point(region, u0 + su, v0 + sv),
        _region_point(region, u0, v0 + sv),
    ]


def _draw_dashed_polygon(draw, points, *, fill, width, dash, gap):
    closed = list(points) + [points[0]]
    for p0, p1 in zip(closed, closed[1:]):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        pos, on = 0.0, True
        while pos < length:
            seg = dash if on else gap
            nxt = min(length, pos + seg)
            if on:
                draw.line(
                    [(x0 + ux * pos, y0 + uy * pos), (x0 + ux * nxt, y0 + uy * nxt)],
                    fill=fill, width=width,
                )
            pos = nxt
            on = not on


def road_frame_crop(original, region):
    e_u, e_v = region["e_u"], region["e_v"]
    u0, v0 = region["u0"], region["v0"]
    ox, oy = _region_point(region, u0, v0)
    affine = (e_u[0], e_v[0], ox, e_u[1], e_v[1], oy)
    size = (max(1, round(region["span_u"])), max(1, round(region["span_v"])))
    if (size[0] * size[1]) / 1e6 > 40.0:
        raise ValueError(
            "arm crop %sx%s (%.1f MP) exceeds 40 MP; junction_center/heading likely out of range"
            % (size[0], size[1], (size[0] * size[1]) / 1e6)
        )
    image = original.transform(
        size, Image.Transform.AFFINE, affine,
        resample=Image.Resampling.BILINEAR, fillcolor=(192, 192, 192),
    )
    image = _paint_crop_outside_original(image, affine, original.size)
    return image, affine


def _src_px_to_crop_px(x_src, y_src, affine):
    a, b, c, d, e, f = affine
    dx, dy = float(x_src) - c, float(y_src) - f
    return a * dx + d * dy, b * dx + e * dy


def _paint_crop_outside_original(image, affine, orig_size, fill=(192, 192, 192)):
    """Gray-out crop pixels whose source lies outside the original image."""
    ow, oh = orig_size
    quad = [
        _src_px_to_crop_px(0, 0, affine),
        _src_px_to_crop_px(ow, 0, affine),
        _src_px_to_crop_px(ow, oh, affine),
        _src_px_to_crop_px(0, oh, affine),
    ]
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).polygon(quad, fill=255)
    pad = Image.new("RGB", image.size, fill)
    return Image.composite(image.convert("RGB"), pad, mask)


def crop_arm_image(
    original, arm,
    *, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
):
    region = arm_crop_region(
        arm, original.width, original.height,
        width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
        stage1_width_frac=stage1_width_frac,
    )
    image, affine = road_frame_crop(original, region)
    return {"image": image, "affine": affine, "region": region, "size": image.size}


def _crop_norm_to_full_norm(pt, affine, crop_size, full_size):
    """Crop [0, 1]×[0, 1] (independent axes) → full-image square-side normalized xy."""
    cw, ch = crop_size
    side = float(max(full_size))
    x, y = _ensure_norm_xy(pt, cw, ch, independent=True)
    x_c = x * float(cw)
    y_c = y * float(ch)
    a, b, c, d, e, f = affine
    x_src = a * x_c + b * y_c + c
    y_src = d * x_c + e * y_c + f
    return [x_src / side, y_src / side]


def apply_crop_refine(arm, refine, crop, full_size):
    """Write refined mouth_edges / heading / length from the crop into full-image coords.

    Width is the span between the two mapped mouth edges. Stage-1 ``junction_center``
    is left unchanged.
    """
    affine, size = crop["affine"], crop["size"]
    ch = float(size[1])
    side = float(max(full_size))
    arm["mouth_edges"] = [
        _crop_norm_to_full_norm(pt, affine, size, full_size) for pt in refine["mouth_edges"]
    ]
    offset = _signed_heading_delta(0.0, refine["heading_offset_deg"])
    arm["heading_offset_deg"] = offset
    _shift_heading(arm, offset)
    _stamp_mouth_fields(arm)
    _ensure_norm_arm_xy(arm, full_size[0], full_size[1])
    arm["length"] = max(1e-6, float(refine["length"]) * ch / side)
    arm["lookback"] = DEFAULT_LOOKBACK_FRAC
    arm.pop("anchor_point", None)
    if refine.get("evidence"):
        arm["evidence"] = refine["evidence"]
    if refine.get("uncertainty") is not None:
        arm["uncertainty"] = refine["uncertainty"]
    return arm


def _overlay_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_label(draw, xy, text, *, font, fill, stroke):
    x, y = xy
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + dx, y + dy), text, font=font, fill=stroke)
    draw.text((x, y), text, font=font, fill=fill)


def render_arm_overlay(
    image, result,
    *, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC, show_seed=True,
):
    """Draw each arm as a heading-aligned rectangle from heading / junction-or-mouth / width / length.

    Boxes are drawn first; point markers are drawn last so they stay visible.
    Stage 2 overlays pass ``show_seed=False`` and draw mouth edges only.
    Stage 1 overlays mark the shared junction_center.
    """
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    width, height = canvas.size
    line_w = max(2, round(min(width, height) / 450))
    dash = max(10, round(min(width, height) / 80))
    gap = max(6, round(dash * 0.55))
    font = _overlay_font(max(16, round(min(width, height) / 42)))
    edge_r = max(6, round(min(width, height) / 110))
    mid_r = max(5, round(edge_r * 0.7))
    showed_extra = False
    point_marks = []

    for i, arm in enumerate(result.get("arms") or []):
        arm = copy.deepcopy(arm)
        _ensure_norm_arm_xy(arm, width, height)
        rgb = _ARM_OVERLAY_COLORS[i % len(_ARM_OVERLAY_COLORS)]
        region = arm_crop_region(
            arm, width, height,
            width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
            stage1_width_frac=stage1_width_frac,
        )
        extra_corners = _region_corners(region)
        true_span = float(region.get("true_span_u") or region["span_u"])
        true_corners = _region_corners(region, true_span)
        has_extra = region["span_u"] > true_span + 1.0
        length_px = region["length_px"]
        mid = _arm_heading(arm)
        ux, uy = _heading_unit(mid)
        if has_extra:
            draw.polygon(extra_corners, fill=(*rgb, 36))
            _draw_dashed_polygon(
                draw, extra_corners, fill=(*rgb, 200), width=max(1, line_w - 1), dash=dash, gap=gap,
            )
            showed_extra = True
        draw.polygon(true_corners, fill=(*rgb, 72))
        draw.line([*true_corners, true_corners[0]], fill=(*rgb, 230), width=line_w)
        ox, oy = _norm_to_px(_arm_origin(arm), width, height)
        lookback_px = -region["v0"]
        bx, by = ox - ux * lookback_px, oy - uy * lookback_px
        ex, ey = ox + ux * length_px, oy + uy * length_px
        draw.line([(bx, by), (ex, ey)], fill=(*rgb, 230), width=max(line_w + 1, round(line_w * 1.6)))
        _draw_label(
            draw, (ex + 6, ey - 8), f"arm {arm.get('id')}",
            font=font, fill=(*rgb, 255), stroke=(0, 0, 0, 220),
        )
        if _has_mouth_edges(arm):
            (i_l, p_l), (i_r, p_r) = _mouth_crop_left_right(arm)
            lx, ly = _norm_to_px(p_l, width, height)
            rx, ry = _norm_to_px(p_r, width, height)
            draw.line([(lx, ly), (rx, ry)], fill=(*rgb, 230), width=max(line_w + 1, round(line_w * 1.6)))
            point_marks.append((rgb, "L", arm.get("id"), (lx, ly)))
            point_marks.append((rgb, "R", arm.get("id"), (rx, ry)))

    if show_seed:
        center_pt = result.get("junction_center")
        if center_pt is None or len(center_pt) < 2:
            for arm in result.get("arms") or []:
                pt = arm.get("junction_center") or arm.get("seed_point")
                if pt is not None and len(pt) >= 2:
                    center_pt = pt
                    break
        if center_pt is not None and len(center_pt) >= 2:
            point_marks.append(
                ((20, 20, 20), "center", None, _norm_to_px(center_pt, width, height))
            )

    legend_font = _overlay_font(max(14, round(min(width, height) / 55)))
    _draw_width_legend(
        draw, (12, 12), font=legend_font, line_w=line_w, dash=dash, gap=gap, show_extra=showed_extra,
    )
    for rgb, name, arm_id, (ax, ay) in point_marks:
        _draw_overlay_point(
            draw, (ax, ay), rgb, name=name, arm_id=arm_id,
            r=mid_r, line_w=line_w, label_gap=edge_r, font=font,
        )
    return Image.alpha_composite(canvas, layer).convert("RGB")


def _draw_overlay_point(draw, xy, rgb, *, name, arm_id, r, line_w, label_gap, font):
    ax, ay = xy
    outline = (*rgb, 255)
    fill = (255, 255, 255, 255)
    if name in ("seed", "center"):
        draw.rectangle(
            (ax - r, ay - r, ax + r, ay + r),
            fill=fill, outline=outline, width=max(2, line_w),
        )
        lx, ly = ax + label_gap + 4, ay + 4
    else:
        draw.ellipse(
            (ax - r, ay - r, ax + r, ay + r),
            fill=fill, outline=outline, width=max(2, line_w),
        )
        ly = ay + 4 if name in ("L", "R") else ay - label_gap - 6
        lx = ax + label_gap + 4
    label = name if arm_id is None else f"{name} {arm_id}"
    _draw_label(
        draw, (lx, ly), label,
        font=font, fill=(*rgb, 255), stroke=(0, 0, 0, 220),
    )


def _draw_width_legend(draw, origin, *, font, line_w, dash, gap, show_extra):
    x, y = origin
    sw, sh = 34, 16
    pad, gap_y, gap_x = 8, 6, 8
    rows = [("width", False)]
    if show_extra:
        rows.append(("extra", True))
    label_w = 0
    for label, _dashed in rows:
        bbox = draw.textbbox((0, 0), label, font=font)
        label_w = max(label_w, bbox[2] - bbox[0])
    box_h = pad * 2 + len(rows) * sh + max(0, len(rows) - 1) * gap_y
    box_w = pad * 2 + sw + gap_x + label_w
    draw.rectangle((x, y, x + box_w, y + box_h), fill=(0, 0, 0, 150), outline=(255, 255, 255, 180))
    cy = y + pad
    sample = (90, 180, 255)
    for label, dashed in rows:
        rx0, ry0, rx1, ry1 = x + pad, cy, x + pad + sw, cy + sh
        draw.rectangle((rx0, ry0, rx1, ry1), fill=(*sample, 36 if dashed else 72))
        corners = [(rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1)]
        if dashed:
            _draw_dashed_polygon(draw, corners, fill=(*sample, 220), width=line_w, dash=max(6, dash // 2), gap=max(4, gap // 2))
        else:
            draw.line([*corners, corners[0]], fill=(*sample, 230), width=line_w)
        _draw_label(
            draw, (rx1 + gap_x, ry0 - 1), label,
            font=font, fill=(255, 255, 255, 255), stroke=(0, 0, 0, 220),
        )
        cy += sh + gap_y


def run_arm_level_stage(llm, image_msg, *, model, tracker, max_tokens=1024, reasoning=None, driving_side=None):
    """Stage 1: identify arms, one junction_center, and per-arm heading_deg."""
    side = normalize_driving_side(driving_side)
    system_prompt = arm_level_prompt(side)
    messages = [image_msg]
    result, _text, response = llm.call_json(
        model=model, system_prompt=system_prompt, messages=messages,
        schema=ARM_LEVEL_SCHEMA, max_tokens=max_tokens, reasoning_effort=reasoning,
    )
    tracker.record("stage1/arms", model, response)
    messages.append(llm.assistant_message(response))
    result["driving_side"] = side
    result["channelized_features"] = enrich_channelized_features(
        result.get("channelized_features"), result.get("arms"), side,
    )
    return {
        "messages": to_jsonable(messages),
        "system": system_prompt,
        "result": result,
    }


def save_arm_crops(
    output_dir, image, result,
    *, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
):
    """Write heading-aligned crops for each arm; return in-memory crop bundles keyed by arm id."""
    crop_dir = Path(output_dir) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops = {}
    for arm in result.get("arms") or []:
        arm_id = arm["id"]
        bundle = crop_arm_image(
            image, arm, width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
            stage1_width_frac=stage1_width_frac,
        )
        rel_path = Path("crops") / f"arm_{arm_id}.png"
        path = Path(output_dir) / rel_path
        bundle["image"].save(path)
        bundle["path"] = str(rel_path)
        crops[arm_id] = bundle
    return crops


def run_arm_refine_one(
    llm, arm, crop, full_size, *, model, tracker, max_tokens, reasoning,
    cost_label=None,
):
    """Fresh conversation: crop image only; refine mouth_edges, heading, and length."""
    backup = _snapshot_arm(arm)
    system_prompt = arm_refine_prompt()
    messages = [llm.image_user_message(crop["image"], arm_crop_user_text(arm.get("id")))]
    parsed = None
    last_issues = []
    for attempt in range(MOUTH_GEOMETRY_RETRY_LIMIT + 1):
        label = cost_label or "stage2/arm_%s" % arm.get("id")
        if attempt:
            label += "_retry"
        parsed, _text, response = llm.call_json(
            model=model, system_prompt=system_prompt, messages=messages,
            schema=ARM_REFINE_SCHEMA, max_tokens=max_tokens, reasoning_effort=reasoning,
        )
        tracker.record(label, model, response)
        messages.append(llm.assistant_message(response))
        cw, ch = crop["size"]
        if parsed.get("mouth_edges"):
            parsed["mouth_edges"] = [
                _ensure_norm_xy(p, cw, ch, independent=True) for p in parsed["mouth_edges"]
            ]
        if parsed.get("length") is not None and float(parsed["length"]) > 1.5:
            parsed["length"] = float(parsed["length"]) / float(ch)
        last_issues = validate_refine_box(parsed, arm.get("id"))
        _restore_arm(arm, backup)
        if not last_issues or attempt >= MOUTH_GEOMETRY_RETRY_LIMIT:
            if last_issues:
                BaseLLM.log(
                    "arm %s stage2 mouth still off after retry: %s; keeping previous geometry"
                    % (arm.get("id"), "; ".join(last_issues))
                )
            else:
                _undo_heading_offset(arm)
                apply_crop_refine(arm, parsed, crop, full_size)
            break
        messages.append(llm.make_user_message(format_refine_box_retry_text(last_issues)))
    return {
        "messages": to_jsonable(messages),
        "system": system_prompt,
        "result": parsed,
        "crop": crop.get("path"),
    }


def run_arm_refine_stage(llm, original, result, crops, *, model, tracker, max_tokens=1024, reasoning=None):
    """Per-arm crop refine of mouth_edges/heading/length; heading_offset adds onto stage-1 heading."""
    full_size = (original.width, original.height)
    stage2 = {}
    for arm in result.get("arms") or []:
        arm_id = arm["id"]
        crop = crops.get(arm_id)
        if crop is None:
            continue
        try:
            stage2[str(arm_id)] = run_arm_refine_one(
                llm, arm, crop, full_size,
                model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
            )
        except Exception as exc:
            BaseLLM.log(f"arm {arm_id} stage2 failed: {exc}")
            stage2[str(arm_id)] = {"error": str(exc), "crop": crop.get("path")}
    return stage2


def retry_through_road_refine_stage(
    llm, original, result, crops, stage2, *, model, tracker, max_tokens=1024, reasoning=None,
):
    """If a through-road pair disagrees on heading or width, redo the same crop call for each arm."""
    stage2 = dict(stage2 or {})
    full_size = (original.width, original.height)
    for round_idx in range(THROUGH_ROAD_RETRY_LIMIT):
        jobs = []
        seen = set()
        for arm_a, arm_b in _iter_through_road_pairs(result):
            issues = through_road_pair_issues(arm_a, arm_b)
            if not issues:
                continue
            BaseLLM.log(
                "through-road %s/%s mismatch (round %s): %s"
                % (arm_a.get("id"), arm_b.get("id"), round_idx + 1, "; ".join(issues))
            )
            for arm in (arm_a, arm_b):
                arm_id = arm["id"]
                if arm_id in seen:
                    continue
                seen.add(arm_id)
                jobs.append(arm)
        if not jobs:
            break
        for arm in jobs:
            arm_id = arm["id"]
            crop = crops.get(arm_id)
            if crop is None:
                continue
            label = "stage2/arm_%s_through_retry" % arm_id
            if round_idx:
                label += "_r%s" % (round_idx + 1)
            try:
                rec = run_arm_refine_one(
                    llm, arm, crop, full_size,
                    model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
                    cost_label=label,
                )
                rec["through_road_retry"] = True
                stage2[str(arm_id)] = rec
            except Exception as exc:
                BaseLLM.log("arm %s through-road retry failed: %s" % (arm_id, exc))
                prev = stage2.get(str(arm_id)) or {}
                prev["through_road_retry_error"] = str(exc)
                stage2[str(arm_id)] = prev
    leftover = []
    for arm_a, arm_b in _iter_through_road_pairs(result):
        leftover.extend(through_road_pair_issues(arm_a, arm_b))
    if leftover:
        BaseLLM.log("through-road still mismatched after retry: %s" % "; ".join(leftover))
    return stage2


def _write_overlay(
    output_dir, image, result, filename,
    *, width_ext_frac, min_width_ext_px=None,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC, show_seed=True,
):
    if image is None or not result:
        return None
    try:
        overlay = render_arm_overlay(
            image, result, width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
            stage1_width_frac=stage1_width_frac, show_seed=show_seed,
        )
        path = Path(output_dir) / filename
        overlay.save(path)
        return path
    except Exception as exc:
        BaseLLM.log(f"overlay {filename} failed: {exc}")
        return None


def save_arm_level_artifacts(
    output_dir, artifact, tracker, *, name=None, source_image=None, image=None,
    width_ext_frac=DEFAULT_WIDTH_EXT_FRAC,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
    min_width_ext_px=None, min_width_ext_cm=None, image_dpi=None,
    crops=None,
):
    paths = {}
    output_dir = Path(output_dir)
    overlay_kw = dict(
        width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
        stage1_width_frac=stage1_width_frac,
    )
    if artifact:
        stage1_result = artifact.get("stage1_result")
        conversation = {
            "system": artifact.get("system"),
            "messages": artifact.get("messages", []),
            "width_ext_frac": width_ext_frac,
            "stage1_width_frac": stage1_width_frac,
            "min_width_ext_cm": min_width_ext_cm,
            "image_dpi": image_dpi,
            "stage2": artifact.get("stage2") or {},
        }
        if name is not None:
            conversation["name"] = name
        if source_image is not None:
            conversation["source_image"] = str(source_image)
        if crops:
            conversation["crops"] = {
                str(arm_id): {"path": crop.get("path"), "affine": list(crop["affine"]), "size": list(crop["size"])}
                for arm_id, crop in crops.items()
            }
        paths["conversation"] = output_dir / "conversation.json"
        dump_json(paths["conversation"], conversation)

        if stage1_result is not None:
            paths["arms_stage1"] = output_dir / "arms_stage1.json"
            dump_json(paths["arms_stage1"], stage1_result)
            paths["overlay_stage1"] = _write_overlay(
                output_dir, image, stage1_result, "overlay_stage1.png", **overlay_kw,
            )

        paths["arms"] = output_dir / "arms.json"
        dump_json(paths["arms"], artifact["result"])

        if artifact.get("stage2") is not None:
            stage2_result = artifact.get("stage2_result") or artifact.get("result")
            paths["arms_stage2"] = output_dir / "arms_stage2.json"
            dump_json(paths["arms_stage2"], stage2_result)
            paths["overlay_stage2"] = _write_overlay(
                output_dir, image, stage2_result, "overlay_stage2.png", **overlay_kw, show_seed=False,
            )
            paths["overlay"] = _write_overlay(
                output_dir, image, artifact.get("result"), "overlay.png", **overlay_kw, show_seed=False,
            )
        elif artifact.get("stage2") is None and stage1_result is None:
            paths["overlay_stage1"] = _write_overlay(
                output_dir, image, artifact.get("result"), "overlay_stage1.png", **overlay_kw,
            )

    paths["cost"] = output_dir / "cost.json"
    dump_json(paths["cost"], tracker.summary())
    return paths


def _load_stage1_artifact(out_dir):
    """Reload a previous stage-1 result so stage 2 can run independently."""
    stage1_path = Path(out_dir) / "arms_stage1.json"
    if not stage1_path.is_file():
        raise FileNotFoundError("stage 2 needs %s" % stage1_path)
    conversation_path = Path(out_dir) / "conversation.json"
    conversation = {}
    if conversation_path.is_file():
        conversation = json.loads(conversation_path.read_text(encoding="utf-8"))
    result = json.loads(stage1_path.read_text(encoding="utf-8"))
    return {
        "system": conversation.get("system"),
        "messages": conversation.get("messages", []),
        "result": result,
        "stage1_result": copy.deepcopy(result),
    }


def run_arm_level(
    model, dataset="nhtsa_2023", case_id="28661", max_tokens=1024, compressed=True, verbose=True,
    width_ext_frac=DEFAULT_WIDTH_EXT_FRAC,
    stage1_width_frac=DEFAULT_STAGE1_WIDTH_FRAC,
    min_width_ext_cm=DEFAULT_MIN_WIDTH_EXT_CM, image_dpi=DEFAULT_IMAGE_DPI,
    artifact_case_dir=None, stage="all",
):
    if stage not in ("all", "1", "2"):
        raise ValueError("stage must be 'all', '1', or '2'")
    cfg = unpack_model_config(model)
    api, source, model_name = cfg["api"], cfg["source"], cfg["model_name"]
    api_key, base_url, proxy, reasoning = cfg["api_key"], cfg["base_url"], cfg["proxy"], cfg["reasoning"]
    side = load_case_driving_side(dataset, case_id)
    min_px = paper_cm_to_px(min_width_ext_cm, image_dpi)

    prev_verbose = set_verbose(verbose)
    _source_image, run_name = resolve_case_inputs(dataset=dataset, case_id=case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else resolve_run_dir("results", run_name)
    out_dir = prepare_output_dir(arm_level_dir(case_dir, model_name, reasoning), clear=(stage in ("all", "1")))

    processed = load_processed_image(case_dir, compressed=compressed)
    image_path = processed["path"]
    image = processed["image"]
    raw_after = processed["raw_bytes"]

    llm = build_llm(api, api_key, base_url, proxy=proxy)
    image_msg = llm.image_user_message(image, FULL_DIAGRAM_USER_TEXT)

    tracker = CostTracker(source=source, silent=True)
    BaseLLM.rule("ARM-LEVEL INFERENCE")
    parts = image_path.parts
    image_display = str(Path(*parts[parts.index("results"):])) if "results" in parts else str(image_path)
    BaseLLM.log(f"image : {image_display} ({image.width}x{image.height}px, {len(raw_after) / (1024 * 1024):.2f} MB)")
    BaseLLM.log(
        f"model : {model_name}  reasoning: {reasoning}  driving_side: {side}  stage: {stage}  "
        f"max_tokens: {max_tokens}  "
        f"width_ext_frac: {width_ext_frac}  stage1_width_frac: {stage1_width_frac}  "
        f"min_width_ext_cm: {min_width_ext_cm}  dpi: {image_dpi}"
    )

    artifact = None
    crops = None
    try:
        if stage == "2":
            artifact = _load_stage1_artifact(out_dir)
        else:
            artifact = run_arm_level_stage(
                llm, image_msg, model=model_name, tracker=tracker,
                max_tokens=max_tokens, reasoning=reasoning, driving_side=side,
            )
        apply_stage1_localization(
            artifact["result"], image.size,
            width_frac=stage1_width_frac,
        )
        artifact["stage1_result"] = copy.deepcopy(artifact["result"])
        crops = save_arm_crops(
            out_dir, image, artifact["stage1_result"],
            width_ext_frac=width_ext_frac, min_width_ext_px=min_px,
            stage1_width_frac=stage1_width_frac,
        )
        n_arms = len((artifact.get("result") or {}).get("arms") or [])
        BaseLLM.log("stage 1 done: arms=%s" % n_arms)
        if stage in ("all", "2"):
            artifact["stage2"] = run_arm_refine_stage(
                llm, image, artifact["result"], crops,
                model=model_name, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
            )
            artifact["stage2"] = retry_through_road_refine_stage(
                llm, image, artifact["result"], crops, artifact["stage2"],
                model=model_name, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
            )
            artifact["stage2_result"] = copy.deepcopy(artifact["result"])
            BaseLLM.log("stage 2 done: refined=%s" % len(artifact.get("stage2") or {}))
            try:
                finalize_through_road_geometry(artifact["result"], drop_offsets=True)
            except Exception as exc:
                BaseLLM.log(f"stage2 finalize failed: {exc}")
        arm_ids = ",".join(str(arm["id"]) for arm in (artifact.get("result") or {}).get("arms") or [])
        n_stage2 = len(artifact.get("stage2") or {})
        BaseLLM.log("stage%s: arms=%s  refined=%s" % (stage, arm_ids or "none", n_stage2))
    finally:
        save_arm_level_artifacts(
            out_dir, artifact, tracker, name=run_name, source_image=image_path, image=image,
            width_ext_frac=width_ext_frac,
            stage1_width_frac=stage1_width_frac,
            min_width_ext_px=min_px, min_width_ext_cm=min_width_ext_cm, image_dpi=image_dpi,
            crops=crops,
        )

    if verbose:
        tracker.print_summary()
    BaseLLM.verbose = prev_verbose
    return artifact


if __name__ == "__main__":
    MODEL = {
        "api": "openai",
        "provider": "openrouter",
        "model": "openai/gpt-5.6-sol",
        "reasoning": "medium",
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="openaerialmap", help="Dataset folder under dataset/, e.g. nhtsa_2023")
    parser.add_argument("--case-id", default="1004", help="Case folder name under the dataset")
    parser.add_argument("--model", default=MODEL["model"])
    parser.add_argument("--reasoning", default=MODEL.get("reasoning", "none"))
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage", choices=("all", "1", "2"), default="all", help="Run stage 1 (junction center + heading), 2 (mouth refine + postprocess), or all")
    parser.add_argument("--width-ext-frac", type=float, default=DEFAULT_WIDTH_EXT_FRAC, help="Expand crop/overlay width by this fraction of the arm width (0.3 = +30%)")
    parser.add_argument("--stage1-width", type=float, default=DEFAULT_STAGE1_WIDTH_FRAC, help="Fixed stage-1 crop width as a fraction of the image side")
    parser.add_argument("--min-width-ext-cm", type=float, default=DEFAULT_MIN_WIDTH_EXT_CM, help="Minimum per-side lateral pad in paper centimeters")
    parser.add_argument("--image-dpi", type=float, default=DEFAULT_IMAGE_DPI, help="Scan DPI used to convert min-width-ext-cm to pixels")
    args = parser.parse_args()

    run_arm_level(
        {
            "api": MODEL["api"],
            "provider": MODEL["provider"],
            "model": args.model,
            "reasoning": args.reasoning,
        },
        dataset=args.dataset,
        case_id=args.case_id,
        max_tokens=args.max_tokens,
        compressed=args.compressed,
        verbose=args.verbose,
        width_ext_frac=args.width_ext_frac,
        stage1_width_frac=args.stage1_width,
        min_width_ext_cm=args.min_width_ext_cm,
        image_dpi=args.image_dpi,
        stage=args.stage,
    )
