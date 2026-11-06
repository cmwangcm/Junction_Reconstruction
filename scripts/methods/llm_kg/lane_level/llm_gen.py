"""Lane-level generation: identifier → evaluator → inspector → organizer."""

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

from PIL import Image

from scripts.methods.image_processing import load_processed_image
from scripts.methods.image_utils import bbox_to_px, load_rgb_image
from scripts.methods.connection_parser import enrich_channelized_features, normalize_driving_side, normalize_movement_candidates
from scripts.methods.llm_kg.lane_level.schemas import BOUNDARY_TYPE, LANE_TYPE, LINE_TYPES, NON_LANE_ITEM_TYPES, SEPARATOR_TYPES, TRAVELS, evaluator_result_schema, initial_result_schema, inspect_patch_schema, movement_codes, organization_result_schema
from scripts.methods.llm_kg.lane_level.prompts import LANE_LEVEL_PROMPT, arm_on_through_road, arm_user_text, evaluator_system_prompt, evaluator_text, inspect_patch_text, inspector_system_prompt, lane_level_prompt, organization_system_prompt, organization_text
from scripts.methods.llm_kg.lane_level.structure import (
    STRUCTURE_RETRY_LIMIT,
    fill_lane_travels,
    format_organization_retry_text,
    format_structure_retry_text,
    raise_structure_error,
    validate_arm_result,
)
from scripts.methods.llm_kg.inspect_metrics import inspect_events_from_agents
from scripts.methods.method_utils import (
    arms_path_for_model,
    default_arms_path,
    lane_level_dir,
    load_case_driving_side,
    prepare_output_dir,
    resolve_case_inputs,
    resolve_run_dir,
    set_verbose,
    unpack_model_config,
)
from scripts.methods.llm_kg.arm_level.llm_gen import (
    CROP_LOOKBACK_FRAC,
    DEFAULT_IMAGE_DPI,
    DEFAULT_MIN_WIDTH_EXT_CM,
    DEFAULT_WIDTH_EXT_FRAC,
    _arm_heading,
    arm_crop_region,
    paper_cm_to_px,
)
from scripts.methods.llm_utils import BaseLLM, CostTracker, build_llm, dump_json, to_jsonable

# Inspect crops should preserve local semantic context rather than isolate only
# the target. Evaluator boxes a tight region; code expands it before the zoom.
LOCATE_BOX_MARGIN = 0.40
LOCATE_BOX_MIN_WIDTH = 0.20
LOCATE_BOX_MIN_HEIGHT = 0.25


def road_frame(heading_deg):
    theta = math.radians(float(heading_deg) % 360.0)
    e_v = (math.cos(theta), -math.sin(theta))
    e_u = (e_v[1], -e_v[0])
    return e_u, e_v


def _dot_offset(point, origin, axis):
    return (point[0] - origin[0]) * axis[0] + (point[1] - origin[1]) * axis[1]


def _region_xy(region, u, v):
    e_u, e_v = region["e_u"], region["e_v"]
    cx, cy = region["center"]
    return cx + u * e_u[0] + v * e_v[0], cy + u * e_u[1] + v * e_v[1]


def _tighten_extra_for_axis(x0, vcomp, lo, hi, extra_lo, extra_hi):
    """x = x0 - extra * vcomp must stay in [lo, hi]. Tighten extra_lo/extra_hi."""
    if abs(vcomp) < 1e-12:
        if x0 < lo or x0 > hi:
            return extra_lo, -1.0
        return extra_lo, extra_hi
    # extra * vcomp <= x0 - lo ; extra * vcomp >= x0 - hi
    a = (x0 - lo) / vcomp
    b = (x0 - hi) / vcomp
    if vcomp > 0:
        return max(extra_lo, b), min(extra_hi, a)
    return max(extra_lo, a), min(extra_hi, b)


def _junction_extra_inside_image(region, extra, width, height):
    """Largest extra lookback that keeps the junction-side edge inside the image."""
    extra_lo = 0.0
    extra_hi = float(extra)
    if extra_hi <= 0:
        return 0.0
    e_v = region["e_v"]
    u0 = float(region["u0"])
    v0 = float(region["v0"])
    span_u = float(region["span_u"])
    w, h = float(width), float(height)
    for u in (u0, u0 + span_u):
        x0, y0 = _region_xy(region, u, v0)
        extra_lo, extra_hi = _tighten_extra_for_axis(x0, e_v[0], 0.0, w, extra_lo, extra_hi)
        extra_lo, extra_hi = _tighten_extra_for_axis(y0, e_v[1], 0.0, h, extra_lo, extra_hi)
        if extra_hi < extra_lo:
            return 0.0
    return max(0.0, min(float(extra), extra_hi))


CROP_FAR_EXT = 0.0  # past the arm endpoint
CROP_JUNCTION_EXT = 1.0  # extra lookback as a fraction of current crop height, toward the junction


def initial_crop_region(
    arm, width, height,
    *, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, lookback_ext=CROP_LOOKBACK_FRAC, far_ext=CROP_FAR_EXT,
    min_width_ext_px=None,
):
    """Rectangle along the arm heading from heading / anchor / width / length.

    ``width_ext_frac`` expands stored width the same way as arm-level crop/overlay.
    """
    region = arm_crop_region(
        arm, width, height,
        width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
        lookback_frac=lookback_ext,
    )
    if far_ext:
        region = dict(region)
        region["span_v"] = float(region["span_v"]) + float(region["length_px"]) * float(far_ext)
    return region


def extend_crop_region_toward_junction(region, junction_ext=CROP_JUNCTION_EXT, *, image_size=None):
    """Copy a crop region and grow span_v toward the junction (negative v / top of the rotated crop).

    ``junction_ext`` is a fraction of the current crop height: 0.5 adds 50% more lookback
    and leaves the far (arm) edge unchanged. If ``image_size`` is (width, height), extra
    lookback is clamped so the junction-side edge stays inside the original image.
    """
    out = dict(region)
    extra = float(out["span_v"]) * float(junction_ext)
    if image_size is not None:
        extra = _junction_extra_inside_image(out, extra, image_size[0], image_size[1])
    out["v0"] = float(out["v0"]) - extra
    out["span_v"] = float(out["span_v"]) + extra
    return out


def road_frame_crop(original, region):
    e_u, e_v = region["e_u"], region["e_v"]
    center_x, center_y = region["center"]
    u0, v0 = region["u0"], region["v0"]
    span_u, span_v = region["span_u"], region["span_v"]
    ox = center_x + u0 * e_u[0] + v0 * e_v[0]
    oy = center_y + u0 * e_u[1] + v0 * e_v[1]
    affine = (e_u[0], e_v[0], ox, e_u[1], e_v[1], oy)
    size = (max(1, round(span_u)), max(1, round(span_v)))
    image = original.transform(
        size, Image.Transform.AFFINE, affine,
        resample=Image.Resampling.BICUBIC, fillcolor=(255, 255, 255),
    )
    return image, affine


def _save_initial_crop(original, region, crops, crop_dir):
    image, affine = road_frame_crop(original, region)
    crop_id = "view_01"
    path = crop_dir / f"{crop_id}.png"
    image.save(path)
    crops[crop_id] = {"id": crop_id, "auto": True, "affine": list(affine), "size": image.size, "path": str(path)}
    return image, crop_id


def _save_zoom(base_image, box, crops, crop_dir, *, reason="", log=True, transform=None, heading_deg=None):
    px_box = bbox_to_px(box, base_image.width, base_image.height)
    x0, y0, x1, y1 = px_box
    image = base_image.crop((x0, y0, x1, y1))
    if transform is not None:
        image = transform(image)
    crop_id = f"view_{len(crops) + 1:02d}"
    path = crop_dir / f"{crop_id}.png"
    image.save(path)
    request = {"box": list(box), "pixel_box": list(px_box), "reason": reason}
    if transform is not None:
        request["rotated"] = True
    if heading_deg is not None:
        request["heading_deg"] = float(heading_deg)
    crops[crop_id] = {"id": crop_id, "auto": False, "request": request, "affine": None, "size": image.size, "path": str(path)}
    if log:
        BaseLLM.log(f"[crop] {crop_id}: {image.width}x{image.height}px zoom")
    return image, crop_id


def _assign_region_ids(spans):
    return [{**span, "region_id": f"R{i}"} for i, span in enumerate(spans or [], start=1)]


def _normalize_cue(cue):
    kind = cue.get("kind")
    entry = {
        "kind": kind,
        "orientation": cue.get("orientation"),
        "reasoning": cue.get("reasoning") or cue.get("evidence"),
    }
    if kind == "arrow":
        entry["movements"] = normalize_movement_candidates(cue.get("movements"))
    return entry


def _keep_crop_id(entry, item):
    if item.get("crop_id"):
        entry["crop_id"] = item["crop_id"]
    return entry


def _keep_element_id(entry, item):
    """Keep a code-assigned element_id."""
    if entry.get("type") == LANE_TYPE:
        entry.pop("element_id", None)
        return entry
    if item.get("element_id"):
        entry["element_id"] = item["element_id"]
    return entry


def _normalize_cross_section(cross_section, *, keep_element_id=False):
    """Keep LLM structural/cue fields. lane.travel is filled in code after organization."""
    normalized = []
    for item in cross_section or []:
        item_type = item.get("type")
        if item_type == LANE_TYPE:
            entry = {
                "type": item_type,
                "cues": [_normalize_cue(cue) for cue in (item.get("cues") or [])],
            }
        elif item_type in NON_LANE_ITEM_TYPES:
            entry = {
                "type": item_type,
                "evidence": item.get("evidence") or item.get("reasoning"),
            }
            if item_type in SEPARATOR_TYPES:
                entry["color"] = item.get("color")
        else:
            entry = {"type": item_type}
        if keep_element_id:
            entry = _keep_element_id(entry, item)
        else:
            entry.pop("element_id", None)
        normalized.append(_keep_crop_id(entry, item))
    return normalized


def _fill_result_travels(result, *, driving_side=None):
    """Fill every lane.travel from opposing_separator. Returns (result, fill_issues)."""
    if not result:
        return result, []
    updated, issues = fill_lane_travels(
        result.get("cross_section"),
        result.get("opposing_separator"),
        driving_side=driving_side,
    )
    return {**result, "cross_section": updated}, issues


def _assign_element_ids(cross_section, generation_id):
    """Stamp element_id=generation_id-order_id on non-lane items in left-to-right / patch order."""
    order = 0
    stamped = []
    for item in cross_section or []:
        entry = dict(item)
        if entry.get("type") == LANE_TYPE:
            entry.pop("element_id", None)
        else:
            order += 1
            entry["element_id"] = f"{int(generation_id)}-{order}"
        stamped.append(entry)
    return stamped


def _fill_missing_crop_ids(cross_section, crop_id):
    return [{**item, "crop_id": item.get("crop_id") or crop_id} for item in (cross_section or [])]


def _one_way_travel_of(opposing_separator):
    if not isinstance(opposing_separator, dict):
        return None
    travel = opposing_separator.get("one_way_travel")
    return travel if travel in TRAVELS else None


def _resolve_opposing_separator(cross_section, opposing_separator):
    """Normalize outer opposing_separator; keep LLM cut if it cites a valid line or median."""
    reasoning = ""
    one_way_travel = _one_way_travel_of(opposing_separator)
    if isinstance(opposing_separator, dict):
        reasoning = str(opposing_separator.get("reasoning") or "").strip()
    if not opposing_separator or not isinstance(opposing_separator, dict):
        return {"element_id": None, "one_way_travel": one_way_travel, "reasoning": reasoning}
    eid = opposing_separator.get("element_id")
    if not eid:
        return {"element_id": None, "one_way_travel": one_way_travel, "reasoning": reasoning}
    eid = str(eid)
    for item in cross_section or []:
        if str(item.get("element_id") or "") != eid:
            continue
        if item.get("type") in LINE_TYPES or item.get("type") == "median":
            return {"element_id": eid, "one_way_travel": None, "reasoning": reasoning}
        return {"element_id": None, "one_way_travel": one_way_travel, "reasoning": reasoning}
    return {"element_id": None, "one_way_travel": one_way_travel, "reasoning": reasoning}


def _discard_draft_organization(result):
    """Drop opposing_separator drafts when this agent does not assign organization."""
    if not result:
        return result
    cleared = []
    for item in result.get("cross_section") or []:
        entry = dict(item)
        if entry.get("type") == LANE_TYPE:
            entry.pop("travel", None)
        cleared.append(entry)
    return {
        **result,
        "cross_section": cleared,
        "opposing_separator": {"element_id": None, "one_way_travel": None, "reasoning": ""},
    }


def _finalize_visual_result(agent):
    """Stamp element_id and leave organization unset before the organizer runs."""
    result = agent.get("result")
    if not result:
        return result
    result = _discard_draft_organization(result)
    result = {
        **result,
        "cross_section": _assign_element_ids(result.get("cross_section"), 1),
    }
    agent["result"] = result
    return result


def _lane_movements(item):
    for cue in item.get("cues") or []:
        if cue.get("kind") == "arrow":
            return cue.get("movements")
    return None


def _arm_result(arm, crop_id, cross_section, opposing_separator=None, *, keep_element_id=False):
    cs = _normalize_cross_section(cross_section, keep_element_id=keep_element_id)
    cs = _fill_missing_crop_ids(cs, crop_id)
    opp = _resolve_opposing_separator(cs, opposing_separator)
    heading = _arm_heading(arm)
    out = {
        "arm_id": arm["id"],
        "heading_deg": heading,
        "width": arm["width"],
        "length": arm["length"],
        "crop_id": crop_id,
        "cross_section": cs,
        "opposing_separator": opp,
    }
    if arm.get("heading_deg_min") is not None and arm.get("heading_deg_max") is not None:
        out["heading_deg_min"] = arm["heading_deg_min"]
        out["heading_deg_max"] = arm["heading_deg_max"]
    return out


def _coerce_internal_boundaries(items, *, replace_start, replace_end, host_len):
    """Keep type=boundary only if this span owns a mouth end; otherwise use edge."""
    owns_left = int(replace_start) == 0
    owns_right = int(replace_end) == int(host_len)
    coerced = []
    n = len(items or [])
    for i, item in enumerate(items or []):
        entry = dict(item)
        if entry.get("type") == BOUNDARY_TYPE:
            keep = (owns_left and i == 0) or (owns_right and i == n - 1)
            if not keep:
                entry["type"] = "edge"
                entry["color"] = None
        coerced.append(entry)
    return coerced


def _apply_patches(cross_section, patches):
    """Splice confirmed span replacements. Apply right-to-left so indices stay valid."""
    updated = list(cross_section or [])
    host_len = len(updated)
    ordered = sorted(patches or [], key=lambda patch: int(patch["replace_start"]), reverse=True)
    for patch in ordered:
        if not patch.get("confirmed"):
            continue
        start = int(patch["replace_start"])
        end = int(patch["replace_end"])
        items = patch.get("items")
        if not isinstance(items, list):
            raise ValueError("patch items must be an array")
        if start < 0 or end < start or end > len(updated):
            raise ValueError(f"patch range [{start}:{end}] out of range for length {len(updated)}")
        items = _coerce_internal_boundaries(
            items, replace_start=start, replace_end=end, host_len=host_len,
        )
        items = _normalize_cross_section(items)
        if patch.get("crop_id"):
            items = [
                {**item, "crop_id": item.get("crop_id") or patch["crop_id"]}
                for item in items
            ]
        updated = updated[:start] + items + updated[end:]
    return updated


def _validate_region_box(region):
    box = tuple(float(region[key]) for key in ("x0", "y0", "x1", "y1"))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("inspect box needs x1 > x0 and y1 > y0")
    if min(box) < 0.0 or max(box) > 1.0:
        raise ValueError("inspect box must lie in [0, 1] of the source image")
    return box


def _expand_norm_box(
    box,
    margin,
    *,
    min_width=LOCATE_BOX_MIN_WIDTH,
    min_height=LOCATE_BOX_MIN_HEIGHT,
):
    """Expand a tight normalized cue box while retaining nearby scene context.

    ``margin`` is applied on every side as a fraction of the tight box width/height.
    A minimum normalized crop size is then enforced so a very small cue does not
    become an object-only crop with no lane/road context. The final box is shifted,
    not shrunk, when possible to stay inside [0, 1].
    """
    x0, y0, x1, y1 = [float(v) for v in box]
    m = max(0.0, float(margin))

    width = x1 - x0
    height = y1 - y0
    if width <= 0.0 or height <= 0.0:
        raise ValueError("inspect box must have positive width and height")

    pad_x = m * width
    pad_y = m * height
    x0 -= pad_x
    x1 += pad_x
    y0 -= pad_y
    y1 += pad_y

    target_w = min(1.0, max(x1 - x0, max(0.0, float(min_width))))
    target_h = min(1.0, max(y1 - y0, max(0.0, float(min_height))))
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)

    x0 = cx - 0.5 * target_w
    x1 = cx + 0.5 * target_w
    y0 = cy - 0.5 * target_h
    y1 = cy + 0.5 * target_h

    if x0 < 0.0:
        x1 -= x0
        x0 = 0.0
    if x1 > 1.0:
        x0 -= x1 - 1.0
        x1 = 1.0
    if y0 < 0.0:
        y1 -= y0
        y0 = 0.0
    if y1 > 1.0:
        y0 -= y1 - 1.0
        y1 = 1.0

    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(1.0, x1), min(1.0, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("expanded inspect box is empty after context expansion/clamp")
    return (x0, y0, x1, y1)


def _union_norm_box(boxes):
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def _spans_overlap(a, b):
    return int(a["replace_start"]) < int(b["replace_end"]) and int(b["replace_start"]) < int(a["replace_end"])


def _group_overlapping_critiques(critiques, *, margin=LOCATE_BOX_MARGIN, log=True):
    """Merge overlapping critiques into conflict groups for one inspect+patch each.

    Two critiques join the same group only if their replace spans overlap
    (adjacent [a,b) and [b,c) do not). Box overlap alone does not merge.
    The group uses the union span and a crop that is the union of every
    member crop box. Singletons keep the original region_id; multi-member
    groups are G1, G2, … left-to-right.
    """
    items = list(critiques or [])
    if not items:
        return []
    tight = [_validate_region_box(item) for item in items]
    expanded = [
        _expand_norm_box(
            box, margin,
            min_width=LOCATE_BOX_MIN_WIDTH,
            min_height=LOCATE_BOX_MIN_HEIGHT,
        )
        for box in tight
    ]
    parent = list(range(len(items)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def unite(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _spans_overlap(items[i], items[j]):
                unite(i, j)

    buckets = {}
    for i in range(len(items)):
        buckets.setdefault(find(i), []).append(i)

    groups = []
    for idxs in buckets.values():
        idxs = sorted(idxs, key=lambda i: (int(items[i]["replace_start"]), int(items[i]["replace_end"]), i))
        members = [items[i] for i in idxs]
        start = min(int(member["replace_start"]) for member in members)
        end = max(int(member["replace_end"]) for member in members)
        x0, y0, x1, y1 = _union_norm_box([tight[i] for i in idxs])
        crop_box = _union_norm_box([expanded[i] for i in idxs]) if len(members) > 1 else None
        kinds = [member.get("kind") or "" for member in members]
        kind = kinds[0] if len(set(kinds)) == 1 else "conflict_group"
        inspect_content = str(members[0].get("inspect_content") or members[0].get("note") or "")
        visual_anchor = str(members[0].get("visual_anchor") or "")
        reason = str(members[0].get("reason") or "")
        if len(members) > 1:
            inspect_content = " | ".join(
                f"{member.get('region_id')} {member.get('kind')}: "
                f"{member.get('inspect_content') or member.get('note') or ''}".strip()
                for member in members
            )
            visual_anchor = " | ".join(
                f"{member.get('region_id')}: {member.get('visual_anchor') or ''}".strip()
                for member in members
            )
            reason = "overlapping structural spans merged into one conflict group"
        groups.append({
            "region_id": members[0].get("region_id") or "",
            "kind": kind,
            "replace_start": start,
            "replace_end": end,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "crop_box": list(crop_box) if crop_box is not None else None,
            "visual_anchor": visual_anchor,
            "inspect_content": inspect_content,
            "note": inspect_content,
            "reason": reason,
            "members": members,
        })
    groups.sort(key=lambda group: (int(group["replace_start"]), int(group["replace_end"])))
    group_idx = 0
    for group in groups:
        if len(group["members"]) > 1:
            group_idx += 1
            group["region_id"] = f"G{group_idx}"
            if log:
                labels = [
                    f"{member.get('region_id')}:{member.get('kind')}"
                    f"[{member['replace_start']}:{member['replace_end']}]"
                    for member in group["members"]
                ]
                BaseLLM.log(
                    f"[crop] conflict group {group['region_id']} "
                    f"[{group['replace_start']}:{group['replace_end']}] = {labels}",
                    indent=1,
                )
    return groups


def _materialize_regions(agent, regions, *, margin=LOCATE_BOX_MARGIN, crop_dir=None, log=True, base_image=None):
    save_dir = crop_dir or agent["crop_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    source = agent["base_image"] if base_image is None else base_image
    crops = []
    for region in regions or []:
        llm_box = _validate_region_box(region)
        if region.get("crop_box"):
            box = _validate_region_box({
                "x0": region["crop_box"][0],
                "y0": region["crop_box"][1],
                "x1": region["crop_box"][2],
                "y1": region["crop_box"][3],
            })
        else:
            box = _expand_norm_box(
                llm_box,
                margin,
                min_width=LOCATE_BOX_MIN_WIDTH,
                min_height=LOCATE_BOX_MIN_HEIGHT,
            )
        start = int(region["replace_start"])
        end = int(region["replace_end"])
        heading = (agent.get("arm") or {}).get("heading_deg") if agent.get("crop_transform") else None
        image, crop_id = _save_zoom(
            source, box, agent["crops"], save_dir,
            reason=region.get("reason", f"inspect [{start}:{end}]"), log=log,
            transform=agent.get("crop_transform"),
            heading_deg=heading,
        )
        crops.append({
            "crop_id": crop_id,
            "region_id": region.get("region_id") or "",
            "kind": region.get("kind") or "cue",
            "image": image,
            "replace_start": start,
            "replace_end": end,
            "box": list(box),
            "llm_box": list(llm_box),
            "crop_box": list(region["crop_box"]) if region.get("crop_box") else None,
            "margin": float(margin),
            "min_width_frac": float(LOCATE_BOX_MIN_WIDTH),
            "min_height_frac": float(LOCATE_BOX_MIN_HEIGHT),
            "visual_anchor": str(region.get("visual_anchor") or ""),
            "note": str(region.get("note") or region.get("inspect_content") or ""),
            "inspect_content": str(region.get("inspect_content") or region.get("note") or ""),
            "members": list(region.get("members") or []),
        })
    return crops


def _lane_result_doc(lane_results):
    arms = []
    for arm_id in sorted(lane_results):
        arms.append(_strip_lane_source_type(lane_results[arm_id]))
    return {"arms": arms}


def run_rule_initial_crop(original, arm, crop_dir, *, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None):
    region = initial_crop_region(
        arm, original.width, original.height,
        width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
    )
    crops = {}
    crop_image, crop_id = _save_initial_crop(original, region, crops, crop_dir)
    return {"crop_image": crop_image, "crop_id": crop_id, "crops": crops}


def run_rule_extended_crop(
    original, arm, crop_dir, *, crop_id="view_extended", junction_ext=CROP_JUNCTION_EXT,
    width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None,
):
    """Same mouth crop with the top (junction) edge extended by ``junction_ext`` of crop height.

    Extra lookback is clamped to the original image; it does not grow into empty fill.
    """
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    region = extend_crop_region_toward_junction(
        initial_crop_region(
            arm, original.width, original.height,
            width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px,
        ),
        junction_ext=junction_ext,
        image_size=(original.width, original.height),
    )
    image, affine = road_frame_crop(original, region)
    path = crop_dir / f"{crop_id}.png"
    image.save(path)
    meta = {
        "id": crop_id,
        "auto": True,
        "extended": True,
        "junction_ext": float(junction_ext),
        "affine": list(affine),
        "size": image.size,
        "path": str(path),
    }
    return {"crop_image": image, "crop_id": crop_id, "crops": {crop_id: meta}}


def create_arm_agent(llm, arm, crop_dir, crop_bundle, *, channelized_features=None, through_roads=None, driving_side=None, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None):
    crop_image = crop_bundle["crop_image"]
    return {
        "arm": arm,
        "messages": [llm.image_user_message(crop_image, arm_user_text(arm, channelized_features, through_roads, driving_side=driving_side))],
        "crops": crop_bundle["crops"],
        "base_image": crop_image,
        "crop_id": crop_bundle["crop_id"],
        "crop_dir": crop_dir,
        "result": None,
        "allow_lr": not arm_on_through_road(arm["id"], through_roads),
        "refinement_rounds": 0,
        "inspected": False,
        "patch_events": [],
        "critiques": [],
        "assign_organization": True,
        "system_prompt": lane_level_prompt(
            driving_side, channelized_features=channelized_features, arm_id=arm.get("id"),
        ),
        "evaluator_system_prompt": None,
        "evaluator_messages": [],
        "inspector_system_prompt": None,
        "inspector_messages": [],
        "organization_system_prompt": None,
        "organization_messages": [],
        "llm_calls": [],
        "driving_side": driving_side or "left",
        "channelized_features": list(channelized_features or []),
        "through_roads": list(through_roads or []),
        "width_ext_frac": float(width_ext_frac),
        "min_width_ext_px": min_width_ext_px,
    }


def _call_json(llm, messages, *, model, tracker, label, schema, max_tokens, reasoning, agent=None, append_request=True, system_prompt=None, messages_key="messages"):
    prompt = system_prompt
    if prompt is None and agent is not None:
        prompt = agent.get("system_prompt")
    if prompt is None:
        prompt = LANE_LEVEL_PROMPT
    parsed, _text, response = llm.call_json(model=model, system_prompt=prompt, messages=messages, schema=schema, max_tokens=max_tokens, reasoning_effort=reasoning)
    tracker.record(label, model, response)
    if agent is not None:
        assistant = llm.assistant_message(response)
        log = agent.setdefault(messages_key, [])
        if append_request:
            log.extend(messages)
        log.append(assistant)
        agent.setdefault("llm_calls", []).append({
            "system": prompt or "",
            "messages": to_jsonable(list(messages) + [assistant]),
        })
    return parsed


def _merge_text_into_user_message(message, text):
    """Append text onto an existing user message (plain string or multimodal blocks)."""
    content = message.get("content")
    if isinstance(content, str):
        message["content"] = f"{content}\n\n{text}" if content else text
        return
    if isinstance(content, list):
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "text" and "text" in block:
                existing = block.get("text") or ""
                block["text"] = f"{existing}\n\n{text}" if existing else text
                return
        content.append({"type": "text", "text": text})
        return
    message["content"] = text


def _user_message_text(message):
    content = (message or {}).get("content")
    if isinstance(content, str):
        return content or ""
    if isinstance(content, list):
        return "\n".join(
            block.get("text") or ""
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _fresh_image_user_messages(llm, image, seed_messages, extra_text=None):
    """Standalone image+text user turn from the seed request, plus optional retry text."""
    text = "\n\n".join(
        _user_message_text(msg)
        for msg in (seed_messages or [])
        if (msg or {}).get("role") == "user"
    )
    if extra_text:
        text = f"{text}\n\n{extra_text}" if text else extra_text
    return [llm.image_user_message(image, text)]


def _append_or_merge_user_text(llm, agent, text):
    """Add a follow-up user turn; merge into the last user message when no assistant yet.

    Agent messages must already be seeded (e.g. by ``create_arm_agent``). A plain append
    before any assistant reply would create two consecutive user roles — merge instead.
    """
    messages = agent.get("messages")
    if not messages:
        raise ValueError(f"arm {agent['arm']['id']}: agent messages are empty; seed them before appending")
    if messages[-1].get("role") == "user":
        _merge_text_into_user_message(messages[-1], text)
        return
    messages.append(llm.make_user_message(text))

def _driving_side_of(agent):
    return normalize_driving_side(agent.get("driving_side"))


def _accept_structure_or_issues(result, agent, *, stage, check_travel=True, check_organization=None, check_arrow_order=None):
    """Return (result, issues). Empty issues means accepted."""
    issues = validate_arm_result(
        result, driving_side=_driving_side_of(agent), check_travel=check_travel,
        check_organization=check_organization, check_arrow_order=check_arrow_order,
    )
    if issues:
        BaseLLM.log(
            f"arm {agent['arm']['id']}: {stage} structure invalid — {'; '.join(issues)}",
            indent=1,
        )
    return result, issues


def run_identifier(llm, agent, *, model, tracker, max_tokens=2048, reasoning=None):
    """Visual identification: mouth + cues only. No opposing_separator."""
    arm = agent["arm"]
    schema = initial_result_schema(
        allow_lr=agent["allow_lr"],
    )
    side = _driving_side_of(agent)
    seed = list(agent.get("messages") or [])
    if not seed:
        raise ValueError(f"arm {arm['id']}: identifier seed messages are empty")
    last_issues = []
    for attempt in range(STRUCTURE_RETRY_LIMIT + 1):
        label = "identifier" if attempt == 0 else "identifier_retry"
        extra = None
        if attempt:
            extra = format_structure_retry_text(
                last_issues, driving_side=side, context="visual identification (cross_section + cues)",
                check_travel=False, check_organization=False,
            )
        call_messages = seed if extra is None else _fresh_image_user_messages(
            llm, agent["base_image"], seed, extra,
        )
        parsed = _call_json(
            llm, call_messages,
            model=model, tracker=tracker, label=label,
            schema=schema,
            max_tokens=max_tokens, reasoning=reasoning, agent=agent,
            append_request=extra is not None,
        )
        result = _arm_result(
            arm, agent["crop_id"], parsed.get("cross_section"),
            None,
        )
        result = _discard_draft_organization(result)
        result, issues = _accept_structure_or_issues(
            result, agent, stage=label,
            check_travel=False,
            check_organization=False,
            check_arrow_order=None,
        )
        if not issues:
            agent["result"] = result
            return result
        last_issues = issues
        if attempt >= STRUCTURE_RETRY_LIMIT:
            break
    raise_structure_error(last_issues, arm_id=arm["id"], stage="identifier")


def _bind_evaluator_critiques(cross_section, critiques):
    n = len(cross_section or [])
    bound = []
    for critic in critiques or []:
        start = int(critic["replace_start"])
        end = int(critic["replace_end"])
        if start < 0 or end < start or end > n:
            raise ValueError(f"critic span [{start}:{end}] out of range for length {n}")
        bound.append({
            "kind": str(critic.get("kind") or ""),
            "replace_start": start,
            "replace_end": end,
            "x0": float(critic["x0"]),
            "y0": float(critic["y0"]),
            "x1": float(critic["x1"]),
            "y1": float(critic["y1"]),
            "visual_anchor": str(critic.get("visual_anchor") or ""),
            "inspect_content": str(critic.get("inspect_content") or ""),
            "reason": str(critic.get("reason") or ""),
            "note": str(critic.get("inspect_content") or ""),
        })
    return _assign_region_ids(bound)


def _patch_change(before, patch):
    start = int(patch["replace_start"])
    end = int(patch["replace_end"])
    return {
        "region_id": patch.get("region_id") or "",
        "crop_id": patch.get("crop_id") or "",
        "kind": patch.get("kind") or "",
        "replace_start": start,
        "replace_end": end,
        "confirmed": bool(patch.get("confirmed")),
        "reasoning": patch.get("reasoning") or "",
        "before": list((before or [])[start:end]),
        "after": list(patch.get("items") or []),
    }


def _crop_event_meta(crop):
    members = [
        {
            "region_id": member.get("region_id") or "",
            "kind": member.get("kind") or "",
            "replace_start": int(member["replace_start"]),
            "replace_end": int(member["replace_end"]),
            "reason": member.get("reason") or "",
            "inspect_content": member.get("inspect_content") or member.get("note") or "",
        }
        for member in (crop.get("members") or [])
    ]
    return {
        "crop_id": crop["crop_id"],
        "region_id": crop.get("region_id") or "",
        "kind": crop.get("kind") or "",
        "replace_start": crop["replace_start"],
        "replace_end": crop["replace_end"],
        "box": crop["box"],
        "llm_box": crop.get("llm_box"),
        "crop_box": crop.get("crop_box"),
        "margin": crop.get("margin"),
        "visual_anchor": crop.get("visual_anchor") or "",
        "inspect_content": crop.get("inspect_content") or crop.get("note") or "",
        "members": members,
    }


def _record_patch_event(agent, *, stage, targets, crops, applied, before):
    agent["patch_events"].append({
        "stage": stage,
        "targets": list(targets or []),
        "crops": [_crop_event_meta(crop) for crop in (crops or [])],
        "applied": [_patch_change(before, patch) for patch in (applied or [])],
    })


def run_evaluator(llm, agent, *, model, tracker, max_tokens=2048, reasoning=None, label="evaluator"):
    """Compare the visual reading to the arm crop; return inspect-worthy critiques."""
    result = agent.get("result") or {}
    cross_section = result.get("cross_section") or []
    system = evaluator_system_prompt()
    agent["evaluator_system_prompt"] = system
    prompt = evaluator_text(cross_section)
    parsed = _call_json(
        llm, [llm.image_user_message(agent["base_image"], prompt)],
        model=model, tracker=tracker, label=label,
        schema=evaluator_result_schema(),
        max_tokens=max_tokens, reasoning=reasoning, agent=agent,
        system_prompt=system, messages_key="evaluator_messages",
    )
    critiques = _bind_evaluator_critiques(cross_section, parsed.get("critiques") or [])
    agent["critiques"] = critiques
    arm_id = agent["arm"]["id"]
    if not critiques:
        BaseLLM.log(f"arm {arm_id}: evaluator found no inspect targets")
    else:
        labels = [
            f"{c['region_id']}:{c['kind']}[{c['replace_start']}:{c['replace_end']}]"
            for c in critiques
        ]
        BaseLLM.log(f"arm {arm_id}: evaluator critiques={labels}")
    return critiques


def _inspect_one_crop(llm, agent, crop, *, model, tracker, stage, max_tokens, reasoning):
    expected_region_id = str(crop.get("region_id") or "")
    schema = inspect_patch_schema(allow_lr=agent["allow_lr"])
    side = _driving_side_of(agent)
    prompt = inspect_patch_text(agent["result"]["cross_section"], crop)
    system = agent.get("inspector_system_prompt") or inspector_system_prompt()
    agent["inspector_system_prompt"] = system
    base_cs = list(agent["result"]["cross_section"])
    last_issues = []
    for attempt in range(STRUCTURE_RETRY_LIMIT + 1):
        label = stage if attempt == 0 else f"{stage}_retry"
        extra = None
        if attempt:
            extra = format_structure_retry_text(
                last_issues, driving_side=side, context="inspect patch (span items)",
                check_travel=False, check_organization=False,
            )
        user_text = prompt if extra is None else f"{prompt}\n\n{extra}"
        call_messages = [llm.image_user_message(crop["image"], user_text)]
        parsed = _call_json(
            llm, call_messages,
            model=model, tracker=tracker, label=label,
            schema=schema, max_tokens=max_tokens, reasoning=reasoning, agent=agent,
            system_prompt=system, messages_key="inspector_messages",
        )
        raw = parsed.get("patch")
        if not raw:
            last_issues = ["patch payload missing; return a complete patch object"]
            if attempt >= STRUCTURE_RETRY_LIMIT:
                break
            continue
        region_id = str(raw.get("region_id") or "")
        if region_id != expected_region_id:
            raise ValueError(f"patch region_id mismatch; expected {expected_region_id!r}, got {region_id!r}")
        confirmed = bool(raw.get("confirmed"))
        items = list(raw.get("items") or [])
        reason = str(raw.get("reasoning") or "").strip()
        if confirmed and not items:
            last_issues = ["confirmed=true requires replacement items"]
            if attempt >= STRUCTURE_RETRY_LIMIT:
                break
            continue
        if not reason:
            last_issues = ["patch reasoning is required, including when confirmed=false"]
            if attempt >= STRUCTURE_RETRY_LIMIT:
                break
            continue
        if not confirmed:
            items = []
        patch = {
            "region_id": region_id,
            "crop_id": crop["crop_id"],
            "kind": crop.get("kind") or "",
            "replace_start": int(crop["replace_start"]),
            "replace_end": int(crop["replace_end"]),
            "confirmed": confirmed,
            "items": items,
            "reasoning": reason,
        }
        if not confirmed:
            return patch
        updated = _apply_patches(base_cs, [patch])
        candidate = _discard_draft_organization(
            _arm_result(agent["arm"], agent["crop_id"], updated, None),
        )
        candidate, issues = _accept_structure_or_issues(
            candidate, agent, stage=label,
            check_travel=False, check_organization=False, check_arrow_order=None,
        )
        if not issues:
            agent["result"] = candidate
            return patch
        last_issues = issues
        if attempt >= STRUCTURE_RETRY_LIMIT:
            break
    raise_structure_error(
        last_issues or ["inspect patch failed structure validation"],
        arm_id=agent["arm"]["id"],
        stage=stage,
    )


def run_inspector(llm, agent, critiques, *, model, tracker, max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN, crop_dir=None, log=True, stage_prefix="inspector"):
    """Group overlapping critiques, then inspect+patch each conflict group once."""
    if not critiques:
        return agent.get("result")
    arm = agent["arm"]
    before_all = list(agent["result"]["cross_section"])
    agent["inspected"] = True
    groups = _group_overlapping_critiques(critiques, margin=margin, log=log)
    crops = _materialize_regions(agent, groups, margin=margin, crop_dir=crop_dir, log=log)
    ordered = sorted(crops, key=lambda crop: int(crop["replace_start"]), reverse=True)
    patches = []
    for i, crop in enumerate(ordered):
        patch = _inspect_one_crop(
            llm, agent, crop,
            model=model, tracker=tracker, stage=f"{stage_prefix}_{i + 1}",
            max_tokens=max_tokens, reasoning=reasoning,
        )
        patches.append(patch)
    _record_patch_event(
        agent, stage="inspector",
        targets=[group.get("region_id") for group in groups],
        crops=crops, applied=patches, before=before_all,
    )
    confirmed = sum(1 for patch in patches if patch.get("confirmed"))
    if log:
        BaseLLM.log(
            f"arm {arm['id']}: inspector confirmed {confirmed}/{len(patches)} "
            f"over {len(crops)} crop(s); cross_section now {len(agent['result']['cross_section'])} items"
        )
    return agent.get("result")


def _apply_arrow_updates(cross_section, updates, *, allow_lr):
    """Rewrite arrow.movements on existing lane arrows. Returns (cs, issues)."""
    updated = copy.deepcopy(list(cross_section or []))
    allowed = set(movement_codes(allow_lr=allow_lr))
    issues = []
    for raw in updates or []:
        if not isinstance(raw, dict):
            issues.append("arrow_updates item must be an object")
            continue
        try:
            idx = int(raw["index"])
        except (KeyError, TypeError, ValueError):
            issues.append("arrow_updates item missing integer index")
            continue
        if idx < 0 or idx >= len(updated):
            issues.append(f"arrow_updates index {idx} out of range for length {len(updated)}")
            continue
        item = updated[idx]
        if item.get("type") != LANE_TYPE:
            issues.append(f"arrow_updates index {idx} is type={item.get('type')!r}, not lane")
            continue
        movements = normalize_movement_candidates(raw.get("movements"))
        if not movements:
            issues.append(f"arrow_updates index {idx}: movements must be a non-empty arrow code list")
            continue
        bad = [code for code in movements if code not in allowed]
        if bad:
            issues.append(f"arrow_updates index {idx}: unsupported movement codes {bad}")
            continue
        cues = list(item.get("cues") or [])
        arrow_at = next((i for i, cue in enumerate(cues) if cue.get("kind") == "arrow"), None)
        if arrow_at is None:
            issues.append(f"arrow_updates index {idx}: lane has no arrow cue")
            continue
        arrow = dict(cues[arrow_at])
        arrow["movements"] = movements
        cues[arrow_at] = arrow
        updated[idx] = {**item, "cues": cues}
    return updated, issues


def _apply_organization_reading(result, parsed, *, arm, crop_id, driving_side=None, allow_lr=True):
    """Attach opposing_separator, apply arrow movement updates, fill lane.travel."""
    opp = parsed.get("opposing_separator")
    if not isinstance(opp, dict):
        return result, ["organization missing opposing_separator"]
    cs, update_issues = _apply_arrow_updates(
        result.get("cross_section"), parsed.get("arrow_updates") or [],
        allow_lr=allow_lr,
    )
    if update_issues:
        return result, update_issues
    candidate = _arm_result(arm, crop_id, cs, opp, keep_element_id=True)
    candidate, fill_issues = _fill_result_travels(candidate, driving_side=driving_side)
    if fill_issues:
        return candidate, fill_issues
    issues = validate_arm_result(candidate, driving_side=driving_side, check_travel=True)
    return candidate, issues


def run_organization(llm, agent, *, model, tracker, max_tokens=2048, reasoning=None):
    """Identify opposing_separator and optional arrow updates; fill travel in code."""
    result = _finalize_visual_result(agent)
    if not result or not agent.get("assign_organization"):
        return result
    arm = agent["arm"]
    side = _driving_side_of(agent)
    schema = organization_result_schema(allow_lr=agent["allow_lr"])
    system = organization_system_prompt(
        side,
        channelized_features=agent.get("channelized_features"),
        arm_id=arm.get("id"),
    )
    agent["organization_system_prompt"] = system
    agent["organization_messages"] = []
    last_issues = []
    for attempt in range(STRUCTURE_RETRY_LIMIT + 1):
        label = "organization" if attempt == 0 else "organization_retry"
        prompt = organization_text(
            result.get("cross_section"),
            opposing_separator=None,
            issues=last_issues if attempt else None,
            channelized_features=agent.get("channelized_features"),
            through_roads=agent.get("through_roads"),
            driving_side=side,
            arm_id=arm.get("id"),
        )
        if attempt:
            prompt = prompt + "\n\n" + format_organization_retry_text(
                last_issues, driving_side=side,
            )
        call_messages = [llm.image_user_message(agent["base_image"], prompt)]
        parsed = _call_json(
            llm, call_messages,
            model=model, tracker=tracker, label=label,
            schema=schema, max_tokens=max_tokens, reasoning=reasoning, agent=agent,
            system_prompt=system,
            messages_key="organization_messages",
        )
        candidate, apply_issues = _apply_organization_reading(
            result, parsed, arm=arm, crop_id=agent["crop_id"], driving_side=side,
            allow_lr=agent["allow_lr"],
        )
        if apply_issues:
            last_issues = apply_issues
            if attempt >= STRUCTURE_RETRY_LIMIT:
                break
            continue
        candidate, accept_issues = _accept_structure_or_issues(
            candidate, agent, stage=label, check_travel=True, check_organization=True,
        )
        if not accept_issues:
            agent["result"] = candidate
            n_updates = len(parsed.get("arrow_updates") or [])
            extra = f" ({n_updates} arrow update(s))" if n_updates else ""
            BaseLLM.log(f"arm {arm['id']}: organization identified{extra}")
            return candidate
        last_issues = accept_issues
        if attempt >= STRUCTURE_RETRY_LIMIT:
            break
    raise_structure_error(last_issues, arm_id=arm["id"], stage="organization")


def complete_arm_reading(llm, agent, *, model, tracker, max_tokens=2048, reasoning=None):
    """Organization: opposing_separator plus optional arrow updates; code fills travel."""
    return run_organization(
        llm, agent, model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
    )


def run_evaluate_and_inspect(llm, agent, *, model, tracker, max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN, max_inspect_rounds=1, crop_dir=None):
    """Run evaluator→inspector up to ``max_inspect_rounds`` times.

    Each evaluator call may emit multiple critiques; all of them are inspected
    in that round. Stop early if a round returns no critiques.
    """
    result = agent.get("result")
    for round_idx in range(max(0, int(max_inspect_rounds))):
        eval_label = "evaluator" if round_idx == 0 else f"evaluator_{round_idx + 1}"
        critiques = run_evaluator(
            llm, agent, model=model, tracker=tracker, max_tokens=max_tokens,
            reasoning=reasoning, label=eval_label,
        )
        if not critiques:
            break
        inspect_prefix = "inspector" if round_idx == 0 else f"inspector_{round_idx + 1}"
        result = run_inspector(
            llm, agent, critiques,
            model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
            margin=margin, crop_dir=crop_dir, stage_prefix=inspect_prefix,
        )
    return result


def agent_artifact(agent, *, model=None):
    """Per-arm conversation: one list entry per LLM call (system + messages)."""
    return list(agent.get("llm_calls") or [])


def _build_cost(trackers_by_arm):
    cost = {}
    merged = CostTracker(silent=True)
    for arm_id in sorted(trackers_by_arm):
        summary = trackers_by_arm[arm_id].summary()
        cost[str(arm_id)] = summary
        merged.steps.extend(summary["steps"])
    # Prefer ``totals`` to match CostTracker.summary(); keep ``total`` for older readers.
    totals = merged.totals()
    cost["totals"] = totals
    cost["total"] = totals
    return cost


def _build_conversation(agents_by_arm, *, model):
    conversation = {}
    for arm_id in sorted(agents_by_arm):
        conversation[str(arm_id)] = agent_artifact(agents_by_arm[arm_id], model=model)
    return conversation


def _drop_legacy_status(obj):
    """Drop leftover per-element status keys from older lane-level artifacts."""
    if isinstance(obj, dict):
        return {key: _drop_legacy_status(value) for key, value in obj.items() if key != "status"}
    if isinstance(obj, list):
        return [_drop_legacy_status(item) for item in obj]
    return obj


def _strip_lane_source_type(arm):
    """Drop arm-level-only fields and leftover element status from older artifacts."""
    if not isinstance(arm, dict):
        return arm
    arm = _drop_legacy_status(arm)
    arm.pop("channelized_source_type", None)
    return arm


def _lanes_doc(image_path, lane_results, channelized_features, through_roads, driving_side=None):
    arms = []
    for arm_id in sorted(lane_results):
        arms.append(_strip_lane_source_type(lane_results[arm_id]))
    return {
        "source_image": str(image_path),
        "driving_side": normalize_driving_side(driving_side),
        "arms": arms,
        "channelized_features": channelized_features,
        "through_roads": through_roads,
    }


def _agents_touched_by_refine(agents_by_arm):
    """Arms that actually ran a refine turn (exclude untouched generation agents)."""
    return {
        arm_id: agent for arm_id, agent in (agents_by_arm or {}).items()
        if int(agent.get("refinement_rounds") or 0) > 0 or (agent.get("patch_events") or [])
    }


def _stage_lanes_from_agents(
    agents_by_arm, key, image_path, channelized_features, through_roads, driving_side,
):
    """Build a lanes.json-shaped doc from per-arm snapshots stored on agents."""
    results = {}
    for arm_id in sorted(agents_by_arm or {}):
        snap = (agents_by_arm[arm_id] or {}).get(key)
        if isinstance(snap, dict) and snap.get("cross_section") is not None:
            results[arm_id] = _strip_lane_source_type(copy.deepcopy(snap))
    if not results:
        return None
    return _lanes_doc(image_path, results, channelized_features, through_roads, driving_side=driving_side)


def _save_lane_bundle(
    root, *, agents_by_arm, trackers_by_arm, model, lanes_doc=None,
    identifier_doc=None, inspect_doc=None, only_refined=False, write_conversation=True,
):
    """Write cost (and optional conversation/lanes) under ``root``.

    Cost always aggregates every tracker (including failed / non-passing runs).
    ``only_refined`` only filters conversation agents.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    agents = _agents_touched_by_refine(agents_by_arm) if only_refined else (agents_by_arm or {})
    conversation = _build_conversation(agents, model=model) if write_conversation else {}
    cost = _build_cost(trackers_by_arm or {})
    if write_conversation:
        dump_json(root / "conversation.json", conversation)
        inspect_events = inspect_events_from_agents(agents)
        dump_json(root / "inspect_events.json", inspect_events)
    dump_json(root / "cost.json", cost)
    if lanes_doc is not None:
        dump_json(root / "lanes.json", lanes_doc)
    if identifier_doc is not None:
        dump_json(root / "identifier.json", identifier_doc)
    if inspect_doc is not None:
        dump_json(root / "inspect.json", inspect_doc)
    return conversation, cost


def _merged_tracker(trackers_by_arm, *, source=None):
    merged = CostTracker(source=source, silent=True)
    for arm_id in sorted(trackers_by_arm):
        merged.steps.extend(trackers_by_arm[arm_id].steps)
    return merged


def run_arm_agent(llm, original, arm, crop_dir, *, model, tracker, max_tokens=2048, channelized_features=None, through_roads=None, reasoning=None, margin=LOCATE_BOX_MARGIN, crop_and_inspect=True, max_inspect_rounds=1, driving_side=None, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_px=None, on_visual_done=None):
    crop_bundle = run_rule_initial_crop(original, arm, crop_dir, width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px)
    agent = create_arm_agent(llm, arm, crop_dir, crop_bundle, channelized_features=channelized_features, through_roads=through_roads, driving_side=driving_side, width_ext_frac=width_ext_frac, min_width_ext_px=min_width_ext_px)
    run_identifier(llm, agent, model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning)
    agent["identifier_result"] = copy.deepcopy(agent.get("result"))
    if on_visual_done:
        on_visual_done(agent)
    try:
        if crop_and_inspect:
            run_evaluate_and_inspect(
                llm, agent, model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning,
                margin=margin, max_inspect_rounds=max_inspect_rounds,
            )
        else:
            BaseLLM.log(f"arm {arm['id']}: evaluator+inspector skipped")
        agent["inspect_result"] = copy.deepcopy(agent.get("result"))
        if on_visual_done:
            on_visual_done(agent)
        complete_arm_reading(llm, agent, model=model, tracker=tracker, max_tokens=max_tokens, reasoning=reasoning)
        return agent
    except Exception:
        if on_visual_done:
            on_visual_done(agent)
        raise


def _load_lane_arm_result(loaded_arms, arm_id, *, source="lanes.json"):
    for lane_arm in loaded_arms or []:
        if int(lane_arm.get("arm_id", -1)) == int(arm_id):
            return _strip_lane_source_type(lane_arm)
    raise ValueError(f"{source} has no arm_id={arm_id}")


def run_lane_level(llm, image_path, arms, output_dir, *, model, max_tokens=2048, skip_generation=False, channelized_features=None, through_roads=None, reasoning=None, source=None, margin=LOCATE_BOX_MARGIN, crop_and_inspect=True, max_inspect_rounds=1, driving_side=None, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_cm=DEFAULT_MIN_WIDTH_EXT_CM, image_dpi=DEFAULT_IMAGE_DPI, arm_retry_limit=1):
    original = load_rgb_image(image_path)
    min_px = paper_cm_to_px(min_width_ext_cm, image_dpi)
    crop_root = output_dir / "crops"
    lane_results = {}
    agents_by_arm = {}
    trackers_by_arm = {}
    channelized_features = list(channelized_features or [])
    through_roads = list(through_roads or [])
    side = normalize_driving_side(driving_side)
    channelized_features = enrich_channelized_features(channelized_features, arms, side)
    cost = {}
    conversation = {}
    result = None
    try:
        if skip_generation:
            lanes_path = Path(output_dir) / "lanes.json"
            loaded = json.loads(lanes_path.read_text(encoding="utf-8"))
            if loaded.get("channelized_features") is not None:
                channelized_features = list(loaded["channelized_features"])
            if loaded.get("through_roads") is not None:
                through_roads = list(loaded["through_roads"])
            if loaded.get("driving_side") is not None:
                side = normalize_driving_side(loaded.get("driving_side"))
            channelized_features = enrich_channelized_features(channelized_features, arms, side)
            BaseLLM.log(f"[generation] skipped; loaded {lanes_path}")
            for arm in arms:
                BaseLLM.rule(f"LANE-LEVEL ARM {arm['id']} (reload)")
                crop_dir = crop_root / f"arm_{arm['id']}"
                if crop_dir.exists():
                    shutil.rmtree(crop_dir)
                crop_dir.mkdir(parents=True, exist_ok=True)
                tracker = CostTracker(source=source, silent=True)
                crop_bundle = run_rule_initial_crop(original, arm, crop_dir, width_ext_frac=width_ext_frac, min_width_ext_px=min_px)
                agent = create_arm_agent(llm, arm, crop_dir, crop_bundle, channelized_features=channelized_features, through_roads=through_roads, driving_side=side, width_ext_frac=width_ext_frac, min_width_ext_px=min_px)
                agent["result"] = _load_lane_arm_result(loaded.get("arms"), arm["id"])
                agent["inspected"] = True
                agents_by_arm[arm["id"]] = agent
                trackers_by_arm[arm["id"]] = tracker
                lane_results[arm["id"]] = agent["result"]
        else:
            arm_attempts = max(0, int(arm_retry_limit)) + 1
            for arm in arms:
                BaseLLM.rule(f"LANE-LEVEL ARM {arm['id']}")
                crop_dir = crop_root / f"arm_{arm['id']}"
                tracker = CostTracker(source=source, silent=True)
                trackers_by_arm[arm["id"]] = tracker
                agent = None
                def _remember_visual(current, *, arm_id=arm["id"]):
                    agents_by_arm[arm_id] = current

                for attempt in range(arm_attempts):
                    if crop_dir.exists():
                        shutil.rmtree(crop_dir)
                    crop_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        agent = run_arm_agent(
                            llm, original, arm, crop_dir,
                            model=model, tracker=tracker, max_tokens=max_tokens,
                            channelized_features=channelized_features, through_roads=through_roads, reasoning=reasoning,
                            margin=margin, crop_and_inspect=crop_and_inspect, max_inspect_rounds=max_inspect_rounds, driving_side=side,
                            width_ext_frac=width_ext_frac, min_width_ext_px=min_px,
                            on_visual_done=_remember_visual,
                        )
                        break
                    except FileNotFoundError:
                        raise
                    except Exception as exc:
                        if attempt < arm_attempts - 1:
                            print(
                                "[retry] arm %s attempt %s/%s: %s"
                                % (arm["id"], attempt + 2, arm_attempts, exc),
                                flush=True,
                            )
                            continue
                        raise
                agents_by_arm[arm["id"]] = agent
                lane_results[arm["id"]] = agent["result"]
        result = _lanes_doc(image_path, lane_results, channelized_features, through_roads, driving_side=side)
    finally:
        if trackers_by_arm and not skip_generation:
            lanes_doc = result
            if lanes_doc is None and lane_results:
                lanes_doc = _lanes_doc(image_path, lane_results, channelized_features, through_roads, driving_side=side)
            identifier_doc = _stage_lanes_from_agents(
                agents_by_arm, "identifier_result", image_path,
                channelized_features, through_roads, side,
            )
            inspect_doc = _stage_lanes_from_agents(
                agents_by_arm, "inspect_result", image_path,
                channelized_features, through_roads, side,
            )
            conversation, cost = _save_lane_bundle(
                output_dir, agents_by_arm=agents_by_arm, trackers_by_arm=trackers_by_arm, model=model,
                lanes_doc=lanes_doc, identifier_doc=identifier_doc, inspect_doc=inspect_doc,
            )
    tracker = _merged_tracker(trackers_by_arm, source=source)
    return result, tracker, {
        "cost": cost,
        "conversation": conversation,
        "agents_by_arm": agents_by_arm,
        "trackers_by_arm": trackers_by_arm,
        "lane_results": lane_results,
        "driving_side": side,
    }


def run_lanes(model, dataset="nhtsa_2023", case_id="28661", arms_from=None, arms_reasoning=None, skip_generation=False, max_tokens=2048, compressed=True, verbose=True, crop_and_inspect=True, max_inspect_rounds=1, artifact_case_dir=None, width_ext_frac=DEFAULT_WIDTH_EXT_FRAC, min_width_ext_cm=DEFAULT_MIN_WIDTH_EXT_CM, image_dpi=DEFAULT_IMAGE_DPI, arm_retry_limit=1, arms_path=None, image_case_dir=None):
    cfg = unpack_model_config(model)
    api, source, model_name = cfg["api"], cfg["source"], cfg["model_name"]
    api_key, base_url, proxy, reasoning = cfg["api_key"], cfg["base_url"], cfg["proxy"], cfg["reasoning"]
    prev_verbose = set_verbose(verbose)
    _image_path, run_name = resolve_case_inputs(dataset=dataset, case_id=case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else resolve_run_dir("results", run_name)
    image_root = Path(image_case_dir) if image_case_dir is not None else case_dir
    arms_reason = arms_reasoning if arms_reasoning is not None else reasoning
    if arms_path is not None:
        arms_path = Path(arms_path)
    elif arms_from:
        arms_path = arms_path_for_model(image_root, arms_from, arms_reason)
    else:
        arms_path = default_arms_path(image_root, model_name, reasoning)
    processed = load_processed_image(image_root, compressed=compressed)
    image_path = processed["path"]
    image = processed["image"]
    raw_after = processed["raw_bytes"]
    arm_result = json.loads(arms_path.read_text(encoding="utf-8"))
    arms = arm_result["arms"]
    channelized_features = list(arm_result.get("channelized_features") or [])
    through_roads = list(arm_result.get("through_roads") or [])
    side = load_case_driving_side(dataset, case_id)
    out_dir = lane_level_dir(case_dir, model_name, reasoning)
    out_dir = prepare_output_dir(out_dir, clear=not skip_generation)
    BaseLLM.rule("LANE-LEVEL GENERATION")
    parts = image_path.parts
    image_display = str(Path(*parts[parts.index("results"):])) if "results" in parts else str(image_path)
    BaseLLM.log(f"image : {image_display} ({image.width}x{image.height}px, {len(raw_after) / (1024 * 1024):.2f} MB)")
    BaseLLM.log(f"arms  : {arms_path}")
    BaseLLM.log(f"model : {model_name}  reasoning: {reasoning}  driving_side: {side}")
    BaseLLM.log(
        f"crop-and-inspect: {crop_and_inspect}  "
        f"width_ext_frac: {width_ext_frac}  "
        f"min_width_ext_cm: {min_width_ext_cm}  dpi: {image_dpi}"
    )
    llm = build_llm(api, api_key, base_url, proxy=proxy)
    result, tracker, gen_meta = run_lane_level(
        llm, image_path, arms, out_dir,
        model=model_name,
        max_tokens=max_tokens,
        skip_generation=skip_generation,
        channelized_features=channelized_features, through_roads=through_roads,
        reasoning=reasoning, source=source,
        crop_and_inspect=crop_and_inspect, max_inspect_rounds=max_inspect_rounds,
        driving_side=side,
        width_ext_frac=width_ext_frac,
        min_width_ext_cm=min_width_ext_cm,
        image_dpi=image_dpi,
        arm_retry_limit=arm_retry_limit,
    )
    BaseLLM.log(f"saved generation lanes -> {out_dir}")
    if verbose:
        tracker.print_summary()
    BaseLLM.verbose = prev_verbose
    return {
        "result": result,
        "tracker": tracker,
        "gen_meta": gen_meta,
        "output_dir": out_dir,
        "lanes_path": Path(out_dir) / "lanes.json",
        "arms_path": arms_path,
        "run_name": run_name,
    }


if __name__ == "__main__":
    MODEL = {
        "api": "openai",
        "provider": "openrouter",
        "model": "openai/gpt-5.6-sol",
        "reasoning": "medium",
    }
    parser = argparse.ArgumentParser(description="Read the lane cross-section of each arm-level result.")
    parser.add_argument("--dataset", default="openaerialmap", help="Dataset folder under dataset/, e.g. nhtsa_2023")
    parser.add_argument("--case-id", default="1002", help="Case folder name under the dataset")
    parser.add_argument("--arms-from", default="gpt-5.6-sol", help="Arm-level model name to read arms.json from")
    parser.add_argument("--arms-reasoning", default=None, help="Reasoning effort for arms-from dir (default: same as --reasoning / MODEL)")
    parser.add_argument("--model", default=MODEL["model"])
    parser.add_argument("--reasoning", default=MODEL.get("reasoning", "none"))
    parser.add_argument("--crop-and-inspect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--max-inspect-rounds", type=int, default=1,
        help="How many evaluator→inspector passes to run per arm (not a critique count cap)",
    )
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--width-ext-frac",
        type=float,
        default=DEFAULT_WIDTH_EXT_FRAC,
        help="Expand crop width by this fraction of the arm width (0.3 = +30%; same as arm-level --width-ext-frac)",
    )
    parser.add_argument("--min-width-ext-cm", type=float, default=DEFAULT_MIN_WIDTH_EXT_CM, help="Minimum per-side lateral pad in paper centimeters")
    parser.add_argument("--image-dpi", type=float, default=DEFAULT_IMAGE_DPI, help="Scan DPI used to convert min-width-ext-cm to pixels")
    args = parser.parse_args()
    run_lanes(
        {
            "api": MODEL["api"],
            "provider": MODEL["provider"],
            "model": args.model,
            "reasoning": args.reasoning,
        },
        dataset=args.dataset,
        case_id=args.case_id,
        arms_from=args.arms_from,
        arms_reasoning=args.arms_reasoning,
        crop_and_inspect=args.crop_and_inspect,
        max_inspect_rounds=args.max_inspect_rounds,
        max_tokens=args.max_tokens,
        compressed=args.compressed,
        verbose=args.verbose,
        width_ext_frac=args.width_ext_frac,
        min_width_ext_cm=args.min_width_ext_cm,
        image_dpi=args.image_dpi,
    )
