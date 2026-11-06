"""Three-stage junction-level inference with deterministic square crops.

Stage 1: locate distinct junction candidate centers from the full scene, then merge any
         candidates that cover the same physical shared area.
Stage 2: create a fixed-size square crop around each remaining candidate and locally
         verify/refine its center. Nearby confirmed centers are collapsed to one junction.
Stage 3: create a fixed-size final square box around each verified center, then use the
         original full image plus all verified centers to infer direct inter-junction
         links. A link is either a driving connecting_road or a parallel median join.

Junction box size comes from configuration constants, so scale stays consistent.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from scripts.methods.connection_parser import DEFAULT_ARM_LENGTH, DEFAULT_JUNCTION_RADIUS, arm_heading_midpoint
from scripts.methods.image_processing import load_processed_image, process_case_image
from scripts.methods.image_utils import load_rgb_image, save_image_under_limit
from scripts.methods.llm_kg.junction_level.prompts import (
    CANDIDATE_DEDUP_USER_TEXT,
    CANDIDATE_PROMPT,
    CANDIDATE_USER_TEXT,
    CONNECTION_PROMPT,
    VERIFY_PROMPT,
    VERIFY_USER_TEXT_TEMPLATE,
    build_connection_user_text,
)
from scripts.methods.llm_kg.junction_level.schemas import (
    CONNECTION_TYPE_MEDIAN,
    JUNCTION_CANDIDATE_LEVEL_SCHEMA,
    JUNCTION_CONNECTION_LEVEL_SCHEMA,
    JUNCTION_VERIFY_SCHEMA,
    normalize_connection_type,
    normalize_lane_type,
)
from scripts.methods.method_utils import (
    arms_path_for_model,
    dump_json,
    junction_level_dir,
    junction_work_dir,
    prepare_output_dir,
    resolve_case_inputs,
    resolve_run_dir,
    set_verbose,
    unpack_model_config,
    xodr_result_dir_for_model,
)
from scripts.methods.llm_utils import BaseLLM, CostTracker, build_llm, to_jsonable


# All sizes below are fractions of the processed square image side.
#
# Stage 2 needs a relatively large square because its purpose is to verify whether
# the candidate is a true junction using sufficient surrounding roadway context.
VERIFY_BOX_SIZE = 0.4

# The final square is the standardized image passed to downstream arm/lane stages.
# Tune this single global hyperparameter on a validation set if more/less surrounding
# roadway context is desired.
FINAL_BOX_SIZE = 0.3

# Near-identical seeds (same point reported twice). 0.05 ≈ 20 m at 400 m / side.
CANDIDATE_DUP_DIST = 0.05

# After verification, refined centers this close are the same physical junction.
# 0.08 ≈ 32 m at 400 m / side; distinct nearby junctions in a scene are typically farther.
CONFIRMED_MERGE_DIST = 0.08

MIN_CROP_PX = 64
METERS_PER_IMAGE_SIDE = 400.0
# Keep the OpenDRIVE segment positive only. Do not inflate short inter-junction gaps.
MIN_CONNECTOR_LENGTH_M = 1.0
MIN_JUNCTION_RADIUS_M = 20.0
MAX_ARM_OPPOSITE_TOL_DEG = 20.0  # |arm_a heading - arm_b heading| must be 180° ± this

_OVERLAY_COLORS = (
    (255, 64, 64),
    (32, 140, 255),
    (32, 190, 90),
    (255, 170, 24),
    (190, 80, 220),
    (24, 190, 190),
)


def _normalize_heading_deg(heading_deg):
    return float(heading_deg) % 360.0


def _signed_heading_delta(from_deg, to_deg):
    return (_normalize_heading_deg(to_deg) - _normalize_heading_deg(from_deg) + 180.0) % 360.0 - 180.0


def _heading_of_image_vec(dx, dy):
    """Heading of an image-space vector (x right, y down; 0°=right, 90°=up)."""
    return math.degrees(math.atan2(-float(dy), float(dx))) % 360.0


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


def _draw_dashed_line(draw, p0, p1, *, fill, width, dash=18, gap=10):
    x0, y0 = p0
    x1, y1 = p1
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0:
        return
    dx, dy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    on = True
    while pos < length:
        seg = dash if on else gap
        end = min(pos + seg, length)
        if on:
            draw.line(
                [(x0 + dx * pos, y0 + dy * pos), (x0 + dx * end, y0 + dy * end)],
                fill=fill,
                width=width,
            )
        pos = end
        on = not on


def box_aabb(box):
    """Return (x_min, y_min, x_max, y_max) from four [x, y] corners."""
    xs = [float(pt[0]) for pt in box]
    ys = [float(pt[1]) for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def box_center(box):
    xmin, ymin, xmax, ymax = box_aabb(box)
    return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0


def junction_center(junction):
    """Conflict-area center if present; otherwise the AABB midpoint."""
    center = junction.get("center") if isinstance(junction, dict) else None
    if isinstance(center, (list, tuple)) and len(center) >= 2:
        return float(center[0]), float(center[1])
    return box_center(junction["box"])


def box_as_corners(xmin, ymin, xmax, ymax):
    return [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]


def _clamp01(value):
    return min(max(float(value), 0.0), 1.0)


def fixed_square_box(center, side_frac):
    """Return a fixed-size normalized square, shifted inward at image boundaries.

    The requested side length is preserved whenever side_frac <= 1.0. Near an image
    boundary the square is translated, rather than clipped asymmetrically, so every
    junction receives the same crop scale.
    """
    side = min(max(float(side_frac), 1e-4), 1.0)
    half = 0.5 * side
    cx = _clamp01(center[0])
    cy = _clamp01(center[1])

    xmin = cx - half
    xmax = cx + half
    ymin = cy - half
    ymax = cy + half

    if xmin < 0.0:
        xmax -= xmin
        xmin = 0.0
    if xmax > 1.0:
        xmin -= (xmax - 1.0)
        xmax = 1.0
    if ymin < 0.0:
        ymax -= ymin
        ymin = 0.0
    if ymax > 1.0:
        ymin -= (ymax - 1.0)
        ymax = 1.0

    xmin = max(0.0, xmin)
    ymin = max(0.0, ymin)
    xmax = min(1.0, xmax)
    ymax = min(1.0, ymax)
    return box_as_corners(xmin, ymin, xmax, ymax)


def normalize_candidate_result(result):
    """Normalize candidate IDs/centers and deterministically create verification squares."""
    result = dict(result or {})
    candidates = []
    seen = set()
    for raw in result.get("candidates") or []:
        cid = int(raw["id"])
        if cid in seen:
            raise ValueError("duplicate candidate id %s" % cid)
        seen.add(cid)

        center = raw.get("center")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            raise ValueError("candidate %s has no valid center" % cid)

        cx, cy = _clamp01(center[0]), _clamp01(center[1])
        candidates.append({
            "id": cid,
            "center": [cx, cy],
            "box": fixed_square_box([cx, cy], VERIFY_BOX_SIZE),
            "evidence": str(raw.get("evidence") or ""),
        })

    if not candidates:
        raise ValueError("candidate stage returned no junction candidates")

    result["candidates"] = candidates
    return result


def _center_dist(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _cluster_by_distance(items, max_dist, center_fn):
    """Union-find clusters of items whose centers are within ``max_dist``."""
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if _center_dist(center_fn(items[i]), center_fn(items[j])) <= max_dist:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    return list(groups.values())


def collapse_nearby_candidates(candidates, max_dist=CANDIDATE_DUP_DIST):
    """Keep one candidate per near-identical seed; average the collapsed centers."""
    if len(candidates) <= 1:
        return list(candidates)

    collapsed = []
    for group in _cluster_by_distance(candidates, max_dist, lambda c: c["center"]):
        group = sorted(group, key=lambda c: int(c["id"]))
        keep = dict(group[0])
        if len(group) > 1:
            cx = sum(float(c["center"][0]) for c in group) / len(group)
            cy = sum(float(c["center"][1]) for c in group) / len(group)
            keep["center"] = [cx, cy]
            keep["box"] = fixed_square_box([cx, cy], VERIFY_BOX_SIZE)
            dropped = [int(c["id"]) for c in group[1:]]
            keep["evidence"] = (
                (keep.get("evidence") or "")
                + " Collapsed duplicate candidates %s." % dropped
            ).strip()
            BaseLLM.log(
                "collapse candidates %s -> %s at %s"
                % (
                    [int(c["id"]) for c in group],
                    keep["id"],
                    [round(cx, 4), round(cy, 4)],
                )
            )
        collapsed.append(keep)
    return sorted(collapsed, key=lambda c: int(c["id"]))


def collapse_nearby_junctions(junctions, max_dist=CONFIRMED_MERGE_DIST):
    """Merge confirmed junctions whose refined centers land on the same shared area."""
    if len(junctions) <= 1:
        return list(junctions)

    merged = []
    for group in _cluster_by_distance(junctions, max_dist, lambda j: j["center"]):
        group = sorted(group, key=lambda j: int(j["id"]))
        keep = dict(group[0])
        if len(group) > 1:
            cx = sum(float(j["center"][0]) for j in group) / len(group)
            cy = sum(float(j["center"][1]) for j in group) / len(group)
            keep["center"] = [cx, cy]
            keep["box"] = fixed_square_box([cx, cy], FINAL_BOX_SIZE)
            dropped = [int(j["id"]) for j in group[1:]]
            keep["evidence"] = (
                (keep.get("evidence") or "")
                + " Merged duplicate candidates %s." % dropped
            ).strip()
            BaseLLM.log(
                "merge confirmed junctions %s -> %s at %s"
                % (
                    [int(j["id"]) for j in group],
                    keep["id"],
                    [round(cx, 4), round(cy, 4)],
                )
            )
        merged.append(keep)
    return sorted(merged, key=lambda j: int(j["id"]))


def normalize_junction_result(result):
    """Normalize verified junctions and direct inter-junction connections."""
    result = dict(result or {})
    junctions = []
    seen = set()

    for raw in result.get("junctions") or []:
        jid = int(raw["id"])
        if jid in seen:
            raise ValueError("duplicate junction id %s" % jid)
        seen.add(jid)

        center = raw.get("center")
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            raise ValueError("junction %s has no valid center" % jid)

        cx, cy = _clamp01(center[0]), _clamp01(center[1])
        item = dict(raw)
        item["id"] = jid
        item["center"] = [cx, cy]
        item["box"] = fixed_square_box([cx, cy], FINAL_BOX_SIZE)
        junctions.append(item)

    if not junctions:
        raise ValueError("junction-level result has no confirmed junctions")

    valid_ids = {j["id"] for j in junctions}
    connections = []
    seen_pairs = set()

    for raw in result.get("connections") or []:
        ids = [int(x) for x in (raw.get("junction_ids") or [])]
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        if ids[0] not in valid_ids or ids[1] not in valid_ids:
            raise ValueError("connection %s references unknown junction ids" % ids)

        pair = tuple(sorted(ids))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        connections.append({
            "junction_ids": [pair[0], pair[1]],
            "connection_type": normalize_connection_type(raw.get("connection_type")),
            "evidence": str(raw.get("evidence") or ""),
        })

    return {
        "junctions": junctions,
        "connections": connections,
        "connections_evidence": str(result.get("connections_evidence") or ""),
    }


def _norm_to_px(value, side):
    return float(value) * float(side)


def exact_crop_px(box, width, height):
    """Pixel crop box from an already-final normalized square box, with no extra padding."""
    side = float(max(width, height))
    xmin, ymin, xmax, ymax = box_aabb(box)

    left = int(round(_norm_to_px(xmin, side)))
    top = int(round(_norm_to_px(ymin, side)))
    right = int(round(_norm_to_px(xmax, side)))
    bottom = int(round(_norm_to_px(ymax, side)))

    left = max(0, min(left, width - 1))
    top = max(0, min(top, height - 1))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    if right - left < MIN_CROP_PX:
        extra = MIN_CROP_PX - (right - left)
        left = max(0, left - extra // 2)
        right = min(width, left + MIN_CROP_PX)
        left = max(0, right - MIN_CROP_PX)

    if bottom - top < MIN_CROP_PX:
        extra = MIN_CROP_PX - (bottom - top)
        top = max(0, top - extra // 2)
        bottom = min(height, top + MIN_CROP_PX)
        top = max(0, bottom - MIN_CROP_PX)

    return left, top, right, bottom


def write_junction_crop_images(full_image, box, images_dir, *, stem="junction"):
    """Crop the deterministic square box, pad only if source-image geometry requires it."""
    images_dir = prepare_output_dir(images_dir, clear=True)
    width, height = full_image.size
    full_side = float(max(width, height))

    left, top, right, bottom = exact_crop_px(box, width, height)
    crop = full_image.crop((left, top, right, bottom)).convert("RGB")
    crop_path = images_dir / ("%s_cropped.png" % stem)
    crop.save(crop_path, format="PNG", optimize=True)

    crop_w, crop_h = crop.size
    square_side = max(crop_w, crop_h)
    pad_x = (square_side - crop_w) // 2
    pad_y = (square_side - crop_h) // 2

    square = Image.new("RGB", (square_side, square_side), (255, 255, 255))
    square.paste(crop, (pad_x, pad_y))

    square_path = images_dir / ("%s_cropped_square.png" % stem)
    square.save(square_path, format="PNG", optimize=True)

    compressed_path, raw_bytes, media_type = save_image_under_limit(
        square, images_dir, square_path.stem,
    )
    compressed = load_rgb_image(compressed_path)

    meta = {
        "full_width": width,
        "full_height": height,
        "full_side": full_side,
        "crop_box_px": [left, top, right, bottom],
        "crop_width": crop_w,
        "crop_height": crop_h,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "square_side": float(square_side),
        "square": str(square_path.resolve()),
        "cropped": str(crop_path.resolve()),
        "compressed": str(Path(compressed_path).resolve()),
        "width": compressed.width,
        "height": compressed.height,
        "bytes": len(raw_bytes),
        "media_type": media_type,
    }
    dump_json(images_dir / "meta.json", meta)
    return meta


def crop_norm_to_full_norm(pt, meta):
    """Map crop-square [0,1] coordinates to full-square normalized coordinates."""
    square_side = float(meta["square_side"])
    full_side = float(meta["full_side"])
    x_px = (
        float(pt[0]) * square_side
        - float(meta["pad_x"])
        + float(meta["crop_box_px"][0])
    )
    y_px = (
        float(pt[1]) * square_side
        - float(meta["pad_y"])
        + float(meta["crop_box_px"][1])
    )
    return [_clamp01(x_px / full_side), _clamp01(y_px / full_side)]


def full_norm_to_crop_norm(pt, meta):
    """Map full-square normalized coordinates to crop-square [0,1] coordinates."""
    full_side = float(meta["full_side"])
    square_side = float(meta["square_side"])
    x_px = (
        float(pt[0]) * full_side
        - float(meta["crop_box_px"][0])
        + float(meta["pad_x"])
    )
    y_px = (
        float(pt[1]) * full_side
        - float(meta["crop_box_px"][1])
        + float(meta["pad_y"])
    )
    return [x_px / square_side, y_px / square_side]


def map_arms_doc_to_full_image(arms_doc, meta):
    """Copy an arms.json and rewrite mouth geometry into full-image normalized coords."""
    if not meta or not arms_doc:
        return arms_doc

    scale = float(meta["square_side"]) / float(meta["full_side"])
    out = json.loads(json.dumps(arms_doc))

    for arm in out.get("arms") or []:
        edges = arm.get("mouth_edges") or []
        if len(edges) == 2:
            arm["mouth_edges"] = [crop_norm_to_full_norm(pt, meta) for pt in edges]

        if arm.get("arm_mouth_center") and len(arm["arm_mouth_center"]) >= 2:
            arm["arm_mouth_center"] = crop_norm_to_full_norm(
                arm["arm_mouth_center"], meta,
            )

        if arm.get("anchor_point") and len(arm["anchor_point"]) >= 2:
            arm["anchor_point"] = crop_norm_to_full_norm(arm["anchor_point"], meta)
        if arm.get("junction_center") and len(arm["junction_center"]) >= 2:
            arm["junction_center"] = crop_norm_to_full_norm(arm["junction_center"], meta)
        if arm.get("seed_point") and len(arm["seed_point"]) >= 2:
            arm["seed_point"] = crop_norm_to_full_norm(arm["seed_point"], meta)

        if arm.get("length") is not None:
            arm["length"] = float(arm["length"]) * scale
        if arm.get("width") is not None:
            arm["width"] = float(arm["width"]) * scale
        if arm.get("lookback") is not None:
            arm["lookback"] = float(arm["lookback"]) * scale

    if out.get("junction_center") and len(out["junction_center"]) >= 2:
        out["junction_center"] = crop_norm_to_full_norm(out["junction_center"], meta)

    return out


def render_junction_overlay(image, result, *, label_prefix="junc"):
    """Draw deterministic boxes/centers and optional connection chords.

    Boxes may overlap. Fills are drawn first, then every outline, so a later
    box never hides another box's border.
    """
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    width, height = canvas.size
    side = float(max(width, height))
    line_w = max(2, round(min(width, height) / 450))
    font = _overlay_font(max(16, round(min(width, height) / 42)))

    def xy(point):
        return float(point[0]) * side, float(point[1]) * side

    items = result.get("junctions")
    if items is None:
        items = result.get("candidates") or []

    drawn = []
    for i, item in enumerate(items):
        rgb = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
        xmin, ymin, xmax, ymax = box_aabb(item["box"])
        corners = [xy(pt) for pt in box_as_corners(xmin, ymin, xmax, ymax)]
        cx, cy = xy(junction_center(item))
        drawn.append((item, rgb, corners, (cx, cy)))

    for _item, rgb, corners, _center in drawn:
        draw.polygon(corners, fill=(*rgb, 48))

    centers = {}
    for item, _rgb, _corners, center in drawn:
        centers[int(item["id"])] = center
    for link in result.get("connections") or []:
        ids = [int(x) for x in (link.get("junction_ids") or [])]
        if len(ids) != 2 or ids[0] not in centers or ids[1] not in centers:
            continue
        kind = normalize_connection_type(link.get("connection_type"))
        if kind == CONNECTION_TYPE_MEDIAN:
            _draw_dashed_line(
                draw,
                centers[ids[0]],
                centers[ids[1]],
                fill=(255, 210, 40, 230),
                width=max(2, line_w),
            )
        else:
            draw.line(
                [centers[ids[0]], centers[ids[1]]],
                fill=(255, 255, 255, 200),
                width=max(2, line_w),
            )

    for _item, rgb, corners, _center in drawn:
        draw.line([*corners, corners[0]], fill=(*rgb, 230), width=line_w)

    r = max(5, round(min(width, height) / 90))
    for item, rgb, _corners, (cx, cy) in drawn:
        draw.ellipse(
            (cx - r, cy - r, cx + r, cy + r),
            fill=(*rgb, 255),
            outline=(255, 255, 255, 255),
            width=max(2, line_w),
        )
        _draw_label(
            draw,
            (cx + r + 4, cy - r - 6),
            "%s %s" % (label_prefix, item["id"]),
            font=font,
            fill=(*rgb, 255),
            stroke=(0, 0, 0, 220),
        )

    return Image.alpha_composite(canvas, layer).convert("RGB")


def _write_overlay(
    output_dir,
    image,
    result,
    filename="overlay.png",
    *,
    label_prefix="junc",
):
    if image is None or not result:
        return None
    try:
        overlay = render_junction_overlay(
            image, result, label_prefix=label_prefix,
        )
        path = Path(output_dir) / filename
        overlay.save(path)
        BaseLLM.log("overlay -> %s" % path)
        return path
    except Exception as exc:
        BaseLLM.log("overlay %s failed: %s" % (filename, exc))
        return None


def save_junction_level_artifacts(
    output_dir,
    artifact,
    tracker,
    *,
    name=None,
    source_image=None,
    image=None,
):
    paths = {}
    output_dir = Path(output_dir)

    if artifact:
        conversation = {
            "messages": artifact.get("messages", []),
            "candidate_messages": artifact.get("candidate_messages", []),
            "verification_messages": artifact.get("verification_messages", {}),
            "connection_messages": artifact.get("connection_messages", []),
            "candidate_result": artifact.get("candidate_result"),
            "candidate_result_raw": artifact.get("candidate_result_raw"),
            "verification_results": artifact.get("verification_results"),
        }
        if name is not None:
            conversation["name"] = name
        if source_image is not None:
            conversation["source_image"] = str(source_image)

        paths["conversation"] = output_dir / "conversation.json"
        dump_json(paths["conversation"], conversation)

        if artifact.get("candidate_result_raw"):
            paths["candidates_raw"] = output_dir / "candidates_raw.json"
            dump_json(paths["candidates_raw"], artifact["candidate_result_raw"])

        if artifact.get("candidate_result"):
            paths["candidates"] = output_dir / "candidates.json"
            dump_json(paths["candidates"], artifact["candidate_result"])
            paths["candidate_overlay"] = _write_overlay(
                output_dir,
                image,
                artifact["candidate_result"],
                filename="candidate_overlay.png",
                label_prefix="cand",
            )

        if artifact.get("verification_results") is not None:
            paths["verifications"] = output_dir / "verifications.json"
            dump_json(paths["verifications"], artifact["verification_results"])

        if artifact.get("result"):
            paths["junctions"] = output_dir / "junctions.json"
            dump_json(paths["junctions"], artifact["result"])
            paths["overlay"] = _write_overlay(
                output_dir,
                image,
                artifact["result"],
                filename="overlay.png",
            )

    paths["cost"] = output_dir / "cost.json"
    dump_json(paths["cost"], tracker.summary())
    return paths


def save_junction_crops(output_dir, image, result, case_dir):
    """Write the fixed-size final square crop for every confirmed junction."""
    crop_dir = Path(output_dir) / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    crops = {}

    for junction in result.get("junctions") or []:
        jid = int(junction["id"])
        work_dir = junction_work_dir(case_dir, jid)

        meta = write_junction_crop_images(
            image,
            junction["box"],
            work_dir / "images",
            stem="junction_%s" % jid,
        )

        preview = load_rgb_image(meta["square"])
        preview_path = crop_dir / ("junction_%s.png" % jid)
        preview.save(preview_path)

        crops[jid] = {
            "path": str(preview_path),
            "work_dir": str(work_dir),
            "meta": meta,
        }
        BaseLLM.log("final fixed crop junction %s -> %s" % (
            jid, work_dir / "images",
        ))

    dump_json(Path(output_dir) / "crops.json", crops)
    return crops


def run_candidate_stage(
    llm,
    image,
    *,
    model,
    tracker,
    max_tokens=1024,
    reasoning=None,
):
    image_msg = llm.image_user_message(image, CANDIDATE_USER_TEXT)
    messages = [image_msg]

    result, _text, response = llm.call_json(
        model=model,
        system_prompt=CANDIDATE_PROMPT,
        messages=messages,
        schema=JUNCTION_CANDIDATE_LEVEL_SCHEMA,
        max_tokens=max_tokens,
        reasoning_effort=reasoning,
    )
    tracker.record("junction_candidates", model, response)
    messages.append(llm.assistant_message(response))

    raw_result = normalize_candidate_result(result)
    raw_result["candidates"] = collapse_nearby_candidates(
        raw_result["candidates"],
    )

    result = raw_result
    if len(raw_result["candidates"]) > 1:
        messages.append(llm.make_user_message(CANDIDATE_DEDUP_USER_TEXT))
        merged, _text, response = llm.call_json(
            model=model,
            system_prompt=CANDIDATE_PROMPT,
            messages=messages,
            schema=JUNCTION_CANDIDATE_LEVEL_SCHEMA,
            max_tokens=max_tokens,
            reasoning_effort=reasoning,
        )
        tracker.record("junction_candidate_dedup", model, response)
        messages.append(llm.assistant_message(response))
        try:
            result = normalize_candidate_result(merged)
            result["candidates"] = collapse_nearby_candidates(
                result["candidates"],
            )
        except ValueError as exc:
            BaseLLM.log("candidate dedup rejected (%s); keeping first-pass list" % exc)
            result = raw_result

    kept_ids = [int(c["id"]) for c in result["candidates"]]
    raw_ids = [int(c["id"]) for c in raw_result["candidates"]]
    if kept_ids != raw_ids:
        BaseLLM.log(
            "candidate dedup %s -> %s"
            % (raw_ids, kept_ids)
        )

    return {
        "messages": to_jsonable(messages),
        "result": result,
        "raw_result": raw_result,
    }


def run_verification_stage(
    llm,
    full_image,
    candidates,
    out_dir,
    *,
    model,
    tracker,
    max_tokens=1024,
    reasoning=None,
):
    """Verify each fixed-size candidate crop and map refined centers to the full image."""
    verify_root = prepare_output_dir(
        Path(out_dir) / "candidate_crops", clear=True,
    )

    confirmed = []
    verification_results = []
    verification_messages = {}

    for candidate in candidates:
        cid = int(candidate["id"])
        candidate_dir = verify_root / ("candidate_%s" % cid)

        # Candidate["box"] is already a deterministic VERIFY_BOX_SIZE square.
        meta = write_junction_crop_images(
            full_image,
            candidate["box"],
            candidate_dir,
            stem="candidate_%s" % cid,
        )

        crop_image = load_rgb_image(meta["compressed"])
        candidate_in_crop = full_norm_to_crop_norm(
            candidate["center"], meta,
        )

        # Verification is intentionally independent of Stage-1 semantic evidence.
        # Passing the candidate explanation here can anchor the verifier to the
        # original hypothesis (e.g., "plausible T-junction") and encourage confirmation.
        user_text = VERIFY_USER_TEXT_TEMPLATE.format(
            candidate_id=cid,
            cx=candidate_in_crop[0],
            cy=candidate_in_crop[1],
        )
        image_msg = llm.image_user_message(crop_image, user_text)
        messages = [image_msg]

        result, _text, response = llm.call_json(
            model=model,
            system_prompt=VERIFY_PROMPT,
            messages=messages,
            schema=JUNCTION_VERIFY_SCHEMA,
            max_tokens=max_tokens,
            reasoning_effort=reasoning,
        )
        tracker.record("junction_verify", model, response)
        messages.append(llm.assistant_message(response))
        verification_messages[str(cid)] = to_jsonable(messages)

        status = str(result.get("status") or "").strip()

        record = {
            "candidate_id": cid,
            "candidate_center": candidate["center"],
            "verification_box": candidate["box"],
            "status": status,
            "local_center": result.get("center"),
            "evidence": str(result.get("evidence") or ""),
            "uncertainty": str(result.get("uncertainty") or ""),
            "crop_meta": meta,
        }

        if status == "confirmed":
            full_center = crop_norm_to_full_norm(
                result["center"], meta,
            )
            final_box = fixed_square_box(
                full_center, FINAL_BOX_SIZE,
            )

            junction = {
                "id": cid,
                "center": full_center,
                "box": final_box,
                "evidence": str(result.get("evidence") or ""),
                "uncertainty": str(result.get("uncertainty") or ""),
            }
            confirmed.append(junction)

            record["full_center"] = full_center
            record["final_box"] = final_box
            BaseLLM.log(
                "candidate %s -> confirmed center=%s"
                % (cid, [round(x, 4) for x in full_center])
            )
        else:
            BaseLLM.log("candidate %s -> rejected" % cid)

        verification_results.append(record)

    confirmed = collapse_nearby_junctions(confirmed)

    if not confirmed:
        raise ValueError(
            "all junction candidates were rejected during local verification"
        )

    return {
        "junctions": confirmed,
        "results": verification_results,
        "messages": verification_messages,
    }


def run_connection_stage(
    llm,
    full_image,
    junctions,
    *,
    model,
    tracker,
    max_tokens=1024,
    reasoning=None,
):
    """Infer direct links from the original full image and fixed verified centers."""
    if len(junctions) <= 1:
        return {
            "messages": [],
            "result": {
                "connections": [],
                "connections_evidence": (
                    "Only one confirmed junction is present."
                ),
            },
        }

    user_text = build_connection_user_text(junctions)
    image_msg = llm.image_user_message(full_image, user_text)
    messages = [image_msg]

    result, _text, response = llm.call_json(
        model=model,
        system_prompt=CONNECTION_PROMPT,
        messages=messages,
        schema=JUNCTION_CONNECTION_LEVEL_SCHEMA,
        max_tokens=max_tokens,
        reasoning_effort=reasoning,
    )
    tracker.record("junction_connections", model, response)
    messages.append(llm.assistant_message(response))

    valid_ids = {int(j["id"]) for j in junctions}
    seen_pairs = set()
    connections = []

    for raw in result.get("connections") or []:
        ids = [int(x) for x in (raw.get("junction_ids") or [])]
        if len(ids) != 2 or ids[0] == ids[1]:
            continue
        if ids[0] not in valid_ids or ids[1] not in valid_ids:
            BaseLLM.log(
                "drop connection %s: unknown confirmed junction id" % ids
            )
            continue

        pair = tuple(sorted(ids))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)

        connections.append({
            "junction_ids": [pair[0], pair[1]],
            "connection_type": normalize_connection_type(
                raw.get("connection_type")
            ),
            "evidence": str(raw.get("evidence") or ""),
        })

    return {
        "messages": to_jsonable(messages),
        "result": {
            "connections": connections,
            "connections_evidence": str(
                result.get("connections_evidence") or ""
            ),
        },
    }


def run_junction_level(
    model,
    dataset="nhtsa_2023",
    case_id="28661",
    max_tokens=1024,
    compressed=True,
    verbose=True,
    output_dir="results",
):
    cfg = unpack_model_config(model)
    api, source, model_name = (
        cfg["api"], cfg["source"], cfg["model_name"]
    )
    api_key, base_url, proxy, reasoning = (
        cfg["api_key"],
        cfg["base_url"],
        cfg["proxy"],
        cfg["reasoning"],
    )

    prev_verbose = set_verbose(verbose)

    _source_image, run_name = resolve_case_inputs(
        dataset=dataset, case_id=case_id,
    )
    case_dir = resolve_run_dir(output_dir, run_name)
    out_dir = prepare_output_dir(
        junction_level_dir(case_dir, model_name, reasoning),
        clear=True,
    )

    junctions_root = case_dir / "junctions"
    if junctions_root.exists():
        shutil.rmtree(junctions_root)

    processed = load_processed_image(
        case_dir, compressed=compressed,
    )
    image_path = processed["path"]
    image = processed["image"]
    raw_after = processed["raw_bytes"]

    llm = build_llm(api, api_key, base_url, proxy=proxy)
    tracker = CostTracker(source=source, silent=True)

    BaseLLM.rule("JUNCTION-LEVEL INFERENCE")
    parts = image_path.parts
    image_display = (
        str(Path(*parts[parts.index("results"):]))
        if "results" in parts
        else str(image_path)
    )
    BaseLLM.log(
        "image : %s (%sx%spx, %.2f MB)"
        % (
            image_display,
            image.width,
            image.height,
            len(raw_after) / (1024 * 1024),
        )
    )
    BaseLLM.log(
        "model : %s  reasoning: %s" % (model_name, reasoning)
    )
    BaseLLM.log(
        "verify square: %.3f  final square: %.3f  "
        "candidate-dup: %.3f  confirmed-merge: %.3f"
        % (
            VERIFY_BOX_SIZE,
            FINAL_BOX_SIZE,
            CANDIDATE_DUP_DIST,
            CONFIRMED_MERGE_DIST,
        )
    )

    artifact = None

    try:
        BaseLLM.rule("STAGE 1 - CANDIDATE CENTER LOCALIZATION")
        candidate_artifact = run_candidate_stage(
            llm,
            image,
            model=model_name,
            tracker=tracker,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )

        BaseLLM.rule("STAGE 2 - FIXED-CROP LOCAL VERIFICATION")
        verification = run_verification_stage(
            llm,
            image,
            candidate_artifact["result"]["candidates"],
            out_dir,
            model=model_name,
            tracker=tracker,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )

        BaseLLM.rule("STAGE 3 - INTER-JUNCTION CONNECTIONS")
        connection_artifact = run_connection_stage(
            llm,
            image,
            verification["junctions"],
            model=model_name,
            tracker=tracker,
            max_tokens=max_tokens,
            reasoning=reasoning,
        )

        final_result = normalize_junction_result({
            "junctions": verification["junctions"],
            "connections": connection_artifact["result"]["connections"],
            "connections_evidence": (
                connection_artifact["result"]["connections_evidence"]
            ),
        })

        all_messages = []
        all_messages.extend(candidate_artifact["messages"])
        for cid in sorted(
            verification["messages"],
            key=lambda x: int(x),
        ):
            all_messages.extend(verification["messages"][cid])
        all_messages.extend(connection_artifact["messages"])

        artifact = {
            "messages": all_messages,
            "candidate_messages": candidate_artifact["messages"],
            "verification_messages": verification["messages"],
            "connection_messages": connection_artifact["messages"],
            "candidate_result": candidate_artifact["result"],
            "candidate_result_raw": candidate_artifact.get("raw_result"),
            "verification_results": verification["results"],
            "result": final_result,
        }

        crops = save_junction_crops(
            out_dir, image, final_result, case_dir,
        )
        artifact["crops"] = {
            str(jid): {
                "path": crop["path"],
                "work_dir": crop["work_dir"],
            }
            for jid, crop in crops.items()
        }

    finally:
        save_junction_level_artifacts(
            out_dir,
            artifact,
            tracker,
            name=run_name,
            source_image=image_path,
            image=image,
        )

    if verbose and artifact:
        BaseLLM.rule("RESULT")
        print(
            json.dumps(
                artifact["result"],
                indent=2,
                ensure_ascii=False,
            ),
            flush=True,
        )
        tracker.print_summary()

    BaseLLM.verbose = prev_verbose
    return artifact




def _arm_heading(arm):
    heading = arm_heading_midpoint(arm)
    if heading is None:
        raise ValueError("arm %s has no heading_deg" % arm.get("id"))
    return heading


def _arm_opposite_err_deg(heading_a, heading_b):
    """Degrees away from exactly opposite (180°). 0 means headings differ by 180°."""
    return abs(abs(_signed_heading_delta(heading_a, heading_b)) - 180.0)


def _axis_mean_heading(heading_a, heading_b):
    """Mean heading of two directions on the same undirected axis."""
    ha = _normalize_heading_deg(heading_a)
    hb = _normalize_heading_deg(heading_b)
    if abs(_signed_heading_delta(ha, hb)) > 90.0:
        hb = (hb + 180.0) % 360.0
    return _normalize_heading_deg(ha + 0.5 * _signed_heading_delta(ha, hb))


def _junction_through_heading(arms):
    """Heading of this junction's through-road (best opposite local arm pair)."""
    best = None
    items = list(arms or [])
    for i, arm_a in enumerate(items):
        ha = _arm_heading(arm_a)
        for arm_b in items[i + 1:]:
            hb = _arm_heading(arm_b)
            err = _arm_opposite_err_deg(ha, hb)
            if err > MAX_ARM_OPPOSITE_TOL_DEG:
                continue
            if best is None or err < best[0]:
                best = (err, ha)
    return None if best is None else best[1]


def _match_opposite_arm_pair(
    arms_a, arms_b, heading_ab, heading_ba, *, used_a, used_b,
    max_opposite_err_deg=MAX_ARM_OPPOSITE_TOL_DEG,
):
    """Pick unused arms whose outward headings differ by ~180°.

    Both ends of a connecting road point away from their own junction, so the two
    mouth headings should be opposite. If several pairs qualify, keep the pair
    that best faces the other junction (chord A→B / B→A).
    """
    best = None
    closest_err = None
    for arm_a in arms_a or []:
        id_a = int(arm_a["id"])
        if id_a in used_a:
            continue
        ha = _arm_heading(arm_a)
        for arm_b in arms_b or []:
            id_b = int(arm_b["id"])
            if id_b in used_b:
                continue
            hb = _arm_heading(arm_b)
            opposite_err = _arm_opposite_err_deg(ha, hb)
            if closest_err is None or opposite_err < closest_err:
                closest_err = opposite_err
            if opposite_err > float(max_opposite_err_deg):
                continue
            face_a = abs(_signed_heading_delta(ha, heading_ab))
            face_b = abs(_signed_heading_delta(hb, heading_ba))
            score = (face_a + face_b, opposite_err)
            if best is None or score < best[0]:
                best = (score, arm_a, arm_b, opposite_err, face_a, face_b)
    if best is None:
        return None, None, closest_err, None
    _score, arm_a, arm_b, opposite_err, face_a, face_b = best
    return arm_a, arm_b, opposite_err, {
        "from": face_a,
        "to": face_b,
        "opposite_err_deg": opposite_err,
    }


def _world_xy(cx, cy, meters_per_side=METERS_PER_IMAGE_SIDE):
    """Image-normalized center -> world meters (x right, y up)."""
    return (float(cx) - 0.5) * meters_per_side, (0.5 - float(cy)) * meters_per_side


def _junction_radius_m(box, meters_per_side=METERS_PER_IMAGE_SIDE):
    xmin, ymin, xmax, ymax = box_aabb(box)
    width_m = (xmax - xmin) * meters_per_side
    height_m = (ymax - ymin) * meters_per_side
    # The crop envelope is larger than the conflict area; cap so connectors stay positive.
    return max(MIN_JUNCTION_RADIUS_M, min(DEFAULT_JUNCTION_RADIUS, 0.25 * min(width_m, height_m)))


def _connector_arm_id(item, key):
    value = item.get(key)
    if value is None:
        return None
    return int(value)


def derive_inter_junction_roads(
    junctions_doc, arms_by_junction, *, meters_per_side=METERS_PER_IMAGE_SIDE, strict=True,
):
    """Match connected junction pairs to opposite local arms (~180° headings).

    ``connecting_road`` links require a matched opposite-arm pair (same as before).
    ``median`` links never consume driving arms: each junction keeps its local
    carriageway, and the strip is only a non-driving join between the two boxes.

    Returns a list of connector records::

        {
          "junction_ids": [a, b],
          "connection_type": "connecting_road" | "median",
          "from_junction": larger_id, "to_junction": smaller_id,
          "from_arm": local arm id at from_junction or None,
          "to_arm": local arm id at to_junction or None,
          "heading_deg": heading from from_junction toward to_junction,
          "length": meters between radii,
          "heading_match_deg": {"from": ..., "to": ..., "opposite_err_deg": ...} or None,
        }
    """
    junctions = {int(j["id"]): j for j in (junctions_doc.get("junctions") or [])}
    used = {jid: set() for jid in junctions}
    connectors = []
    for link in junctions_doc.get("connections") or []:
        ids = [int(x) for x in (link.get("junction_ids") or [])]
        if len(ids) != 2:
            continue
        a, b = ids[0], ids[1]
        if a not in junctions or b not in junctions:
            raise ValueError("inter-junction link %s has unknown ids" % ids)
        connection_type = normalize_connection_type(link.get("connection_type"))
        is_median = connection_type == CONNECTION_TYPE_MEDIAN
        ca = junction_center(junctions[a])
        cb = junction_center(junctions[b])
        heading_ab = _heading_of_image_vec(cb[0] - ca[0], cb[1] - ca[1])
        heading_ba = _heading_of_image_vec(ca[0] - cb[0], ca[1] - cb[1])
        arm_a = arm_b = None
        match_ab = None
        opposite_err = None
        if not is_median:
            arm_a, arm_b, opposite_err, match_ab = _match_opposite_arm_pair(
                (arms_by_junction.get(a) or {}).get("arms"),
                (arms_by_junction.get(b) or {}).get("arms"),
                heading_ab, heading_ba,
                used_a=used[a], used_b=used[b],
            )
            if arm_a is None or arm_b is None:
                message = (
                    "could not match opposite arms for junctions %s-%s "
                    "(need headings 180±%s deg; best_err=%s)"
                    % (a, b, MAX_ARM_OPPOSITE_TOL_DEG, opposite_err)
                )
                if strict:
                    raise ValueError(message)
                BaseLLM.log("skip connector: %s" % message)
                continue
            used[a].add(int(arm_a["id"]))
            used[b].add(int(arm_b["id"]))

        from_jid, to_jid = (a, b) if a > b else (b, a)
        if from_jid == a:
            from_arm, to_arm = arm_a, arm_b
            heading = heading_ab
            match = match_ab
        else:
            from_arm, to_arm = arm_b, arm_a
            heading = heading_ba
            match = None if match_ab is None else {
                "from": match_ab["to"],
                "to": match_ab["from"],
                "opposite_err_deg": match_ab["opposite_err_deg"],
            }

        wa = _world_xy(*ca, meters_per_side)
        wb = _world_xy(*cb, meters_per_side)
        center_dist = math.hypot(wb[0] - wa[0], wb[1] - wa[1])
        radius_from = _junction_radius_m(junctions[from_jid]["box"], meters_per_side)
        radius_to = _junction_radius_m(junctions[to_jid]["box"], meters_per_side)
        length = max(MIN_CONNECTOR_LENGTH_M, center_dist - radius_from - radius_to)
        lane_type = normalize_lane_type(link.get("lane_type"), connection_type=connection_type)
        through_heading = None
        if is_median:
            th_a = _junction_through_heading((arms_by_junction.get(a) or {}).get("arms"))
            th_b = _junction_through_heading((arms_by_junction.get(b) or {}).get("arms"))
            if th_a is not None and th_b is not None:
                through_heading = _axis_mean_heading(th_a, th_b)
            elif th_a is not None:
                through_heading = th_a
            elif th_b is not None:
                through_heading = th_b
        connectors.append({
            "junction_ids": [a, b],
            "connection_type": connection_type,
            "lane_type": lane_type,
            "from_junction": from_jid,
            "to_junction": to_jid,
            "from_arm": None if from_arm is None else int(from_arm["id"]),
            "to_arm": None if to_arm is None else int(to_arm["id"]),
            "heading_deg": heading,
            "through_heading_deg": through_heading,
            "length": length,
            "heading_match_deg": match,
            "evidence": link.get("evidence"),
        })
        if from_arm is None or to_arm is None or match is None:
            BaseLLM.log(
                "%s j%s <-> j%s heading=%.1f length=%.1fm (no facing arms)"
                % (connection_type, from_jid, to_jid, heading, length)
            )
        else:
            BaseLLM.log(
                "%s j%s.arm%s <-> j%s.arm%s heading=%.1f length=%.1fm "
                "opposite_err=%.1f deg face=%.1f/%.1f deg"
                % (
                    connection_type,
                    from_jid, from_arm["id"], to_jid, to_arm["id"], heading, length,
                    match["opposite_err_deg"], match["from"], match["to"],
                )
            )
    return connectors


def _roads_by_id(connections_doc):
    return {int(road["id"]): road for road in (connections_doc.get("roads") or [])}


def _remap_arm_ids(connections_doc, id_map):
    """Rewrite from_arm/to_arm using a complete old→new map (one pass, no collisions)."""
    mapping = {int(old): int(new) for old, new in id_map.items() if new is not None}
    for conn in connections_doc.get("connections") or []:
        conn["from_arm"] = mapping.get(int(conn["from_arm"]), conn["from_arm"])
        conn["to_arm"] = mapping.get(int(conn["to_arm"]), conn["to_arm"])
    return connections_doc


def _drop_arm_connections(junction_doc, arm_ids):
    """Remove junction connections that touch any of the given (global) arm ids."""
    drop = {int(x) for x in arm_ids if x is not None}
    if not drop:
        return junction_doc
    junction_doc["connections"] = [
        conn for conn in (junction_doc.get("connections") or [])
        if int(conn["from_arm"]) not in drop and int(conn["to_arm"]) not in drop
    ]
    return junction_doc


def merge_junction_connections(
    junctions_doc,
    connections_by_junction,
    connectors,
    *,
    driving_side=None,
    meters_per_side=METERS_PER_IMAGE_SIDE,
):
    """Build a multi-junction connections.json from per-junction artifacts + derived links."""
    junctions = {int(j["id"]): j for j in (junctions_doc.get("junctions") or [])}
    road_map = {}
    next_road_id = 1
    merged_roads = []
    merged_junctions = []
    side = driving_side

    connector_arm_keys = set()
    for item in connectors or []:
        from_arm = _connector_arm_id(item, "from_arm")
        to_arm = _connector_arm_id(item, "to_arm")
        if from_arm is not None:
            connector_arm_keys.add((int(item["from_junction"]), from_arm))
        if to_arm is not None:
            connector_arm_keys.add((int(item["to_junction"]), to_arm))

    for jid in sorted(connections_by_junction):
        doc = json.loads(json.dumps(connections_by_junction[jid]))
        if side is None:
            side = doc.get("driving_side")
        local_junctions = doc.get("junctions") or []
        if len(local_junctions) != 1:
            raise ValueError("junction %s connections.json must contain exactly one junction" % jid)
        local = local_junctions[0]
        spec = junctions.get(jid) or {}
        box = spec.get("box")
        radius = _junction_radius_m(box, meters_per_side) if box is not None else DEFAULT_JUNCTION_RADIUS
        cx, cy = junction_center(spec) if spec else (0.5, 0.5)
        wx, wy = _world_xy(cx, cy, meters_per_side)

        local_to_global = {}
        for road in doc.get("roads") or []:
            local_id = int(road["id"])
            key = (int(jid), local_id)
            global_id = next_road_id
            next_road_id += 1
            road_map[key] = global_id
            local_to_global[local_id] = global_id
            if key in connector_arm_keys:
                continue
            mapped = dict(road)
            mapped["id"] = global_id
            if mapped.get("length") is None:
                mapped["length"] = DEFAULT_ARM_LENGTH
            merged_roads.append(mapped)
        _remap_arm_ids(local, local_to_global)

        merged_junctions.append({
            "id": int(jid),
            "radius": radius,
            "x": wx,
            "y": wy,
            "connections": local.get("connections") or [],
        })

    merged_by_id = {int(j["id"]): j for j in merged_junctions}
    merged_connectors = []
    for item in connectors or []:
        from_jid = int(item["from_junction"])
        to_jid = int(item["to_junction"])
        from_local = _connector_arm_id(item, "from_arm")
        to_local = _connector_arm_id(item, "to_arm")
        connection_type = normalize_connection_type(item.get("connection_type"))
        is_median = connection_type == CONNECTION_TYPE_MEDIAN
        from_road = (
            None if from_local is None
            else _roads_by_id(connections_by_junction[from_jid]).get(from_local)
        )
        to_road = (
            None if to_local is None
            else _roads_by_id(connections_by_junction[to_jid]).get(to_local)
        )
        if not is_median and (from_road is None or to_road is None):
            raise ValueError("connector j%s.arm%s / j%s.arm%s missing from connections roads" % (
                from_jid, from_local, to_jid, to_local,
            ))
        cid = next_road_id
        next_road_id += 1
        record = {
            "id": cid,
            "junctions": [from_jid, to_jid],
            "heading_deg": float(item["heading_deg"]),
            "length": float(item["length"]),
            "connection_type": connection_type,
        }
        if is_median:
            record["outgoing_lanes"] = 0
            record["incoming_lanes"] = 0
            record["lane_type"] = (
                normalize_lane_type(item.get("lane_type"), connection_type=connection_type)
                or CONNECTION_TYPE_MEDIAN
            )
            if item.get("through_heading_deg") is not None:
                record["through_heading_deg"] = float(item["through_heading_deg"])
            merged_connectors.append(record)
            drop_from = (
                {road_map[(from_jid, from_local)]}
                if from_local is not None and (from_jid, from_local) in road_map
                else set()
            )
            drop_to = (
                {road_map[(to_jid, to_local)]}
                if to_local is not None and (to_jid, to_local) in road_map
                else set()
            )
            _drop_arm_connections(merged_by_id[from_jid], drop_from)
            _drop_arm_connections(merged_by_id[to_jid], drop_to)
            continue

        # generate_roads treats the smaller-id end as successor and applies
        # odr_left_right_lanes(outgoing, incoming) there. The two mouths of a
        # connecting_road are a HARD constraint: counts must already agree.
        outgoing = int(to_road.get("outgoing_lanes") or 0)
        incoming = int(to_road.get("incoming_lanes") or 0)
        from_out = int(from_road.get("outgoing_lanes") or 0)
        from_in = int(from_road.get("incoming_lanes") or 0)
        if from_out != incoming or from_in != outgoing:
            raise ValueError(
                "HARD connecting-road mismatch: connector %s "
                "j%s.arm%s out/in=%s/%s vs j%s.arm%s in/out=%s/%s"
                % (
                    cid, from_jid, from_local, from_out, from_in,
                    to_jid, to_local, incoming, outgoing,
                )
            )
        record["outgoing_lanes"] = outgoing
        record["incoming_lanes"] = incoming
        merged_connectors.append(record)
        _remap_arm_ids(merged_by_id[from_jid], {road_map[(from_jid, from_local)]: cid})
        _remap_arm_ids(merged_by_id[to_jid], {road_map[(to_jid, to_local)]: cid})

    return {
        "driving_side": side,
        "roads": merged_roads,
        "connectors": merged_connectors,
        "junctions": merged_junctions,
    }


def load_junctions_doc(case_dir, model_name, reasoning="none"):
    path = junction_level_dir(case_dir, model_name, reasoning) / "junctions.json"
    if not path.is_file():
        raise FileNotFoundError("junction-level result not found: %s" % path)
    return json.loads(path.read_text(encoding="utf-8"))


def promote_single_junction_artifacts(case_dir, work_dir, *, crop_meta=None):
    """Copy a one-junction run's arm/lane/refine trees to the case root for evaluation."""
    case_dir = Path(case_dir)
    work_dir = Path(work_dir)
    for name in ("arm_level", "lane_level", "refinement"):
        src = work_dir / name
        dst = case_dir / name
        if not src.is_dir():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    if crop_meta is None:
        return
    for arms_path in sorted((case_dir / "arm_level").glob("*/arms.json")) if (case_dir / "arm_level").is_dir() else []:
        mapped = map_arms_doc_to_full_image(json.loads(arms_path.read_text(encoding="utf-8")), crop_meta)
        dump_json(arms_path, mapped)
    for stage1_path in sorted((case_dir / "arm_level").glob("*/arms_stage1.json")) if (case_dir / "arm_level").is_dir() else []:
        mapped = map_arms_doc_to_full_image(json.loads(stage1_path.read_text(encoding="utf-8")), crop_meta)
        dump_json(stage1_path, mapped)


def assemble_scene_connections(
    case_dir,
    junctions_doc,
    *,
    junction_model,
    junction_reasoning,
    arm_model,
    arm_reasoning,
    refine_model,
    refine_reasoning,
    driving_side=None,
):
    """Derive connectors from arm headings and merge per-junction connections.json.

    A single junction writes back into that junction's ``xodr_result/``.
    Multiple junctions write a scene-level merge under the case-root refinement dir.
    """
    case_dir = Path(case_dir)
    arms_by_junction = {}
    connections_by_junction = {}
    for junction in junctions_doc.get("junctions") or []:
        jid = int(junction["id"])
        work_dir = junction_work_dir(case_dir, jid)
        arms_by_junction[jid] = json.loads(
            arms_path_for_model(work_dir, arm_model, arm_reasoning).read_text(encoding="utf-8")
        )
        xodr_dir = xodr_result_dir_for_model(work_dir, refine_model, refine_reasoning)
        conns_path = xodr_dir / "connections.json"
        if not conns_path.is_file():
            raise FileNotFoundError("junction %s connections.json not found: %s" % (jid, conns_path))
        connections_by_junction[jid] = json.loads(conns_path.read_text(encoding="utf-8"))

    BaseLLM.rule("INTER-JUNCTION ROADS")
    connectors = derive_inter_junction_roads(junctions_doc, arms_by_junction)
    merged = merge_junction_connections(
        junctions_doc, connections_by_junction, connectors, driving_side=driving_side,
    )
    junction_ids = [int(j["id"]) for j in (junctions_doc.get("junctions") or [])]
    if len(junction_ids) == 1:
        out_dir = xodr_result_dir_for_model(
            junction_work_dir(case_dir, junction_ids[0]),
            refine_model,
            refine_reasoning,
        )
    else:
        out_dir = xodr_result_dir_for_model(case_dir, refine_model, refine_reasoning)
    dump_json(out_dir / "connections.json", merged)
    jl_dir = junction_level_dir(case_dir, junction_model, junction_reasoning)
    if (jl_dir / "junctions.json").is_file():
        dump_json(jl_dir / "inter_junction_roads.json", {"connectors": connectors, "merged": True})
    BaseLLM.log("merged connections -> %s" % (out_dir / "connections.json"))
    return {
        "connections_path": out_dir / "connections.json",
        "connectors": connectors,
        "merged": merged,
    }


if __name__ == "__main__":
    MODEL = {
        "api": "openai",
        "provider": "openrouter",
        "model": "openai/gpt-5.6-sol",
        "reasoning": "medium",
    }
    parser = argparse.ArgumentParser(description="Identify junctions in a full scene image.")
    parser.add_argument("--dataset", default="openaerialmap")
    parser.add_argument("--case-id", default="2002")
    parser.add_argument("--model", default=MODEL["model"])
    parser.add_argument("--reasoning", default=MODEL.get("reasoning", "none"))
    parser.add_argument("--max-tokens", type=int, default=10240)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    process_case_image(dataset=args.dataset, case_id=args.case_id)
    run_junction_level(
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
    )