"""Evaluate arm-level, lane-level, and refinement predictions against GT.

Arm ids are unordered: GT and prediction arms are matched by closest
``heading_deg`` (optimal bipartite assignment). Channelized features and
lane checks use the same pred->gt id mapping when arms.json is available.

Stages:
  - arm-level (``arms.json``): arm count, channelized-island ``(from,to)`` pairs,
    through-road unordered arm pairs, heading_deg accuracy (match if |diff| <=
    ``DEFAULT_HEADING_TOL_DEG``). Island and through-road ids are remapped
    through the heading assignment; both report precision/recall.
  - lane-level (``lane_level/.../lanes.json``): outgoing/incoming counts, arrows
    (reads legacy ``arrow`` or cue ``kind=arrow`` / ``movements``).
    Combination codes ``{L,S,R,LS,SR,LR,LSR}`` are atomic movement categories.
    Incoming lanes are paired left-to-right. Arrow **existence** (PREC/REC)
    is separate from **category** accuracy (ACC): ``N_both`` lanes have an
    arrow on both sides, ``N_false`` have pred-only, ``N_miss`` have GT-only.
    PREC = N_both / (N_both + N_false), REC = N_both / (N_both + N_miss).
    ACC = N_both^correct / N_both, exact category match on ``N_both`` only.
    ``incoming_arrow_e2e`` (alias ``incoming_arrow``) scores every heading-
    matched arm; ``incoming_arrow_matched`` keeps arms whose incoming lane
    count matches GT. Coverage is GT arrows on those arms / all GT arrows.
    Lane count is evaluated separately. ``lane_count_accuracy`` requires exact
    incoming and outgoing counts; ``lane_1_accuracy`` is junction-level: 1.0 when
    the sum of |gt-pred| over all incoming/outgoing lane counts on GT arms, with
    each outgoing↔incoming travel swap counted once, is at most 1. Arrows do not
    affect either lane metric.
  - refinement (``refinement/.../lanes.json``): same lane metrics as lane-level

Example::

    python -m scripts.methods.evaluation \\
        --gt dataset/openaerialmap/1002 \\
        --pred scripts/methods/llm_kg/results/openaerialmap/1002/lane_level/openai_gpt_5_6_sol_medium

By default writes stage-scoped bundles:
  - ``lane_level/{model}_{reasoning}/evaluation/`` — arm_level + lane_level
  - ``refinement/{model}_{reasoning}/evaluation/`` — refinement
Each bundle contains ``full.json`` / ``mismatches.json`` / ``metrics.json``.
Pass ``--out`` to write one combined bundle to a single directory instead.
"""

import itertools
import json
from pathlib import Path


DEFAULT_HEADING_TOL_DEG = 10.1
ARMS_JSON_NAME = "arms.json"
LANES_JSON_NAME = "lanes.json"
ARROW_CATEGORIES = ("L", "S", "R", "LS", "SR", "LR", "LSR")


def find_json_under(dir_path, filename, subdirs=None):
    """Find ``filename`` in ``dir_path`` or one of ``subdirs`` (first hit wins)."""
    folder = Path(dir_path)
    if not folder.is_dir():
        raise FileNotFoundError("directory does not exist: %s" % folder)
    candidates = [folder / filename]
    for sub in subdirs or ():
        candidates.append(folder / sub / filename)
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("file not found under %s: %s (tried %s)" % (
        folder,
        filename,
        ", ".join(str(p) for p in candidates),
    ))


def _optional_file(path):
    path = Path(path)
    return path.resolve() if path.is_file() else None


def resolve_side_dir(dir_path):
    """Resolve one GT/pred folder to arms / lanes / refine-lanes paths.

    Returns ``(arms_path, lanes_path, refine_lanes_path)``; missing levels are None.

    Supports:
      - ``.../arm_level/{model}_{reasoning}`` / ``.../lane_level/{model}_{reasoning}`` / ``.../refinement/{model}_{reasoning}``
      - flat folder with ``arms.json`` / ``lanes.json`` (e.g. GT; no refinement)
    """
    folder = Path(dir_path)
    if not folder.is_dir():
        raise FileNotFoundError("directory does not exist: %s" % folder)
    arms_path = None
    lanes_path = None
    refine_lanes_path = None
    parent_name = folder.parent.name if folder.parent else ""
    case_dir = folder.parent.parent if parent_name in ("arm_level", "lane_level", "refinement") else None
    model_name = folder.name if case_dir is not None else None

    if parent_name == "arm_level":
        arms_path = _optional_file(folder / ARMS_JSON_NAME)
        lanes_path = _optional_file(case_dir / "lane_level" / model_name / LANES_JSON_NAME)
        refine_lanes_path = _optional_file(case_dir / "refinement" / model_name / LANES_JSON_NAME)
    elif parent_name == "lane_level":
        lanes_path = _optional_file(folder / LANES_JSON_NAME)
        arms_path = _optional_file(case_dir / "arm_level" / model_name / ARMS_JSON_NAME)
        refine_lanes_path = _optional_file(case_dir / "refinement" / model_name / LANES_JSON_NAME)
    elif parent_name == "refinement":
        refine_lanes_path = _optional_file(folder / LANES_JSON_NAME)
        arms_path = _optional_file(case_dir / "arm_level" / model_name / ARMS_JSON_NAME)
        lanes_path = _optional_file(case_dir / "lane_level" / model_name / LANES_JSON_NAME)
    else:
        arms_path = _optional_file(folder / ARMS_JSON_NAME)
        lanes_path = _optional_file(folder / LANES_JSON_NAME)
        if arms_path is None:
            try:
                arms_path = find_json_under(folder, ARMS_JSON_NAME, subdirs=("arm_level",))
            except FileNotFoundError:
                pass
        if lanes_path is None:
            try:
                lanes_path = find_json_under(folder, LANES_JSON_NAME, subdirs=("lane_level",))
            except FileNotFoundError:
                pass
        refine_lanes_path = _optional_file(folder / "refinement" / LANES_JSON_NAME)
        if refine_lanes_path is None:
            refine_dirs = sorted((folder / "refinement").glob("*/" + LANES_JSON_NAME)) if (folder / "refinement").is_dir() else []
            if len(refine_dirs) == 1:
                refine_lanes_path = refine_dirs[0].resolve()

    if arms_path is None and lanes_path is None and refine_lanes_path is None:
        raise FileNotFoundError(
            "neither %s nor %s found under %s (also checked arm_level/ lane_level/ refinement/)"
            % (ARMS_JSON_NAME, LANES_JSON_NAME, folder)
        )
    return arms_path, lanes_path, refine_lanes_path


def resolve_pred_artifacts(
    case_dir,
    *,
    arm_model,
    lane_model,
    refine_model,
    arm_reasoning="none",
    lane_reasoning="none",
    refine_reasoning="none",
):
    """Resolve pred arms/lanes/refine-lanes under a case dir with per-stage models."""
    from scripts.methods.method_utils import arm_level_dir, lane_level_dir, refinement_dir

    case_dir = Path(case_dir)
    return (
        _optional_file(arm_level_dir(case_dir, arm_model, arm_reasoning) / ARMS_JSON_NAME),
        _optional_file(lane_level_dir(case_dir, lane_model, lane_reasoning) / LANES_JSON_NAME),
        _optional_file(refinement_dir(case_dir, refine_model, refine_reasoning) / LANES_JSON_NAME),
    )


def load_json(path_or_obj):
    if isinstance(path_or_obj, (str, Path)):
        return json.loads(Path(path_or_obj).read_text(encoding="utf-8"))
    return path_or_obj


def angular_diff_deg(a, b):
    diff = abs(float(a) % 360.0 - float(b) % 360.0) % 360.0
    return min(diff, 360.0 - diff)


def normalize_arrow(arrow):
    """Normalize one arrow/movement code; absent/undefined markers become None."""
    if arrow is None:
        return None
    key = str(arrow).upper().strip()
    if key in ("", "UNDEFINED", "NULL", "NONE", "N", "UNRESOLVABLE"):
        return None
    aliases = {"RS": "SR", "LRS": "LSR", "LEFT": "L", "STRAIGHT": "S", "RIGHT": "R"}
    return aliases.get(key, key)


def arrow_movement_labels(code):
    """Canonical arrow class as a singleton set (``LS`` is not expanded to L+S)."""
    normalized = normalize_arrow(code)
    if not normalized:
        return set()
    return {normalized}


def lane_arrow_codes(item):
    """Ranked unique movement codes on this lane.

    Collects every usable ``movements`` entry from ``kind=arrow`` cues, then
    the legacy GT ``arrow`` field. Order is first-seen.
    """
    codes = []
    seen = set()

    def _add(code):
        normalized = normalize_arrow(code)
        if normalized and normalized not in seen:
            seen.add(normalized)
            codes.append(normalized)

    for cue in item.get("cues") or []:
        if cue.get("kind") != "arrow":
            continue
        movements = cue.get("movements")
        if movements is None:
            continue
        if not isinstance(movements, list):
            movements = [movements]
        for code in movements:
            _add(code)
    if not codes and "arrow" in item:
        _add(item.get("arrow"))
    return codes


def lane_arrow_code(item):
    """First ranked arrow code for a lane, or None if absent."""
    codes = lane_arrow_codes(item)
    return codes[0] if codes else None


def lane_arrow_labels(item):
    """Arrow class set for one lane (all ``movements`` candidates)."""
    return set(lane_arrow_codes(item))


def channelized_id(feature):
    """Identity for a channelized feature: (from_arm, to_arm)."""
    if feature.get("from_arm") is not None and feature.get("to_arm") is not None:
        return (int(feature["from_arm"]), int(feature["to_arm"]))
    arm_list = feature.get("arm_list") or []
    if len(arm_list) == 2:
        return (int(arm_list[0]), int(arm_list[1]))
    raise ValueError("channelized feature missing from_arm/to_arm or arm_list: %s" % feature)


def arms_by_id(payload):
    mapping = {}
    for arm in payload.get("arms") or []:
        arm_id = arm.get("id", arm.get("arm_id"))
        if arm_id is None:
            continue
        mapping[int(arm_id)] = arm
    return mapping


def arm_heading_deg(arm):
    if arm is None:
        return None
    if arm.get("heading_deg") is not None:
        return float(arm["heading_deg"])
    if arm.get("heading_deg_min") is None or arm.get("heading_deg_max") is None:
        return None
    hmin = float(arm["heading_deg_min"]) % 360.0
    hmax = float(arm["heading_deg_max"]) % 360.0
    span = (hmax - hmin) % 360.0
    return (hmin + span / 2.0) % 360.0


def clip_arm_mapping(mapping, gt_arms, pred_arms):
    """Keep heading pairs whose ids exist in both dicts; leftover ids are unmatched.

    ``evaluate()`` reuses the arms.json mapping for lane-level. A failed pipeline
    may write a partial ``lanes.json`` (only completed arms), which would otherwise
    KeyError on a missing pred id.
    """
    if mapping is None:
        return match_arms_by_heading(gt_arms, pred_arms)
    gt_ids = set(gt_arms)
    pred_ids = set(pred_arms)
    pairs = [
        pair for pair in (mapping.get("pairs") or [])
        if pair.get("gt_arm_id") in gt_ids and pair.get("pred_arm_id") in pred_ids
    ]
    used_gt = {pair["gt_arm_id"] for pair in pairs}
    used_pred = {pair["pred_arm_id"] for pair in pairs}
    pred_to_gt = {pair["pred_arm_id"]: pair["gt_arm_id"] for pair in pairs}
    return {
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": {gt_id: pred_id for pred_id, gt_id in pred_to_gt.items()},
        "pairs": pairs,
        "unmatched_gt": sorted(gt_ids - used_gt),
        "unmatched_pred": sorted(pred_ids - used_pred),
        "total_diff_deg": mapping.get("total_diff_deg"),
    }


def match_arms_by_heading(gt_arms, pred_arms):
    """Match GT/pred arms by minimizing total heading angular difference.

    Returns a dict with ``pred_to_gt``, ``gt_to_pred``, ``pairs``, and unmatched ids.
    When counts differ, only ``min(n_gt, n_pred)`` arms are paired.
    """
    gt_ids = sorted(gt_arms)
    pred_ids = sorted(pred_arms)
    pairs = []
    if not gt_ids or not pred_ids:
        return {
            "pred_to_gt": {},
            "gt_to_pred": {},
            "pairs": [],
            "unmatched_gt": gt_ids,
            "unmatched_pred": pred_ids,
            "total_diff_deg": None,
        }

    if len(gt_ids) <= len(pred_ids):
        fixed_ids, free_ids, fixed_is_gt = gt_ids, pred_ids, True
    else:
        fixed_ids, free_ids, fixed_is_gt = pred_ids, gt_ids, False

    best_cost = None
    best_pairs = None
    for free_perm in itertools.permutations(free_ids, len(fixed_ids)):
        candidate = []
        cost = 0.0
        for fixed_id, free_id in zip(fixed_ids, free_perm):
            gt_id = fixed_id if fixed_is_gt else free_id
            pred_id = free_id if fixed_is_gt else fixed_id
            gt_h = arm_heading_deg(gt_arms[gt_id])
            pred_h = arm_heading_deg(pred_arms[pred_id])
            if gt_h is None or pred_h is None:
                diff = 180.0
            else:
                diff = angular_diff_deg(gt_h, pred_h)
            cost += diff
            candidate.append((gt_id, pred_id, diff, gt_h, pred_h))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_pairs = candidate

    pred_to_gt = {}
    gt_to_pred = {}
    for gt_id, pred_id, diff, gt_h, pred_h in best_pairs:
        pred_to_gt[pred_id] = gt_id
        gt_to_pred[gt_id] = pred_id
        pairs.append({
            "gt_arm_id": gt_id,
            "pred_arm_id": pred_id,
            "gt_heading_deg": gt_h,
            "pred_heading_deg": pred_h,
            "diff_deg": diff,
        })

    matched_gt = set(gt_to_pred)
    matched_pred = set(pred_to_gt)
    return {
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
        "pairs": pairs,
        "unmatched_gt": [arm_id for arm_id in gt_ids if arm_id not in matched_gt],
        "unmatched_pred": [arm_id for arm_id in pred_ids if arm_id not in matched_pred],
        "total_diff_deg": best_cost,
    }


def remap_channelized_ids(features, pred_to_gt):
    """Map pred channelized (from,to) into GT id space. Unmappable pairs are dropped to errors."""
    remapped = set()
    errors = []
    for feature in features or []:
        try:
            from_arm, to_arm = channelized_id(feature)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if from_arm not in pred_to_gt or to_arm not in pred_to_gt:
            errors.append("unmapped channelized arms: from_arm=%s to_arm=%s" % (from_arm, to_arm))
            continue
        remapped.add((pred_to_gt[from_arm], pred_to_gt[to_arm]))
    return remapped, errors


def through_road_id(item):
    """Identity for a through road: sorted ``(arm_a, arm_b)``."""
    arm_ids = item.get("arm_ids") if isinstance(item, dict) else item
    if not arm_ids or len(arm_ids) != 2:
        raise ValueError("through road must have two arm_ids: %s" % item)
    arm_a, arm_b = int(arm_ids[0]), int(arm_ids[1])
    if arm_a == arm_b:
        raise ValueError("through road arm_ids must be distinct: %s" % item)
    return (arm_a, arm_b) if arm_a < arm_b else (arm_b, arm_a)


def remap_through_road_ids(roads, pred_to_gt):
    """Map pred through-road pairs into GT id space. Unmappable pairs are errors."""
    remapped = set()
    errors = []
    for item in roads or []:
        try:
            arm_a, arm_b = through_road_id(item)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if arm_a not in pred_to_gt or arm_b not in pred_to_gt:
            errors.append("unmapped through-road arms: %s %s" % (arm_a, arm_b))
            continue
        gt_a, gt_b = pred_to_gt[arm_a], pred_to_gt[arm_b]
        remapped.add((gt_a, gt_b) if gt_a < gt_b else (gt_b, gt_a))
    return remapped, errors


def _pair_report(gt_set, pred_set, pred_raw, errors, *, directed):
    """Shared gt/pred pair match block for channelized or through-roads."""

    def as_items(pairs):
        if directed:
            return [{"from_arm": a, "to_arm": b} for a, b in sorted(pairs)]
        return [{"arm_ids": [a, b]} for a, b in sorted(pairs)]

    return {
        "gt_ids": as_items(gt_set),
        "pred_ids_raw": as_items(pred_raw),
        "pred_ids": as_items(pred_set),
        "missing_in_pred": as_items(gt_set - pred_set),
        "extra_in_pred": as_items(pred_set - gt_set),
        "errors": errors,
        "match": gt_set == pred_set and not errors,
    }


def mouth_lanes(arm):
    """LTR mouth lanes (``type=lane``) with canonical arrow class codes."""
    lanes = []
    if arm is None:
        return lanes
    for item in arm.get("cross_section") or []:
        if item.get("type") != "lane":
            continue
        candidates = lane_arrow_codes(item)
        code = candidates[0] if candidates else None
        lanes.append({
            "travel": item.get("travel"),
            "code": code,
            "candidates": candidates,
            "labels": list(candidates),
        })
    return lanes


def incoming_mouth_lanes(arm):
    """LTR incoming mouth lanes used for pavement-arrow matching."""
    return [lane for lane in mouth_lanes(arm) if lane.get("travel") == "incoming"]


def lane_stats(arm):
    lanes = mouth_lanes(arm)
    incoming = [item for item in lanes if item.get("travel") == "incoming"]
    return {
        "outgoing_count": sum(1 for item in lanes if item.get("travel") == "outgoing"),
        "incoming_count": len(incoming),
        "incoming_arrows": [item.get("code") for item in incoming],
        "lanes": lanes,
    }


def arm_way_kind(arm):
    """``one_way`` / ``two_way`` from mouth-lane travels; ``None`` if undetermined.

    Two-way: both incoming and outgoing motor lanes. One-way: all motor lanes share
    a single travel direction. Ignores ``opposing_separator`` so GT (often without
    that field) and predictions are scored under the same rule.
    """
    travels = {
        lane.get("travel")
        for lane in mouth_lanes(arm)
        if lane.get("travel") in ("incoming", "outgoing")
    }
    if not travels:
        return None
    if travels == {"incoming", "outgoing"}:
        return "two_way"
    if travels <= {"incoming"} or travels <= {"outgoing"}:
        return "one_way"
    return None


def way_kind_match_counts(gt, pred, arm_mapping=None):
    """Lane-count EMR counts stratified by GT arm one-way / two-way type.

    For each GT arm with a determined way kind, count whether the heading-matched
    prediction has exact incoming and outgoing lane counts. Unmatched GT arms
    count as incorrect. GT arms with undetermined travel are skipped.
    """
    gt = load_json(gt)
    pred = load_json(pred)
    gt_arms = arms_by_id(gt)
    pred_arms = arms_by_id(pred)
    mapping = clip_arm_mapping(
        arm_mapping or match_arms_by_heading(gt_arms, pred_arms),
        gt_arms,
        pred_arms,
    )
    gt_to_pred = mapping.get("gt_to_pred") or {}
    out = {
        "way_one_correct": 0,
        "way_one_n": 0,
        "way_two_correct": 0,
        "way_two_n": 0,
    }
    for gt_id, gt_arm in gt_arms.items():
        gt_kind = arm_way_kind(gt_arm)
        if gt_kind is None:
            continue
        pred_id = gt_to_pred.get(gt_id)
        pred_arm = pred_arms.get(pred_id) if pred_id is not None else None
        gt_stats = lane_stats(gt_arm)
        lane_match = False
        if pred_arm is not None:
            pred_stats = lane_stats(pred_arm)
            lane_match = (
                gt_stats["outgoing_count"] == pred_stats["outgoing_count"]
                and gt_stats["incoming_count"] == pred_stats["incoming_count"]
            )
        if gt_kind == "one_way":
            out["way_one_n"] += 1
            if lane_match:
                out["way_one_correct"] += 1
        else:
            out["way_two_n"] += 1
            if lane_match:
                out["way_two_correct"] += 1
    return out


def evaluate_arm_level(gt, pred, heading_tol_deg=DEFAULT_HEADING_TOL_DEG, arm_mapping=None):
    gt = load_json(gt)
    pred = load_json(pred)

    gt_arms = arms_by_id(gt)
    pred_arms = arms_by_id(pred)
    gt_ids = sorted(gt_arms)
    pred_ids = sorted(pred_arms)
    mapping = arm_mapping or match_arms_by_heading(gt_arms, pred_arms)
    pred_to_gt = mapping["pred_to_gt"]

    arm_count = {
        "gt": len(gt_ids),
        "pred": len(pred_ids),
        "gt_ids": gt_ids,
        "pred_ids": pred_ids,
        "match": len(gt_ids) == len(pred_ids) and not mapping["unmatched_gt"] and not mapping["unmatched_pred"],
    }

    gt_cf = []
    cf_errors = []
    for feature in gt.get("channelized_features") or []:
        try:
            gt_cf.append(channelized_id(feature))
        except ValueError as exc:
            cf_errors.append("gt: %s" % exc)
    gt_cf_set = set(gt_cf)
    pred_cf_set, pred_cf_errors = remap_channelized_ids(pred.get("channelized_features") or [], pred_to_gt)
    cf_errors.extend("pred: %s" % err for err in pred_cf_errors)
    pred_cf_raw = []
    for feature in pred.get("channelized_features") or []:
        try:
            pred_cf_raw.append(channelized_id(feature))
        except ValueError:
            continue
    channelized = _pair_report(gt_cf_set, pred_cf_set, set(pred_cf_raw), cf_errors, directed=True)

    gt_tr = []
    tr_errors = []
    for item in gt.get("through_roads") or []:
        try:
            gt_tr.append(through_road_id(item))
        except ValueError as exc:
            tr_errors.append("gt: %s" % exc)
    gt_tr_set = set(gt_tr)
    pred_tr_set, pred_tr_errors = remap_through_road_ids(pred.get("through_roads") or [], pred_to_gt)
    tr_errors.extend("pred: %s" % err for err in pred_tr_errors)
    pred_tr_raw = []
    for item in pred.get("through_roads") or []:
        try:
            pred_tr_raw.append(through_road_id(item))
        except ValueError:
            continue
    through_roads = _pair_report(gt_tr_set, pred_tr_set, set(pred_tr_raw), tr_errors, directed=False)

    headings = []
    heading_matches = []
    for pair in mapping["pairs"]:
        diff = pair["diff_deg"]
        matched = diff is not None and diff <= float(heading_tol_deg)
        headings.append({
            "gt_arm_id": pair["gt_arm_id"],
            "pred_arm_id": pair["pred_arm_id"],
            "gt_heading_deg": pair["gt_heading_deg"],
            "pred_heading_deg": pair["pred_heading_deg"],
            "diff_deg": diff,
            "heading_tol_deg": float(heading_tol_deg),
            "match": matched,
            "status": "ok" if matched else "heading_mismatch",
        })
        heading_matches.append(matched)
    for arm_id in mapping["unmatched_gt"]:
        headings.append({
            "gt_arm_id": arm_id,
            "pred_arm_id": None,
            "gt_heading_deg": arm_heading_deg(gt_arms[arm_id]),
            "pred_heading_deg": None,
            "diff_deg": None,
            "match": False,
            "status": "missing_in_pred",
        })
        heading_matches.append(False)
    for arm_id in mapping["unmatched_pred"]:
        headings.append({
            "gt_arm_id": None,
            "pred_arm_id": arm_id,
            "gt_heading_deg": None,
            "pred_heading_deg": arm_heading_deg(pred_arms[arm_id]),
            "diff_deg": None,
            "match": False,
            "status": "missing_in_gt",
        })
        heading_matches.append(False)

    passed = bool(arm_count["match"] and channelized["match"] and all(heading_matches))
    return {
        "arm_mapping": {
            "pred_to_gt": {str(k): v for k, v in sorted(pred_to_gt.items())},
            "pairs": mapping["pairs"],
            "unmatched_gt": mapping["unmatched_gt"],
            "unmatched_pred": mapping["unmatched_pred"],
            "total_diff_deg": mapping["total_diff_deg"],
        },
        "arm_count": arm_count,
        "channelized_features": channelized,
        "through_roads": through_roads,
        "headings": headings,
        "passed": passed,
        "summary": {
            "arm_count_match": arm_count["match"],
            "channelized_ids_match": channelized["match"],
            "through_road_ids_match": through_roads["match"],
            "heading_match_count": sum(1 for x in heading_matches if x),
            "heading_total": len(heading_matches),
            "passed": passed,
        },
    }


def evaluate_lane_level(gt, pred, arm_mapping=None):
    gt = load_json(gt)
    pred = load_json(pred)

    gt_arms = arms_by_id(gt)
    pred_arms = arms_by_id(pred)
    mapping = clip_arm_mapping(
        arm_mapping or match_arms_by_heading(gt_arms, pred_arms),
        gt_arms,
        pred_arms,
    )

    arm_reports = []
    arm_matches = []

    for pair in mapping["pairs"]:
        gt_id = pair["gt_arm_id"]
        pred_id = pair["pred_arm_id"]
        gt_arm = gt_arms[gt_id]
        pred_arm = pred_arms[pred_id]
        gt_stats = lane_stats(gt_arm)
        pred_stats = lane_stats(pred_arm)
        outgoing_match = gt_stats["outgoing_count"] == pred_stats["outgoing_count"]
        incoming_match = gt_stats["incoming_count"] == pred_stats["incoming_count"]
        arrows = evaluate_arm_arrows(gt_arm, pred_arm)
        matched = outgoing_match and incoming_match and arrows["match"]
        arm_reports.append({
            "gt_arm_id": gt_id,
            "pred_arm_id": pred_id,
            "heading_diff_deg": pair["diff_deg"],
            "outgoing": {
                "gt": gt_stats["outgoing_count"],
                "pred": pred_stats["outgoing_count"],
                "match": outgoing_match,
            },
            "incoming": {
                "gt": gt_stats["incoming_count"],
                "pred": pred_stats["incoming_count"],
                "match": incoming_match,
            },
            "arrows": arrows,
            "match": matched,
            "status": "ok" if matched else "mismatch",
        })
        arm_matches.append(matched)

    for arm_id in mapping["unmatched_gt"]:
        gt_stats = lane_stats(gt_arms[arm_id])
        arrows = evaluate_arm_arrows(gt_arms[arm_id], None)
        arm_reports.append({
            "gt_arm_id": arm_id,
            "pred_arm_id": None,
            "outgoing": {"gt": gt_stats["outgoing_count"], "pred": None, "match": False},
            "incoming": {"gt": gt_stats["incoming_count"], "pred": None, "match": False},
            "arrows": arrows,
            "match": False,
            "status": "missing_in_pred",
        })
        arm_matches.append(False)
    for arm_id in mapping["unmatched_pred"]:
        pred_stats = lane_stats(pred_arms[arm_id])
        arrows = evaluate_arm_arrows(None, pred_arms[arm_id])
        arm_reports.append({
            "gt_arm_id": None,
            "pred_arm_id": arm_id,
            "outgoing": {"gt": None, "pred": pred_stats["outgoing_count"], "match": False},
            "incoming": {"gt": None, "pred": pred_stats["incoming_count"], "match": False},
            "arrows": arrows,
            "match": False,
            "status": "missing_in_gt",
        })
        arm_matches.append(False)

    passed = bool(arm_matches) and all(arm_matches)
    return {
        "arm_mapping": {
            "pred_to_gt": {str(k): v for k, v in sorted(mapping["pred_to_gt"].items())},
            "pairs": mapping["pairs"],
            "unmatched_gt": mapping["unmatched_gt"],
            "unmatched_pred": mapping["unmatched_pred"],
            "total_diff_deg": mapping["total_diff_deg"],
        },
        "arms": arm_reports,
        "passed": passed,
        "summary": {
            "arm_match_count": sum(1 for x in arm_matches if x),
            "arm_total": len(arm_matches),
            "outgoing_match_count": sum(1 for r in arm_reports if r.get("outgoing") and r["outgoing"]["match"]),
            "incoming_match_count": sum(1 for r in arm_reports if r.get("incoming") and r["incoming"]["match"]),
            "arrow_match_count": sum(1 for r in arm_reports if r.get("arrows") and r["arrows"]["match"]),
            "passed": passed,
        },
    }


def evaluate(
    gt_arms=None,
    pred_arms=None,
    gt_lanes=None,
    pred_lanes=None,
    pred_refine_lanes=None,
    heading_tol_deg=DEFAULT_HEADING_TOL_DEG,
):
    """Evaluate available stages; omit a stage by leaving its paths/objects as None.

    ``lane_level`` and ``refinement`` both compare against the same ``gt_lanes``
    using ``evaluate_lane_level`` (identical metric definitions).
    """
    result = {"arm_level": None, "lane_level": None, "refinement": None, "passed": True}
    shared_mapping = None
    if gt_arms is not None and pred_arms is not None:
        gt_arms_obj = load_json(gt_arms)
        pred_arms_obj = load_json(pred_arms)
        shared_mapping = match_arms_by_heading(arms_by_id(gt_arms_obj), arms_by_id(pred_arms_obj))
        result["arm_level"] = evaluate_arm_level(
            gt_arms_obj, pred_arms_obj, heading_tol_deg=heading_tol_deg, arm_mapping=shared_mapping,
        )
        result["passed"] = result["passed"] and result["arm_level"]["passed"]

    gt_lanes_obj = load_json(gt_lanes) if gt_lanes is not None else None

    if gt_lanes_obj is not None and pred_lanes is not None:
        pred_lanes_obj = load_json(pred_lanes)
        lane_mapping = shared_mapping
        if lane_mapping is None:
            lane_mapping = match_arms_by_heading(arms_by_id(gt_lanes_obj), arms_by_id(pred_lanes_obj))
        result["lane_level"] = evaluate_lane_level(gt_lanes_obj, pred_lanes_obj, arm_mapping=lane_mapping)
        result["passed"] = result["passed"] and result["lane_level"]["passed"]

    if gt_lanes_obj is not None and pred_refine_lanes is not None:
        pred_refine_obj = load_json(pred_refine_lanes)
        refine_mapping = shared_mapping
        if refine_mapping is None:
            refine_mapping = match_arms_by_heading(arms_by_id(gt_lanes_obj), arms_by_id(pred_refine_obj))
        result["refinement"] = evaluate_lane_level(gt_lanes_obj, pred_refine_obj, arm_mapping=refine_mapping)
        result["passed"] = result["passed"] and result["refinement"]["passed"]

    if result["arm_level"] is None and result["lane_level"] is None and result["refinement"] is None:
        raise ValueError("provide at least one of arm-level, lane-level, or refinement gt/pred pairs")
    return result


def _arm_ref(report_item):
    return {"gt_arm_id": report_item.get("gt_arm_id"), "pred_arm_id": report_item.get("pred_arm_id")}


def presence_error_type(gt_val, pred_val):
    """Classify a scalar/value mismatch: gt_only / pred_only / mismatch."""
    gt_has = gt_val is not None
    pred_has = pred_val is not None
    if gt_has and not pred_has:
        return "gt_only"
    if pred_has and not gt_has:
        return "pred_only"
    if gt_has and pred_has and gt_val != pred_val:
        return "mismatch"
    return None


def align_lanes_spatial(gt_lanes, pred_lanes):
    """Order-preserving spatial matching of LTR mouth lanes.

    Both sequences are left-aligned in a shared frame of length ``max(n, m)``,
    so extras on the longer side stay unmatched on the right. The DP matches
    ``min(n, m)`` lanes (match cost in [0, 1], skip cost 1).
    """
    gt_lanes = list(gt_lanes or [])
    pred_lanes = list(pred_lanes or [])
    n, m = len(gt_lanes), len(pred_lanes)
    if n == 0 and m == 0:
        return []
    span = float(max(n, m))
    pos_g = [(i + 0.5) / span for i in range(n)]
    pos_p = [(j + 0.5) / span for j in range(m)]
    skip = 1.0
    inf = 1e9
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    prev = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = i * skip
        prev[i][0] = ("skip_gt", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j * skip
        prev[0][j] = ("skip_pred", 0, j - 1)
    for i in range(n):
        for j in range(m):
            skip_gt = dp[i][j + 1] + skip
            skip_pred = dp[i + 1][j] + skip
            match = dp[i][j] + abs(pos_g[i] - pos_p[j])
            best = match
            action = ("match", i, j)
            if skip_gt < best:
                best = skip_gt
                action = ("skip_gt", i, j + 1)
            if skip_pred < best:
                best = skip_pred
                action = ("skip_pred", i + 1, j)
            dp[i + 1][j + 1] = best
            prev[i + 1][j + 1] = action
    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        action, pi, pj = prev[i][j]
        if action == "match":
            pairs.append({
                "gt_lane": pi + 1,
                "pred_lane": pj + 1,
                "gt": gt_lanes[pi],
                "pred": pred_lanes[pj],
                "status": "matched",
            })
            i, j = pi, pj
        elif action == "skip_gt":
            pairs.append({
                "gt_lane": pi + 1,
                "pred_lane": None,
                "gt": gt_lanes[pi],
                "pred": None,
                "status": "unmatched_gt",
            })
            i, j = pi, pj
        else:
            pairs.append({
                "gt_lane": None,
                "pred_lane": pj + 1,
                "gt": None,
                "pred": pred_lanes[pj],
                "status": "unmatched_pred",
            })
            i, j = pi, pj
    pairs.reverse()
    return pairs


def _incoming_arrow_codes(lanes):
    """Readable incoming arrow codes in LTR mouth order (lanes without arrows omitted)."""
    codes = []
    for lane in lanes or []:
        code = _lane_identified_arrow(lane)
        if code:
            codes.append(code)
    return codes


def _lane_identified_arrow(lane):
    """Single identified movement category, or None if the lane has no pavement arrow."""
    if not lane:
        return None
    code = normalize_arrow(lane.get("code"))
    if code:
        return code
    for cand in lane.get("candidates") or lane.get("labels") or []:
        code = normalize_arrow(cand)
        if code:
            return code
    return None


def _arrow_pair_event(gt_lane, pred_lane):
    """Classify one LTR-matched incoming lane: both / false / miss / none / sub."""
    gt_code = _lane_identified_arrow(gt_lane)
    pred_code = _lane_identified_arrow(pred_lane)
    if gt_code and pred_code:
        if gt_code == pred_code:
            return "both", gt_code, pred_code
        return "sub", gt_code, pred_code
    if pred_code and not gt_code:
        return "fp", None, pred_code
    if gt_code and not pred_code:
        return "fn", gt_code, None
    return "none", None, None


def _arrow_count_block(n_both, n_false, n_miss, n_both_correct, errors, *, n_gt, n_pred):
    match = n_gt == n_pred and n_false == 0 and n_miss == 0 and n_both_correct == n_both
    counts = {
        "tp": int(n_both),
        "fp": int(n_false),
        "fn": int(n_miss),
        "n_both": int(n_both),
        "n_false": int(n_false),
        "n_miss": int(n_miss),
        "n_both_correct": int(n_both_correct),
    }
    return {
        "e2e": counts,
        "matched": counts,
        "errors": errors,
        "match": match,
        "n_both": int(n_both),
        "n_false": int(n_false),
        "n_miss": int(n_miss),
        "n_both_correct": int(n_both_correct),
    }


def score_arrow_multiset(gt_lanes, pred_lanes):
    """Score incoming arrows by existence then category on LTR-matched lanes.

    ``LS``, ``LR``, ``SR``, and ``LSR`` are independent categories, not unions of
    L/S/R. Each incoming lane is paired by mouth order. Existence counts:
    ``N_both`` (both sides have an arrow), ``N_false`` (pred only), ``N_miss``
    (GT only). Category accuracy is exact-code match on ``N_both`` only.
    Extra incoming lanes on the longer side are unmatched (miss or false).
    Lanes with no pavement arrow on either side contribute nothing.
    """
    gt_lanes = list(gt_lanes or [])
    pred_lanes = list(pred_lanes or [])
    n_pair = min(len(gt_lanes), len(pred_lanes))
    n_both = n_false = n_miss = n_both_correct = 0
    errors = []

    def _record(kind, gt_code, pred_code):
        nonlocal n_both, n_false, n_miss, n_both_correct
        if kind == "both":
            n_both += 1
            n_both_correct += 1
        elif kind == "sub":
            n_both += 1
            errors.append({
                "label": gt_code,
                "pred_label": pred_code,
                "error_type": "sub",
            })
        elif kind == "fn":
            n_miss += 1
            errors.append({"label": gt_code, "error_type": "fn"})
        elif kind == "fp":
            n_false += 1
            payload = {"error_type": "fp"}
            if pred_code:
                payload["pred_label"] = pred_code
            errors.append(payload)

    for idx in range(n_pair):
        kind, gt_code, pred_code = _arrow_pair_event(gt_lanes[idx], pred_lanes[idx])
        _record(kind, gt_code, pred_code)
    for lane in gt_lanes[n_pair:]:
        kind, gt_code, pred_code = _arrow_pair_event(lane, None)
        _record(kind, gt_code, pred_code)
    for lane in pred_lanes[n_pair:]:
        kind, gt_code, pred_code = _arrow_pair_event(None, lane)
        _record(kind, gt_code, pred_code)

    scored = _arrow_count_block(
        n_both, n_false, n_miss, n_both_correct, errors,
        n_gt=len(gt_lanes), n_pred=len(pred_lanes),
    )
    scored["gt_labels"] = _incoming_arrow_codes(gt_lanes)
    scored["pred_labels"] = _incoming_arrow_codes(pred_lanes)
    return scored


def score_arrow_alignment(pairs):
    """Score existence then category on already-paired incoming lanes."""
    n_both = n_false = n_miss = n_both_correct = 0
    errors = []
    n_gt = n_pred = 0
    for pair in pairs:
        status = pair.get("status")
        if status == "matched":
            n_gt += 1
            n_pred += 1
            kind, gt_code, pred_code = _arrow_pair_event(pair.get("gt"), pair.get("pred"))
        elif status == "unmatched_gt":
            n_gt += 1
            kind, gt_code, pred_code = _arrow_pair_event(pair.get("gt"), None)
        elif status == "unmatched_pred":
            n_pred += 1
            kind, gt_code, pred_code = _arrow_pair_event(None, pair.get("pred"))
        else:
            continue
        if kind == "both":
            n_both += 1
            n_both_correct += 1
        elif kind == "sub":
            n_both += 1
            errors.append({
                **pair, "label": gt_code, "pred_label": pred_code, "error_type": "sub",
            })
        elif kind == "fn":
            n_miss += 1
            errors.append({**pair, "label": gt_code, "error_type": "fn"})
        elif kind == "fp":
            n_false += 1
            payload = {**pair, "error_type": "fp"}
            if pred_code:
                payload["pred_label"] = pred_code
            errors.append(payload)
    scored = _arrow_count_block(
        n_both, n_false, n_miss, n_both_correct, errors,
        n_gt=n_gt, n_pred=n_pred,
    )
    return scored


def evaluate_arm_arrows(gt_arm, pred_arm):
    """Score incoming mouth arrows: existence P/R, then category ACC on N_both."""
    gt_lanes = incoming_mouth_lanes(gt_arm)
    pred_lanes = incoming_mouth_lanes(pred_arm) if pred_arm is not None else []
    scored = score_arrow_multiset(gt_lanes, pred_lanes)
    return {
        "gt": gt_lanes,
        "pred": pred_lanes if pred_arm is not None else None,
        "gt_labels": scored["gt_labels"],
        "pred_labels": scored["pred_labels"],
        "e2e": scored["e2e"],
        "matched": scored["matched"],
        "n_both": scored["n_both"],
        "n_false": scored["n_false"],
        "n_miss": scored["n_miss"],
        "n_both_correct": scored["n_both_correct"],
        "errors": scored["errors"],
        "match": scored["match"],
    }


def align_arrow_sequences(gt_arrows, pred_arrows):
    """Backward-compatible wrapper: LTR existence/category score over arrow codes."""

    def _as_lanes(arrows):
        if arrows is None:
            return []
        lanes = []
        for code in arrows:
            normalized = normalize_arrow(code)
            if normalized:
                lanes.append({
                    "travel": "incoming",
                    "code": normalized,
                    "labels": [normalized],
                })
        return lanes

    scored = score_arrow_multiset(_as_lanes(gt_arrows), _as_lanes(pred_arrows))
    return {
        "gt_labels": scored["gt_labels"],
        "pred_labels": scored["pred_labels"],
        "errors": scored["errors"],
        "match": scored["match"],
        "cost": scored["e2e"]["fp"] + scored["e2e"]["fn"],
        "e2e": scored["e2e"],
        "matched": scored["matched"],
    }


def channelized_error_items(cf):
    """Channelized mismatches with error_type: gt_only / pred_only."""
    items = []
    for pair in cf.get("missing_in_pred") or []:
        items.append({**pair, "error_type": "gt_only"})
    for pair in cf.get("extra_in_pred") or []:
        items.append({**pair, "error_type": "pred_only"})
    return items


def through_road_error_items(roads):
    """Through-road mismatches with error_type: gt_only / pred_only."""
    items = []
    for pair in roads.get("missing_in_pred") or []:
        items.append({**pair, "error_type": "gt_only"})
    for pair in roads.get("extra_in_pred") or []:
        items.append({**pair, "error_type": "pred_only"})
    return items


def _safe_div(num, den):
    if not den:
        return None
    return float(num) / float(den)


def _arrow_matched_coverage(matched_tp, matched_fn, e2e_tp, e2e_fn):
    """GT arrows on incoming-count-matched arms / all GT arrows."""
    gt_n = int(e2e_tp or 0) + int(e2e_fn or 0)
    if not gt_n:
        return None
    matched_gt = int(matched_tp or 0) + int(matched_fn or 0)
    return round(matched_gt / float(gt_n), 6)


def _prf1(tp, fp, fn, n_both_correct=None):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    n_both = int(tp)
    accuracy = None if n_both_correct is None else _safe_div(n_both_correct, n_both)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "n_both": n_both,
        "n_false": int(fp),
        "n_miss": int(fn),
        "n_both_correct": None if n_both_correct is None else int(n_both_correct),
        "precision": None if precision is None else round(precision, 6),
        "recall": None if recall is None else round(recall, 6),
        "f1": None if f1 is None else round(f1, 6),
        "accuracy": None if accuracy is None else round(accuracy, 6),
    }


def _mean_abs(values):
    if not values:
        return None, 0
    return round(sum(values) / len(values), 6), len(values)


def _lane_count_abs_and_ape(gt_val, pred_val):
    """Absolute error for MAE, and APE for MAPE (None when undefined).

    MAPE uses |pred-gt|/|gt| when gt is present and non-zero. gt=0 with pred=0
    contributes 0; gt=0 with pred!=0 is skipped. Pred-only rows have no APE.
    """
    if gt_val is not None and pred_val is not None:
        abs_err = abs(gt_val - pred_val)
    elif gt_val is not None:
        abs_err = abs(gt_val)
    elif pred_val is not None:
        abs_err = abs(pred_val)
    else:
        return None, None
    if gt_val is None:
        return abs_err, None
    if gt_val == 0:
        return abs_err, (0.0 if abs_err == 0 else None)
    return abs_err, abs_err / float(abs(gt_val))


def _lane_travel_swaps(gt_out, pred_out, gt_in, pred_in):
    """Outgoing↔incoming substitutions on one arm: ``(out_to_in, in_to_out)``.

    A lane counted as outgoing in GT and incoming in pred (or the reverse) is
    one swap, not two independent count errors.
    """
    if None in (gt_out, pred_out, gt_in, pred_in):
        return 0, 0
    d_out = int(pred_out) - int(gt_out)
    d_in = int(pred_in) - int(gt_in)
    out_to_in = min(max(-d_out, 0), max(d_in, 0))
    in_to_out = min(max(d_out, 0), max(-d_in, 0))
    return int(out_to_in), int(in_to_out)


def _count_leftover_lanes(gt_val, pred_val, leftover, matched):
    """How many leftover out/in lanes to count after travel swaps.

    Both sides present: ``leftover`` (|gt-pred| minus swaps). Missing pred uses
    GT count; extra pred uses pred count. One arm can contribute several lanes.
    """
    if leftover is None:
        if matched:
            return 0
        if gt_val is not None:
            return int(gt_val)
        if pred_val is not None:
            return int(pred_val)
        return 1
    return max(0, int(leftover))


def _lane_count_total_abs_error(gt_out, pred_out, gt_in, pred_in):
    """Lane-count error on one arm: travel swaps count once, leftover counts add.

    Missing pred/GT still uses the remaining count as the error.
    """
    if gt_out is None or gt_in is None:
        total = 0.0
        for gt_val, pred_val in ((gt_out, pred_out), (gt_in, pred_in)):
            if gt_val is None:
                continue
            abs_err, _ = _lane_count_abs_and_ape(gt_val, pred_val)
            if abs_err is not None:
                total += abs_err
        return total
    if pred_out is None or pred_in is None:
        total = 0.0
        for gt_val, pred_val in ((gt_out, pred_out), (gt_in, pred_in)):
            abs_err, _ = _lane_count_abs_and_ape(gt_val, pred_val)
            if abs_err is not None:
                total += abs_err
        return total
    out_to_in, in_to_out = _lane_travel_swaps(gt_out, pred_out, gt_in, pred_in)
    raw = abs(int(gt_out) - int(pred_out)) + abs(int(gt_in) - int(pred_in))
    return float(raw - out_to_in - in_to_out)


def _lane_count_within_tolerance(gt_out, pred_out, gt_in, pred_in, *, tolerance=1):
    """True when total incoming+outgoing lane-count error is at most ``tolerance``."""
    return _lane_count_total_abs_error(gt_out, pred_out, gt_in, pred_in) <= tolerance


def _arm_count_hit(metrics):
    """1 if predicted arm count equals GT, else 0. None if unknown."""
    if metrics.get("arm_count_accuracy") is not None:
        return 1.0 if metrics["arm_count_accuracy"] else 0.0
    gt, pred = metrics.get("arm_count_gt"), metrics.get("arm_count_pred")
    if gt is not None and pred is not None:
        return 1.0 if gt == pred else 0.0
    mae = metrics.get("arm_count_mae")
    if mae is not None:
        return 1.0 if mae == 0 else 0.0
    return None


def _directed_pairs(ids):
    return {(int(item["from_arm"]), int(item["to_arm"])) for item in ids or []}


def _undirected_pairs(ids):
    return {through_road_id(item) for item in ids or []}


def _compute_arm_level_metrics(arm):
    headings = arm.get("headings") or []
    heading_n = len(headings)
    heading_correct = sum(1 for item in headings if item.get("match"))
    heading_accuracy = None if not heading_n else round(heading_correct / float(heading_n), 6)
    gt_n = arm["arm_count"]["gt"]
    pred_n = arm["arm_count"]["pred"]
    arm_count_correct = int(gt_n == pred_n)
    gt_cf = _directed_pairs((arm.get("channelized_features") or {}).get("gt_ids"))
    pred_cf = _directed_pairs((arm.get("channelized_features") or {}).get("pred_ids"))
    gt_tr = _undirected_pairs((arm.get("through_roads") or {}).get("gt_ids"))
    pred_tr = _undirected_pairs((arm.get("through_roads") or {}).get("pred_ids"))
    return {
        "passed": arm.get("passed"),
        "heading_deg_accuracy": heading_accuracy,
        "heading_deg_correct": int(heading_correct),
        "heading_deg_n": int(heading_n),
        "arm_count_mae": abs(gt_n - pred_n),
        "arm_count_accuracy": float(arm_count_correct),
        "arm_count_correct": arm_count_correct,
        "arm_count_n": 1,
        "arm_count_gt": gt_n,
        "arm_count_pred": pred_n,
        "channelized_features": _prf1(
            tp=len(gt_cf & pred_cf),
            fp=len(pred_cf - gt_cf),
            fn=len(gt_cf - pred_cf),
        ),
        "through_roads": _prf1(
            tp=len(gt_tr & pred_tr),
            fp=len(pred_tr - gt_tr),
            fn=len(gt_tr - pred_tr),
        ),
    }


def _is_gt_lane_arm(item):
    """True for GT arms (matched or missing in pred). Extra pred arms are excluded."""
    if item.get("gt_arm_id") is not None:
        return True
    outgoing = item.get("outgoing") or {}
    incoming = item.get("incoming") or {}
    return outgoing.get("gt") is not None or incoming.get("gt") is not None


def _arrow_n_both_correct(block):
    if not block:
        return 0
    if block.get("n_both_correct") is not None:
        return int(block["n_both_correct"])
    return 0


def _compute_lanes_metrics(lane):
    abs_sum = 0.0
    term_n = 0
    arm_n = 0
    correct = 0
    incoming_correct = 0
    junction_lane_error = 0.0
    e2e_tp = e2e_fp = e2e_fn = 0
    matched_tp = matched_fp = matched_fn = 0
    e2e_correct = matched_correct = 0
    for item in lane["arms"]:
        outgoing = item.get("outgoing") or {}
        incoming = item.get("incoming") or {}
        gt_out, pred_out = outgoing.get("gt"), outgoing.get("pred")
        gt_in, pred_in = incoming.get("gt"), incoming.get("pred")
        if _is_gt_lane_arm(item):
            arm_n += 1
            out_abs, _ = _lane_count_abs_and_ape(gt_out, pred_out)
            in_abs, _ = _lane_count_abs_and_ape(gt_in, pred_in)
            if out_abs is not None:
                abs_sum += out_abs
                term_n += 1
            if in_abs is not None:
                abs_sum += in_abs
                term_n += 1
            if outgoing.get("match") and incoming.get("match"):
                correct += 1
            if incoming.get("match"):
                incoming_correct += 1
            junction_lane_error += _lane_count_total_abs_error(
                gt_out, pred_out, gt_in, pred_in
            )

        arrows = item.get("arrows") or {}
        e2e = arrows.get("e2e")
        if e2e is None:
            scored = score_arrow_multiset(arrows.get("gt") or [], arrows.get("pred") or [])
            e2e = scored["e2e"]
        e2e_tp += (e2e or {}).get("tp", 0)
        e2e_fp += (e2e or {}).get("fp", 0)
        e2e_fn += (e2e or {}).get("fn", 0)
        e2e_correct += _arrow_n_both_correct(e2e)
        if incoming.get("match"):
            matched_tp += (e2e or {}).get("tp", 0)
            matched_fp += (e2e or {}).get("fp", 0)
            matched_fn += (e2e or {}).get("fn", 0)
            matched_correct += _arrow_n_both_correct(e2e)

    lane_1_correct = None
    if arm_n:
        lane_1_correct = int(junction_lane_error <= 1.0)

    return {
        "passed": lane.get("passed"),
        "lane_count_mae": None if not term_n else round(abs_sum / term_n, 6),
        "lane_count_n": int(arm_n),
        "lane_count_accuracy": None if not arm_n else round(correct / float(arm_n), 6),
        "lane_count_correct": int(correct),
        "incoming_count_correct": int(incoming_correct),
        "lane_1_error": None if not arm_n else round(junction_lane_error, 6),
        "lane_1_accuracy": None if lane_1_correct is None else float(lane_1_correct),
        "lane_1_correct": lane_1_correct,
        "incoming_arrow_e2e": _prf1(e2e_tp, e2e_fp, e2e_fn, n_both_correct=e2e_correct),
        "incoming_arrow_matched": _prf1(
            matched_tp, matched_fp, matched_fn, n_both_correct=matched_correct
        ),
        "incoming_arrow_matched_coverage": _arrow_matched_coverage(
            matched_tp, matched_fn, e2e_tp, e2e_fn
        ),
        "incoming_arrow": _prf1(e2e_tp, e2e_fp, e2e_fn, n_both_correct=e2e_correct),
    }


def compute_metrics(report):
    """Derive scalar metrics nested by stage from a full evaluation report."""
    metrics = {
        "passed": report.get("passed"),
        "arm_level": None,
        "lane_level": None,
        "refinement": None,
    }
    if report.get("arm_level") is not None:
        metrics["arm_level"] = _compute_arm_level_metrics(report["arm_level"])
    if report.get("lane_level") is not None:
        metrics["lane_level"] = _compute_lanes_metrics(report["lane_level"])
    if report.get("refinement") is not None:
        metrics["refinement"] = _compute_lanes_metrics(report["refinement"])
    return metrics


def _aggregate_arm_level_metrics(items):
    if not items:
        return None
    heading_correct = heading_n = 0
    arm_count_errs = []
    arm_count_correct = arm_count_n = 0
    cf_tp = cf_fp = cf_fn = 0
    tr_tp = tr_fp = tr_fn = 0
    passed_count = 0
    for metrics in items:
        if metrics.get("passed"):
            passed_count += 1
        n = int(metrics.get("heading_deg_n") or 0)
        if n:
            if metrics.get("heading_deg_correct") is not None:
                heading_correct += int(metrics["heading_deg_correct"])
            elif metrics.get("heading_deg_accuracy") is not None:
                heading_correct += int(round(float(metrics["heading_deg_accuracy"]) * n))
            heading_n += n
        if metrics.get("arm_count_mae") is not None:
            arm_count_errs.append(metrics["arm_count_mae"])
        k = int(metrics.get("arm_count_n") or 0)
        if metrics.get("arm_count_correct") is not None and k:
            arm_count_correct += int(metrics["arm_count_correct"])
            arm_count_n += k
        else:
            hit = _arm_count_hit(metrics)
            if hit is not None:
                arm_count_correct += int(hit)
                arm_count_n += 1
        cf = metrics.get("channelized_features") or {}
        cf_tp += cf.get("tp", 0)
        cf_fp += cf.get("fp", 0)
        cf_fn += cf.get("fn", 0)
        tr = metrics.get("through_roads") or {}
        tr_tp += tr.get("tp", 0)
        tr_fp += tr.get("fp", 0)
        tr_fn += tr.get("fn", 0)
    arm_count_mae, _ = _mean_abs(arm_count_errs)
    return {
        "n_cases": len(items),
        "n_passed": passed_count,
        "heading_deg_accuracy": None if not heading_n else round(heading_correct / float(heading_n), 6),
        "heading_deg_correct": int(heading_correct),
        "heading_deg_n": int(heading_n),
        "arm_count_mae": arm_count_mae,
        "arm_count_accuracy": None if not arm_count_n else round(arm_count_correct / float(arm_count_n), 6),
        "arm_count_correct": int(arm_count_correct),
        "arm_count_n": int(arm_count_n),
        "channelized_features": _prf1(cf_tp, cf_fp, cf_fn),
        "through_roads": _prf1(tr_tp, tr_fp, tr_fn),
    }


def _lane_count_mae_weight(metrics):
    """Return (mae, n_terms) for micro-averaging. n_terms is 2*M when both in/out exist."""
    arm_n = int(metrics.get("lane_count_n") or 0)
    if metrics.get("lane_count_mae") is not None and arm_n:
        return metrics["lane_count_mae"], 2 * arm_n
    out_mae = metrics.get("outgoing_lane_count_mae")
    in_mae = metrics.get("incoming_lane_count_mae")
    out_n = int(metrics.get("outgoing_lane_count_n") or 0)
    in_n = int(metrics.get("incoming_lane_count_n") or 0)
    term_n = out_n + in_n
    if out_mae is not None and in_mae is not None and term_n:
        return (out_mae * out_n + in_mae * in_n) / term_n, term_n
    if out_mae is not None and out_n:
        return out_mae, out_n
    if in_mae is not None and in_n:
        return in_mae, in_n
    return None, 0


def _aggregate_lanes_metrics(items):
    if not items:
        return None
    mae_sum = term_n = 0.0
    acc_correct = acc_n = 0.0
    tol1_correct = tol1_n = 0.0
    e2e_tp = e2e_fp = e2e_fn = 0
    matched_tp = matched_fp = matched_fn = 0
    e2e_correct = matched_correct = 0
    passed_count = 0
    for metrics in items:
        if metrics.get("passed"):
            passed_count += 1
        mae, n_terms = _lane_count_mae_weight(metrics)
        if mae is not None and n_terms:
            mae_sum += mae * n_terms
            term_n += n_terms
        arm_n = int(metrics.get("lane_count_n") or 0)
        if metrics.get("lane_count_correct") is not None and arm_n:
            acc_correct += int(metrics["lane_count_correct"])
            acc_n += arm_n
        elif metrics.get("lane_count_accuracy") is not None and arm_n:
            acc_correct += float(metrics["lane_count_accuracy"]) * arm_n
            acc_n += arm_n
        if metrics.get("lane_1_correct") is not None:
            tol1_correct += int(metrics["lane_1_correct"])
            tol1_n += 1
        elif metrics.get("lane_1_accuracy") is not None:
            tol1_correct += 1 if float(metrics["lane_1_accuracy"]) >= 1.0 - 1e-9 else 0
            tol1_n += 1
        e2e = metrics.get("incoming_arrow_e2e") or metrics.get("incoming_arrow") or {}
        matched = metrics.get("incoming_arrow_matched") or {}
        e2e_tp += e2e.get("tp", 0)
        e2e_fp += e2e.get("fp", 0)
        e2e_fn += e2e.get("fn", 0)
        e2e_correct += _arrow_n_both_correct(e2e)
        matched_tp += matched.get("tp", 0)
        matched_fp += matched.get("fp", 0)
        matched_fn += matched.get("fn", 0)
        matched_correct += _arrow_n_both_correct(matched)
    return {
        "n_cases": len(items),
        "n_passed": passed_count,
        "lane_count_mae": None if not term_n else round(mae_sum / term_n, 6),
        "lane_count_n": int(round(term_n / 2.0)) if term_n else 0,
        "lane_count_accuracy": None if not acc_n else round(acc_correct / acc_n, 6),
        "lane_count_correct": int(round(acc_correct)),
        "lane_1_accuracy": None if not tol1_n else round(tol1_correct / tol1_n, 6),
        "lane_1_correct": int(round(tol1_correct)),
        "incoming_arrow_e2e": _prf1(e2e_tp, e2e_fp, e2e_fn, n_both_correct=e2e_correct),
        "incoming_arrow_matched": _prf1(
            matched_tp, matched_fp, matched_fn, n_both_correct=matched_correct
        ),
        "incoming_arrow_matched_coverage": _arrow_matched_coverage(
            matched_tp, matched_fn, e2e_tp, e2e_fn
        ),
        "incoming_arrow": _prf1(e2e_tp, e2e_fp, e2e_fn, n_both_correct=e2e_correct),
    }


def lanes_metrics_for_stage(metrics, stage):
    """Metrics section stored under ``stage``."""
    return (metrics or {}).get(stage)


def _collect_stage_lanes(items, stage):
    return [section for section in (lanes_metrics_for_stage(m, stage) for m in items) if section]


def aggregate_metrics(metrics_list):
    """Micro-average nested case metrics into one batch summary by stage."""
    items = [m for m in metrics_list if m]
    if not items:
        return None
    passed_count = sum(1 for metrics in items if metrics.get("passed"))
    return {
        "n_cases": len(items),
        "n_passed": passed_count,
        "arm_level": _aggregate_arm_level_metrics([m["arm_level"] for m in items if m.get("arm_level")]),
        "lane_level": _aggregate_lanes_metrics(_collect_stage_lanes(items, "lane_level")),
        "refinement": _aggregate_lanes_metrics(_collect_stage_lanes(items, "refinement")),
        "layer_fea": _aggregate_lanes_metrics(_collect_stage_lanes(items, "layer_fea")),
    }


def build_arrow_confusion_matrix(arrow_metrics):
    """Keep existence (N_both/N_false/N_miss) and category-accuracy counts."""
    arrow = arrow_metrics or {}
    tp = int(arrow.get("tp") or arrow.get("n_both") or 0)
    fp = int(arrow.get("fp") or arrow.get("n_false") or 0)
    fn = int(arrow.get("fn") or arrow.get("n_miss") or 0)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_both": int(arrow.get("n_both") or tp),
        "n_false": int(arrow.get("n_false") or fp),
        "n_miss": int(arrow.get("n_miss") or fn),
        "n_both_correct": arrow.get("n_both_correct"),
        "precision": arrow.get("precision"),
        "recall": arrow.get("recall"),
        "f1": arrow.get("f1"),
        "accuracy": arrow.get("accuracy"),
    }


def format_arrow_confusion_matrix(confusion):
    """Return a one-line N_both/N_false/N_miss and P/R/ACC summary."""
    if not confusion:
        return "(empty)"

    def _line(key, block):
        return (
            "%s N_both=%s N_false=%s N_miss=%s N_both_correct=%s P=%s R=%s ACC=%s"
            % (
                key,
                block.get("n_both", block.get("tp")),
                block.get("n_false", block.get("fp")),
                block.get("n_miss", block.get("fn")),
                block.get("n_both_correct"),
                block.get("precision"),
                block.get("recall"),
                block.get("accuracy"),
            )
        )

    if "e2e" in confusion or "matched" in confusion:
        lines = []
        for key in ("e2e", "matched"):
            lines.append(_line(key, confusion.get(key) or {}))
        return "\n".join(lines)
    return _line("arrow", confusion)


def _arm_level_mismatches(arm):
    out = {}
    if not arm["arm_count"]["match"]:
        out["arm_count"] = {"gt": arm["arm_count"]["gt"], "pred": arm["arm_count"]["pred"]}
    heading_bad = []
    for item in arm["headings"]:
        if item["match"]:
            continue
        heading_bad.append({
            **_arm_ref(item),
            "gt": item["gt_heading_deg"],
            "pred": item["pred_heading_deg"],
            "diff_deg": item["diff_deg"],
        })
    if heading_bad:
        out["heading_deg"] = heading_bad
    cf = arm["channelized_features"]
    if not cf["match"]:
        cf_items = channelized_error_items(cf)
        if cf["errors"]:
            for err in cf["errors"]:
                cf_items.append({"error_type": "error", "message": err})
        if cf_items:
            out["channelized_features"] = cf_items
    roads = arm.get("through_roads") or {}
    if roads and not roads.get("match", True):
        tr_items = through_road_error_items(roads)
        if roads.get("errors"):
            for err in roads["errors"]:
                tr_items.append({"error_type": "error", "message": err})
        if tr_items:
            out["through_roads"] = tr_items
    return out


def _lanes_mismatches(lane):
    out = {}
    outgoing_bad = []
    incoming_bad = []
    travel_bad = []
    arrows_bad = []
    for item in lane["arms"]:
        outgoing = item.get("outgoing") or {}
        incoming = item.get("incoming") or {}
        arrows = item.get("arrows") or {}
        gt_out, pred_out = outgoing.get("gt"), outgoing.get("pred")
        gt_in, pred_in = incoming.get("gt"), incoming.get("pred")
        out_to_in, in_to_out = _lane_travel_swaps(gt_out, pred_out, gt_in, pred_in)
        for _ in range(out_to_in):
            travel_bad.append({**_arm_ref(item), "from": "outgoing", "to": "incoming"})
        for _ in range(in_to_out):
            travel_bad.append({**_arm_ref(item), "from": "incoming", "to": "outgoing"})
        leftover_out = (
            None if None in (gt_out, pred_out)
            else abs(int(gt_out) - int(pred_out)) - out_to_in - in_to_out
        )
        leftover_in = (
            None if None in (gt_in, pred_in)
            else abs(int(gt_in) - int(pred_in)) - out_to_in - in_to_out
        )
        totals_match = (
            None not in (gt_out, pred_out, gt_in, pred_in)
            and int(gt_out) + int(gt_in) == int(pred_out) + int(pred_in)
        )
        if totals_match or out_to_in or in_to_out:
            n_out_err = 0
            n_in_err = 0
        else:
            n_out_err = _count_leftover_lanes(
                gt_out, pred_out, leftover_out, outgoing.get("match"),
            )
            n_in_err = _count_leftover_lanes(
                gt_in, pred_in, leftover_in, incoming.get("match"),
            )
        for _ in range(n_out_err):
            outgoing_bad.append({**_arm_ref(item), "gt": gt_out, "pred": pred_out})
        for _ in range(n_in_err):
            incoming_bad.append({**_arm_ref(item), "gt": gt_in, "pred": pred_in})
        if not incoming.get("match"):
            continue
        lane_errors = arrows.get("errors")
        if lane_errors is None and not arrows.get("match", True):
            scored = score_arrow_multiset(arrows.get("gt") or [], arrows.get("pred") or [])
            lane_errors = scored["errors"]
        fps, fns, rest = [], [], []
        for lane_err in lane_errors or []:
            kind = lane_err.get("error_type")
            if kind == "fp":
                fps.append(lane_err)
            elif kind == "fn":
                fns.append(lane_err)
            else:
                rest.append(lane_err)
        fps = fps[min(out_to_in, len(fps)):]
        fns = fns[min(in_to_out, len(fns)):]
        for lane_err in rest + fns + fps:
            compact = {
                "label": lane_err.get("label"),
                "error_type": lane_err.get("error_type"),
            }
            if lane_err.get("pred_label"):
                compact["pred_label"] = lane_err["pred_label"]
            arrows_bad.append({**_arm_ref(item), **compact})
    if outgoing_bad:
        out["outgoing_lane_count"] = outgoing_bad
    if incoming_bad:
        out["incoming_lane_count"] = incoming_bad
    if travel_bad:
        out["lane_travel"] = travel_bad
    if arrows_bad:
        out["incoming_arrows"] = arrows_bad
    return out


def n_lane_mismatches(lane_report):
    """Count leftover out/in lanes, travel swaps, and arrow errors (one each)."""
    section = _lanes_mismatches(lane_report) if lane_report else {}
    return sum(len(section.get(key) or []) for key in (
        "outgoing_lane_count", "incoming_lane_count", "lane_travel", "incoming_arrows",
    ))


def lanes_at_most_one_mismatch(gt, pred, arm_mapping=None):
    """True when leftover lane/travel/arrow errors <= 1 (includes a full match)."""
    if not gt or not pred:
        return None
    try:
        return n_lane_mismatches(evaluate_lane_level(gt, pred, arm_mapping=arm_mapping)) <= 1
    except Exception:
        return False


def mismatch_report(report):
    """Keep only mismatched metrics, nested by stage."""
    out = {
        "passed": report["passed"],
        "arm_level": None,
        "lane_level": None,
        "refinement": None,
    }
    if report.get("arm_level") is not None:
        section = _arm_level_mismatches(report["arm_level"])
        out["arm_level"] = section or {}
    if report.get("lane_level") is not None:
        section = _lanes_mismatches(report["lane_level"])
        out["lane_level"] = section or {}
    if report.get("refinement") is not None:
        section = _lanes_mismatches(report["refinement"])
        out["refinement"] = section or {}
    return out


def slice_report(report, stages):
    """Keep only selected stages from a full evaluation report."""
    out = {
        "arm_level": None,
        "lane_level": None,
        "refinement": None,
    }
    passed = True
    any_stage = False
    for stage in stages:
        section = report.get(stage)
        out[stage] = section
        if section is not None:
            any_stage = True
            passed = passed and bool(section.get("passed"))
    out["passed"] = passed if any_stage else bool(report.get("passed"))
    return out


def dump_evaluation_dir(eval_dir, report):
    """Write ``full.json`` / ``mismatches.json`` / ``metrics.json`` under ``eval_dir``."""
    eval_dir = Path(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)
    mismatches = mismatch_report(report)
    metrics = compute_metrics(report)
    dump_paths = {
        "full": eval_dir / "full.json",
        "mismatches": eval_dir / "mismatches.json",
        "metrics": eval_dir / "metrics.json",
    }
    dump_paths["full"].write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dump_paths["mismatches"].write_text(
        json.dumps(mismatches, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    dump_paths["metrics"].write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "report": report,
        "mismatches": mismatches,
        "metrics": metrics,
        "paths": {key: str(path) for key, path in dump_paths.items()},
    }


def _default_stage_eval_dirs(
    case_dir,
    lane_model,
    refine_model,
    *,
    lane_reasoning="none",
    refine_reasoning="none",
):
    from scripts.methods.method_utils import lane_level_dir, refinement_dir

    return (
        lane_level_dir(case_dir, lane_model, lane_reasoning) / "evaluation",
        refinement_dir(case_dir, refine_model, refine_reasoning) / "evaluation",
    )


def _resolve_stage_eval_dirs_from_pred(pred_dir):
    """Infer lane/refinement evaluation dirs from a stage model folder or case folder.

    When ``pred_dir`` is already a ``{model}_{reasoning}`` stage folder, reuse that
    folder name as the sibling key under lane_level/ and refinement/.
    """
    folder = Path(pred_dir)
    parent_name = folder.parent.name if folder.parent else ""
    if parent_name in ("arm_level", "lane_level", "refinement"):
        case_dir = folder.parent.parent
        model_dir = folder.name
        return (
            case_dir / "lane_level" / model_dir / "evaluation",
            case_dir / "refinement" / model_dir / "evaluation",
        )

    lane_eval = None
    refine_eval = None
    lane_root = folder / "lane_level"
    refine_root = folder / "refinement"
    if lane_root.is_dir():
        lane_models = sorted(p for p in lane_root.iterdir() if p.is_dir())
        if len(lane_models) == 1:
            lane_eval = lane_models[0] / "evaluation"
    if refine_root.is_dir():
        refine_models = sorted(p for p in refine_root.iterdir() if p.is_dir())
        if len(refine_models) == 1:
            refine_eval = refine_models[0] / "evaluation"
    if lane_eval is None and refine_eval is None:
        return folder / "evaluation", None
    return lane_eval, refine_eval


def run_evaluation(
    gt_dir,
    pred_dir=None,
    *,
    case_dir=None,
    arm_model=None,
    lane_model=None,
    refine_model=None,
    arm_reasoning="none",
    lane_reasoning="none",
    refine_reasoning="none",
    heading_tol_deg=DEFAULT_HEADING_TOL_DEG,
    out_dir=None,
    refine_out_dir=None,
):
    """Score arm, lane, and refinement artifacts against GT.

    Paths come from ``pred_dir``, or from ``case_dir`` plus each stage's model.
    With ``out_dir`` set and ``refine_out_dir`` left unset, write one bundle there.
    """
    from scripts.methods.method_utils import lane_level_dir

    gt_dir = Path(gt_dir)
    gt_arms, gt_lanes, _gt_refine = resolve_side_dir(gt_dir)

    lane_eval_dir = None
    refine_eval_dir = None
    if case_dir is not None and arm_model is not None and lane_model is not None and refine_model is not None:
        pred_arms, pred_lanes, pred_refine_lanes = resolve_pred_artifacts(
            case_dir,
            arm_model=arm_model,
            lane_model=lane_model,
            refine_model=refine_model,
            arm_reasoning=arm_reasoning,
            lane_reasoning=lane_reasoning,
            refine_reasoning=refine_reasoning,
        )
        lane_eval_dir, refine_eval_dir = _default_stage_eval_dirs(
            case_dir,
            lane_model,
            refine_model,
            lane_reasoning=lane_reasoning,
            refine_reasoning=refine_reasoning,
        )
    else:
        if pred_dir is None:
            raise ValueError("provide pred_dir or case_dir with arm_model/lane_model/refine_model")
        pred_dir = Path(pred_dir)
        pred_arms, pred_lanes, pred_refine_lanes = resolve_side_dir(pred_dir)
        lane_eval_dir, refine_eval_dir = _resolve_stage_eval_dirs_from_pred(pred_dir)

    if gt_arms is None or pred_arms is None:
        gt_arms = None
        pred_arms = None
    if gt_lanes is None:
        pred_lanes = None
        pred_refine_lanes = None
    if gt_arms is None and gt_lanes is None:
        raise FileNotFoundError("no arms.json/lanes.json under GT %s" % gt_dir)
    if pred_arms is None and pred_lanes is None and pred_refine_lanes is None:
        raise FileNotFoundError("no pred arms/lanes/refine artifacts resolved")

    report = evaluate(
        gt_arms=gt_arms,
        pred_arms=pred_arms,
        gt_lanes=gt_lanes,
        pred_lanes=pred_lanes,
        pred_refine_lanes=pred_refine_lanes,
        heading_tol_deg=heading_tol_deg,
    )
    mismatches = mismatch_report(report)
    metrics = compute_metrics(report)

    paths = {"lane_level": None, "refinement": None, "layer_fea": None, "combined": None}

    # Explicit single-directory override (CLI --out).
    if out_dir is not None and refine_out_dir is None:
        dumped = dump_evaluation_dir(out_dir, report)
        paths["combined"] = dumped["paths"]
        return {
            "report": report,
            "mismatches": mismatches,
            "metrics": metrics,
            "paths": paths,
            "lane_paths": None,
            "refine_paths": None,
        }

    if out_dir is not None:
        lane_eval_dir = Path(out_dir)
    if refine_out_dir is not None:
        refine_eval_dir = Path(refine_out_dir)

    lane_paths = None
    refine_paths = None
    if lane_eval_dir is not None and (report.get("arm_level") is not None or report.get("lane_level") is not None):
        lane_bundle = dump_evaluation_dir(lane_eval_dir, slice_report(report, ("arm_level", "lane_level")))
        lane_paths = lane_bundle["paths"]
        paths["lane_level"] = lane_paths
    if refine_eval_dir is not None and report.get("refinement") is not None:
        refine_bundle = dump_evaluation_dir(refine_eval_dir, slice_report(report, ("refinement",)))
        refine_paths = refine_bundle["paths"]
        paths["refinement"] = refine_paths

    fea_paths = None
    fea_root = Path(refine_eval_dir).parent / "layer_fea" if refine_eval_dir is not None else None
    if fea_root is None and case_dir is not None and refine_model is not None:
        from scripts.methods.method_utils import refinement_dir as _refinement_dir
        fea_root = _refinement_dir(Path(case_dir), refine_model, refine_reasoning) / "layer_fea"
    if fea_root is not None and gt_lanes is not None:
        fea_lanes_path = _optional_file(fea_root / LANES_JSON_NAME)
        if fea_lanes_path is not None:
            fea_report = evaluate(
                gt_arms=gt_arms,
                pred_arms=pred_arms,
                gt_lanes=gt_lanes,
                pred_refine_lanes=fea_lanes_path,
                heading_tol_deg=heading_tol_deg,
            )
            fea_bundle = dump_evaluation_dir(
                fea_root / "evaluation", slice_report(fea_report, ("refinement",)),
            )
            fea_paths = fea_bundle["paths"]
            paths["layer_fea"] = fea_paths

    if lane_paths is None and refine_paths is None:
        fallback = Path(out_dir) if out_dir is not None else (
            Path(pred_dir) / "evaluation"
            if pred_dir is not None
            else lane_level_dir(case_dir, lane_model, lane_reasoning) / "evaluation"
        )
        dumped = dump_evaluation_dir(fallback, report)
        paths["combined"] = dumped["paths"]

    return {
        "report": report,
        "mismatches": mismatches,
        "metrics": metrics,
        "paths": paths,
        "lane_paths": lane_paths,
        "refine_paths": refine_paths,
        "fea_paths": fea_paths,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate arm-level / lane-level / refinement against ground truth.")
    parser.add_argument("--gt", default="dataset/openaerialmap/1003")
    parser.add_argument("--pred", default="scripts/methods/llm_kg/results/openaerialmap/1003/lane_level/openai_gpt_5_6_sol_medium")
    parser.add_argument("--heading-tol", type=float, default=DEFAULT_HEADING_TOL_DEG)
    parser.add_argument(
        "--out",
        default=None,
        help="Write one combined evaluation bundle here (default: stage dirs under lane_level/ and refinement/)",
    )
    args = parser.parse_args()

    result = run_evaluation(args.gt, args.pred, heading_tol_deg=args.heading_tol, out_dir=args.out)
    print(json.dumps(result["mismatches"], indent=2, ensure_ascii=False))
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    for stage, stage_paths in (result.get("paths") or {}).items():
        if not stage_paths:
            continue
        for key, path in stage_paths.items():
            print("saved[%s]: %s" % (stage, path), flush=True)
    raise SystemExit(0 if result["report"]["passed"] else 2)
