"""Stage-wise reconstruction / identification metrics used by eval.ipynb and the pipeline.

Stages
------
- **stage1**: after lane-level generation
- **stage2-1**: after layer-1 (through-road) consistency refinement
- **stage2-2**: after layer-2 (LLM re-identify) consistency refinement, before the solver
- **stage3**: after constraint-based refinement (feasibility-solver min-edit)
"""

import math
from pathlib import Path

from scripts.data_reader import DATASET_ROOT
from scripts.methods.evaluation import (
    DEFAULT_HEADING_TOL_DEG,
    arms_by_id,
    compute_metrics,
    evaluate_arm_level,
    evaluate_lane_level,
    load_json,
    match_arms_by_heading,
    n_lane_mismatches,
    way_kind_match_counts,
    _is_gt_lane_arm,
    _lanes_mismatches,
)
from scripts.methods.method_utils import (
    arm_level_dir,
    junction_level_dir,
    lane_level_dir,
    list_junction_artifact_roots,
    load_json_if_exists,
    refinement_dir,
    resolve_case_dir,
    resolve_output_root,
)

METHOD = "llm_kg"
METRIC_STAGES = ("1", "2-1", "2-2", "3")
RECONSTRUCTION_METRICS = (
    "gen acc",
    "rec acc",
    "rec@1 acc",
)
ARM_METRICS = (
    "arm count emr",
    "heading emr",
)
LANE_METRICS = (
    "lane count mae",
    "lane count emr",
    "matched arrow precision",
    "matched arrow recall",
    "matched arrow accuracy",
)
WAY_KIND_ARM_TYPES = (
    "one-way",
    "two-way",
)
WAY_KIND_METRICS = (
    "lane count emr",
)
COMPARE_WAY_KIND_COLUMNS = [
    "dataset", "arm type", "metric", "DI", "HI", "HAI", "KHAI-1", "KHAI-2", "KHAI",
]
RESOURCE_METRICS = (
    "time",
    "cost",
)
MATCHED_ARROW_COUNT_METRICS = (
    "matched arms",
    "N_both",
    "N_false",
    "N_miss",
    "N_both_correct",
)
LANE_ERROR_MIX_METRICS = (
    "outgoing lane count",
    "incoming lane count",
    "lane travel",
    "arrow substitution",
    "arrow FN",
    "arrow FP",
)
LANE_ERROR_METRICS = LANE_ERROR_MIX_METRICS
LANE_ERROR_SNAP = {
    "outgoing lane count": "err_out",
    "incoming lane count": "err_in",
    "lane travel": "err_travel",
    "arrow substitution": "err_arrow_sub",
    "arrow FN": "err_arrow_fn",
    "arrow FP": "err_arrow_fp",
}
LANE_ERROR_RATE = {
    "outgoing lane count": ("err_out", "err_n_out"),
    "incoming lane count": ("err_in", "err_n_in"),
    "lane travel": ("err_travel", "err_n_lanes"),
}
LANE_SNAP_KEYS = (
    "lane_abs",
    "lane_terms",
    "lane_correct",
    "lane_arms",
    "way_one_correct",
    "way_one_n",
    "way_two_correct",
    "way_two_n",
    "arrow_matched_arms",
    "arrow_tp",
    "arrow_fp",
    "arrow_fn",
    "arrow_matched_tp",
    "arrow_matched_fp",
    "arrow_matched_fn",
    "arrow_correct",
    "arrow_matched_correct",
    "err_n_out",
    "err_n_in",
    "err_n_lanes",
) + tuple(LANE_ERROR_SNAP.values())
SUMMARY_COLUMNS = ["dataset", "method", "metric"] + ["stage%s" % stage for stage in METRIC_STAGES]
COMPARE_COLUMNS = ["dataset", "metric", "DI", "HI", "HAI", "KHAI-1", "KHAI-2", "KHAI"]
COMPARE_KHAI_MIX_COLUMNS = ["dataset"] + list(LANE_ERROR_MIX_METRICS)
COMPARE_EMPTY = {
    "DI": None,
    "HI": None,
    "HAI": None,
    "KHAI-1": None,
    "KHAI-2": None,
    "KHAI": None,
}
DISPLAY_DATASET = {
    "nhtsa": "NHTSA Sketches",
    "openaerialmap": "Open Aerial Maps",
}
DISPLAY_METRIC = {
    "gen acc": "GENER SR",
    "rec acc": "RECON SR",
    "rec@1 acc": "RECON@1 SR",
    "arm count mae": r"$\text{MAE}_\text{arm,count}$",
    "arm count emr": r"$\text{EMR}_\text{arm,count}$",
    "heading emr": r"$\text{EMR}_\text{arm,heading}$",
    "lane count mae": r"$\text{MAE}_\text{lane,count}$",
    "lane count emr": r"$\text{EMR}_\text{lane,count}$",
    "matched arrow precision": r"$\text{PREC}_\text{arrow}$",
    "matched arrow recall": r"$\text{REC}_\text{arrow}$",
    "matched arrow accuracy": r"$\text{ACC}_\text{arrow}$",
    "outgoing lane count": r"$\text{ERR}_\text{out}$",
    "incoming lane count": r"$\text{ERR}_\text{in}$",
    "lane travel": r"$\text{ERR}_\text{in/out}$",
    "arrow substitution": r"$\text{ERR}_\text{arrow,mis}$",
    "arrow FN": r"$\text{ERR}_\text{arrow,FN}$",
    "arrow FP": r"$\text{ERR}_\text{arrow,FP}$",
}
DISPLAY_METRIC_REVERSE = {value: key for key, value in DISPLAY_METRIC.items()}
MAE_METRICS = {
    "arm count mae",
    "lane count mae",
}
RATE_METRICS = {
    "gen acc",
    "rec acc",
    "rec@1 acc",
    "arm count emr",
    "heading emr",
    "lane count emr",
    "arrow precision",
    "arrow recall",
    "arrow F1",
    "arrow accuracy",
    "matched arrow coverage",
    "matched arrow precision",
    "matched arrow recall",
    "matched arrow F1",
    "matched arrow accuracy",
    *LANE_ERROR_MIX_METRICS,
}


def dataset_family(name):
    name = str(name or "")
    if name.startswith("nhtsa"):
        return "nhtsa"
    if name == "openaerialmap":
        return "openaerialmap"
    return None


def _as_bool(value):
    if value is None:
        return None
    if isinstance(value, float):
        try:
            import math
            if math.isnan(value):
                return None
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "none", "nan"):
        return None
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return bool(value)


def _flag_acc(value):
    flag = _as_bool(value)
    return None if flag is None else float(flag)


def _first_existing(*paths):
    for path in paths:
        if path is not None and Path(path).is_file():
            return Path(path)
    return None


def _pass_rate(values):
    flags = [value for value in values if value is not None]
    if not flags:
        return None
    return round(sum(flags) / float(len(flags)), 4)


def _and_flags(flags):
    if not flags:
        return None
    if any(value is None for value in flags):
        return None
    return float(all(flags))


def _n_mismatches(lane_report):
    """Count leftover out/in lanes, travel swaps, and arrow errors (one each)."""
    return n_lane_mismatches(lane_report)


def _lane_error_counts(lane_report):
    """Per-lane leftover/travel counts, matched-arm arrow errors, and GT denominators."""
    section = _lanes_mismatches(lane_report) if lane_report else {}
    n_out = n_in = 0
    incoming_ok = set()
    for item in (lane_report or {}).get("arms") or []:
        if (item.get("incoming") or {}).get("match"):
            incoming_ok.add((item.get("gt_arm_id"), item.get("pred_arm_id")))
        if not _is_gt_lane_arm(item):
            continue
        gt_out = (item.get("outgoing") or {}).get("gt")
        gt_in = (item.get("incoming") or {}).get("gt")
        if gt_out is not None:
            n_out += int(gt_out)
        if gt_in is not None:
            n_in += int(gt_in)
    arrow_sub = arrow_fn = arrow_fp = 0
    for item in section.get("incoming_arrows") or []:
        if (item.get("gt_arm_id"), item.get("pred_arm_id")) not in incoming_ok:
            continue
        kind = item.get("error_type")
        if kind == "sub":
            arrow_sub += 1
        elif kind in ("fn", "unmatched_gt"):
            arrow_fn += 1
        elif kind in ("fp", "unmatched_pred"):
            arrow_fp += 1
    return {
        "err_out": len(section.get("outgoing_lane_count") or []),
        "err_in": len(section.get("incoming_lane_count") or []),
        "err_travel": len(section.get("lane_travel") or []),
        "err_arrow_sub": arrow_sub,
        "err_arrow_fn": arrow_fn,
        "err_arrow_fp": arrow_fp,
        "err_n_out": n_out,
        "err_n_in": n_in,
        "err_n_lanes": n_out + n_in,
    }


def _prf(tp, fp, fn):
    tp, fp, fn = int(tp or 0), int(fp or 0), int(fn or 0)
    precision = None if (tp + fp) == 0 else tp / float(tp + fp)
    recall = None if (tp + fn) == 0 else tp / float(tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _round4(value):
    return None if value is None else round(float(value), 4)


def _ratio(num, den):
    if den is None or float(den) == 0:
        return None
    return round(float(num) / float(den), 4)


def _mean_nonnull(values):
    values = [value for value in values if value is not None]
    if not values:
        return None
    return round(sum(values) / float(len(values)), 4)


def _add_cost(left, right):
    if left is None and right is None:
        return None
    return float(left or 0.0) + float(right or 0.0)


def _cost_usd_from_doc(data):
    if not isinstance(data, dict):
        return None
    for key in ("totals", "total"):
        block = data.get(key)
        if isinstance(block, dict) and block.get("cost_usd") is not None:
            return float(block["cost_usd"])
    if data.get("cost_usd") is not None:
        return float(data["cost_usd"])
    return None


def _cost_usd_from_path(path):
    return _cost_usd_from_doc(load_json_if_exists(path) or {})


def _file_mtime(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_mtime
    return None


def _case_dir_from_item(item):
    if item.get("case_dir"):
        return Path(item["case_dir"])
    output_dir = item.get("output_dir")
    dataset = item.get("dataset")
    case_id = item.get("case_id")
    if output_dir and dataset is not None and case_id is not None:
        return resolve_case_dir(output_dir, dataset, case_id)
    return None


def _count_assistant_messages(messages):
    n = 0
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            n += 1
    return n


def _entry_assistant_calls(entry):
    if isinstance(entry, list):
        return sum(
            _count_assistant_messages(call.get("messages") if isinstance(call, dict) else None)
            for call in entry
        )
    if not isinstance(entry, dict):
        return 0
    n = _count_assistant_messages(entry.get("messages"))
    if not entry.get("messages"):
        visual = entry.get("visual") or {}
        n += _count_assistant_messages(visual.get("messages"))
    organization = entry.get("organization") or {}
    n += _count_assistant_messages(organization.get("messages"))
    return n


def _assistant_calls(conversation, arm_id):
    if not isinstance(conversation, dict):
        return 0
    arm_key = str(arm_id)
    n = 0
    for round_doc in conversation.values():
        if not isinstance(round_doc, dict):
            continue
        n += _entry_assistant_calls(round_doc.get(arm_key))
    if n:
        return n
    return _entry_assistant_calls(conversation.get(arm_key))


def _refine_step_bucket(stage_name):
    text = str(stage_name or "").lower()
    if text.startswith("check") or text.startswith("layer1") or text.startswith("layer_1"):
        return "2-1"
    if (
        text.startswith("fix")
        or text.startswith("layer2")
        or text.startswith("layer_2")
        or text.startswith("layer3")
        or text.startswith("layer_3")
    ):
        return "2-2"
    return None


def _steps_cost(steps):
    total = 0.0
    priced = False
    for step in steps or []:
        if isinstance(step, dict) and step.get("cost_usd") is not None:
            total += float(step["cost_usd"])
            priced = True
    return total if priced else 0.0


def _split_refine_llm_cost(refine_dir):
    """Split refinement/cost.json across layer-1 vs layer-2.

    Newer runs tag steps ``check_*`` / ``fix_*``. Older inspect-patch runs share
    labels, so steps are sliced using assistant-message counts in each layer's
    ``conversation.json``.
    """
    refine_dir = Path(refine_dir)
    cost_doc = load_json_if_exists(refine_dir / "cost.json") or {}
    l1_conv = load_json_if_exists(refine_dir / "layer_1" / "conversation.json") or {}
    l2_conv = load_json_if_exists(refine_dir / "layer_2" / "conversation.json") or {}
    has_l1 = (refine_dir / "layer_1" / "lanes.json").is_file()
    has_l2 = (refine_dir / "layer_2" / "lanes.json").is_file()
    c21 = 0.0
    c22 = 0.0
    priced = False
    for arm_id, block in cost_doc.items():
        if arm_id in ("totals", "total") or not isinstance(block, dict):
            continue
        steps = list(block.get("steps") or [])
        if not steps:
            continue
        priced = True
        buckets = [_refine_step_bucket((step or {}).get("stage")) for step in steps]
        if steps and all(bucket is not None for bucket in buckets):
            for step, bucket in zip(steps, buckets):
                amount = _steps_cost([step])
                if bucket == "2-1":
                    c21 += amount
                else:
                    c22 += amount
            continue
        n1 = _assistant_calls(l1_conv, arm_id)
        n2 = _assistant_calls(l2_conv, arm_id)
        if n1 or n2:
            split_at = min(max(n1, 0), len(steps))
            c21 += _steps_cost(steps[:split_at])
            c22 += _steps_cost(steps[split_at:])
            continue
        amount = _steps_cost(steps)
        if has_l2 and not has_l1:
            c22 += amount
        else:
            c21 += amount
    if not priced:
        return 0.0, 0.0
    return c21, c22


def _stage_times_for_root(root, elapsed_sec, lane_model, lane_reasoning, refine_model, refine_reasoning):
    lane_mtime = _file_mtime(lane_level_dir(root, lane_model, lane_reasoning) / "lanes.json")
    refine_dir = refinement_dir(root, refine_model, refine_reasoning)
    stamps = {
        "2-1": _file_mtime(refine_dir / "layer_1" / "lanes.json"),
        "2-2": _file_mtime(refine_dir / "layer_2" / "lanes.json"),
        "3": _file_mtime(refine_dir / "layer_fea" / "lanes.json"),
    }
    times = {"1": None, "2-1": 0.0, "2-2": 0.0, "3": 0.0}
    prev = lane_mtime
    for stage in ("2-1", "2-2", "3"):
        ts = stamps[stage]
        if ts is None or prev is None:
            times[stage] = 0.0
            continue
        times[stage] = max(0.0, float(ts - prev))
        prev = ts
    later = times["2-1"] + times["2-2"] + times["3"]
    if elapsed_sec is not None:
        times["1"] = max(0.0, float(elapsed_sec) - later)
    return times


def collect_case_stage_resources(item):
    """Incremental wall-clock (s) and USD for stage1 / 2-1 / 2-2 / 3."""
    empty = {
        "time_stage1": item.get("elapsed_sec"),
        "time_stage2-1": 0.0 if item.get("elapsed_sec") is not None else None,
        "time_stage2-2": 0.0 if item.get("elapsed_sec") is not None else None,
        "time_stage3": 0.0 if item.get("elapsed_sec") is not None else None,
        "cost_stage1": item.get("cost_usd"),
        "cost_stage2-1": 0.0 if item.get("cost_usd") is not None else None,
        "cost_stage2-2": 0.0 if item.get("cost_usd") is not None else None,
        "cost_stage3": 0.0 if item.get("cost_usd") is not None else None,
    }
    case_dir = _case_dir_from_item(item)
    if case_dir is None or not case_dir.is_dir():
        return empty
    arm_model = item.get("arm_model") or item.get("model")
    lane_model = item.get("lane_model") or arm_model
    refine_model = item.get("refine_model") or lane_model
    junction_model = item.get("junction_model") or arm_model
    if not lane_model:
        return empty
    arm_reasoning = item.get("arm_reasoning") or "none"
    lane_reasoning = item.get("lane_reasoning") or "none"
    refine_reasoning = item.get("refine_reasoning") or "none"
    junction_reasoning = item.get("junction_reasoning") or "none"
    roots = list_junction_artifact_roots(case_dir)
    if not roots:
        roots = [case_dir]

    junction_cost = _cost_usd_from_path(
        junction_level_dir(case_dir, junction_model, junction_reasoning) / "cost.json"
    )
    arm_cost = None
    lane_cost = None
    c21 = 0.0
    c22 = 0.0
    fea_cost = None
    times = None
    for root in roots:
        arm_cost = _add_cost(arm_cost, _cost_usd_from_path(
            arm_level_dir(root, arm_model, arm_reasoning) / "cost.json"
        ))
        lane_cost = _add_cost(lane_cost, _cost_usd_from_path(
            lane_level_dir(root, lane_model, lane_reasoning) / "cost.json"
        ))
        refine_dir = refinement_dir(root, refine_model, refine_reasoning)
        layer_c21, layer_c22 = _split_refine_llm_cost(refine_dir)
        c21 += layer_c21
        c22 += layer_c22
        fea_cost = _add_cost(fea_cost, _cost_usd_from_path(refine_dir / "layer_fea" / "cost.json"))
        if times is None:
            times = _stage_times_for_root(
                root, item.get("elapsed_sec"),
                lane_model, lane_reasoning, refine_model, refine_reasoning,
            )
    if times is None:
        times = _stage_times_for_root(
            case_dir, item.get("elapsed_sec"),
            lane_model, lane_reasoning, refine_model, refine_reasoning,
        )
    stage1_cost = None
    for value in (junction_cost, arm_cost, lane_cost):
        stage1_cost = _add_cost(stage1_cost, value)
    return {
        "time_stage1": times.get("1"),
        "time_stage2-1": times.get("2-1"),
        "time_stage2-2": times.get("2-2"),
        "time_stage3": times.get("3"),
        "cost_stage1": stage1_cost,
        "cost_stage2-1": c21,
        "cost_stage2-2": c22,
        "cost_stage3": 0.0 if fea_cost is None else fea_cost,
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "%.4f" % value
    return str(value)


def format_table(records, columns):
    """Plain-text aligned table (no pandas)."""
    if not records:
        return "(empty)"
    rows = [{col: _cell(record.get(col)) for col in columns} for record in records]
    widths = {col: max(len(str(col)), max(len(row[col]) for row in rows)) for col in columns}
    header = "  ".join(str(col).ljust(widths[col]) for col in columns)
    rule = "  ".join("-" * widths[col] for col in columns)
    lines = [header, rule]
    for row in rows:
        lines.append("  ".join(row[col].ljust(widths[col]) for col in columns))
    return "\n".join(lines)


def eval_lane_snapshot(gt_lanes, pred_path, arm_mapping=None):
    """Reconstruction flags plus identification stats for one lanes.json snapshot."""
    if pred_path is None:
        return None
    pred_doc = load_json(pred_path)
    report = evaluate_lane_level(gt_lanes, pred_doc, arm_mapping=arm_mapping)
    section = compute_metrics({"lane_level": report}).get("lane_level") or {}
    n_mis = _n_mismatches(report)
    lane_err = section.get("lane_1_error")
    arm_n = int(section.get("lane_count_n") or 0)
    mae = section.get("lane_count_mae")
    terms = 2 * arm_n if arm_n else 0
    abs_sum = lane_err if lane_err is not None else (None if mae is None or not terms else mae * terms)
    e2e = section.get("incoming_arrow_e2e") or section.get("incoming_arrow") or {}
    matched = section.get("incoming_arrow_matched") or {}
    errors = _lane_error_counts(report)
    way = way_kind_match_counts(gt_lanes, pred_doc, arm_mapping=arm_mapping)
    return {
        "rec": float(bool(report.get("passed"))),
        "rec1": float(n_mis <= 1),
        "rec1lane": None if lane_err is None else float(float(lane_err) <= 1.0 + 1e-9),
        "lane_abs": abs_sum,
        "lane_terms": terms,
        "lane_correct": section.get("lane_count_correct"),
        "lane_arms": arm_n,
        "way_one_correct": way.get("way_one_correct") or 0,
        "way_one_n": way.get("way_one_n") or 0,
        "way_two_correct": way.get("way_two_correct") or 0,
        "way_two_n": way.get("way_two_n") or 0,
        "arrow_matched_arms": section.get("incoming_count_correct"),
        "arrow_tp": e2e.get("tp") or 0,
        "arrow_fp": e2e.get("fp") or 0,
        "arrow_fn": e2e.get("fn") or 0,
        "arrow_matched_tp": matched.get("tp") or 0,
        "arrow_matched_fp": matched.get("fp") or 0,
        "arrow_matched_fn": matched.get("fn") or 0,
        "arrow_correct": e2e.get("n_both_correct") or 0,
        "arrow_matched_correct": matched.get("n_both_correct") or 0,
        **errors,
    }


def eval_arm_snapshot(gt_arms, pred_arms):
    if not gt_arms or not pred_arms:
        return None
    report = evaluate_arm_level(gt_arms, pred_arms, heading_tol_deg=DEFAULT_HEADING_TOL_DEG)
    section = compute_metrics({"arm_level": report}).get("arm_level") or {}
    gt_n, pred_n = section.get("arm_count_gt"), section.get("arm_count_pred")
    mae = section.get("arm_count_mae")
    if mae is None and gt_n is not None and pred_n is not None:
        mae = abs(pred_n - gt_n)
    hit = None
    if gt_n is not None and pred_n is not None:
        hit = float(gt_n == pred_n)
    elif section.get("arm_count_correct") is not None:
        hit = float(section["arm_count_correct"])
    heading_correct = int(section.get("heading_deg_correct") or 0)
    heading_n = int(section.get("heading_deg_n") or 0)
    return {
        "arm_count_mae": mae,
        "arm_count_emr": hit,
        "heading_correct": heading_correct,
        "heading_n": heading_n,
        "heading_match": heading_n > 0 and heading_correct == heading_n,
        "arm_level_match": bool(hit) and (heading_n == 0 or heading_correct == heading_n),
    }


def arm_ident_fields(arm_stats):
    """Sum arm-level identification counts across junctions for one case."""
    heading_correct = sum((snap or {}).get("heading_correct") or 0 for snap in arm_stats)
    heading_n = sum((snap or {}).get("heading_n") or 0 for snap in arm_stats)
    return {
        "arm count mae": _mean_key(arm_stats, "arm_count_mae"),
        "arm count emr": _mean_key(arm_stats, "arm_count_emr"),
        "heading_correct": heading_correct,
        "heading_n": heading_n,
        "heading_match": None if not heading_n else heading_correct == heading_n,
        "arm_level_match": _and_match_flags(
            [(snap or {}).get("arm_level_match") for snap in arm_stats]
        ),
    }


def _and_match_flags(flags):
    """False if any flag is False; True if every present flag is True; else None."""
    present = [flag for flag in flags if flag is not None]
    if not present:
        return None
    return all(bool(flag) for flag in present)


def _snap_lane_match(snap):
    """True when incoming/outgoing counts and travel fully match GT."""
    if not snap:
        return None
    if (snap.get("err_out") or 0) or (snap.get("err_in") or 0) or (snap.get("err_travel") or 0):
        return False
    arms = snap.get("lane_arms")
    if not arms:
        return None
    return int(snap.get("lane_correct") or 0) == int(arms)


def _snap_arrow_match(snap):
    """True when incoming arrows match in existence and category."""
    if not snap:
        return None
    if (
        (snap.get("err_arrow_sub") or 0)
        or (snap.get("err_arrow_fn") or 0)
        or (snap.get("err_arrow_fp") or 0)
        or (snap.get("arrow_fp") or 0)
        or (snap.get("arrow_fn") or 0)
    ):
        return False
    return int(snap.get("arrow_correct") or 0) == int(snap.get("arrow_tp") or 0)


def _sum_dicts(items, keys):
    out = {key: 0.0 for key in keys}
    n = 0
    for item in items:
        if not item:
            continue
        n += 1
        for key in keys:
            value = item.get(key)
            if value is not None:
                out[key] += float(value)
    return out if n else None


def _mean_key(items, key):
    values = [item.get(key) for item in items if item and item.get(key) is not None]
    if not values:
        return None
    return sum(values) / float(len(values))


def snapshot_paths(root, lane_model, lane_reasoning, refine_model, refine_reasoning):
    """``lanes.json`` path for each metric stage."""
    lane_dir = lane_level_dir(root, lane_model, lane_reasoning)
    refine_dir = refinement_dir(root, refine_model, refine_reasoning)
    return {
        "1": _first_existing(lane_dir / "lanes.json"),
        "2-1": _first_existing(refine_dir / "layer_1" / "lanes.json"),
        "2-2": _first_existing(refine_dir / "layer_2" / "lanes.json"),
        "3": _first_existing(refine_dir / "lanes.json"),
    }


def _output_dir_of(item):
    if item.get("output_dir"):
        return item["output_dir"]
    case_dir = item.get("case_dir")
    if case_dir:
        path = Path(case_dir)
        if path.parent.name and path.parent.parent:
            return path.parent.parent
    return resolve_output_root("results")


def evaluate_case_artifacts(item):
    """Per-junction arm stats plus per-stage lane snapshots."""
    dataset = item.get("dataset")
    case_id = item.get("case_id")
    case_dir = resolve_case_dir(_output_dir_of(item), dataset, case_id)
    gt_lanes = load_json_if_exists(DATASET_ROOT / dataset / str(case_id) / "lanes.json")
    gt_arms = load_json_if_exists(DATASET_ROOT / dataset / str(case_id) / "arms.json")
    lane_model = item.get("lane_model")
    lane_reasoning = item.get("lane_reasoning") or "none"
    refine_model = item.get("refine_model") or lane_model
    refine_reasoning = item.get("refine_reasoning") or "none"
    arm_model = item.get("arm_model") or lane_model
    arm_reasoning = item.get("arm_reasoning") or "none"

    arm_stats = []
    lane_stats = {stage: [] for stage in METRIC_STAGES}
    for root in list_junction_artifact_roots(case_dir):
        arms_doc = load_json_if_exists(arm_level_dir(root, arm_model, arm_reasoning) / "arms.json")
        mapping = None
        if gt_arms and arms_doc:
            try:
                mapping = match_arms_by_heading(arms_by_id(gt_arms), arms_by_id(arms_doc))
            except Exception:
                mapping = None
            arm_stats.append(eval_arm_snapshot(gt_arms, arms_doc))
        if not gt_lanes:
            continue
        snapshots = snapshot_paths(
            root, lane_model, lane_reasoning, refine_model, refine_reasoning,
        )
        for stage, path in snapshots.items():
            lane_stats[stage].append(eval_lane_snapshot(gt_lanes, path, mapping))
    return arm_stats, lane_stats


def _complete_snaps(snaps):
    """Every junction snap for one stage, or None."""
    snaps = list(snaps or [])
    if not snaps or any(snap is None for snap in snaps):
        return None
    return snaps


def _ident_from_lane_snaps(snaps):
    if not snaps:
        return None
    return _sum_dicts(snaps, LANE_SNAP_KEYS)


def case_stage_row(item, *, method=METHOD):
    """One case: reconstruction flags, arm identification, lane-identification sums."""
    family = dataset_family(item.get("dataset"))
    arm_stats, lane_stats = evaluate_case_artifacts(item)
    rec = {}
    for stage, snaps in lane_stats.items():
        complete = _complete_snaps(snaps)
        rec[stage] = (
            None, None, None
        ) if complete is None else (
            _and_flags([snap["rec"] for snap in complete]),
            _and_flags([snap["rec1"] for snap in complete]),
            _and_flags([snap["rec1lane"] for snap in complete]),
        )
    rec_s1, one_s1, lane_s1 = rec["1"]
    rec_s21, one_s21, lane_s21 = rec["2-1"]
    rec_s22, one_s22, lane_s22 = rec["2-2"]
    rec_s3, one_s3, lane_s3 = rec["3"]

    final_snaps = _complete_snaps(lane_stats.get("3")) or []

    out = {
        "dataset": item.get("dataset"),
        "family": family,
        "source_dataset": item.get("dataset"),
        "case_id": item.get("case_id"),
        "method": method,
        "gen acc_stage1": _flag_acc(item.get("lane_level_valid")),
        "gen acc_stage2-1": _flag_acc(item.get("refine_l1_valid")),
        "gen acc_stage2-2": _flag_acc(item.get("refine_l2_valid")),
        "gen acc_stage3": _flag_acc(item.get("is_final_valid")),
        "rec acc_stage1": rec_s1,
        "rec acc_stage2-1": rec_s21,
        "rec acc_stage2-2": rec_s22,
        "rec acc_stage3": rec_s3,
        "rec@1 acc_stage1": one_s1,
        "rec@1 acc_stage2-1": one_s21,
        "rec@1 acc_stage2-2": one_s22,
        "rec@1 acc_stage3": one_s3,
        "rec@1lane acc_stage1": lane_s1,
        "rec@1lane acc_stage2-1": lane_s21,
        "rec@1lane acc_stage2-2": lane_s22,
        "rec@1lane acc_stage3": lane_s3,
        **arm_ident_fields(arm_stats),
        "lane_match": _and_match_flags(
            [_snap_lane_match(snap) for snap in final_snaps]
        ),
        "arrow_match": _and_match_flags(
            [_snap_arrow_match(snap) for snap in final_snaps]
        ),
        "time": item.get("elapsed_sec"),
        "cost": item.get("cost_usd"),
    }
    out.update(collect_case_stage_resources(item))
    for stage in METRIC_STAGES:
        ident = _ident_from_lane_snaps(_complete_snaps(lane_stats[stage])) or {}
        for key in LANE_SNAP_KEYS:
            out["%s_s%s" % (key, stage)] = ident.get(key)
    return out


def stage_field_names():
    names = []
    for metric in RECONSTRUCTION_METRICS:
        for stage in METRIC_STAGES:
            names.append("%s_stage%s" % (metric, stage))
    names.extend([
        "arm count mae",
        "arm count emr",
        "heading_correct",
        "heading_n",
        "heading_match",
        "arm_level_match",
        "lane_match",
        "arrow_match",
        "time",
        "cost",
    ])
    for metric in RESOURCE_METRICS:
        for stage in METRIC_STAGES:
            names.append("%s_stage%s" % (metric, stage))
    for stage in METRIC_STAGES:
        for key in LANE_SNAP_KEYS:
            names.append("%s_s%s" % (key, stage))
    return names


def _lane_error_total(row, stage):
    return sum(row.get("%s_s%s" % (key, stage)) or 0 for key in LANE_ERROR_SNAP.values())


def stage_value(metric, rows, stage):
    if metric == "arm count mae":
        return _mean_nonnull([row.get("arm count mae") for row in rows])
    if metric == "arm count emr":
        return _mean_nonnull([row.get("arm count emr") for row in rows])
    if metric == "heading emr":
        return _ratio(
            sum(row.get("heading_correct") or 0 for row in rows),
            sum(row.get("heading_n") or 0 for row in rows),
        )
    if metric == "time":
        return _mean_nonnull([row.get("time_stage%s" % stage) for row in rows])
    if metric == "cost":
        return _mean_nonnull([row.get("cost_stage%s" % stage) for row in rows])
    if metric == "lane count mae":
        return _ratio(
            sum(row.get("lane_abs_s%s" % stage) or 0 for row in rows),
            sum(row.get("lane_terms_s%s" % stage) or 0 for row in rows),
        )
    if metric == "lane count emr":
        return _ratio(
            sum(row.get("lane_correct_s%s" % stage) or 0 for row in rows),
            sum(row.get("lane_arms_s%s" % stage) or 0 for row in rows),
        )
    if metric == "matched arrow coverage":
        return _ratio(
            sum(
                (row.get("arrow_matched_tp_s%s" % stage) or 0)
                + (row.get("arrow_matched_fn_s%s" % stage) or 0)
                for row in rows
            ),
            sum(
                (row.get("arrow_tp_s%s" % stage) or 0)
                + (row.get("arrow_fn_s%s" % stage) or 0)
                for row in rows
            ),
        )
    if metric == "errors":
        return int(sum(_lane_error_total(row, stage) for row in rows))
    if metric in LANE_ERROR_RATE:
        num_key, den_key = LANE_ERROR_RATE[metric]
        return _ratio(
            sum(row.get("%s_s%s" % (num_key, stage)) or 0 for row in rows),
            sum(row.get("%s_s%s" % (den_key, stage)) or 0 for row in rows),
        )
    if metric in LANE_ERROR_SNAP:
        return _ratio(
            sum(row.get("%s_s%s" % (LANE_ERROR_SNAP[metric], stage)) or 0 for row in rows),
            sum(
                (row.get("arrow_tp_s%s" % stage) or 0)
                + (row.get("arrow_fn_s%s" % stage) or 0)
                for row in rows
            ),
        )
    if metric in (
        "arrow precision", "arrow recall", "arrow F1", "arrow accuracy",
        "matched arrow precision", "matched arrow recall", "matched arrow F1",
        "matched arrow accuracy",
    ):
        prefix = "arrow_matched" if metric.startswith("matched ") else "arrow"
        tp = sum(row.get("%s_tp_s%s" % (prefix, stage)) or 0 for row in rows)
        fp = sum(row.get("%s_fp_s%s" % (prefix, stage)) or 0 for row in rows)
        fn = sum(row.get("%s_fn_s%s" % (prefix, stage)) or 0 for row in rows)
        precision, recall, f1 = _prf(tp, fp, fn)
        correct_key = "arrow_matched_correct" if prefix == "arrow_matched" else "arrow_correct"
        accuracy = _ratio(
            sum(row.get("%s_s%s" % (correct_key, stage)) or 0 for row in rows),
            tp,
        )
        name = metric[len("matched "):] if metric.startswith("matched ") else metric
        return {
            "arrow precision": _round4(precision),
            "arrow recall": _round4(recall),
            "arrow F1": _round4(f1),
            "arrow accuracy": _round4(accuracy),
        }[name]
    return _pass_rate([row.get("%s_stage%s" % (metric, stage)) for row in rows])


def _group_rows(rows, dataset_key, *, default_method=METHOD):
    grouped = []
    order = []
    buckets = {}
    for row in rows:
        dataset = row.get(dataset_key) if dataset_key != "dataset" else row.get("dataset")
        if dataset_key == "family":
            dataset = row.get("family")
        if dataset is None:
            continue
        method = row.get("method") or default_method
        key = (dataset, method)
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    for key in order:
        grouped.append((key, buckets[key]))
    return grouped


def build_reconstruction_table(rows, *, dataset_key="dataset", method=METHOD):
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for metric in RECONSTRUCTION_METRICS:
            record = {"dataset": dataset, "method": row_method, "metric": metric}
            for stage in METRIC_STAGES:
                record["stage%s" % stage] = stage_value(metric, group, stage)
            records.append(record)
    return records


def build_arm_table(rows, *, dataset_key="dataset", method=METHOD):
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        record = {"dataset": dataset, "method": row_method}
        for metric in ARM_METRICS:
            record[metric] = stage_value(metric, group, "1")
        records.append(record)
    return records


def build_resource_table(rows, *, dataset_key="dataset", method=METHOD):
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for metric in RESOURCE_METRICS:
            record = {"dataset": dataset, "method": row_method, "metric": metric}
            for stage in METRIC_STAGES:
                value = stage_value(metric, group, stage)
                if value is None:
                    record["stage%s" % stage] = None
                elif metric == "time":
                    record["stage%s" % stage] = round(float(value), 1)
                else:
                    record["stage%s" % stage] = round(float(value), 4)
            records.append(record)
    return records


def build_lane_table(rows, *, dataset_key="dataset", method=METHOD):
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for metric in LANE_METRICS:
            record = {"dataset": dataset, "method": row_method, "metric": metric}
            for stage in METRIC_STAGES:
                record["stage%s" % stage] = stage_value(metric, group, stage)
            records.append(record)
    return records


def build_way_kind_table(rows, *, dataset_key="dataset", method=METHOD):
    """Lane-count EMR stratified by GT arm type (one-way / two-way)."""
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for arm_type in WAY_KIND_ARM_TYPES:
            record = {
                "dataset": dataset,
                "method": row_method,
                "arm type": arm_type,
                "metric": "lane count emr",
            }
            for stage in METRIC_STAGES:
                if arm_type == "one-way":
                    record["stage%s" % stage] = _ratio(
                        sum(row.get("way_one_correct_s%s" % stage) or 0 for row in group),
                        sum(row.get("way_one_n_s%s" % stage) or 0 for row in group),
                    )
                else:
                    record["stage%s" % stage] = _ratio(
                        sum(row.get("way_two_correct_s%s" % stage) or 0 for row in group),
                        sum(row.get("way_two_n_s%s" % stage) or 0 for row in group),
                    )
            records.append(record)
    return records


def pivot_way_kind_compare_table(records, *, dataset_order=None, arm_types=None):
    """Reshape way-kind rows into compare columns with an ``arm type`` field."""
    slots = {}
    seen_datasets = []
    seen_types = []
    for rec in records:
        dataset = rec.get("dataset")
        arm_type = rec.get("arm type")
        metric = rec.get("metric")
        method = rec.get("method")
        if dataset is None or arm_type is None or metric is None:
            continue
        if dataset not in seen_datasets:
            seen_datasets.append(dataset)
        if arm_type not in seen_types:
            seen_types.append(arm_type)
        key = (dataset, arm_type, metric)
        slot = slots.get(key)
        if slot is None:
            slot = {"dataset": dataset, "arm type": arm_type, "metric": metric}
            slot.update(COMPARE_EMPTY)
            slots[key] = slot
        if method == "llm_e2e":
            slot["DI"] = rec.get("stage1")
        elif method == "llm_noagent":
            slot["HI"] = rec.get("stage1")
        elif method == "llm_kg":
            slot["HAI"] = rec.get("stage1")
            slot["KHAI-1"] = rec.get("stage2-1")
            slot["KHAI-2"] = rec.get("stage2-2")
            slot["KHAI"] = rec.get("stage3")
    datasets = list(dataset_order) if dataset_order is not None else seen_datasets
    types = list(arm_types) if arm_types is not None else list(WAY_KIND_ARM_TYPES)
    ordered = []
    used = set()
    for dataset in datasets:
        for arm_type in types:
            key = (dataset, arm_type, "lane count emr")
            if key in slots:
                ordered.append(slots[key])
                used.add(key)
    for key, slot in slots.items():
        if key not in used:
            ordered.append(slot)
    return ordered


def build_error_table(rows, *, dataset_key="dataset", method=METHOD):
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for metric in LANE_ERROR_METRICS:
            record = {"dataset": dataset, "method": row_method, "metric": metric}
            for stage in METRIC_STAGES:
                record["stage%s" % stage] = stage_value(metric, group, stage)
            records.append(record)
    return records


def build_error_mix_table(rows, *, dataset_key="dataset", method=METHOD):
    """Share of each error type among remaining items (sums to 1 per column)."""
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for metric in LANE_ERROR_MIX_METRICS:
            record = {"dataset": dataset, "method": row_method, "metric": metric}
            for stage in METRIC_STAGES:
                record["stage%s" % stage] = _ratio(
                    sum(row.get("%s_s%s" % (LANE_ERROR_SNAP[metric], stage)) or 0 for row in group),
                    sum(_lane_error_total(row, stage) for row in group),
                )
            records.append(record)
    return records


def matched_arrow_stage_counts(rows, stage):
    both = sum(row.get("arrow_matched_tp_s%s" % stage) or 0 for row in rows)
    false = sum(row.get("arrow_matched_fp_s%s" % stage) or 0 for row in rows)
    miss = sum(row.get("arrow_matched_fn_s%s" % stage) or 0 for row in rows)
    correct = sum(row.get("arrow_matched_correct_s%s" % stage) or 0 for row in rows)
    return {
        "matched arms": int(sum(row.get("arrow_matched_arms_s%s" % stage) or 0 for row in rows)),
        "N_both": int(both),
        "N_false": int(false),
        "N_miss": int(miss),
        "N_both_correct": int(correct),
    }


def build_matched_arrow_count_table(rows, *, dataset_key="dataset", method=METHOD):
    records = []
    for (dataset, row_method), group in _group_rows(rows, dataset_key, default_method=method):
        for metric in MATCHED_ARROW_COUNT_METRICS:
            record = {"dataset": dataset, "method": row_method, "metric": metric}
            for stage in METRIC_STAGES:
                record["stage%s" % stage] = matched_arrow_stage_counts(group, stage)[metric]
            records.append(record)
    return records


def pivot_compare_table(records, *, metric_order=None, dataset_order=None):
    """Reshape ``[dataset, method, metric, stage*]`` into compare columns.

    Mapping:
    - DI: ``llm_e2e`` stage1
    - HI: ``llm_noagent`` stage1
    - HAI: ``llm_kg`` stage1
    - KHAI-1 / KHAI-2 / KHAI: ``llm_kg`` stage2-1 / stage2-2 / stage3
    """
    slots = {}
    seen_datasets = []
    seen_metrics = []
    for rec in records:
        dataset = rec.get("dataset")
        metric = rec.get("metric")
        method = rec.get("method")
        if dataset is None or metric is None:
            continue
        if dataset not in seen_datasets:
            seen_datasets.append(dataset)
        if metric not in seen_metrics:
            seen_metrics.append(metric)
        key = (dataset, metric)
        slot = slots.get(key)
        if slot is None:
            slot = {"dataset": dataset, "metric": metric}
            slot.update(COMPARE_EMPTY)
            slots[key] = slot
        if method == "llm_e2e":
            slot["DI"] = rec.get("stage1")
        elif method == "llm_noagent":
            slot["HI"] = rec.get("stage1")
        elif method == "llm_kg":
            slot["HAI"] = rec.get("stage1")
            slot["KHAI-1"] = rec.get("stage2-1")
            slot["KHAI-2"] = rec.get("stage2-2")
            slot["KHAI"] = rec.get("stage3")
    datasets = list(dataset_order) if dataset_order is not None else seen_datasets
    metrics = list(metric_order) if metric_order is not None else seen_metrics
    ordered = []
    used = set()
    for dataset in datasets:
        for metric in metrics:
            key = (dataset, metric)
            if key in slots:
                ordered.append(slots[key])
                used.add(key)
    for key, slot in slots.items():
        if key not in used:
            ordered.append(slot)
    return ordered


def pivot_arm_compare_table(records, *, dataset_order=None):
    """Wide arm table ``[dataset, method, ...]`` → compare columns.

    Arm geometry does not change in later kg stages, so KHAI-* copy HAI.
    """
    melted = []
    for rec in records:
        for metric in ARM_METRICS:
            value = rec.get(metric)
            melted.append({
                "dataset": rec.get("dataset"),
                "method": rec.get("method"),
                "metric": metric,
                "stage1": value,
                "stage2-1": value,
                "stage2-2": value,
                "stage3": value,
            })
    return pivot_compare_table(
        melted,
        metric_order=ARM_METRICS,
        dataset_order=dataset_order,
    )


def pivot_khai_error_mix_table(records, *, dataset_order=None):
    """One row per dataset; error types as columns; KHAI (stage3) only."""
    by_ds = {}
    seen = []
    for rec in records:
        if rec.get("method") not in (None, "llm_kg"):
            continue
        dataset = rec.get("dataset")
        metric = rec.get("metric")
        if dataset is None or metric is None:
            continue
        if dataset not in by_ds:
            by_ds[dataset] = {"dataset": dataset}
            seen.append(dataset)
        by_ds[dataset][metric] = rec.get("stage3")
    order = list(dataset_order) if dataset_order is not None else seen
    ordered = []
    used = set()
    for dataset in order:
        if dataset in by_ds:
            ordered.append(by_ds[dataset])
            used.add(dataset)
    for dataset in seen:
        if dataset not in used:
            ordered.append(by_ds[dataset])
    return ordered


def display_dataset(name):
    return DISPLAY_DATASET.get(name, name)


def display_metric(name):
    return DISPLAY_METRIC.get(name, name)


def apply_display_labels(records):
    out = []
    for rec in records:
        item = {}
        for key, value in rec.items():
            if key == "dataset":
                item["dataset"] = DISPLAY_DATASET.get(value, value)
            elif key == "metric":
                item["metric"] = DISPLAY_METRIC.get(value, value)
            elif key == "arm type":
                item["arm type"] = value
            elif key in DISPLAY_METRIC:
                item[DISPLAY_METRIC[key]] = value
            else:
                item[key] = value
        out.append(item)
    return out


def _latex_metric_cell(name):
    shown = display_metric(name)
    if isinstance(shown, str) and shown.startswith("$"):
        return shown
    return _latex_escape(shown)


def _latex_header(col):
    shown = display_metric(col)
    if isinstance(shown, str) and shown.startswith("$"):
        return r"\textbf{%s}" % shown
    text = str(shown)
    if text:
        text = text[0].upper() + text[1:]
    return r"\textbf{%s}" % _latex_escape(text)


def _latex_escape(text):
    return (
        str(text)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def canonical_metric(name):
    if name in DISPLAY_METRIC:
        return name
    return DISPLAY_METRIC_REVERSE.get(name, name)


def format_compare_value(value, metric, *, latex=False, force=None):
    if value is None:
        return "--"
    try:
        num = float(value)
    except (TypeError, ValueError):
        text = str(value)
        if latex:
            if text.startswith("$"):
                return text
            return _latex_escape(text)
        return text
    if math.isnan(num):
        return "--"
    key = canonical_metric(metric)
    if force == "count" or (force is None and key in MATCHED_ARROW_COUNT_METRICS):
        return "%d" % int(round(num))
    if force == "rate" or (force is None and key in RATE_METRICS):
        text = "%.2f" % (100.0 * num)
        return (text + r"\%") if latex else (text + "%")
    if key == "time":
        return "%.1f" % num
    if key == "cost":
        return "%.4f" % num
    if key in MAE_METRICS or "mae" in key:
        return "%.3f" % num
    return "%.4f" % num


def _latex_value(value, metric, *, force=None):
    return format_compare_value(value, metric, latex=True, force=force)


def format_compare_latex(records, columns=None, *, caption=None, label=None, force=None):
    """Booktabs ``tabular`` for compare-column records (``COMPARE_COLUMNS``)."""
    columns = list(columns or COMPARE_COLUMNS)
    text_cols = {"dataset", "metric", "arm type"}
    align = "".join("l" if col in text_cols else "c" for col in columns)
    lines = [r"\begin{table}[!htbp]", r"\centering"]
    if caption:
        lines.append(r"\caption{%s}" % caption)
    if label:
        lines.append(r"\label{%s}" % label)
    lines.append(r"\begin{tabular}{%s}" % align)
    lines.append(r"\toprule")
    lines.append(
        " & ".join(_latex_header(col) for col in columns) + r" \\"
    )
    lines.append(r"\midrule")
    prev_dataset = object()
    for i, rec in enumerate(records or []):
        dataset = rec.get("dataset")
        if i and dataset != prev_dataset:
            lines.append(r"\midrule")
        prev_dataset = dataset
        row_metric = rec.get("metric")
        cells = []
        for col in columns:
            if col in text_cols:
                raw = rec.get(col)
                if raw is None:
                    cells.append("")
                elif col == "dataset":
                    cells.append(_latex_escape(display_dataset(raw)))
                elif col == "metric":
                    cells.append(_latex_metric_cell(raw))
                elif col == "arm type":
                    cells.append(_latex_escape(raw))
                else:
                    cells.append(_latex_escape(raw))
            else:
                cells.append(_latex_value(rec.get(col), row_metric if row_metric is not None else col, force=force))
        lines.append(" & ".join(cells) + r" \\")
        lines.append(r"\addlinespace[0.5em]")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def format_stage_metrics_tables(rows, *, dataset_key="dataset"):
    recon = build_reconstruction_table(rows, dataset_key=dataset_key)
    arm = build_arm_table(rows, dataset_key=dataset_key)
    lane = build_lane_table(rows, dataset_key=dataset_key)
    way = build_way_kind_table(rows, dataset_key=dataset_key)
    errors = build_error_table(rows, dataset_key=dataset_key)
    error_mix = build_error_mix_table(rows, dataset_key=dataset_key)
    resource = build_resource_table(rows, dataset_key=dataset_key)
    parts = [
        "reconstruction",
        format_table(recon, SUMMARY_COLUMNS),
        "arm-level identification",
        format_table(arm, ["dataset", "method"] + list(ARM_METRICS)),
        "lane-level identification",
        format_table(lane, SUMMARY_COLUMNS),
        "one-way / two-way EMR",
        format_table(way, ["dataset", "method", "arm type", "metric"] + ["stage%s" % s for s in METRIC_STAGES]),
        "error rates",
        format_table(errors, SUMMARY_COLUMNS),
        "error mix",
        format_table(error_mix, SUMMARY_COLUMNS),
        "time / cost (mean per case, incremental)",
        format_table(resource, SUMMARY_COLUMNS),
    ]
    return "\n".join(parts)


def _stage_triplet(row, metric):
    return " ".join(
        "stage%s=%s" % (stage, _cell(row.get("%s_stage%s" % (metric, stage))))
        for stage in METRIC_STAGES
    )


def _fmt_num(value, digits):
    if value is None:
        return "-"
    return ("%." + str(digits) + "f") % float(value)


def format_case_resource_metrics(row):
    """Per-case time/cost lines (totals plus incremental stages)."""
    if not row:
        return ["[metrics] time  (none)", "[metrics] cost  (none)"]
    time_parts = [row.get("time_stage%s" % stage) for stage in METRIC_STAGES]
    cost_parts = [row.get("cost_stage%s" % stage) for stage in METRIC_STAGES]
    time_total = row.get("time")
    if time_total is None and any(value is not None for value in time_parts):
        time_total = sum(float(value or 0.0) for value in time_parts)
    cost_total = None
    if any(value is not None for value in cost_parts):
        cost_total = sum(float(value or 0.0) for value in cost_parts)
    elif row.get("cost") is not None:
        cost_total = row.get("cost")
    time_bits = ["total=%s" % _fmt_num(time_total, 1)]
    cost_bits = ["total=%s" % _fmt_num(cost_total, 4)]
    for stage in METRIC_STAGES:
        time_bits.append("stage%s=%s" % (stage, _fmt_num(row.get("time_stage%s" % stage), 1)))
        cost_bits.append("stage%s=%s" % (stage, _fmt_num(row.get("cost_stage%s" % stage), 4)))
    return [
        "[metrics] time  %s" % " ".join(time_bits),
        "[metrics] cost  %s" % " ".join(cost_bits),
    ]


def format_case_stage_metrics(row):
    """Compact per-case log (same names as the summary tables)."""
    if not row:
        return "[metrics] (none)"
    heading = _ratio(row.get("heading_correct"), row.get("heading_n"))
    lines = [
        "[metrics] reconstruction",
        "  gen acc       %s" % _stage_triplet(row, "gen acc"),
        "  rec acc       %s" % _stage_triplet(row, "rec acc"),
        "  rec@1 acc     %s" % _stage_triplet(row, "rec@1 acc"),
        "[metrics] arm-level identification  arm count emr=%s  heading emr=%s"
        % (
            _cell(_round4(row.get("arm count emr"))),
            _cell(heading),
        ),
    ]
    lane_bits = []
    for stage in METRIC_STAGES:
        mae = _ratio(row.get("lane_abs_s%s" % stage), row.get("lane_terms_s%s" % stage))
        emr = _ratio(row.get("lane_correct_s%s" % stage), row.get("lane_arms_s%s" % stage))
        precision, recall, _f1 = _prf(
            row.get("arrow_tp_s%s" % stage),
            row.get("arrow_fp_s%s" % stage),
            row.get("arrow_fn_s%s" % stage),
        )
        m_p, m_r, _m_f = _prf(
            row.get("arrow_matched_tp_s%s" % stage),
            row.get("arrow_matched_fp_s%s" % stage),
            row.get("arrow_matched_fn_s%s" % stage),
        )
        acc = _ratio(
            row.get("arrow_correct_s%s" % stage),
            row.get("arrow_tp_s%s" % stage),
        )
        m_acc = _ratio(
            row.get("arrow_matched_correct_s%s" % stage),
            row.get("arrow_matched_tp_s%s" % stage),
        )
        p_s, r_s, a_s = _round4(precision), _round4(recall), _round4(acc)
        mp_s, mr_s, ma_s = _round4(m_p), _round4(m_r), _round4(m_acc)
        if p_s is None and r_s is None and a_s is None:
            prf = "-"
        else:
            prf = "%s/%s/%s" % (_cell(p_s), _cell(r_s), _cell(a_s))
        if mp_s is None and mr_s is None and ma_s is None:
            matched_prf = "-"
        else:
            matched_prf = "%s/%s/%s" % (_cell(mp_s), _cell(mr_s), _cell(ma_s))
        lane_bits.append(
            "stage%s mae=%s emr=%s P/R/ACC=%s matched P/R/ACC=%s"
            % (stage, _cell(mae), _cell(emr), prf, matched_prf)
        )
    lines.append("[metrics] lane-level identification  %s" % " | ".join(lane_bits))
    lines.extend(format_case_resource_metrics(row))
    return "\n".join(lines)


def pipeline_item_from_summary(summary, *, output_dir=None):
    """Build a ``collect_local_case_metrics``-like item from ``run_one_case`` summary."""
    item = dict(summary or {})
    if output_dir is not None:
        item["output_dir"] = str(output_dir)
    elif summary and summary.get("case_dir") and not item.get("output_dir"):
        item["output_dir"] = str(Path(summary["case_dir"]).parent.parent)
    return item
