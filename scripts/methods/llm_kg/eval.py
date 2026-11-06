"""Score generated llm_kg artifacts against GT and aggregate case metrics.

The generation pipeline in ``main.py`` calls ``stage_evaluation`` per case and
``evaluate_outputs`` after a batch. Run on existing outputs without generation::

    python -m scripts.methods.llm_kg.eval

``DATASETS`` is the list used by this eval entry.
"""

import copy
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from scripts.data_reader import DATASET_ROOT
from scripts.methods.evaluation import (
    DEFAULT_HEADING_TOL_DEG,
    aggregate_metrics,
    arms_by_id,
    compute_metrics,
    evaluate_lane_level,
    lanes_at_most_one_mismatch,
    lanes_metrics_for_stage,
    match_arms_by_heading,
    run_evaluation,
)
from scripts.methods.llm_kg.inspect_metrics import write_inspect_metrics_csv
from scripts.methods.llm_kg.stage_metrics import (
    build_arm_table,
    build_lane_table,
    build_reconstruction_table,
    case_stage_row,
    format_case_stage_metrics,
    format_stage_metrics_tables,
    stage_field_names,
)
from scripts.methods.llm_utils import BaseLLM
from scripts.methods.method_utils import (
    arm_level_dir,
    dump_json,
    junction_level_dir,
    lane_level_dir,
    list_junction_artifact_roots,
    list_pipeline_summary_paths,
    load_json_if_exists,
    model_dir_name,
    parse_model_dir_name,
    refinement_dir,
    resolve_case_dir,
    resolve_model_dir,
    resolve_output_root,
    set_verbose,
    xodr_result_dir_for_model,
)

DATASETS = ["nhtsa_2022", "nhtsa_2023", "nhtsa_2024", "openaerialmap"]

# Dropped from case_metrics.csv: aliases of elapsed/cost, empty junction fields, multi-junction-only l3.
CASE_METRICS_CSV_OMIT = {
    "time",
    "cost",
    "junction_model",
    "junction_reasoning",
    "junction_level_usd",
    "refine_l3_pass",
    "arm_level_match",
    "lane_match",
    "arrow_match",
}

CASE_METRICS_CSV_FIELDS = [
    "dataset",
    "case_id",
    "completed_at",
    "arm_model",
    "arm_reasoning",
    "lane_model",
    "lane_reasoning",
    "refine_model",
    "refine_reasoning",
    "arm_level_valid",
    "arm_level_match",
    "lane_level_valid",
    "lane_level_match",
    "lane_match",
    "arrow_match",
    "refine_l1_pass",
    "refine_l1_valid",
    "refine_l1_match",
    "refine_l2_pass",
    "refine_l2_valid",
    "refine_l2_match",
    "refine_fea_pass",
    "refine_fea_valid",
    "refine_fea_match",
    "refine_fea_edits",
    "refine_fea_score",
    "refine_fea_n_edits",
    "is_valid",
    "is_final_valid",
    "is_match",
    "is_one_mismatch",
    "fail_reason",
    "elapsed_sec",
    "cost_usd",
    "arm_level_usd",
    "lane_level_usd",
    "refine_level_usd",
] + [name for name in stage_field_names() if name not in CASE_METRICS_CSV_OMIT]

CASE_METRICS_SIMPLE_CSV_FIELDS = [
    "dataset",
    "case_id",
    "completed_at",
    "model",
    "reasoning",
    "is_valid",
    "is_final_valid",
    "is_match",
    "is_one_mismatch",
    "heading_match",
    "arrow_match",
    "lane_match",
    "arm_level_valid",
    "arm_level_match",
    "lane_level_valid",
    "lane_level_match",
    "refine_l1_pass",
    "refine_l1_valid",
    "refine_l1_match",
    "refine_l2_pass",
    "refine_l2_valid",
    "refine_l2_match",
    "refine_fea_pass",
    "refine_fea_valid",
    "refine_fea_match",
    "fail_reason",
    "elapsed_sec",
    "cost_usd",
]


def _load_cost_usd(cost_path):
    """Read stage cost_usd from cost.json.

    Arm-level uses CostTracker.summary() → ``totals.cost_usd``.
    Lane/refine use _build_cost() → historically ``total.cost_usd``, now also ``totals``.
    """
    path = Path(cost_path)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ("totals", "total"):
        block = data.get(key)
        if isinstance(block, dict) and block.get("cost_usd") is not None:
            return float(block["cost_usd"])
    if data.get("cost_usd") is not None:
        return float(data["cost_usd"])
    return None


def collect_case_cost(
    case_dir,
    arm_model,
    lane_model=None,
    refine_model=None,
    *,
    arm_reasoning="none",
    lane_reasoning="none",
    refine_reasoning="none",
    junction_model=None,
    junction_reasoning="none",
):
    """Sum junction / arm / lane / refine cost_usd from saved cost.json files (even if stages failed)."""
    case_dir = Path(case_dir)
    lane_model = lane_model or arm_model
    refine_model = refine_model or lane_model
    junction_model = junction_model or arm_model
    roots = list_junction_artifact_roots(case_dir)

    def _stage_cost(root, stage_dir, model_name, reasoning):
        return _load_cost_usd(stage_dir(root, model_name, reasoning) / "cost.json")

    junction_cost = _load_cost_usd(junction_level_dir(case_dir, junction_model, junction_reasoning) / "cost.json")
    arm_cost = None
    lane_cost = None
    refine_cost = None
    for root in roots:
        arm_cost = _add_cost(arm_cost, _stage_cost(root, arm_level_dir, arm_model, arm_reasoning))
        lane_cost = _add_cost(lane_cost, _stage_cost(root, lane_level_dir, lane_model, lane_reasoning))
        refine_cost = _add_cost(refine_cost, _stage_cost(root, refinement_dir, refine_model, refine_reasoning))
    total = 0.0
    priced = False
    for value in (junction_cost, arm_cost, lane_cost, refine_cost):
        if value is not None:
            total += value
            priced = True
    return {
        "junction_level_usd": junction_cost,
        "arm_level_usd": arm_cost,
        "lane_level_usd": lane_cost,
        "refine_level_usd": refine_cost,
        "cost_usd": round(total, 6) if priced else None,
    }


def _add_cost(left, right):
    if left is None and right is None:
        return None
    return float(left or 0.0) + float(right or 0.0)


def stage_evaluation(config, dataset, case_id, case_dir):
    gt_dir = DATASET_ROOT / dataset / str(case_id)
    BaseLLM.rule("EVALUATION")
    roots = list_junction_artifact_roots(case_dir)
    collected = []
    last_error = None
    for root in roots:
        try:
            result = run_evaluation(
                gt_dir,
                case_dir=root,
                arm_model=config["arm_model"],
                lane_model=config["lane_model"],
                refine_model=config["refine_model"],
                arm_reasoning=config["arm_reasoning"],
                lane_reasoning=config["lane_reasoning"],
                refine_reasoning=config["refine_reasoning"],
                heading_tol_deg=config.get("heading_tol_deg", DEFAULT_HEADING_TOL_DEG),
            )
        except FileNotFoundError as exc:
            last_error = exc
            continue
        metrics_one = dict(result["metrics"] or {})
        fea_paths = (result.get("paths") or {}).get("layer_fea") or {}
        fea_metrics = load_json_if_exists(fea_paths.get("metrics")) if fea_paths else None
        metrics_one["layer_fea"] = _fea_eval_section(fea_metrics)
        collected.append(metrics_one)
    if not collected:
        raise last_error or FileNotFoundError(
            "no pred arms/lanes/refine artifacts under %s" % case_dir
        )
    metrics = collected[0] if len(collected) == 1 else aggregate_metrics(collected)
    BaseLLM.log(
        "eval stages: arm=%s lane=%s refine=%s fea=%s passed=%s"
        % (
            metrics.get("arm_level") is not None,
            metrics.get("lane_level") is not None,
            metrics.get("refinement") is not None,
            metrics.get("layer_fea") is not None,
            metrics.get("passed"),
        )
    )
    return metrics


COMPLETED_AT_FMT = "%Y-%m-%d %H:%M"


def format_completed_at(when=None):
    """Wall-clock completion time truncated to the minute."""
    dt = when if isinstance(when, datetime) else datetime.now()
    return dt.strftime(COMPLETED_AT_FMT)


def load_completed_at(paths):
    """Prefer ``completed_at`` in pipeline_summary.json; else newest file mtime."""
    newest = None
    newest_mtime = None
    for path in paths:
        path = Path(path)
        value = (load_json_if_exists(path) or {}).get("completed_at")
        if value:
            return str(value)
        if path.is_file():
            mtime = path.stat().st_mtime
            if newest_mtime is None or mtime > newest_mtime:
                newest = path
                newest_mtime = mtime
    if newest is None:
        return None
    return format_completed_at(datetime.fromtimestamp(newest_mtime))


def load_elapsed_sec(paths):
    """Wall-clock seconds from the first ``pipeline_summary.json`` that recorded it."""
    for path in paths:
        value = (load_json_if_exists(path) or {}).get("elapsed_sec")
        if value is not None:
            return float(value)
    return None


def _layer_entered_refine(refine_dir, layer_folder):
    layer_dir = Path(refine_dir) / layer_folder
    if not layer_dir.is_dir():
        return False
    if (layer_dir / "result.json").is_file() or (layer_dir / "conversation.json").is_file():
        return True
    return any(layer_dir.glob("round_*"))


def _layer_had_inconsistency(refine_dir, layer_folder):
    """True if this layer found an inconsistency (the repair path actually ran).

    L1 always *checks* every case. A clean first-pass (``passed`` and
    ``refine_rounds==0``) is not a real entry. L2 is the same on the
    connectivity check. ``layer_fea`` is entered only when the min-edit
    solver was invoked after L2 still failed.
    """
    layer_dir = Path(refine_dir) / layer_folder
    if layer_folder == "layer_fea":
        return (
            (layer_dir / "feasibility_solver.json").is_file()
            or (layer_dir / "result.json").is_file()
            or _layer_entered_refine(refine_dir, layer_folder)
        )
    payload = load_json_if_exists(layer_dir / "result.json")
    if payload is not None:
        if payload.get("passed") is False:
            return True
        if payload.get("validation_error"):
            return True
        if int(payload.get("refine_rounds") or 0) > 0:
            return True
        return False
    return _layer_entered_refine(refine_dir, layer_folder)


def _post_refine_layer_passed(refine_dir, layer_folder, *, arms_path, name):
    """Re-run the layer check on saved post-refine lanes (legacy artifacts without result.json)."""
    from scripts.methods.llm_kg.refinement.config_build import build_config
    from scripts.methods.llm_kg.refinement import layer1_tr_check, layer2_con_fix

    lanes_doc = load_json_if_exists(Path(refine_dir) / "lanes.json")
    arms_doc = load_json_if_exists(arms_path)
    if not lanes_doc or not arms_doc:
        return None
    try:
        config = build_config(
            arms_doc.get("arms") or [],
            lanes_doc,
            name,
            channelized_features=lanes_doc.get("channelized_features"),
            through_roads=lanes_doc.get("through_roads"),
            driving_side=lanes_doc.get("driving_side"),
        )
        lane_results = {
            int(arm["arm_id"]): arm
            for arm in (lanes_doc.get("arms") or [])
            if arm.get("arm_id") is not None
        }
        if layer_folder == "layer_1":
            return layer1_tr_check.check(
                config, lanes_doc.get("through_roads") or [], lane_results=lane_results,
            ) is None
        if layer_folder == "layer_2":
            return layer2_con_fix.check(config, name) is None
        if layer_folder == "layer_3":
            return None
    except Exception:
        return None
    return None


def _layer_check_passed(refine_dir, layer_folder, *, arms_path=None, name=None):
    """Whether this layer resolved its inconsistency. None if it did not truly enter."""
    if not _layer_had_inconsistency(refine_dir, layer_folder):
        return None
    payload = load_json_if_exists(Path(refine_dir) / layer_folder / "result.json")
    if payload is not None:
        return bool(payload.get("passed"))
    return _post_refine_layer_passed(refine_dir, layer_folder, arms_path=arms_path, name=name)


def _apply_layer_conversation(lanes_doc, conversation):
    """Patch a lanes snapshot with the last per-arm result from a layer conversation."""
    if not lanes_doc:
        return None
    doc = copy.deepcopy(lanes_doc)
    if not conversation:
        return doc
    arms = {int(arm["arm_id"]): arm for arm in (doc.get("arms") or []) if arm.get("arm_id") is not None}
    round_keys = sorted(conversation, key=lambda key: int(key) if str(key).isdigit() else 0)
    for round_key in round_keys:
        round_doc = conversation.get(round_key) or {}
        if not isinstance(round_doc, dict):
            continue
        for arm_key, turn in round_doc.items():
            if not isinstance(turn, dict):
                continue
            result = turn.get("result")
            if not isinstance(result, dict) or not result.get("cross_section"):
                continue
            arm_id = result.get("arm_id")
            aid = int(arm_id if arm_id is not None else arm_key)
            if aid not in arms:
                continue
            merged = dict(arms[aid])
            merged.update(result)
            arms[aid] = merged
    ordered = []
    for arm in (lanes_doc.get("arms") or []):
        aid = arm.get("arm_id")
        if aid is None:
            continue
        ordered.append(arms[int(aid)])
    doc["arms"] = ordered
    return doc


def _layer_lanes_snapshot(refine_dir, layer_folder, base_lanes=None):
    """Saved lanes for this layer, or a conversation patch on ``base_lanes``."""
    saved = load_json_if_exists(Path(refine_dir) / layer_folder / "lanes.json")
    if saved and saved.get("arms"):
        return saved
    conversation = load_json_if_exists(Path(refine_dir) / layer_folder / "conversation.json")
    if conversation and base_lanes:
        patched = _apply_layer_conversation(base_lanes, conversation)
        if patched and json.dumps(patched.get("arms"), sort_keys=True, default=str) != json.dumps(
            base_lanes.get("arms"), sort_keys=True, default=str
        ):
            return patched
    return None


def _lanes_connection_valid(arms_doc, lanes_doc, name):
    """True if fill/connections can be solved from this lanes snapshot."""
    if not lanes_doc or not (lanes_doc.get("arms") or []):
        return None
    if not arms_doc:
        return False
    from scripts.methods.llm_kg.refinement.config_build import build_config
    from scripts.methods.llm_kg.refinement.fill_solver import try_solve

    try:
        config = build_config(
            arms_doc.get("arms") or [],
            lanes_doc,
            name,
            channelized_features=lanes_doc.get("channelized_features"),
            through_roads=lanes_doc.get("through_roads"),
            driving_side=lanes_doc.get("driving_side"),
        )
        _filled, _connections, exc = try_solve(config, name=name, diagnose=False)
        return exc is None
    except Exception:
        return False


def _lanes_match_gt(gt_lanes, pred_lanes, arm_mapping=None):
    """True if pred lane counts/arrows fully match GT (same as evaluation lane_level.passed)."""
    if not gt_lanes or not pred_lanes:
        return None
    try:
        return bool(evaluate_lane_level(gt_lanes, pred_lanes, arm_mapping=arm_mapping).get("passed"))
    except Exception:
        return False


def _same_lanes_snapshot(left, right):
    if not left or not right:
        return False
    if left is right:
        return True
    return json.dumps(left.get("arms"), sort_keys=True, default=str) == json.dumps(
        right.get("arms"), sort_keys=True, default=str
    )


def _stage_passed_flag(metrics, stage):
    section = (metrics or {}).get(stage)
    if not section:
        return None
    if "passed" not in section:
        return None
    return bool(section.get("passed"))


_FAIL_REASON_MAX = 2000
_UNSET = object()


def _is_connections_wrapper(error):
    return "did not produce valid connections" in str(error or "")


def _format_validation_error(payload, *, prefix=None):
    """Compact one-line text from a validation_error dict for CSV."""
    if not payload:
        return None
    title = " ".join(str(payload.get("title") or "").split())
    parts = []
    seen = set()
    for issue in payload.get("issues") or []:
        constraint = issue.get("constraint") or ""
        message = " ".join(str(issue.get("message") or "").split()).rstrip(".")
        loc = issue.get("location") or {}
        arm = loc.get("arm")
        bit = message or constraint
        if not bit:
            continue
        if arm is not None:
            bit = "arm %s: %s" % (arm, bit)
        if bit in seen:
            continue
        seen.add(bit)
        parts.append(bit)
    body = "; ".join(parts)
    text = " | ".join(item for item in (title, body) if item) or None
    if not text:
        return None
    if prefix:
        text = "%s: %s" % (prefix, text)
    if len(text) > _FAIL_REASON_MAX:
        text = text[: _FAIL_REASON_MAX - 3] + "..."
    return text


def _case_fail_reason(*, refine_dir, xodr_dir, l2_payload=None, pipeline_error=_UNSET, failed_stage=_UNSET, case_root=None, artifact_root=None):
    """Exception text (timeout/connection/...) or L2/xodr validation failure."""
    if pipeline_error is _UNSET:
        refine_dir = Path(refine_dir)
        if artifact_root is None:
            artifact_root = refine_dir.parent.parent
        candidates = list_pipeline_summary_paths(
            artifact_root, refine_dir=refine_dir, case_root=case_root,
        )
        summary = {}
        for path in candidates:
            loaded = load_json_if_exists(path) or {}
            if loaded.get("error") or loaded.get("failed_stage") or loaded.get("fail_reason"):
                summary = loaded
                break
            if loaded and not summary:
                summary = loaded
        pipeline_error = summary.get("error")
        if failed_stage is _UNSET:
            failed_stage = summary.get("failed_stage")
    if pipeline_error is _UNSET:
        pipeline_error = None
    if failed_stage is _UNSET:
        failed_stage = None
    exception_text = " ".join(str(pipeline_error).split()) if pipeline_error else None
    if exception_text and failed_stage and not str(exception_text).startswith("%s:" % failed_stage):
        exception_text = "%s: %s" % (failed_stage, exception_text)

    l2_text = None
    if l2_payload is not None and not l2_payload.get("passed"):
        l2_text = _format_validation_error(l2_payload.get("validation_error"), prefix="l2")
    xodr_text = _format_validation_error(
        load_json_if_exists(Path(xodr_dir) / "validation_error.json"),
        prefix="connections",
    )

    if exception_text and not _is_connections_wrapper(exception_text):
        return exception_text
    if l2_text:
        return l2_text
    if xodr_text:
        return xodr_text
    return exception_text or None


_VALIDATION_FLAGS = (
    "arm_level_valid",
    "lane_level_valid",
    "lane_level_match",
    "refine_l1_pass",
    "refine_l1_valid",
    "refine_l1_match",
    "refine_l2_pass",
    "refine_l2_valid",
    "refine_l2_match",
    "refine_fea_pass",
    "refine_fea_valid",
    "refine_fea_match",
    "refine_l3_pass",
    "is_valid",
    "is_final_valid",
    "is_match",
    "is_one_mismatch",
)


def _all_true_flag(values):
    """AND over junctions: False if any failed, True if all passed, else None."""
    if any(value is False for value in values):
        return False
    if all(value is True for value in values):
        return True
    return None


def _case_root_and_name(artifact_root):
    """Map a per-junction work dir back to the case root and a display name."""
    artifact_root = Path(artifact_root)
    if artifact_root.parent.name == "junctions":
        case_dir = artifact_root.parent.parent
        return case_dir, "%s/%s/j%s" % (case_dir.parent.name, case_dir.name, artifact_root.name)
    return artifact_root, "%s/%s" % (artifact_root.parent.name, artifact_root.name)


def _merge_validation_statuses(statuses):
    if not statuses:
        merged = {key: None for key in _VALIDATION_FLAGS}
        merged["fail_reason"] = None
        merged.update(_summarize_feasibility_repair(None))
        return merged
    if len(statuses) == 1:
        return statuses[0]
    merged = {
        key: _all_true_flag([status.get(key) for status in statuses])
        for key in _VALIDATION_FLAGS
    }
    reasons = [status.get("fail_reason") for status in statuses if status.get("fail_reason")]
    merged["fail_reason"] = "; ".join(reasons) if reasons else None
    merged.update(_merge_fea_repair_fields(statuses))
    return merged


def _summarize_feasibility_repair(solver_payload):
    """CSV-facing summary of the chosen layer_fea min-edit (None fields if solver did not run)."""
    empty = {
        "refine_fea_edits": None,
        "refine_fea_score": None,
        "refine_fea_n_edits": None,
    }
    if not solver_payload:
        return empty
    from scripts.methods.llm_kg.refinement.feasibility_solver import format_repair_edits

    edits = solver_payload.get("edits") or []
    lines = format_repair_edits(edits)
    if lines:
        text = " | ".join(lines)
    elif solver_payload.get("ok"):
        text = ""
    else:
        text = solver_payload.get("note") or ""
    return {
        "refine_fea_edits": text,
        "refine_fea_score": solver_payload.get("score"),
        "refine_fea_n_edits": len(edits),
    }


def _join_status_text(statuses, key):
    values = [status.get(key) for status in statuses]
    if all(value is None for value in values):
        return None
    if len(statuses) == 1:
        return values[0]
    parts = []
    for status in statuses:
        value = status.get(key)
        if value is None or value == "":
            continue
        label = status.get("_junction_label")
        text = str(value)
        parts.append("%s: %s" % (label, text) if label else text)
    if parts:
        return " || ".join(parts)
    return "" if any(value == "" for value in values) else None


def _sum_status_num(statuses, key):
    nums = [status.get(key) for status in statuses if status.get(key) is not None]
    if not nums:
        return None
    if len(statuses) == 1:
        return nums[0]
    return sum(nums)


def _merge_fea_repair_fields(statuses):
    return {
        "refine_fea_edits": _join_status_text(statuses, "refine_fea_edits"),
        "refine_fea_score": _sum_status_num(statuses, "refine_fea_score"),
        "refine_fea_n_edits": _sum_status_num(statuses, "refine_fea_n_edits"),
    }


def case_validation_status(
    case_dir,
    arm_model,
    lane_model=None,
    refine_model=None,
    *,
    arm_reasoning="none",
    lane_reasoning="none",
    refine_reasoning="none",
    lane_metrics=None,
    refine_metrics=None,
    pipeline_error=_UNSET,
    failed_stage=_UNSET,
):
    """Validation and GT-match flags. Multi-junction cases combine per-junction statuses."""
    case_dir = Path(case_dir)
    roots = list_junction_artifact_roots(case_dir)
    if len(roots) == 1:
        return _case_validation_status_at(
            roots[0],
            arm_model,
            lane_model,
            refine_model,
            arm_reasoning=arm_reasoning,
            lane_reasoning=lane_reasoning,
            refine_reasoning=refine_reasoning,
            lane_metrics=lane_metrics,
            refine_metrics=refine_metrics,
            pipeline_error=pipeline_error,
            failed_stage=failed_stage,
        )
    statuses = [
        _case_validation_status_at(
            root,
            arm_model,
            lane_model,
            refine_model,
            arm_reasoning=arm_reasoning,
            lane_reasoning=lane_reasoning,
            refine_reasoning=refine_reasoning,
            pipeline_error=pipeline_error if index == 0 else _UNSET,
            failed_stage=failed_stage if index == 0 else _UNSET,
        )
        for index, root in enumerate(roots)
    ]
    merged = _merge_validation_statuses(statuses)
    if pipeline_error is not _UNSET and pipeline_error:
        merged["fail_reason"] = _case_fail_reason(
            refine_dir=refinement_dir(case_dir, refine_model or arm_model, refine_reasoning),
            xodr_dir=xodr_result_dir_for_model(
                case_dir, refine_model or lane_model or arm_model, refine_reasoning,
            ),
            pipeline_error=pipeline_error,
            failed_stage=failed_stage,
            case_root=case_dir,
            artifact_root=roots[0] if roots else case_dir,
        ) or merged.get("fail_reason")
    return merged


def _case_validation_status_at(
    case_dir,
    arm_model,
    lane_model=None,
    refine_model=None,
    *,
    arm_reasoning="none",
    lane_reasoning="none",
    refine_reasoning="none",
    lane_metrics=None,
    refine_metrics=None,
    pipeline_error=_UNSET,
    failed_stage=_UNSET,
):
    """Validation and GT-match flags for one artifact root."""
    case_dir = Path(case_dir)
    lane_model = lane_model or arm_model
    refine_model = refine_model or lane_model
    refine_dir = refinement_dir(case_dir, refine_model, refine_reasoning)
    lane_dir = lane_level_dir(case_dir, lane_model, lane_reasoning)
    arms_path = arm_level_dir(case_dir, arm_model, arm_reasoning) / "arms.json"
    xodr_dir = xodr_result_dir_for_model(case_dir, refine_model, refine_reasoning)
    case_root, name = _case_root_and_name(case_dir)
    gt_dir = DATASET_ROOT / case_root.parent.name / case_root.name
    gt_lanes = load_json_if_exists(gt_dir / "lanes.json")
    gt_arms = load_json_if_exists(gt_dir / "arms.json")
    arms_doc = load_json_if_exists(arms_path)
    lane_lanes = load_json_if_exists(lane_dir / "lanes.json")
    refine_lanes = load_json_if_exists(refine_dir / "lanes.json")
    l2_payload = load_json_if_exists(refine_dir / "layer_2" / "result.json")
    fea_payload = load_json_if_exists(refine_dir / "layer_fea" / "result.json")
    arm_mapping = None
    if gt_arms and arms_doc:
        try:
            arm_mapping = match_arms_by_heading(arms_by_id(gt_arms), arms_by_id(arms_doc))
        except Exception:
            arm_mapping = None

    l1_lanes = _layer_lanes_snapshot(refine_dir, "layer_1", lane_lanes)
    l2_lanes = _layer_lanes_snapshot(refine_dir, "layer_2", l1_lanes)
    fea_lanes = _layer_lanes_snapshot(refine_dir, "layer_fea", l2_lanes)
    final_lanes = refine_lanes
    connections_exist = (xodr_dir / "connections.json").is_file()
    valid_cache = {}

    def _valid(lanes_doc):
        if not lanes_doc:
            return None
        arms_key = json.dumps(lanes_doc.get("arms"), sort_keys=True, default=str)
        if arms_key in valid_cache:
            return valid_cache[arms_key]
        if (
            connections_exist
            and final_lanes is not None
            and arms_key == json.dumps(final_lanes.get("arms"), sort_keys=True, default=str)
        ):
            valid_cache[arms_key] = True
            return True
        ok = _lanes_connection_valid(arms_doc, lanes_doc, name)
        valid_cache[arms_key] = ok
        return ok

    def _match(pred_lanes):
        return _lanes_match_gt(gt_lanes, pred_lanes, arm_mapping=arm_mapping)

    def _reuse_or_match(pred_lanes):
        if pred_lanes is None:
            return None
        if _same_lanes_snapshot(pred_lanes, lane_lanes) and lane_level_match is not None:
            return lane_level_match
        if _same_lanes_snapshot(pred_lanes, final_lanes) and is_match is not None:
            return is_match
        return _match(pred_lanes)

    lane_level_match = _stage_passed_flag(lane_metrics, "lane_level")
    if lane_level_match is None:
        lane_level_match = _match(lane_lanes)

    is_match = _stage_passed_flag(refine_metrics, "refinement")
    if is_match is None:
        is_match = _stage_passed_flag(lane_metrics, "refinement")
    if is_match is None:
        is_match = lane_level_match if _same_lanes_snapshot(final_lanes, lane_lanes) else _match(final_lanes)
    if is_match:
        is_one_mismatch = True
    else:
        is_one_mismatch = lanes_at_most_one_mismatch(gt_lanes, final_lanes, arm_mapping=arm_mapping)

    refine_l1_match = _reuse_or_match(l1_lanes)
    refine_l2_match = _reuse_or_match(l2_lanes)
    refine_fea_match = _reuse_or_match(fea_lanes)

    is_final_valid = connections_exist if final_lanes is not None else None
    if is_final_valid is False and final_lanes is not None:
        is_final_valid = _valid(final_lanes)
    elif is_final_valid is None and final_lanes is not None:
        is_final_valid = _valid(final_lanes)

    solver_payload = (
        load_json_if_exists(refine_dir / "layer_fea" / "feasibility_solver.json")
        or load_json_if_exists(refine_dir / "layer_2" / "feasibility_solver.json")
    )
    if solver_payload is not None or fea_payload is not None:
        pre_solver = l2_lanes if l2_lanes is not None else lane_lanes
        is_valid = _valid(pre_solver) if pre_solver is not None else None
    else:
        is_valid = is_final_valid

    fea_repair = _summarize_feasibility_repair(solver_payload)
    junction_label = case_dir.name if case_dir.parent.name == "junctions" else None

    refine_l1_pass = _layer_check_passed(refine_dir, "layer_1", arms_path=arms_path, name=name)
    refine_l1_valid = _valid(l1_lanes)
    refine_l2_pass = _layer_check_passed(refine_dir, "layer_2", arms_path=arms_path, name=name)
    refine_l2_valid = _valid(l2_lanes)
    refine_fea_pass = _layer_check_passed(refine_dir, "layer_fea", arms_path=arms_path, name=name)
    refine_fea_valid = _valid(fea_lanes) if fea_lanes is not None or fea_payload is not None else None
    refine_l3_pass = _layer_check_passed(refine_dir, "layer_3", arms_path=arms_path, name=name)

    return {
        "arm_level_valid": True if arms_path.is_file() else None,
        "lane_level_valid": _valid(lane_lanes),
        "lane_level_match": lane_level_match,
        "refine_l1_pass": refine_l1_pass,
        "refine_l1_valid": refine_l1_valid,
        "refine_l1_match": refine_l1_match,
        "refine_l2_pass": refine_l2_pass,
        "refine_l2_valid": refine_l2_valid,
        "refine_l2_match": refine_l2_match,
        "refine_fea_pass": refine_fea_pass,
        "refine_fea_valid": refine_fea_valid,
        "refine_fea_match": refine_fea_match,
        "refine_l3_pass": refine_l3_pass,
        "is_valid": is_valid,
        "is_final_valid": is_final_valid,
        "is_match": is_match,
        "is_one_mismatch": is_one_mismatch,
        "fail_reason": _case_fail_reason(
            refine_dir=refine_dir,
            xodr_dir=xodr_dir,
            l2_payload=l2_payload,
            pipeline_error=pipeline_error,
            failed_stage=failed_stage,
            case_root=case_root,
            artifact_root=case_dir,
        ),
        "_junction_label": junction_label,
        **fea_repair,
    }


def _pass_rate_stats(flags, *, ignore_none=True):
    if ignore_none:
        flags = [flag for flag in flags if flag is not None]
    n_total = len(flags)
    n_passed = sum(1 for flag in flags if flag)
    if n_total == 0:
        rate = None
    else:
        rate = round(n_passed / float(n_total), 6)
    return {"n_passed": n_passed, "n_total": n_total, "pass_rate": rate}


def _case_lane_metric_pass_stats(items, stage, metric_key):
    """Case-level pass rate: metric == 1.0 on the chosen stage snapshot."""
    flags = []
    for item in items:
        section = lanes_metrics_for_stage(item.get("metrics") or {}, stage)
        if not section:
            continue
        value = section.get(metric_key)
        if value is not None:
            flags.append(float(value) >= 1.0 - 1e-9)
    return _pass_rate_stats(flags)


def load_local_metrics(model_dir):
    """Load case metrics and optional full report from local ``evaluation/``."""
    model_dir = Path(model_dir)
    eval_dir = model_dir / "evaluation"
    metrics_path = eval_dir / "metrics.json"
    full_path = eval_dir / "full.json"
    report = None
    if full_path.is_file():
        report = json.loads(full_path.read_text(encoding="utf-8"))

    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return metrics, metrics_path, report

    if report is None:
        raise FileNotFoundError(
            "local evaluation result not found under %s (need metrics.json or full.json)" % eval_dir
        )
    metrics = compute_metrics(report)
    dump_json(metrics_path, metrics)
    return metrics, metrics_path, report


def _fea_eval_section(fea_metrics):
    """layer_fea/evaluation stores the snapshot under the refinement key."""
    if not fea_metrics:
        return None
    if fea_metrics.get("layer_fea") is not None:
        return fea_metrics.get("layer_fea")
    return fea_metrics.get("refinement")


def _passed_from_stages(sections, fallback=True):
    passed = True
    any_stage = False
    for stage in ("arm_level", "lane_level", "refinement"):
        section = sections.get(stage)
        if section is None:
            continue
        any_stage = True
        passed = passed and bool(section.get("passed"))
    return passed if any_stage else bool(fallback)


def _merge_stage_sections(lane_doc=None, refine_doc=None, fea_doc=None, *, include_fea=False):
    """Merge lane / refine / (optional) fea evaluation dicts into one case dict."""
    if lane_doc is None and refine_doc is None and not include_fea:
        return None
    lane_doc = lane_doc or {}
    refine_doc = refine_doc or {}
    merged = {
        "arm_level": lane_doc.get("arm_level"),
        "lane_level": lane_doc.get("lane_level"),
        "refinement": refine_doc.get("refinement")
        if refine_doc.get("refinement") is not None
        else lane_doc.get("refinement"),
    }
    if include_fea:
        merged["layer_fea"] = _fea_eval_section(fea_doc)
    merged["passed"] = _passed_from_stages(
        merged, fallback=lane_doc.get("passed", refine_doc.get("passed", True)),
    )
    return merged


def _merge_stage_metrics(lane_metrics=None, refine_metrics=None, fea_metrics=None):
    """Merge lane-level, refinement, and feasibility evaluation metrics into one case dict."""
    return _merge_stage_sections(
        lane_metrics, refine_metrics, fea_metrics, include_fea=True,
    )


def _merge_stage_reports(lane_report=None, refine_report=None):
    """Merge lane-level and refinement full reports into one case dict."""
    return _merge_stage_sections(lane_report, refine_report)


def _stage_spec(run_models, key, fallback_key=None):
    block = (run_models or {}).get(key)
    if not isinstance(block, dict) and fallback_key:
        block = (run_models or {}).get(fallback_key)
    if not isinstance(block, dict):
        block = {}
    return block.get("model"), (block.get("reasoning") or "none")


def _run_models_from_fields(
    junction_model, junction_reasoning, arm_model, arm_reasoning,
    lane_model, lane_reasoning, refine_model, refine_reasoning,
):
    return {
        "junction-level": {"model": junction_model, "reasoning": junction_reasoning or "none"},
        "arm-level": {"model": arm_model, "reasoning": arm_reasoning or "none"},
        "lane-level": {"model": lane_model, "reasoning": lane_reasoning or "none"},
        "refinement": {"model": refine_model, "reasoning": refine_reasoning or "none"},
    }


def _run_models_identity(run_models):
    junction_model, junction_reasoning = _stage_spec(run_models, "junction-level", "arm-level")
    arm_model, arm_reasoning = _stage_spec(run_models, "arm-level")
    lane_model, lane_reasoning = _stage_spec(run_models, "lane-level", "arm-level")
    refine_model, refine_reasoning = _stage_spec(run_models, "refinement", "lane-level")
    if refine_model is None:
        refine_model, refine_reasoning = _stage_spec(run_models, "refinement", "arm-level")
    return (
        junction_model or arm_model, junction_reasoning,
        arm_model, arm_reasoning,
        lane_model or arm_model, lane_reasoning,
        refine_model or lane_model or arm_model, refine_reasoning,
    )


def _run_models_dir_key(run_models):
    ident = _run_models_identity(run_models)
    if not ident[2] and not ident[4]:
        return None
    return tuple(model_dir_name(ident[i], ident[i + 1]) for i in range(0, 8, 2))


def _run_models_from_summary(data):
    if not isinstance(data, dict) or not data.get("arm_model"):
        return None
    arm_model = data.get("arm_model")
    arm_reasoning = data.get("arm_reasoning") or "none"
    lane_model = data.get("lane_model") or arm_model
    lane_reasoning = data.get("lane_reasoning") or arm_reasoning
    refine_model = data.get("refine_model") or lane_model
    refine_reasoning = data.get("refine_reasoning") or lane_reasoning
    junction_model = data.get("junction_model") or arm_model
    junction_reasoning = data.get("junction_reasoning") or arm_reasoning
    return _run_models_from_fields(
        junction_model, junction_reasoning, arm_model, arm_reasoning,
        lane_model, lane_reasoning, refine_model, refine_reasoning,
    )


def _summary_matches_run(data, run_models):
    parsed = _run_models_from_summary(data)
    if parsed is None:
        return False
    left, right = _run_models_dir_key(parsed), _run_models_dir_key(run_models)
    return left is not None and left == right


def discover_case_run_models(case_dir, extra_run_models=None):
    """Unique pipeline runs on disk, plus ``extra_run_models`` (current MODELS).

    Directory names win over the current MODELS list for labeling: a GPT artifact
    folder is not rewritten as whatever ``MODELS`` happens to be now.
    """
    ordered = []
    seen = set()

    def add(run_models):
        key = _run_models_dir_key(run_models)
        if key is None or key in seen:
            return
        seen.add(key)
        ordered.append(run_models)

    case_dir = Path(case_dir)
    for root in list_junction_artifact_roots(case_dir):
        for path in list_pipeline_summary_paths(root, case_root=case_dir):
            add(_run_models_from_summary(load_json_if_exists(path)))
        for stage in ("arm_level", "lane_level", "refinement"):
            parent = Path(root) / stage
            if not parent.is_dir():
                continue
            for child in parent.iterdir():
                if not child.is_dir() or child.name.startswith("."):
                    continue
                model, reasoning = parse_model_dir_name(child.name)
                if not model:
                    continue
                add(_run_models_from_fields(
                    model, reasoning, model, reasoning, model, reasoning, model, reasoning,
                ))
    for run_models in extra_run_models or []:
        add(run_models)
    return ordered


def _eval_slug(value):
    return str(value or "").replace("/", "_").replace("-", "_").replace(".", "_")


def run_models_match_eval(run_models, model=None, reasoning=None):
    """True if a MODELS / discovered run matches the eval model filter."""
    if not model and not reasoning:
        return True
    names, reasonings = [], []
    if isinstance(run_models, dict):
        if run_models.get("model"):
            names.append(run_models.get("model"))
            reasonings.append(run_models.get("reasoning") or "none")
        for key in ("e2e", "junction-level", "arm-level", "lane-level", "refinement"):
            block = run_models.get(key)
            if isinstance(block, dict) and block.get("model"):
                names.append(block.get("model"))
                reasonings.append(block.get("reasoning") or "none")
    if model and not any(_eval_slug(name) == _eval_slug(model) for name in names if name):
        return False
    if reasoning and not any(
        _eval_slug(value or "none") == _eval_slug(reasoning) for value in reasonings
    ):
        return False
    return True


def item_matches_eval_model(item, model=None, reasoning=None):
    """True if a collected case-metrics item belongs to ``model`` / ``reasoning``."""
    if not model and not reasoning:
        return True
    names = [
        item.get(key)
        for key in ("junction_model", "arm_model", "lane_model", "refine_model", "model")
        if item.get(key)
    ]
    if model and not any(_eval_slug(name) == _eval_slug(model) for name in names):
        return False
    if reasoning:
        values = [
            item.get(key) or "none"
            for key in (
                "junction_reasoning", "arm_reasoning", "lane_reasoning",
                "refine_reasoning", "reasoning",
            )
        ]
        if not any(_eval_slug(value) == _eval_slug(reasoning) for value in values):
            return False
    return True


def collect_local_case_metrics(
    output_dir, dataset, case_ids, run_models_list, *, quiet=False, model=None, reasoning=None,
):
    """Read validation status and evaluation metrics for each case.

    Includes model runs found on disk as well as ``run_models_list``.
    ``model`` / ``reasoning`` keep a single run. A matching ``pipeline_summary.json``
    supplies ``fail_reason`` when evaluation files are absent.
    """
    items = []
    for case_id in case_ids:
        case_dir = resolve_case_dir(output_dir, dataset, case_id)
        for run_models in discover_case_run_models(case_dir, extra_run_models=run_models_list):
            if not run_models_match_eval(run_models, model=model, reasoning=reasoning):
                continue
            arm_model = run_models["arm-level"]["model"]
            lane_model = run_models["lane-level"]["model"]
            refine_model = run_models["refinement"]["model"]
            junction_cfg = run_models.get("junction-level") or run_models["arm-level"]
            junction_model = junction_cfg["model"]
            junction_reasoning = junction_cfg.get("reasoning") or "none"
            arm_reasoning = run_models["arm-level"].get("reasoning") or "none"
            lane_reasoning = run_models["lane-level"].get("reasoning") or "none"
            refine_reasoning = run_models["refinement"].get("reasoning") or "none"
            roots = list_junction_artifact_roots(case_dir)
            merged_parts = []
            report_parts = []
            metrics_path = None
            model_dir = None
            for root in roots:
                root_model_dir = lane_level_dir(root, lane_model, lane_reasoning)
                root_refine_dir = refinement_dir(root, refine_model, refine_reasoning)
                has_lane_eval = (
                    (root_model_dir / "evaluation" / "metrics.json").is_file()
                    or (root_model_dir / "evaluation" / "full.json").is_file()
                )
                has_refine_eval = (
                    (root_refine_dir / "evaluation" / "metrics.json").is_file()
                    or (root_refine_dir / "evaluation" / "full.json").is_file()
                )
                if not has_lane_eval and not has_refine_eval:
                    continue
                lane_metrics = lane_report = refine_metrics = refine_report = None
                if has_lane_eval:
                    lane_metrics, lane_metrics_path, lane_report = load_local_metrics(root_model_dir)
                    if metrics_path is None:
                        metrics_path = lane_metrics_path
                    if model_dir is None:
                        model_dir = root_model_dir
                if has_refine_eval:
                    refine_metrics, refine_metrics_path, refine_report = load_local_metrics(root_refine_dir)
                    if metrics_path is None:
                        metrics_path = refine_metrics_path
                    if model_dir is None:
                        model_dir = root_model_dir
                fea_root = root_refine_dir / "layer_fea"
                has_fea_eval = (
                    (fea_root / "evaluation" / "metrics.json").is_file()
                    or (fea_root / "evaluation" / "full.json").is_file()
                )
                fea_metrics = None
                if has_fea_eval:
                    fea_metrics, _fea_metrics_path, _fea_report = load_local_metrics(fea_root)
                merged_parts.append(_merge_stage_metrics(lane_metrics, refine_metrics, fea_metrics))
                report_parts.append(_merge_stage_reports(lane_report, refine_report))
            summary_paths = []
            for root in roots:
                for path in list_pipeline_summary_paths(
                    root,
                    refine_dir=refinement_dir(root, refine_model, refine_reasoning),
                    case_root=case_dir,
                ):
                    if _summary_matches_run(load_json_if_exists(path), run_models):
                        summary_paths.append(path)
            if not merged_parts and not summary_paths:
                continue
            stamp_paths = list(summary_paths)
            if metrics_path is not None:
                stamp_paths.append(Path(metrics_path))
            completed_at = load_completed_at(stamp_paths)
            elapsed_sec = load_elapsed_sec(summary_paths)
            cost = collect_case_cost(
                case_dir,
                arm_model,
                lane_model,
                refine_model,
                arm_reasoning=arm_reasoning,
                lane_reasoning=lane_reasoning,
                refine_reasoning=refine_reasoning,
                junction_model=junction_model,
                junction_reasoning=junction_reasoning,
            )
            if merged_parts:
                metrics = merged_parts[0] if len(merged_parts) == 1 else aggregate_metrics(merged_parts)
                report = report_parts[0] if len(report_parts) == 1 else None
            else:
                metrics = None
                report = None
            if model_dir is None:
                model_dir = resolve_model_dir(output_dir, dataset, case_id, lane_model, lane_reasoning)
            validation = case_validation_status(
                case_dir,
                arm_model,
                lane_model,
                refine_model,
                arm_reasoning=arm_reasoning,
                lane_reasoning=lane_reasoning,
                refine_reasoning=refine_reasoning,
                lane_metrics=metrics if len(merged_parts) == 1 else None,
                refine_metrics=metrics if len(merged_parts) == 1 else None,
            )
            items.append({
                "dataset": dataset,
                "case_id": case_id,
                "output_dir": str(resolve_output_root(output_dir)),
                "completed_at": completed_at,
                "elapsed_sec": elapsed_sec,
                "model": lane_model,
                "junction_model": junction_model,
                "junction_reasoning": junction_reasoning,
                "arm_model": arm_model,
                "arm_reasoning": arm_reasoning,
                "lane_model": lane_model,
                "lane_reasoning": lane_reasoning,
                "refine_model": refine_model,
                "refine_reasoning": refine_reasoning,
                "model_dir": str(model_dir),
                "metrics_path": str(metrics_path) if metrics_path is not None else None,
                "cost_usd": cost["cost_usd"],
                "junction_level_usd": cost["junction_level_usd"],
                "arm_level_usd": cost["arm_level_usd"],
                "lane_level_usd": cost["lane_level_usd"],
                "refine_level_usd": cost["refine_level_usd"],
                "arm_level_valid": validation["arm_level_valid"],
                "lane_level_valid": validation["lane_level_valid"],
                "lane_level_match": validation["lane_level_match"],
                "refine_l1_pass": validation["refine_l1_pass"],
                "refine_l1_valid": validation["refine_l1_valid"],
                "refine_l1_match": validation["refine_l1_match"],
                "refine_l2_pass": validation["refine_l2_pass"],
                "refine_l2_valid": validation["refine_l2_valid"],
                "refine_l2_match": validation["refine_l2_match"],
                "refine_fea_pass": validation.get("refine_fea_pass"),
                "refine_fea_valid": validation.get("refine_fea_valid"),
                "refine_fea_match": validation.get("refine_fea_match"),
                "refine_fea_edits": validation.get("refine_fea_edits"),
                "refine_fea_score": validation.get("refine_fea_score"),
                "refine_fea_n_edits": validation.get("refine_fea_n_edits"),
                "refine_l3_pass": validation.get("refine_l3_pass"),
                "is_valid": validation["is_valid"],
                "is_final_valid": validation.get("is_final_valid"),
                "is_match": validation["is_match"],
                "is_one_mismatch": validation.get("is_one_mismatch"),
                "fail_reason": validation.get("fail_reason"),
                "metrics": metrics,
                "report": report,
            })
    return items


def aggregate_metrics_by_model(case_metrics_items):
    """Group local case metrics by model and compute per-model aggregates."""
    grouped = defaultdict(list)
    for item in case_metrics_items:
        grouped[item["model"]].append(item)

    by_model = {}
    for model_name in sorted(grouped):
        items = grouped[model_name]
        cost_values = [item["cost_usd"] for item in items if item.get("cost_usd") is not None]
        metrics = aggregate_metrics([item.get("metrics") for item in items if item.get("metrics")])
        arm_stats = _pass_rate_stats([item.get("arm_level_valid") for item in items])
        lane_valid_stats = _pass_rate_stats([item.get("lane_level_valid") for item in items])
        lane_match_stats = _pass_rate_stats([item.get("lane_level_match") for item in items])
        refine_l1_pass_stats = _pass_rate_stats([item.get("refine_l1_pass") for item in items])
        refine_l1_valid_stats = _pass_rate_stats([item.get("refine_l1_valid") for item in items])
        refine_l1_match_stats = _pass_rate_stats([item.get("refine_l1_match") for item in items])
        refine_l2_pass_stats = _pass_rate_stats([item.get("refine_l2_pass") for item in items])
        refine_l2_valid_stats = _pass_rate_stats([item.get("refine_l2_valid") for item in items])
        refine_l2_match_stats = _pass_rate_stats([item.get("refine_l2_match") for item in items])
        refine_fea_pass_stats = _pass_rate_stats([item.get("refine_fea_pass") for item in items])
        refine_fea_valid_stats = _pass_rate_stats([item.get("refine_fea_valid") for item in items])
        refine_fea_match_stats = _pass_rate_stats([item.get("refine_fea_match") for item in items])
        is_valid_stats = _pass_rate_stats([item.get("is_valid") for item in items])
        is_final_valid_stats = _pass_rate_stats([item.get("is_final_valid") for item in items])
        is_match_stats = _pass_rate_stats([item.get("is_match") for item in items])
        is_one_mismatch_stats = _pass_rate_stats([item.get("is_one_mismatch") for item in items])
        lane_lane_count_pass = _case_lane_metric_pass_stats(items, "lane_level", "lane_count_accuracy")
        lane_lane_1_pass = _case_lane_metric_pass_stats(items, "lane_level", "lane_1_accuracy")
        refine_lane_count_pass = _case_lane_metric_pass_stats(items, "refinement", "lane_count_accuracy")
        refine_lane_1_pass = _case_lane_metric_pass_stats(items, "refinement", "lane_1_accuracy")
        fea_lane_count_pass = _case_lane_metric_pass_stats(items, "layer_fea", "lane_count_accuracy")
        fea_lane_1_pass = _case_lane_metric_pass_stats(items, "layer_fea", "lane_1_accuracy")
        by_model[model_name] = {
            "n_cases": len(items),
            "total_cost_usd": round(sum(cost_values), 6) if cost_values else None,
            "arm_level_validation": arm_stats,
            "lane_level_valid": lane_valid_stats,
            "lane_level_match": lane_match_stats,
            "refine_l1_pass": refine_l1_pass_stats,
            "refine_l1_valid": refine_l1_valid_stats,
            "refine_l1_match": refine_l1_match_stats,
            "refine_l2_pass": refine_l2_pass_stats,
            "refine_l2_valid": refine_l2_valid_stats,
            "refine_l2_match": refine_l2_match_stats,
            "refine_fea_pass": refine_fea_pass_stats,
            "refine_fea_valid": refine_fea_valid_stats,
            "refine_fea_match": refine_fea_match_stats,
            "is_valid": is_valid_stats,
            "is_final_valid": is_final_valid_stats,
            "is_match": is_match_stats,
            "is_one_mismatch": is_one_mismatch_stats,
            "lane_lane_count_pass": lane_lane_count_pass,
            "lane_lane_1_pass": lane_lane_1_pass,
            "refine_lane_count_pass": refine_lane_count_pass,
            "refine_lane_1_pass": refine_lane_1_pass,
            "fea_lane_count_pass": fea_lane_count_pass,
            "fea_lane_1_pass": fea_lane_1_pass,
            "metrics": metrics,
            "arrow_confusion": {
                "lane_level": {
                    "e2e": ((metrics or {}).get("lane_level") or {}).get("incoming_arrow_e2e")
                    or ((metrics or {}).get("lane_level") or {}).get("incoming_arrow"),
                    "matched": ((metrics or {}).get("lane_level") or {}).get("incoming_arrow_matched"),
                },
                "refinement": {
                    "e2e": ((metrics or {}).get("refinement") or {}).get("incoming_arrow_e2e")
                    or ((metrics or {}).get("refinement") or {}).get("incoming_arrow"),
                    "matched": ((metrics or {}).get("refinement") or {}).get("incoming_arrow_matched"),
                },
                "layer_fea": {
                    "e2e": ((metrics or {}).get("layer_fea") or {}).get("incoming_arrow_e2e")
                    or ((metrics or {}).get("layer_fea") or {}).get("incoming_arrow"),
                    "matched": ((metrics or {}).get("layer_fea") or {}).get("incoming_arrow_matched"),
                },
            },
        }
    return by_model


def aggregate_metrics_by_dataset(case_metrics_items):
    """Group local case metrics by dataset, then by model."""
    grouped = defaultdict(list)
    for item in case_metrics_items:
        grouped[item.get("dataset") or ""].append(item)
    return {dataset: aggregate_metrics_by_model(items) for dataset, items in grouped.items()}


def flatten_case_metrics_row(item):
    """Flatten one case into a CSV row: identity, funnel flags, cost, stage metrics."""
    row = {
        "dataset": item.get("dataset"),
        "case_id": item.get("case_id"),
        "completed_at": item.get("completed_at"),
        "junction_model": item.get("junction_model"),
        "junction_reasoning": item.get("junction_reasoning"),
        "arm_model": item.get("arm_model"),
        "arm_reasoning": item.get("arm_reasoning"),
        "lane_model": item.get("lane_model"),
        "lane_reasoning": item.get("lane_reasoning"),
        "refine_model": item.get("refine_model"),
        "refine_reasoning": item.get("refine_reasoning"),
        "arm_level_valid": item.get("arm_level_valid"),
        "arm_level_match": None,
        "lane_level_valid": item.get("lane_level_valid"),
        "lane_level_match": item.get("lane_level_match"),
        "lane_match": None,
        "arrow_match": None,
        "refine_l1_pass": item.get("refine_l1_pass"),
        "refine_l1_valid": item.get("refine_l1_valid"),
        "refine_l1_match": item.get("refine_l1_match"),
        "refine_l2_pass": item.get("refine_l2_pass"),
        "refine_l2_valid": item.get("refine_l2_valid"),
        "refine_l2_match": item.get("refine_l2_match"),
        "refine_fea_pass": item.get("refine_fea_pass"),
        "refine_fea_valid": item.get("refine_fea_valid"),
        "refine_fea_match": item.get("refine_fea_match"),
        "refine_fea_edits": item.get("refine_fea_edits"),
        "refine_fea_score": item.get("refine_fea_score"),
        "refine_fea_n_edits": item.get("refine_fea_n_edits"),
        "refine_l3_pass": item.get("refine_l3_pass"),
        "is_valid": item.get("is_valid"),
        "is_final_valid": item.get("is_final_valid"),
        "is_match": item.get("is_match"),
        "is_one_mismatch": item.get("is_one_mismatch"),
        "fail_reason": item.get("fail_reason"),
        "elapsed_sec": item.get("elapsed_sec"),
        "cost_usd": item.get("cost_usd"),
        "junction_level_usd": item.get("junction_level_usd"),
        "arm_level_usd": item.get("arm_level_usd"),
        "lane_level_usd": item.get("lane_level_usd"),
        "refine_level_usd": item.get("refine_level_usd"),
    }
    stage = item.get("_stage_row")
    if stage is None:
        stage = case_stage_row(item)
        item["_stage_row"] = stage
    for key in stage_field_names():
        row[key] = stage.get(key)
    for key in ("arm_level_match", "lane_match", "arrow_match"):
        if stage.get(key) is not None:
            row[key] = stage.get(key)
    return row


def _simple_model_reasoning(row):
    model = row.get("arm_model") or row.get("lane_model") or row.get("refine_model") or row.get("model")
    reasoning = (
        row.get("arm_reasoning") or row.get("lane_reasoning")
        or row.get("refine_reasoning") or row.get("reasoning") or "none"
    )
    return model, reasoning


def flatten_case_metrics_simple_row(row):
    """Keep the case-level funnel / exact-match flags for a compact CSV."""
    model, reasoning = _simple_model_reasoning(row)
    simple = {"model": model, "reasoning": reasoning}
    for key in CASE_METRICS_SIMPLE_CSV_FIELDS:
        if key in ("model", "reasoning"):
            continue
        simple[key] = row.get(key)
    return simple


def write_case_metrics_csv(items, path):
    """Write the full metrics CSV and a compact ``case_metrics_simple.csv`` beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_case_metrics_row(item) for item in items]
    rows.sort(key=lambda row: (
        str(row.get("dataset") or ""),
        str(row.get("case_id") or ""),
        str(row.get("arm_model") or ""),
        str(row.get("lane_model") or ""),
        str(row.get("refine_model") or ""),
    ))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_METRICS_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    simple_path = path.with_name("case_metrics_simple.csv")
    simple_rows = [flatten_case_metrics_simple_row(row) for row in rows]
    with simple_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CASE_METRICS_SIMPLE_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(simple_rows)
    write_inspect_metrics_csv(items, path)
    return path


def case_metrics_csv_path(output_dir, dataset):
    return resolve_output_root(output_dir) / dataset / "case_metrics.csv"


def refresh_case_metrics_csv(output_dir, dataset_case_ids, run_models_list, *, quiet=True):
    """Rescan local evaluation artifacts and overwrite per-dataset ``case_metrics.csv``."""
    items = []
    paths = {}
    for dataset, case_ids in dataset_case_ids.items():
        if not case_ids:
            continue
        dataset_items = collect_local_case_metrics(
            output_dir, dataset, case_ids, run_models_list, quiet=quiet,
        )
        items.extend(dataset_items)
        paths[dataset] = write_case_metrics_csv(
            dataset_items,
            case_metrics_csv_path(output_dir, dataset),
        )
    return items, paths


def log_model_batch_metrics(model_name, payload, rows=None):
    """Print notebook-style stage tables for one model (or n_cases if rows omitted)."""
    BaseLLM.log("%s: n=%s" % (model_name, payload.get("n_cases")))
    if not rows:
        return
    for line in format_stage_metrics_tables(rows).splitlines():
        BaseLLM.log(line)


def _stage_item_from_eval(config, dataset, case_id, metrics, output_dir):
    case_dir = resolve_case_dir(output_dir, dataset, case_id)
    validation = case_validation_status(
        case_dir,
        config["arm_model"],
        config["lane_model"],
        config["refine_model"],
        arm_reasoning=config["arm_reasoning"],
        lane_reasoning=config["lane_reasoning"],
        refine_reasoning=config["refine_reasoning"],
        lane_metrics=metrics,
        refine_metrics=metrics,
    )
    item = {
        "dataset": dataset,
        "case_id": case_id,
        "output_dir": str(resolve_output_root(output_dir)),
        "junction_model": config["junction_model"],
        "junction_reasoning": config["junction_reasoning"],
        "arm_model": config["arm_model"],
        "arm_reasoning": config["arm_reasoning"],
        "lane_model": config["lane_model"],
        "lane_reasoning": config["lane_reasoning"],
        "refine_model": config["refine_model"],
        "refine_reasoning": config["refine_reasoning"],
        "metrics": metrics,
    }
    item.update(validation)
    return item


def config_for_eval(run_models, output_dir, heading_tol_deg=DEFAULT_HEADING_TOL_DEG):
    """Minimal config ``stage_evaluation`` needs from one ``MODELS`` entry."""
    junction = run_models.get("junction-level") or run_models["arm-level"]
    arm = run_models["arm-level"]
    lane = run_models["lane-level"]
    refine = run_models["refinement"]
    return {
        "junction_model": junction["model"],
        "junction_reasoning": junction.get("reasoning") or "none",
        "arm_model": arm["model"],
        "arm_reasoning": arm.get("reasoning") or "none",
        "lane_model": lane["model"],
        "lane_reasoning": lane.get("reasoning") or "none",
        "refine_model": refine["model"],
        "refine_reasoning": refine.get("reasoning") or "none",
        "output_dir": output_dir,
        "heading_tol_deg": heading_tol_deg,
    }


def list_output_case_ids(output_dir, dataset):
    """Case ids that already have a results folder under ``output_dir``."""
    root = resolve_output_root(output_dir) / dataset
    if not root.is_dir():
        return []
    return [
        path.name
        for path in sorted(root.iterdir(), key=lambda p: p.name)
        if path.is_dir() and not path.name.startswith(".")
    ]


def evaluate_outputs(
    output_dir,
    dataset_case_ids,
    run_models_list,
    *,
    reevaluate=False,
    quiet=False,
    heading_tol_deg=DEFAULT_HEADING_TOL_DEG,
):
    """Optionally re-score artifacts, then rewrite CSVs and aggregate by model.

    ``quiet=True`` suppresses per-case progress (``[eval]``, ``EVALUATION``
    banners). Errors and the final batch summary still print.

    Returns ``(case_metrics_items, csv_paths, metrics_by_dataset, stage_metrics_by_dataset)``.
    ``metrics_by_dataset`` maps ``dataset -> model -> aggregate payload``.
    ``stage_metrics_by_dataset`` maps ``dataset -> {reconstruction, arm, lane}`` tables.
    """
    prev_verbose = set_verbose(False) if quiet else None
    try:
        if reevaluate:
            for dataset, case_ids in dataset_case_ids.items():
                for case_id in case_ids:
                    case_dir = resolve_case_dir(output_dir, dataset, case_id)
                    if not case_dir.is_dir():
                        if not quiet:
                            print(
                                "[eval skip] dataset=%s case_id=%s (no output dir)"
                                % (dataset, case_id),
                                flush=True,
                            )
                        continue
                    for run_models in run_models_list:
                        config = config_for_eval(run_models, output_dir, heading_tol_deg)
                        if not quiet:
                            print(
                                "[eval] dataset=%s case_id=%s arm=%s lane=%s refine=%s"
                                % (
                                    dataset,
                                    case_id,
                                    config["arm_model"],
                                    config["lane_model"],
                                    config["refine_model"],
                                ),
                                flush=True,
                            )
                        try:
                            metrics = stage_evaluation(config, dataset, case_id, case_dir)
                        except Exception as exc:
                            print(
                                "[eval error] dataset=%s case_id=%s %s"
                                % (dataset, case_id, exc),
                                flush=True,
                            )
                            continue
                        if not quiet:
                            item = _stage_item_from_eval(
                                config, dataset, case_id, metrics, output_dir,
                            )
                            print(format_case_stage_metrics(case_stage_row(item)), flush=True)

        items, csv_paths = refresh_case_metrics_csv(
            output_dir, dataset_case_ids, run_models_list, quiet=quiet,
        )
    finally:
        if prev_verbose is not None:
            set_verbose(prev_verbose)
    metrics_by_dataset = {}
    stage_metrics_by_dataset = {}
    for dataset in dataset_case_ids:
        ds_items = [item for item in items if item.get("dataset") == dataset]
        by_model = aggregate_metrics_by_model(ds_items)
        metrics_by_dataset[dataset] = by_model
        rows = [item.get("_stage_row") or case_stage_row(item) for item in ds_items]
        stage_metrics_by_dataset[dataset] = {
            "reconstruction": build_reconstruction_table(rows),
            "arm": build_arm_table(rows),
            "lane": build_lane_table(rows),
        }
        BaseLLM.rule("BATCH METRICS  dataset=%s" % dataset)
        csv_path = csv_paths.get(dataset)
        if csv_path is not None:
            BaseLLM.log("wrote case metrics csv: %s (%s rows)" % (csv_path, len(ds_items)))
        if not rows:
            BaseLLM.log("(no case metrics)")
            continue
        for model_name, payload in by_model.items():
            model_rows = [row for row, item in zip(rows, ds_items) if item.get("model") == model_name]
            log_model_batch_metrics(model_name, payload, rows=model_rows)
    return items, csv_paths, metrics_by_dataset, stage_metrics_by_dataset


def main():
    """Re-evaluate existing pipeline outputs and print aggregates."""
    from scripts.methods.llm_kg.main import MODELS, OUTPUT_DIR, VERBOSE

    set_verbose(VERBOSE)
    dataset_case_ids = {
        dataset: list_output_case_ids(OUTPUT_DIR, dataset) for dataset in DATASETS
    }
    for dataset, case_ids in dataset_case_ids.items():
        BaseLLM.rule("EVAL DATASET %s" % dataset)
        BaseLLM.log("output cases: %d  ids=%s" % (len(case_ids), ",".join(case_ids) or "(none)"))
    items, csv_paths, _metrics_by_dataset, stage_metrics_by_dataset = evaluate_outputs(
        OUTPUT_DIR, dataset_case_ids, MODELS, reevaluate=True, quiet=False,
    )
    payload = {
        "n_cases": len(items),
        "case_metrics_csv": {dataset: str(path) for dataset, path in csv_paths.items()},
        "stage_metrics": stage_metrics_by_dataset,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
