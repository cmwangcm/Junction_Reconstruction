"""Scene-level feasibility repair after per-junction solve_junction.

Runs only when a case has more than one junction. Operators and Scheme B
scoring are the same as ``solve_junction``, keyed by ``(junction_id, arm_id)``.
The oracle is each junction's ``try_solve`` plus Layer-3 connector-end checks.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.methods.image_processing import resolve_processed_image
from scripts.methods.llm_kg.refinement.common import (
    dump_json,
    refinement_layer_dir,
    save_layer_result,
)
from scripts.methods.llm_kg.refinement.feasibility_rank import rank_scene_repair_candidates
from scripts.methods.llm_kg.refinement.feasibility_solver import (
    format_repair_edits,
    solve_scene,
)
from scripts.methods.llm_kg.refinement.layer2_con_fix import _lanes_doc_from_results
from scripts.methods.llm_kg.refinement.layer3_jun_fix import (
    locate_inter_junction_pairs,
    pair_log_label,
)
from scripts.methods.llm_kg.refinement.llm_refine import emit_xodr_artifacts
from scripts.methods.llm_utils import BaseLLM, CostTracker, build_llm
from scripts.methods.method_utils import (
    arms_path_for_model,
    load_case_driving_side,
    refinement_dir,
    resolve_case_inputs,
    unpack_model_config,
    xodr_result_dir_for_model,
)


def _lane_results_from_doc(lane_doc):
    return {
        int(arm["arm_id"]): arm
        for arm in (lane_doc.get("arms") or [])
        if arm.get("arm_id") is not None
    }


def _resolve_scene_image(case_dir, config, *, dataset, case_id):
    compressed = bool(config.get("use_compressed_image", False))
    try:
        path = resolve_processed_image(case_dir, compressed=compressed)
        if Path(path).is_file():
            return Path(path)
    except FileNotFoundError:
        pass
    try:
        image_path, _run_name = resolve_case_inputs(dataset, case_id)
        if Path(image_path).is_file():
            return Path(image_path)
    except Exception:
        pass
    return None


def _load_junction_bundle(work_dir, config, *, driving_side):
    refine_model = config["refine_model"]
    refine_reasoning = config["refine_reasoning"]
    ref_root = refinement_dir(work_dir, refine_model, refine_reasoning)
    lanes_path = ref_root / "lanes.json"
    if not lanes_path.is_file():
        raise FileNotFoundError("scene feasibility needs refinement lanes.json: %s" % lanes_path)
    lane_doc = json.loads(lanes_path.read_text(encoding="utf-8"))
    if not lane_doc.get("arms"):
        raise FileNotFoundError("scene feasibility lanes.json has no arms: %s" % lanes_path)
    arms_path = arms_path_for_model(work_dir, config["arm_model"], config["arm_reasoning"])
    arms_doc = json.loads(arms_path.read_text(encoding="utf-8"))
    lane_doc = dict(lane_doc)
    lane_doc["driving_side"] = lane_doc.get("driving_side") or driving_side
    return {
        "work_dir": Path(work_dir),
        "ref_root": ref_root,
        "lanes_path": lanes_path,
        "lane_doc": lane_doc,
        "lane_results": _lane_results_from_doc(lane_doc),
        "arms_doc": arms_doc,
        "arms": arms_doc.get("arms") or [],
        "channelized_features": lane_doc.get("channelized_features"),
        "through_roads": lane_doc.get("through_roads"),
    }


def _persist_scene_layer(
    refinement_root, *, passed, repair, lanes_doc, rank_cost=None,
    rank_conversation=None, pairs=None,
):
    extra = {}
    if repair is not None:
        extra["feasibility_repair"] = repair.to_dict()
    if pairs is not None:
        extra["pairs"] = [
            {"left": item.get("left"), "right": item.get("right")}
            for item in pairs
        ]
    save_layer_result(
        refinement_root, "fea_scene",
        passed=bool(passed), refine_rounds=0,
        validation_error=None,
        lanes_doc=lanes_doc,
        extra=extra or None,
    )
    if refinement_root is None:
        return
    fea_dir = refinement_layer_dir(refinement_root, "fea_scene")
    if repair is not None:
        dump_json(fea_dir / "feasibility_solver.json", repair.to_dict())
    dump_json(fea_dir / "cost.json", rank_cost if rank_cost is not None else {"steps": [], "totals": {}})
    dump_json(fea_dir / "conversation.json", rank_conversation if rank_conversation is not None else {})
    junctions = (lanes_doc or {}).get("junctions") or {}
    for jid, doc in junctions.items():
        dump_json(fea_dir / "junctions" / str(jid) / "lanes.json", doc)


def _combined_lanes_doc(bundles, *, driving_side, repaired_by_jid=None):
    junctions = {}
    for jid, bundle in bundles.items():
        results = (repaired_by_jid or {}).get(jid) or bundle["lane_results"]
        junctions[str(jid)] = _lanes_doc_from_results(
            results,
            channelized_features=bundle["channelized_features"],
            through_roads=bundle["through_roads"],
            driving_side=driving_side,
            extra=bundle["lane_doc"],
        )
    return {"driving_side": driving_side, "junctions": junctions}


def run_scene_feasibility(
    config,
    dataset,
    case_id,
    junctions_doc,
    work_items,
    *,
    case_dir,
    run_name=None,
):
    """Min-edit scene repair. Writes ``layer_fea_scene`` and re-emits xodr mouths."""
    junctions = list((junctions_doc or {}).get("junctions") or [])
    if len(junctions) <= 1:
        return {"passed": True, "skipped": True, "reason": "single_junction"}
    if config.get("skip_fea_solver"):
        return {"passed": True, "skipped": True, "reason": "skip_fea_solver"}

    case_dir = Path(case_dir)

    side = load_case_driving_side(dataset, case_id)
    bundles = {}
    arms_by_junction = {}
    for jid, work_dir in work_items:
        if jid is None or work_dir is None:
            continue
        bundle = _load_junction_bundle(work_dir, config, driving_side=side)
        bundles[int(jid)] = bundle
        arms_by_junction[int(jid)] = bundle["arms_doc"]

    if len(bundles) <= 1:
        return {"passed": True, "skipped": True, "reason": "single_junction"}

    pairs = locate_inter_junction_pairs(junctions_doc, arms_by_junction)
    BaseLLM.rule("SCENE FEASIBILITY")
    BaseLLM.log("inter-junction pairs: %s" % (
        ", ".join(pair_log_label(item) for item in pairs) or "(none)"
    ))

    scene_spec = {
        jid: {
            "lane_results": bundle["lane_results"],
            "arms": bundle["arms"],
            "channelized_features": bundle["channelized_features"],
            "through_roads": bundle["through_roads"],
            "name": "j%s" % jid,
        }
        for jid, bundle in bundles.items()
    }

    BaseLLM.log("[validate:feasibility_scene] start")
    repair = solve_scene(scene_spec, pairs, driving_side=side)
    original_lanes = _combined_lanes_doc(bundles, driving_side=side)

    refine_cfg = unpack_model_config(_model_cfg_from_pipeline(config))
    llm = build_llm(
        refine_cfg["api"], refine_cfg["api_key"], refine_cfg["base_url"],
        proxy=refine_cfg["proxy"],
    )
    rank_tracker = CostTracker(source=refine_cfg["source"], silent=True)
    image_path = _resolve_scene_image(
        case_dir, config, dataset=dataset, case_id=case_id,
    )
    if image_path is None:
        try:
            image_path, _ = resolve_case_inputs(dataset, case_id)
        except Exception:
            image_path = None

    rank_conversation = None
    if repair.ok and (repair.candidates or []):
        BaseLLM.log(
            f"[validate:feasibility_scene] {len(repair.candidates)} candidate(s); "
            f"trials={repair.trials}"
        )
        for cand in repair.candidates:
            BaseLLM.log(
                f"candidate {cand.candidate_id} score={cand.score} "
                f"({len(cand.edits)} edit(s))",
                indent=1,
            )
            for line in format_repair_edits(cand.edits):
                BaseLLM.log(line, indent=2)
        repair, rank_conversation = rank_scene_repair_candidates(
            repair,
            llm=llm,
            model=refine_cfg["model_name"],
            original_lanes=original_lanes,
            image_path=image_path,
            tracker=rank_tracker,
            reasoning=refine_cfg["reasoning"],
            max_tokens=min(int(config.get("refine_max_tokens") or 2048), 4096),
        )

    if not repair.ok:
        BaseLLM.log(
            "[validate:feasibility_scene] FAIL"
            + (f" ({repair.note})" if repair.note else "")
            + f" trials={repair.trials}"
        )
    else:
        BaseLLM.log(
            f"[validate:feasibility_scene] chosen score={repair.score} "
            f"({len(repair.edits)} edit(s); source={repair.rank_source or 'score'}; "
            f"trials={repair.trials})"
        )
        for line in format_repair_edits(repair.edits):
            BaseLLM.log(line, indent=1)

    repaired_by_jid = repair.lane_results if repair.ok else None
    combined = _combined_lanes_doc(
        bundles, driving_side=side, repaired_by_jid=repaired_by_jid,
    )
    if rank_conversation is None:
        rank_conversation = {
            "stage": "feasibility_rank_scene",
            "model": refine_cfg["model_name"],
            "skipped": "solver_failed" if not repair.ok else "no_candidates",
        }
    if image_path and "source_image" not in rank_conversation:
        rank_conversation["source_image"] = str(image_path)

    scene_root = refinement_dir(case_dir, config["refine_model"], config["refine_reasoning"])
    _persist_scene_layer(
        scene_root,
        passed=bool(repair.ok),
        repair=repair,
        lanes_doc=combined,
        rank_cost=rank_tracker.summary(),
        rank_conversation=rank_conversation,
        pairs=pairs,
    )
    if rank_tracker.steps:
        rank_tracker.print_summary()

    if not repair.ok:
        raise RuntimeError(
            "scene feasibility solver failed: %s" % (repair.note or "no feasible repair")
        )

    updated = []
    if repair.edits:
        for jid, bundle in bundles.items():
            results = (repair.lane_results or {}).get(jid) or bundle["lane_results"]
            lane_doc = dict(bundle["lane_doc"])
            lane_doc["arms"] = [results[arm_id] for arm_id in sorted(results)]
            lane_doc["driving_side"] = side
            dump_json(bundle["ref_root"] / "lanes.json", lane_doc)
            xodr_dir = xodr_result_dir_for_model(
                bundle["work_dir"], config["refine_model"], config["refine_reasoning"],
            )
            artifacts = emit_xodr_artifacts(
                xodr_dir,
                arms=bundle["arms"],
                lane_result=lane_doc,
                name=run_name or ("j%s" % jid),
                channelized_features=lane_doc.get("channelized_features"),
                through_roads=lane_doc.get("through_roads"),
                refine_rounds=0,
                driving_side=side,
            )
            if not artifacts["accepted"]:
                raise RuntimeError(
                    "scene feasibility did not produce valid connections for junction %s: %s"
                    % (jid, xodr_dir / "validation_error.json")
                )
            updated.append(jid)
        BaseLLM.log("scene feasibility rewrote junctions: %s" % ",".join(str(x) for x in updated))
    else:
        BaseLLM.log("scene already pair-balanced; no mouth rewrite")

    return {
        "passed": True,
        "skipped": False,
        "score": repair.score,
        "edits": list(repair.edits or []),
        "updated_junctions": updated,
        "trials": repair.trials,
    }


def _model_cfg_from_pipeline(config):
    return {
        "api": config["refine_api"],
        "provider": config["refine_provider"],
        "model": config["refine_model"],
        "reasoning": config["refine_reasoning"],
    }
