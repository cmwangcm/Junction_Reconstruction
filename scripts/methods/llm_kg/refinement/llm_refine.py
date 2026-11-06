"""Lane-level refinement entry: load lanes → layer1 check + layer2 fix → xodr artifacts."""

import argparse
import copy
import json
import shutil
from pathlib import Path

from scripts.methods.connection_parser import enrich_channelized_features
from scripts.methods.image_utils import load_rgb_image
from scripts.methods.llm_kg.arm_level.llm_gen import (
    DEFAULT_IMAGE_DPI,
    DEFAULT_MIN_WIDTH_EXT_CM,
    DEFAULT_WIDTH_EXT_FRAC,
    paper_cm_to_px,
)
from scripts.methods.llm_kg.lane_level import llm_gen as gen
from scripts.methods.llm_kg.refinement import layer1_tr_check, layer2_con_fix
from scripts.methods.llm_kg.refinement.arm_refine import refine_arm_agent
from scripts.methods.llm_kg.refinement.common import (
    log_validation_error,
    save_refine_artifacts,
)
from scripts.methods.llm_kg.refinement.config_build import build_config
from scripts.methods.llm_kg.refinement.fill_solver import try_solve
from scripts.methods.llm_kg.refinement.loop import RefineContext
from scripts.methods.method_utils import (
    arm_level_image_path,
    arms_path_for_model,
    default_arms_path,
    lanes_path_for_model,
    load_case_driving_side,
    prepare_output_dir,
    refinement_dir,
    resolve_case_inputs,
    resolve_run_dir,
    set_verbose,
    unpack_model_config,
)
from scripts.methods.llm_utils import BaseLLM, CostTracker, build_llm, dump_json

# Re-exports for callers (main.py, tests).
__all__ = [
    "build_config",
    "build_refine_context_from_lanes",
    "emit_xodr_artifacts",
    "refine_arm_agent",
    "run_layered_refinement",
    "run_refine_from_lanes",
    "run_scene_layer3",
]


def _copy_lane_results(lane_results):
    return {arm_id: copy.deepcopy(result) for arm_id, result in (lane_results or {}).items()}


def _restore_lane_results(ctx, snapshot):
    """Put lane-level readings back after a non-valid layer-1 attempt."""
    restored = _copy_lane_results(snapshot)
    ctx.lane_results.clear()
    ctx.lane_results.update(restored)
    for arm_id, agent in list(ctx.agents_by_arm.items()):
        if arm_id in restored:
            agent["result"] = copy.deepcopy(restored[arm_id])


def _config_connection_valid(config, name):
    """True when fill/connections can be solved from this config (same bar as lane_level_valid)."""
    _filled, _connections, exc = try_solve(config, name=name, diagnose=False)
    return exc is None


def run_layered_refinement(
    *,
    config,
    through_roads,
    max_refine_rounds,
    llm,
    agents_by_arm,
    trackers_by_arm,
    lane_results,
    arms,
    all_arm_ids,
    channelized_features,
    name,
    model,
    max_tokens,
    reasoning,
    margin,
    max_inspect_rounds,
    refinement_root=None,
    driving_side=None,
    crop_and_inspect=True,
    image_path=None,
    skip_fea_solver=False,
):
    """L1 checks the lanes. L2 runs when both L1 and lane-level fail the connection check. The solver runs when L2 still fails."""
    if refinement_root is not None:
        refinement_root = prepare_output_dir(refinement_root, clear=True)
    side = driving_side or (config or {}).get("driving_side") or "left"
    if config is not None and config.get("driving_side") is None:
        config = dict(config)
        config["driving_side"] = side
    ctx = RefineContext(
        llm=llm, agents_by_arm=agents_by_arm, trackers_by_arm=trackers_by_arm,
        lane_results=lane_results, arms=arms, all_arm_ids=all_arm_ids,
        channelized_features=channelized_features, through_roads=through_roads,
        name=name, model=model, max_tokens=max_tokens, reasoning=reasoning,
        margin=margin, max_inspect_rounds=max_inspect_rounds,
        refinement_root=refinement_root, driving_side=side,
        crop_and_inspect=crop_and_inspect,
        image_path=image_path,
    )
    refine_by_layer = {"check": 0, "fix": 0}
    lane_level_snapshot = _copy_lane_results(ctx.lane_results)
    lane_level_valid = _config_connection_valid(config, name)

    def _result(passed, config, validation_error):
        status = "PASS" if passed else "FAIL"
        BaseLLM.log(f"[validate] {status} after layered refinement (rounds={refine_by_layer})")
        out = {
            "passed": passed,
            "config": config,
            "refine_rounds": sum(refine_by_layer.values()),
            "refine_rounds_by_layer": refine_by_layer,
            "validation_error": validation_error,
            "filled_config": None,
            "connections": None,
            "fill_adjustments": None,
            "refinement_root": refinement_root,
        }
        return out

    BaseLLM.log("[validate] layer-1 check (always; even if lane-level is already valid)")
    check_out = layer1_tr_check.run(
        ctx, config, through_roads,
        max_refine_rounds=max_refine_rounds,
    )
    refine_by_layer["check"] = check_out["refine_rounds"]
    config = check_out["config"]
    l1_valid = _config_connection_valid(config, name)

    if l1_valid:
        BaseLLM.log("[validate] layer-1 result is valid; keep it and skip layer-2")
        return _result(True, config, None)

    if lane_level_valid:
        BaseLLM.log("[validate] layer-1 result is not valid; restore valid lane-level and skip layer-2")
        _restore_lane_results(ctx, lane_level_snapshot)
        config = ctx.rebuild_config()
        return _result(True, config, None)

    BaseLLM.log("[validate] layer-1 and lane-level are both not valid; keep layer-1 and run layer-2")
    fix_out = layer2_con_fix.run(
        ctx, config, max_refine_rounds=max_refine_rounds, skip_fea_solver=skip_fea_solver,
    )
    refine_by_layer["fix"] = fix_out["refine_rounds"]
    config = fix_out["config"]
    if not fix_out["passed"]:
        return _result(False, config, fix_out["validation_error"])
    return _result(True, config, None)


def emit_xodr_artifacts(xodr_dir, *, arms, lane_result, name, channelized_features=None, through_roads=None, refine_rounds=0, driving_side=None):
    """Build ``xodr_config.json`` then try_solve into fill/connections (or validation_error)."""
    xodr_dir = Path(xodr_dir)
    xodr_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(arms, lane_result, name, channelized_features=channelized_features, through_roads=through_roads, driving_side=driving_side)
    dump_json(xodr_dir / "xodr_config.json", config)
    filled_config, connections, exc = try_solve(config, name=name)
    loop_out = {
        "accepted": exc is None,
        "refine_rounds": refine_rounds,
        "filled_config": filled_config if exc is None else None,
        "connections": connections if exc is None else None,
        "validation_error": None if exc is None else log_validation_error(exc),
    }
    save_refine_artifacts(xodr_dir, loop_out)
    return {"config": config, **loop_out}


def load_lanes_doc(lanes_path):
    lanes_path = Path(lanes_path)
    if not lanes_path.is_file():
        raise FileNotFoundError(f"lanes.json not found: {lanes_path}")
    doc = json.loads(lanes_path.read_text(encoding="utf-8"))
    for arm in doc.get("arms") or []:
        if isinstance(arm, dict):
            arm.pop("channelized_source_type", None)
    return doc


def _lane_arms_for_doc(lane_results):
    arms = []
    for arm_id in sorted(lane_results):
        arm = lane_results[arm_id]
        if isinstance(arm, dict):
            arm.pop("channelized_source_type", None)
        arms.append(arm)
    return arms


def _load_lane_arm_result(loaded_arms, arm_id):
    for lane_arm in loaded_arms or []:
        if int(lane_arm.get("arm_id", -1)) == int(arm_id):
            if isinstance(lane_arm, dict):
                lane_arm.pop("channelized_source_type", None)
            return lane_arm
    raise ValueError(f"lanes.json has no arm_id={arm_id}")


def build_refine_context_from_lanes(
    model,
    *,
    dataset="openaerialmap",
    case_id="1001",
    arms_from=None,
    arms_reasoning=None,
    lanes_from=None,
    lanes_reasoning=None,
    max_tokens=10240,
    compressed=True,
    artifact_case_dir=None,
    lanes_path=None,
    llm=None,
    max_inspect_rounds=1,
    crop_and_inspect=True,
    width_ext_frac=DEFAULT_WIDTH_EXT_FRAC,
    min_width_ext_cm=DEFAULT_MIN_WIDTH_EXT_CM,
    image_dpi=DEFAULT_IMAGE_DPI,
):
    """Load arms/lanes and build a RefineContext without running L1/L2."""
    cfg = unpack_model_config(model)
    api, source, model_name = cfg["api"], cfg["source"], cfg["model_name"]
    api_key, base_url, proxy, reasoning = cfg["api_key"], cfg["base_url"], cfg["proxy"], cfg["reasoning"]
    _image_path, run_name = resolve_case_inputs(dataset=dataset, case_id=case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else resolve_run_dir("results", run_name)
    side = load_case_driving_side(dataset, case_id)
    arms_reason = arms_reasoning if arms_reasoning is not None else reasoning
    lanes_reason = lanes_reasoning if lanes_reasoning is not None else reasoning

    if arms_from:
        arms_path = arms_path_for_model(case_dir, arms_from, arms_reason)
    else:
        arms_path = default_arms_path(case_dir)
    image_path = arm_level_image_path(arms_path, compressed=compressed)
    arm_result = json.loads(arms_path.read_text(encoding="utf-8"))
    arms = arm_result["arms"]
    arms_by_id = {arm["id"]: arm for arm in arms}

    source_lanes_path = Path(lanes_path) if lanes_path is not None else lanes_path_for_model(
        case_dir, lanes_from, lanes_reason,
    )
    loaded = load_lanes_doc(source_lanes_path)
    channelized_features = enrich_channelized_features(
        loaded.get("channelized_features") or arm_result.get("channelized_features") or [],
        arms,
        side,
    )
    through_roads = list(loaded.get("through_roads") or arm_result.get("through_roads") or [])

    ref_root = refinement_dir(case_dir, model_name, reasoning)
    crop_root = Path(ref_root) / "crops"
    if llm is None:
        llm = build_llm(api, api_key, base_url, proxy=proxy)

    original = {"image": None}
    min_px = paper_cm_to_px(min_width_ext_cm, image_dpi)
    all_arm_ids = [arm["id"] for arm in arms]
    lane_results = {}
    trackers_by_arm = {}
    for arm in arms:
        lane_results[arm["id"]] = _load_lane_arm_result(loaded.get("arms"), arm["id"])
        trackers_by_arm[arm["id"]] = CostTracker(source=source, silent=True)

    def _ensure_original():
        if original["image"] is None:
            original["image"] = load_rgb_image(image_path)
        return original["image"]

    def _build_agent_for_refine(arm_id):
        arm = arms_by_id[arm_id]
        crop_dir = crop_root / f"arm_{arm_id}"
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_bundle = gen.run_rule_initial_crop(
            _ensure_original(), arm, crop_dir, width_ext_frac=width_ext_frac, min_width_ext_px=min_px,
        )
        agent = gen.create_arm_agent(
            llm, arm, crop_dir, crop_bundle,
            channelized_features=channelized_features, through_roads=through_roads,
            driving_side=side,
            width_ext_frac=width_ext_frac,
            min_width_ext_px=min_px,
        )
        agent["result"] = lane_results[arm_id]
        agent["inspected"] = True
        agent["original_image"] = _ensure_original()
        return agent

    ctx = RefineContext(
        llm=llm, agents_by_arm=_LazyAgentsByArm(_build_agent_for_refine),
        trackers_by_arm=trackers_by_arm, lane_results=lane_results, arms=arms,
        all_arm_ids=all_arm_ids, channelized_features=channelized_features,
        through_roads=through_roads, name=run_name, model=model_name,
        max_tokens=max_tokens, reasoning=reasoning, margin=gen.LOCATE_BOX_MARGIN,
        max_inspect_rounds=max_inspect_rounds, refinement_root=ref_root,
        driving_side=side, crop_and_inspect=crop_and_inspect,
        image_path=image_path,
    )
    return {
        "ctx": ctx,
        "arms_doc": arm_result,
        "run_name": run_name,
        "case_dir": case_dir,
        "ref_root": ref_root,
        "source": source,
        "model_name": model_name,
        "reasoning": reasoning,
        "side": side,
        "image_path": image_path,
        "source_lanes_path": source_lanes_path,
        "llm": llm,
    }


def _junction_lanes_changed(ctx):
    return any(
        int(agent.get("refinement_rounds") or 0) > 0 or (agent.get("patch_events") or [])
        for agent in ctx.agents_by_arm.values()
    )


def _persist_layer3_junction(bundle):
    """Write updated lanes.json and re-emit xodr when layer-3 changed this junction.

    If the new reading is not locally fillable, keep the pre-layer-3 lanes.json so
    scene feasibility does not start from a broken mouth.
    """
    ctx = bundle["ctx"]
    if not _junction_lanes_changed(ctx):
        return False
    config = build_config(
        ctx.arms,
        gen._lane_result_doc(ctx.lane_results),
        ctx.name,
        channelized_features=ctx.channelized_features,
        through_roads=ctx.through_roads,
        driving_side=ctx.driving_side,
    )
    _filled, _conns, exc = try_solve(config, name=ctx.name, diagnose=False)
    if exc is not None:
        BaseLLM.log(
            "layer-3 skip persist %s: local fill failed after re-identify; keep prior lanes.json"
            % (ctx.name,)
        )
        return False
    artifacts = emit_xodr_artifacts(
        Path(bundle["ref_root"]) / "xodr_result",
        arms=ctx.arms,
        lane_result=gen._lane_result_doc(ctx.lane_results),
        name=ctx.name,
        channelized_features=ctx.channelized_features,
        through_roads=ctx.through_roads,
        refine_rounds=sum(
            int(agent.get("refinement_rounds") or 0)
            for agent in ctx.agents_by_arm.values()
        ),
        driving_side=ctx.driving_side,
    )
    if not artifacts.get("accepted"):
        BaseLLM.log(
            "layer-3 skip persist %s: local fill failed after re-identify; keep prior lanes.json"
            % (ctx.name,)
        )
        return False
    result = {
        "source_image": str(bundle["image_path"]),
        "driving_side": ctx.driving_side,
        "arms": _lane_arms_for_doc(ctx.lane_results),
        "channelized_features": ctx.channelized_features,
        "through_roads": ctx.through_roads,
    }
    dump_json(Path(bundle["ref_root"]) / "lanes.json", result)
    BaseLLM.log(
        "layer-3 updated junction lanes -> %s"
        % (Path(bundle["ref_root"]) / "lanes.json")
    )
    return True


def run_scene_layer3(
    model,
    *,
    dataset,
    case_id,
    junctions_doc,
    work_items,
    arms_from,
    arms_reasoning,
    lanes_from,
    lanes_reasoning,
    max_refine_rounds=1,
    crop_and_inspect=True,
    max_inspect_rounds=1,
    max_tokens=10240,
    compressed=True,
    scene_case_dir=None,
    width_ext_frac=DEFAULT_WIDTH_EXT_FRAC,
    min_width_ext_cm=DEFAULT_MIN_WIDTH_EXT_CM,
    image_dpi=DEFAULT_IMAGE_DPI,
):
    """Scene-level layer-3 HARD pair constraints across confirmed junctions."""
    from scripts.methods.llm_kg.refinement.layer3_jun_fix import (
        locate_inter_junction_pairs,
        pair_log_label,
        run as run_layer3,
    )

    junctions = list((junctions_doc or {}).get("junctions") or [])
    if len(junctions) <= 1:
        BaseLLM.log("[validate:layer3] skipped (one junction)")
        return {"passed": True, "skipped": True, "refine_rounds": 0}

    cfg = unpack_model_config(model)
    api, source, model_name = cfg["api"], cfg["source"], cfg["model_name"]
    api_key, base_url, proxy, reasoning = (
        cfg["api_key"], cfg["base_url"], cfg["proxy"], cfg["reasoning"],
    )
    llm = build_llm(api, api_key, base_url, proxy=proxy)
    scene_root = refinement_dir(scene_case_dir, model_name, reasoning)
    scene_root.mkdir(parents=True, exist_ok=True)

    bundles = {}
    arms_by_junction = {}
    for jid, work_dir in work_items:
        if jid is None or work_dir is None:
            continue
        refine_lanes = Path(refinement_dir(work_dir, model_name, reasoning)) / "lanes.json"
        lanes_path = refine_lanes if refine_lanes.is_file() else None
        bundle = build_refine_context_from_lanes(
            model,
            dataset=dataset,
            case_id=case_id,
            arms_from=arms_from,
            arms_reasoning=arms_reasoning,
            lanes_from=lanes_from,
            lanes_reasoning=lanes_reasoning,
            max_tokens=max_tokens,
            compressed=compressed,
            artifact_case_dir=work_dir,
            lanes_path=lanes_path,
            llm=llm,
            max_inspect_rounds=max_inspect_rounds,
            crop_and_inspect=crop_and_inspect,
            width_ext_frac=width_ext_frac,
            min_width_ext_cm=min_width_ext_cm,
            image_dpi=image_dpi,
        )
        bundles[int(jid)] = bundle
        arms_by_junction[int(jid)] = bundle["arms_doc"]

    BaseLLM.rule("LAYER-3 INTER-JUNCTION HARD CONSTRAINT")
    pairs = locate_inter_junction_pairs(junctions_doc, arms_by_junction)
    BaseLLM.log("inter-junction pairs: %s" % (
        ", ".join(pair_log_label(item) for item in pairs) or "(none)"
    ))
    contexts = {jid: bundle["ctx"] for jid, bundle in bundles.items()}
    out = run_layer3(
        contexts, pairs,
        scene_root=scene_root,
        max_refine_rounds=max_refine_rounds,
    )
    updated = []
    for jid, bundle in bundles.items():
        if _persist_layer3_junction(bundle):
            updated.append(jid)
    if updated:
        BaseLLM.log("layer-3 rewrote junctions: %s" % ",".join(str(x) for x in updated))
    out["updated_junctions"] = updated
    out["source"] = source
    return out


class _LazyAgentsByArm(dict):
    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def __missing__(self, arm_id):
        agent = self._factory(arm_id)
        self[arm_id] = agent
        return agent


def run_refine_from_lanes(
    model,
    *,
    dataset="openaerialmap",
    case_id="1001",
    arms_from=None,
    arms_reasoning=None,
    lanes_from=None,
    lanes_reasoning=None,
    max_refine_rounds=2,
    crop_and_inspect=True,
    max_inspect_rounds=1,
    max_tokens=10240,
    compressed=True,
    verbose=True,
    artifact_case_dir=None,
    width_ext_frac=DEFAULT_WIDTH_EXT_FRAC,
    min_width_ext_cm=DEFAULT_MIN_WIDTH_EXT_CM,
    image_dpi=DEFAULT_IMAGE_DPI,
    skip_fea_solver=False,
):
    """Standalone refine with separate input models and refine LLM.
    """
    cfg = unpack_model_config(model)
    api, source, model_name = cfg["api"], cfg["source"], cfg["model_name"]
    api_key, base_url, proxy, reasoning = cfg["api_key"], cfg["base_url"], cfg["proxy"], cfg["reasoning"]
    prev_verbose = set_verbose(verbose)
    _image_path, run_name = resolve_case_inputs(dataset=dataset, case_id=case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else resolve_run_dir("results", run_name)
    side = load_case_driving_side(dataset, case_id)
    arms_reason = arms_reasoning if arms_reasoning is not None else reasoning
    lanes_reason = lanes_reasoning if lanes_reasoning is not None else reasoning

    if arms_from:
        arms_path = arms_path_for_model(case_dir, arms_from, arms_reason)
    else:
        arms_path = default_arms_path(case_dir)
    image_path = arm_level_image_path(arms_path, compressed=compressed)
    arm_result = json.loads(arms_path.read_text(encoding="utf-8"))
    arms = arm_result["arms"]
    arms_by_id = {arm["id"]: arm for arm in arms}

    source_lanes_path = lanes_path_for_model(case_dir, lanes_from, lanes_reason)
    loaded = load_lanes_doc(source_lanes_path)
    channelized_features = enrich_channelized_features(
        loaded.get("channelized_features") or arm_result.get("channelized_features") or [],
        arms,
        side,
    )
    through_roads = list(loaded.get("through_roads") or arm_result.get("through_roads") or [])

    ref_root = refinement_dir(case_dir, model_name, reasoning)
    out_lanes_path = Path(ref_root) / "lanes.json"
    xodr_dir = Path(ref_root) / "xodr_result"
    crop_root = Path(ref_root) / "crops"

    BaseLLM.rule("LANE-LEVEL REFINE")
    BaseLLM.log(f"arms={arms_from} lanes={lanes_from} refine={model_name}/{reasoning} driving_side={side}")
    BaseLLM.log(
        f"crop-and-inspect: {crop_and_inspect}  width_ext_frac: {width_ext_frac}  "
        f"min_width_ext_cm: {min_width_ext_cm}  dpi: {image_dpi}"
    )

    llm = build_llm(api, api_key, base_url, proxy=proxy)
    original = {"image": None}
    min_px = paper_cm_to_px(min_width_ext_cm, image_dpi)
    all_arm_ids = [arm["id"] for arm in arms]
    lane_results = {}
    trackers_by_arm = {}
    for arm in arms:
        lane_results[arm["id"]] = _load_lane_arm_result(loaded.get("arms"), arm["id"])
        trackers_by_arm[arm["id"]] = CostTracker(source=source, silent=True)

    def _ensure_original():
        if original["image"] is None:
            original["image"] = load_rgb_image(image_path)
        return original["image"]

    def _build_agent_for_refine(arm_id):
        arm = arms_by_id[arm_id]
        crop_dir = crop_root / f"arm_{arm_id}"
        if crop_dir.exists():
            shutil.rmtree(crop_dir)
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_bundle = gen.run_rule_initial_crop(
            _ensure_original(), arm, crop_dir, width_ext_frac=width_ext_frac, min_width_ext_px=min_px,
        )
        agent = gen.create_arm_agent(
            llm, arm, crop_dir, crop_bundle,
            channelized_features=channelized_features, through_roads=through_roads,
            driving_side=side,
            width_ext_frac=width_ext_frac,
            min_width_ext_px=min_px,
        )
        agent["result"] = lane_results[arm_id]
        agent["inspected"] = True
        agent["original_image"] = _ensure_original()
        return agent

    agents_by_arm = _LazyAgentsByArm(_build_agent_for_refine)
    config = build_config(
        arms, gen._lane_result_doc(lane_results), run_name,
        channelized_features=channelized_features, through_roads=through_roads,
        driving_side=side,
    )
    layered = None
    artifacts = None
    result = {
        "source_image": str(loaded.get("source_image") or image_path),
        "driving_side": side,
        "arms": _lane_arms_for_doc(lane_results),
        "channelized_features": channelized_features,
        "through_roads": through_roads,
    }
    conversation, cost = {}, {}
    try:
        layered = run_layered_refinement(
            config=config, through_roads=through_roads, max_refine_rounds=max_refine_rounds,
            llm=llm, agents_by_arm=agents_by_arm, trackers_by_arm=trackers_by_arm,
            lane_results=lane_results, arms=arms, all_arm_ids=all_arm_ids,
            channelized_features=channelized_features, name=run_name, model=model_name,
            max_tokens=max_tokens, reasoning=reasoning, margin=gen.LOCATE_BOX_MARGIN,
            max_inspect_rounds=max_inspect_rounds,
            crop_and_inspect=crop_and_inspect,
            refinement_root=ref_root, driving_side=side,
            image_path=image_path,
            skip_fea_solver=skip_fea_solver,
        )
        result = {
            "source_image": str(loaded.get("source_image") or image_path),
            "driving_side": side,
            "arms": _lane_arms_for_doc(lane_results),
            "channelized_features": channelized_features,
            "through_roads": through_roads,
        }
        BaseLLM.rule("XODR_CONFIG (post-refine)")
        artifacts = emit_xodr_artifacts(
            xodr_dir,
            arms=arms,
            lane_result=gen._lane_result_doc(lane_results),
            name=run_name,
            channelized_features=channelized_features,
            through_roads=through_roads,
            refine_rounds=layered["refine_rounds"],
            driving_side=side,
        )
    finally:
        result = {
            "source_image": str(loaded.get("source_image") or image_path),
            "driving_side": side,
            "arms": _lane_arms_for_doc(lane_results),
            "channelized_features": channelized_features,
            "through_roads": through_roads,
        }
        conversation, cost = gen._save_lane_bundle(
            ref_root, agents_by_arm=agents_by_arm, trackers_by_arm=trackers_by_arm, model=model_name, lanes_doc=result,
            only_refined=True, write_conversation=False,
        )

    if layered is None:
        raise RuntimeError("refinement failed before layered result was produced")
    if artifacts is None:
        raise RuntimeError("refinement failed before xodr artifacts were produced")

    refined_trackers = {
        arm_id: trackers_by_arm[arm_id]
        for arm_id in gen._agents_touched_by_refine(agents_by_arm)
        if arm_id in trackers_by_arm
    }
    gen._merged_tracker(refined_trackers, source=source).print_summary()
    BaseLLM.verbose = prev_verbose
    return {
        "result": result,
        "config": artifacts["config"],
        "refine_meta": {
            **layered,
            "filled_config": artifacts["filled_config"],
            "connections": artifacts["connections"],
            "validation_error": layered["validation_error"] or artifacts["validation_error"],
            "connections_validated": bool(artifacts["accepted"] and layered["passed"]),
        },
        "source_lanes_path": source_lanes_path,
        "lanes_path": out_lanes_path,
        "xodr_dir": xodr_dir,
        "refinement_root": layered.get("refinement_root") or ref_root,
        "arms_path": arms_path,
        "conversation": conversation,
        "cost": cost,
    }


if __name__ == "__main__":
    MODEL = {
        "api": "openai",
        "provider": "openrouter",
        "model": "openai/gpt-5.6-sol",
        "reasoning": "medium",
    }
    
    parser = argparse.ArgumentParser(description="Standalone layered refine over an existing lanes.json.")
    parser.add_argument("--dataset", default="openaerialmap")
    parser.add_argument("--case-id", default="1005")
    parser.add_argument("--arms-from", default="gpt-5.6-sol", help="Arm-level model name to read arms.json from")
    parser.add_argument("--arms-reasoning", default=None, help="Reasoning effort for arms-from dir (default: same as refine reasoning)")
    parser.add_argument("--lanes-from", default="gpt-5.6-sol", help="Lane-level model name to read lanes.json from")
    parser.add_argument("--lanes-reasoning", default=None, help="Reasoning effort for lanes-from dir (default: same as refine reasoning)")
    parser.add_argument("--model", default=MODEL["model"], help="Refine LLM model name (output under refinement/<model>_<reasoning>/)")
    parser.add_argument("--reasoning", default=MODEL.get("reasoning", "none"))
    parser.add_argument("--max-refine-rounds", type=int, default=1)
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
    run_refine_from_lanes(
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
        lanes_from=args.lanes_from,
        lanes_reasoning=args.lanes_reasoning,
        max_refine_rounds=args.max_refine_rounds,
        crop_and_inspect=args.crop_and_inspect,
        max_inspect_rounds=args.max_inspect_rounds,
        max_tokens=args.max_tokens,
        compressed=args.compressed,
        verbose=args.verbose,
        width_ext_frac=args.width_ext_frac,
        min_width_ext_cm=args.min_width_ext_cm,
        image_dpi=args.image_dpi,
    )
