"""Batch pipeline: junction_level -> arm-level -> lane-level -> refinement -> generate_roads -> carla -> evaluation.

Each case calls ``eval.stage_evaluation``. Per-case logs and the batch summary
use the same stage-wise metrics as ``eval.ipynb`` (gen/rec acc, arm identification,
lane-count / arrow metrics at stage1, stage2-1, stage2-2, stage3).
Per-dataset ``case_metrics.csv`` is rewritten after every case and again at the
end via ``eval.evaluate_outputs``. To re-score existing outputs without generation::

    python -m scripts.methods.llm_kg.eval

Artifact layout under ``{output_dir}/{dataset}/{case_id}/``::

    images/                 # shared processed full-scene images for the case
    junction_level/{model}_{reasoning}/
        junctions.json         # id, axis-aligned box, inter-junction links
        overlay.png
        crops/junction_{id}.png
        inter_junction_roads.json
    junctions/{id}/            # per-junction crop + arm/lane/refine (when junction_level ran)
        images/
        arm_level/{model}_{reasoning}/
        lane_level/{model}_{reasoning}/
        refinement/{model}_{reasoning}/
            xodr_result/           # connections.json + scenario.xodr
        pipeline_summary.json  # sibling of arm/lane/refine; always written (success or abort)
    arm_level/{model}_{reasoning}/
        arms.json              # after stage 2 + through-road postprocess
        arms_stage1.json       # stage-1 LLM junction_center + heading (no through-road postprocess)
        arms_stage2.json       # stage-2 LLM refine (no through-road postprocess)
        overlay_stage1.png
        overlay_stage2.png     # mouth edges, no center marker
        overlay.png            # after through-road postprocess
        crops/arm_{id}.png
    lane_level/{model}_{reasoning}/
        identifier.json        # after identifier only
        inspect.json           # after identifier + evaluator + inspector
        lanes.json             # after organizer (travel filled)
        evaluation/             # arm_level + lane_level eval
            full.json
            mismatches.json
            metrics.json
    refinement/{model}_{reasoning}/
        lanes.json              # post-refine cross-sections (legacy case-root layout)
        evaluation/             # refinement eval
            full.json
            mismatches.json
            metrics.json
        layer_1/, layer_2/, layer_fea/, layer_fea_scene/, layer_3/
        layer_fea/evaluation/  # GT mismatch for feasibility_solver lanes
        xodr_result/            # merged connections + scenario.xodr (multi-junction only)
            xodr_config.json
            xodr_config_fill.json
            connections.json
            validation_error.json   # on failure
            scenario.xodr
            carla_validation_error.json  # on CARLA failure
    pipeline_summary.json       # sibling of arm/lane/refine for legacy case-root layout

    Also rewrites ``{output_dir}/{dataset}/case_metrics.csv`` after each case (one row per sample).

Run from the repository root::

    python -m scripts.methods.llm_kg.main
    python -m scripts.methods.llm_kg.main --parallel 4
    python -m scripts.methods.llm_kg.eval   # re-score existing outputs only
"""

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.data_reader import DATASET_ROOT
from scripts.methods.evaluation import DEFAULT_HEADING_TOL_DEG
from scripts.methods.llm_kg.eval import (
    _lanes_connection_valid,
    case_validation_status,
    collect_case_cost,
    evaluate_outputs,
    format_completed_at,
    refresh_case_metrics_csv,
    stage_evaluation,
)
from scripts.methods.llm_kg.stage_metrics import (
    case_stage_row,
    format_case_stage_metrics,
    pipeline_item_from_summary,
)
from scripts.methods.llm_utils import BaseLLM
from scripts.methods.method_utils import (
    arms_path_for_model,
    connections_path_for_model,
    dump_json,
    junction_work_dir,
    junctions_path_for_model,
    lane_level_dir,
    lanes_path_for_model,
    list_junction_artifact_roots,
    load_case_driving_side,
    pipeline_summary_path,
    refinement_dir,
    resolve_case_inputs,
    resolve_run_dir,
    scenario_xodr_path_for_model,
    set_verbose,
    xodr_result_dir_for_model,
)

OUTPUT_DIR = "results"
VERBOSE = True

SKIP_JUNCTION_LEVEL = True
SKIP_ARM_LEVEL = False
SKIP_LANE_LEVEL = False
SKIP_REFINEMENT = False
SKIP_FEA_SOLVER = False  # keep the existing refinement/lanes.json
SKIP_GENERATE_ROADS = False
SKIP_CARLA_VALIDATE = True
RANDOM_CASE_COUNT = None  # per-dataset random sample size; None or 0 = all cases
RANDOM_CASE_SEED = 1
CASE_PARALLELISM = 4 # concurrent cases; 1 = sequential
HEADING_TOL_DEG = DEFAULT_HEADING_TOL_DEG
# Extra attempts after a failed LLM or validation stage. Lane-level retries the failed arm.
CASE_STAGE_RETRY_LIMIT = 1

MAX_EDGE = None
USE_COMPRESSED_IMAGE = False
JUNCTION_MAX_TOKENS = 102400
ARM_MAX_TOKENS = 102400
WIDTH_EXT_FRAC = 0.4  # crop/overlay width expansion; 0.4 = 40% wider than stored arm width
STAGE1_WIDTH_FRAC = 0.50  # fixed stage-1 crop width, fraction of image side
MIN_WIDTH_EXT_CM = 0.5  # floor on per-side lateral pad, in original-scan centimeters
IMAGE_DPI = 300.0  # scan DPI used to convert MIN_WIDTH_EXT_CM to pixels
LANE_MAX_TOKENS = 102400
REFINE_MAX_TOKENS = 102400
MAX_REFINE_ROUNDS = 1
LANE_CROP_AND_INSPECT = True
REFINE_CROP_AND_INSPECT = False
LANE_MAX_INSPECT_ROUNDS = 1
REFINE_MAX_INSPECT_ROUNDS = 1

RUN_ESMINI = False
ESMINI_PATH = "esmini"
SPAWN_VEHICLES = 7
SPAWN_DURATION = 60.0


def make_config(run_models):
    """Build pipeline config from one MODELS entry.

    ``run_models`` keys: ``junction-level`` (optional), ``arm-level``, ``lane-level``, ``refinement``;
    each value is ``{api, provider, model, reasoning}``.
    """
    junction = run_models.get("junction-level") or run_models["arm-level"]
    arm = run_models["arm-level"]
    lane = run_models["lane-level"]
    refine = run_models["refinement"]
    return {
        "junction_api": junction["api"],
        "junction_provider": junction["provider"],
        "junction_model": junction["model"],
        "junction_reasoning": junction.get("reasoning") or "none",
        "arm_api": arm["api"],
        "arm_provider": arm["provider"],
        "arm_model": arm["model"],
        "arm_reasoning": arm.get("reasoning") or "none",
        "lane_api": lane["api"],
        "lane_provider": lane["provider"],
        "lane_model": lane["model"],
        "lane_reasoning": lane.get("reasoning") or "none",
        "refine_api": refine["api"],
        "refine_provider": refine["provider"],
        "refine_model": refine["model"],
        "refine_reasoning": refine.get("reasoning") or "none",
        "output_dir": OUTPUT_DIR,
        "verbose": VERBOSE,
        "skip_junction_level": SKIP_JUNCTION_LEVEL,
        "skip_arm_level": SKIP_ARM_LEVEL,
        "skip_lane_level": SKIP_LANE_LEVEL,
        "skip_refinement": SKIP_REFINEMENT,
        "skip_fea_solver": SKIP_FEA_SOLVER,
        "skip_generate_roads": SKIP_GENERATE_ROADS,
        "skip_carla_validate": SKIP_CARLA_VALIDATE,
        "heading_tol_deg": HEADING_TOL_DEG,
        "max_edge": MAX_EDGE,
        "use_compressed_image": USE_COMPRESSED_IMAGE,
        "junction_max_tokens": JUNCTION_MAX_TOKENS,
        "arm_max_tokens": ARM_MAX_TOKENS,
        "width_ext_frac": WIDTH_EXT_FRAC,
        "stage1_width_frac": STAGE1_WIDTH_FRAC,
        "min_width_ext_cm": MIN_WIDTH_EXT_CM,
        "image_dpi": IMAGE_DPI,
        "lane_max_tokens": LANE_MAX_TOKENS,
        "refine_max_tokens": REFINE_MAX_TOKENS,
        "max_refine_rounds": MAX_REFINE_ROUNDS,
        "lane_crop_and_inspect": lane.get("crop_and_inspect", LANE_CROP_AND_INSPECT),
        "refine_crop_and_inspect": refine.get("crop_and_inspect", REFINE_CROP_AND_INSPECT),
        "lane_max_inspect_rounds": lane.get("max_inspect_rounds", LANE_MAX_INSPECT_ROUNDS),
        "refine_max_inspect_rounds": refine.get("max_inspect_rounds", REFINE_MAX_INSPECT_ROUNDS),
        "run_esmini": RUN_ESMINI,
        "esmini_path": ESMINI_PATH,
        "spawn_vehicles": SPAWN_VEHICLES,
        "spawn_duration": SPAWN_DURATION,
    }


def list_case_ids_with_arms_json(dataset):
    """Return case ids under ``dataset/<dataset>/`` that contain ``arms.json``."""
    root = DATASET_ROOT / dataset
    if not root.is_dir():
        return []
    case_ids = []
    for case_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not case_dir.is_dir() or case_dir.name.startswith("."):
            continue
        if (case_dir / "arms.json").is_file():
            case_ids.append(case_dir.name)
    return case_ids


def select_case_ids(case_ids, count=None, seed=0):
    """Return ``count`` case ids sampled without replacement; ``None``/0 keeps all.

    Selection is reproducible given ``seed``. Sampled ids keep the original sorted order.
    """
    case_ids = list(case_ids)
    if count is None or int(count) <= 0 or int(count) >= len(case_ids):
        return case_ids
    chosen = set(random.Random(seed).sample(case_ids, int(count)))
    return [case_id for case_id in case_ids if case_id in chosen]


def _model_cfg(config, *, role="arm"):
    prefix = {"junction": "junction", "arm": "arm", "lane": "lane", "refine": "refine"}.get(role)
    if prefix is None:
        raise ValueError("unknown model role: %r" % role)
    return {
        "api": config["%s_api" % prefix],
        "provider": config["%s_provider" % prefix],
        "model": config["%s_model" % prefix],
        "reasoning": config["%s_reasoning" % prefix],
    }


def _run_dir_for(config, dataset, case_id):
    _image_path, run_name = resolve_case_inputs(dataset=dataset, case_id=case_id)
    run_dir = resolve_run_dir(config["output_dir"], run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_name


def list_junction_work_items(case_dir, junctions_doc=None):
    """``(junction_id, work_dir)`` pairs for arm / lane / refine.

    Prefer ids from ``junctions_doc``. If that document is absent, scan
    ``{case_dir}/junctions/``. If neither exists, return ``[(None, None)]``;
    the skip-junction path then creates ``junctions/1`` from the full image.
    """
    case_dir = Path(case_dir)
    if junctions_doc is not None:
        return [
            (int(junction["id"]), junction_work_dir(case_dir, int(junction["id"])))
            for junction in (junctions_doc.get("junctions") or [])
        ]
    items = []
    for path in list_junction_artifact_roots(case_dir):
        if path.resolve() == case_dir.resolve():
            continue
        try:
            items.append((int(path.name), path))
        except ValueError:
            continue
    return items if items else [(None, None)]


def _full_image_box():
    return [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def _full_image_junctions_doc(work_items):
    """Minimal junctions.json so assemble_scene can promote a skip-junction run."""
    return {
        "junctions": [
            {"id": int(jid), "box": _full_image_box()}
            for jid, _work_dir in work_items
            if jid is not None
        ]
    }


def _ensure_junction_crop_images(config, dataset, case_id, case_dir, work_items, junctions_doc=None):
    """Write per-junction images. Independent of skip_junction_level LLM.

    With no ``junctions.json``, the processed full scene is written only under
    ``junctions/{id}/images`` (no case-root ``images/``). When boxes exist,
    the full scene is processed at case root and then cropped per junction.
    """
    from scripts.methods.image_processing import load_processed_image, process_case_image
    from scripts.methods.llm_kg.junction_level.llm_gen import write_junction_crop_images

    boxes = {}
    for junction in (junctions_doc or {}).get("junctions") or []:
        jid = int(junction["id"])
        boxes[jid] = junction.get("box") or _full_image_box()

    use_full_scene_in_junction = not boxes
    if not use_full_scene_in_junction:
        stage_image_processing(config, dataset, case_id)
        processed = load_processed_image(case_dir, compressed=False)

    ensured = []
    for jid, work_dir in work_items:
        if jid is None:
            jid = 1
            work_dir = junction_work_dir(case_dir, jid)
        work_dir = Path(work_dir)
        images_dir = work_dir / "images"
        if use_full_scene_in_junction:
            process_case_image(
                dataset=dataset,
                case_id=case_id,
                output_dir=config["output_dir"],
                max_edge=config.get("max_edge"),
                images_dir=images_dir,
            )
        else:
            write_junction_crop_images(
                processed["image"],
                boxes.get(int(jid), _full_image_box()),
                images_dir,
                stem="junction_%s" % jid,
            )
        ensured.append((int(jid), work_dir))
    return ensured


def stage_image_processing(config, dataset, case_id):
    from scripts.methods.image_processing import process_case_image

    return process_case_image(
        dataset=dataset,
        case_id=case_id,
        output_dir=config["output_dir"],
        max_edge=config["max_edge"],
    )


def stage_junction_level(config, dataset, case_id):
    from scripts.methods.llm_kg.junction_level.llm_gen import run_junction_level

    junction_model = _model_cfg(config, role="junction")
    run_dir, _run_name = _run_dir_for(config, dataset, case_id)

    if config["skip_junction_level"]:
        junctions_path = junctions_path_for_model(run_dir, config["junction_model"], config["junction_reasoning"])
        BaseLLM.rule("JUNCTION-LEVEL SKIPPED")
        BaseLLM.log("loaded: %s" % junctions_path)
        return json.loads(junctions_path.read_text(encoding="utf-8")), junctions_path

    stage_image_processing(config, dataset, case_id)
    artifact = run_junction_level(
        junction_model,
        dataset=dataset,
        case_id=case_id,
        max_tokens=config["junction_max_tokens"],
        compressed=config.get("use_compressed_image", True),
        verbose=config["verbose"],
        output_dir=config["output_dir"],
    )
    junctions_path = junctions_path_for_model(run_dir, config["junction_model"], config["junction_reasoning"])
    return artifact["result"], junctions_path


def stage_arm_level(config, dataset, case_id, *, artifact_case_dir=None):
    from scripts.methods.llm_kg.arm_level.llm_gen import run_arm_level

    arm_model = _model_cfg(config, role="arm")
    run_dir, _run_name = _run_dir_for(config, dataset, case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else run_dir

    if config["skip_arm_level"]:
        arms_path = arms_path_for_model(case_dir, config["arm_model"], config["arm_reasoning"])
        BaseLLM.rule("ARM-LEVEL SKIPPED")
        BaseLLM.log("loaded: %s" % arms_path)
        return json.loads(arms_path.read_text(encoding="utf-8")), arms_path

    if artifact_case_dir is None:
        stage_image_processing(config, dataset, case_id)
    artifact = run_arm_level(
        arm_model,
        dataset=dataset,
        case_id=case_id,
        max_tokens=config["arm_max_tokens"],
        compressed=config.get("use_compressed_image", True),
        verbose=config["verbose"],
        width_ext_frac=config.get("width_ext_frac", WIDTH_EXT_FRAC),
        stage1_width_frac=config.get("stage1_width_frac", STAGE1_WIDTH_FRAC),
        min_width_ext_cm=config.get("min_width_ext_cm", MIN_WIDTH_EXT_CM),
        image_dpi=config.get("image_dpi", IMAGE_DPI),
        artifact_case_dir=artifact_case_dir,
    )
    arms_path = arms_path_for_model(case_dir, config["arm_model"], config["arm_reasoning"])
    return artifact["result"], arms_path


def stage_lane_level(config, dataset, case_id, *, artifact_case_dir=None):
    """Generate lane-level cross-sections only (no refine / no xodr emit)."""
    from scripts.methods.llm_kg.lane_level.llm_gen import run_lanes

    lane_model = _model_cfg(config, role="lane")
    run_dir, _run_name = _run_dir_for(config, dataset, case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else run_dir

    if config["skip_lane_level"]:
        lanes_path = lanes_path_for_model(case_dir, config["lane_model"], config["lane_reasoning"])
        BaseLLM.rule("LANE-LEVEL SKIPPED")
        BaseLLM.log("loaded: %s" % lanes_path)
        return lanes_path

    out = run_lanes(
        lane_model,
        dataset=dataset,
        case_id=case_id,
        arms_from=config["arm_model"],
        arms_reasoning=config["arm_reasoning"],
        skip_generation=False,
        max_tokens=config["lane_max_tokens"],
        crop_and_inspect=config["lane_crop_and_inspect"],
        max_inspect_rounds=config["lane_max_inspect_rounds"],
        compressed=config.get("use_compressed_image", True),
        verbose=config["verbose"],
        artifact_case_dir=artifact_case_dir,
        width_ext_frac=config.get("width_ext_frac", WIDTH_EXT_FRAC),
        min_width_ext_cm=config.get("min_width_ext_cm", MIN_WIDTH_EXT_CM),
        image_dpi=config.get("image_dpi", IMAGE_DPI),
        arm_retry_limit=CASE_STAGE_RETRY_LIMIT,
    )
    return out["lanes_path"]


def _lane_results_from_doc(lane_doc):
    return {
        int(arm["arm_id"]): arm
        for arm in (lane_doc.get("arms") or [])
        if arm.get("arm_id") is not None
    }


def _lanes_doc_with_results(lane_doc, lane_results, *, driving_side=None):
    doc = dict(lane_doc)
    doc["arms"] = [lane_results[arm_id] for arm_id in sorted(lane_results)]
    if driving_side is not None:
        doc["driving_side"] = driving_side
    return doc


def _load_lanes_doc(paths, *, error):
    for path in paths:
        if not path.is_file():
            continue
        lane_doc = json.loads(path.read_text(encoding="utf-8"))
        if lane_doc.get("arms"):
            return lane_doc, path
    raise FileNotFoundError(error)


def _load_skip_refine_lanes(case_dir, config):
    """Pre-solver lanes: layer_2, then layer_1, then lane-level."""
    ref_root = refinement_dir(case_dir, config["refine_model"], config["refine_reasoning"])
    lane_doc, path = _load_lanes_doc(
        [
            ref_root / "layer_2" / "lanes.json",
            ref_root / "layer_1" / "lanes.json",
            lane_level_dir(case_dir, config["lane_model"], config["lane_reasoning"]) / "lanes.json",
        ],
        error="skip_refinement needs layer_2, layer_1, or lane-level lanes.json under %s" % case_dir,
    )
    return lane_doc, path, ref_root


def _load_existing_refine_lanes(case_dir, config):
    """Refinement ``lanes.json`` used when the solver is skipped."""
    ref_root = refinement_dir(case_dir, config["refine_model"], config["refine_reasoning"])
    path = ref_root / "lanes.json"
    lane_doc, path = _load_lanes_doc(
        [path],
        error="skip_fea_solver needs existing refinement lanes.json: %s" % path,
    )
    return lane_doc, path, ref_root


def _stage_refinement_skip_llm(config, dataset, case_id, *, case_dir, run_name, xodr_dir):
    """Skip L1/L2 LLM. Reuse refinement/lanes.json when skip_fea_solver, else run the solver."""
    from scripts.methods.llm_kg.refinement.feasibility_rank import resolve_junction_image_path
    from scripts.methods.llm_kg.refinement.layer2_con_fix import run_feasibility_repair
    from scripts.methods.llm_kg.refinement.llm_refine import emit_xodr_artifacts
    from scripts.methods.llm_utils import CostTracker, build_llm
    from scripts.methods.method_utils import unpack_model_config

    BaseLLM.rule("REFINEMENT SKIPPED (LLM)")
    arms = json.loads(
        arms_path_for_model(case_dir, config["arm_model"], config["arm_reasoning"]).read_text(encoding="utf-8")
    )["arms"]
    side = load_case_driving_side(dataset, case_id)

    def _emit(doc):
        artifacts = emit_xodr_artifacts(
            xodr_dir,
            arms=arms,
            lane_result=doc,
            name=run_name,
            channelized_features=doc.get("channelized_features"),
            through_roads=doc.get("through_roads"),
            refine_rounds=0,
            driving_side=side,
        )
        if not artifacts["accepted"]:
            raise RuntimeError("did not produce valid connections: %s" % (xodr_dir / "validation_error.json"))
        return connections_path_for_model(case_dir, config["refine_model"], config["refine_reasoning"])

    if config.get("skip_fea_solver"):
        lane_doc, source_path, ref_root = _load_existing_refine_lanes(case_dir, config)
        lane_doc = dict(lane_doc)
        lane_doc["driving_side"] = lane_doc.get("driving_side") or side
        BaseLLM.rule("FEASIBILITY SOLVER SKIPPED")
        BaseLLM.log("loaded existing lanes.json: %s" % source_path)
        return _emit(lane_doc)

    lane_doc, source_path, ref_root = _load_skip_refine_lanes(case_dir, config)
    lane_doc = dict(lane_doc)
    lane_doc["driving_side"] = lane_doc.get("driving_side") or side
    BaseLLM.log("feasibility_solver input: %s" % source_path)

    if _lanes_connection_valid({"arms": arms}, lane_doc, run_name):
        BaseLLM.log("input already valid; skip feasibility_solver")
        dump_json(ref_root / "lanes.json", lane_doc)
        return _emit(lane_doc)

    lane_results = _lane_results_from_doc(lane_doc)

    refine_cfg = unpack_model_config(_model_cfg(config, role="refine"))
    llm = build_llm(
        refine_cfg["api"], refine_cfg["api_key"], refine_cfg["base_url"],
        proxy=refine_cfg["proxy"],
    )
    tracker = CostTracker(source=refine_cfg["source"], silent=True)
    image_path = resolve_junction_image_path(
        source_lanes=lane_doc,
        arms_path=arms_path_for_model(case_dir, config["arm_model"], config["arm_reasoning"]),
    )
    repair = run_feasibility_repair(
        lane_results,
        arms=arms,
        channelized_features=lane_doc.get("channelized_features"),
        through_roads=lane_doc.get("through_roads"),
        name=run_name,
        driving_side=side,
        refinement_root=ref_root,
        source_lanes=lane_doc,
        skip_noop=True,
        llm=llm,
        model=refine_cfg["model_name"],
        reasoning=refine_cfg["reasoning"],
        max_tokens=min(int(config.get("refine_max_tokens") or 2048), 4096),
        image_path=image_path,
        tracker=tracker,
    )
    lane_doc = _lanes_doc_with_results(lane_doc, lane_results, driving_side=side)
    dump_json(ref_root / "lanes.json", lane_doc)
    if repair.ok and repair.edits:
        BaseLLM.log("feasibility_solver applied -> %s" % (ref_root / "lanes.json"))
    else:
        BaseLLM.log("feasibility_solver failed -> %s" % (ref_root / "lanes.json"))
    return _emit(lane_doc)


def stage_refinement(config, dataset, case_id, *, artifact_case_dir=None):
    """Refine lanes and write xodr artifacts under refinement/.

    ``skip_refinement`` skips the L1/L2 loop. ``skip_fea_solver`` reuses
    ``refinement/lanes.json``. Otherwise the solver reads layer_2, then layer_1,
    then lane-level lanes, and leaves a connection-valid snapshot unchanged.
    """
    from scripts.methods.llm_kg.refinement.llm_refine import run_refine_from_lanes

    refine_model = _model_cfg(config, role="refine")
    run_dir, run_name = _run_dir_for(config, dataset, case_id)
    case_dir = Path(artifact_case_dir) if artifact_case_dir is not None else run_dir
    xodr_dir = xodr_result_dir_for_model(case_dir, config["refine_model"], config["refine_reasoning"])

    if config["skip_refinement"]:
        return _stage_refinement_skip_llm(
            config, dataset, case_id,
            case_dir=case_dir, run_name=run_name, xodr_dir=xodr_dir,
        )

    refine_out = run_refine_from_lanes(
        refine_model,
        dataset=dataset,
        case_id=case_id,
        arms_from=config["arm_model"],
        arms_reasoning=config["arm_reasoning"],
        lanes_from=config["lane_model"],
        lanes_reasoning=config["lane_reasoning"],
        max_refine_rounds=config["max_refine_rounds"],
        crop_and_inspect=config["refine_crop_and_inspect"],
        max_inspect_rounds=config["refine_max_inspect_rounds"],
        max_tokens=config["refine_max_tokens"],
        compressed=config.get("use_compressed_image", True),
        verbose=config["verbose"],
        skip_fea_solver=config.get("skip_fea_solver", False),
        artifact_case_dir=artifact_case_dir,
        width_ext_frac=config.get("width_ext_frac", WIDTH_EXT_FRAC),
        min_width_ext_cm=config.get("min_width_ext_cm", MIN_WIDTH_EXT_CM),
        image_dpi=config.get("image_dpi", IMAGE_DPI),
    )
    ref_conns = Path(refine_out["xodr_dir"]) / "connections.json"
    if not ref_conns.is_file() or not (refine_out.get("refine_meta") or {}).get("connections_validated"):
        raise RuntimeError(
            "refinement did not produce valid connections: %s"
            % (Path(refine_out["xodr_dir"]) / "validation_error.json")
        )
    return ref_conns


def stage_scene_feasibility(config, dataset, case_id, junctions_doc, work_items):
    """Joint min-edit so connector mouths stay consistent. No-op for one junction."""
    from scripts.methods.llm_kg.refinement.scene_feasibility import run_scene_feasibility

    run_dir, run_name = _run_dir_for(config, dataset, case_id)
    junctions = list((junctions_doc or {}).get("junctions") or [])
    if len(junctions) <= 1:
        BaseLLM.rule("SCENE FEASIBILITY SKIPPED")
        BaseLLM.log("junctions=%s" % len(junctions))
        return {"passed": True, "skipped": True, "reason": "single_junction"}
    if config.get("skip_fea_solver"):
        BaseLLM.rule("SCENE FEASIBILITY SKIPPED")
        BaseLLM.log("skip_fea_solver=True")
        return {"passed": True, "skipped": True, "reason": "skip_fea_solver"}

    return run_scene_feasibility(
        config, dataset, case_id, junctions_doc, work_items,
        case_dir=run_dir, run_name=run_name,
    )


def stage_layer3_check(config, dataset, case_id, junctions_doc, work_items):
    """Scene-level HARD inter-junction mouth consistency. No-op unless more than one junction."""
    from scripts.methods.llm_kg.refinement.llm_refine import run_scene_layer3

    run_dir, _run_name = _run_dir_for(config, dataset, case_id)
    junctions = list((junctions_doc or {}).get("junctions") or [])
    if config["skip_refinement"] or len(junctions) <= 1:
        BaseLLM.rule("LAYER-3 SKIPPED")
        BaseLLM.log("skip refinement=%s junctions=%s" % (config["skip_refinement"], len(junctions)))
        return {"passed": True, "skipped": True, "refine_rounds": 0}

    return run_scene_layer3(
        _model_cfg(config, role="refine"),
        dataset=dataset,
        case_id=case_id,
        junctions_doc=junctions_doc,
        work_items=work_items,
        arms_from=config["arm_model"],
        arms_reasoning=config["arm_reasoning"],
        lanes_from=config["lane_model"],
        lanes_reasoning=config["lane_reasoning"],
        max_refine_rounds=config["max_refine_rounds"],
        crop_and_inspect=config["refine_crop_and_inspect"],
        max_inspect_rounds=config["refine_max_inspect_rounds"],
        max_tokens=config["refine_max_tokens"],
        compressed=config.get("use_compressed_image", True),
        scene_case_dir=run_dir,
        width_ext_frac=config.get("width_ext_frac", WIDTH_EXT_FRAC),
        min_width_ext_cm=config.get("min_width_ext_cm", MIN_WIDTH_EXT_CM),
        image_dpi=config.get("image_dpi", IMAGE_DPI),
    )


def stage_assemble_scene(config, dataset, case_id, junctions_doc):
    """Derive inter-junction roads from boxes + arm headings; merge connections.json."""
    from scripts.methods.llm_kg.junction_level.llm_gen import assemble_scene_connections

    run_dir, _run_name = _run_dir_for(config, dataset, case_id)
    assembled = assemble_scene_connections(
        run_dir,
        junctions_doc,
        junction_model=config["junction_model"],
        junction_reasoning=config["junction_reasoning"],
        arm_model=config["arm_model"],
        arm_reasoning=config["arm_reasoning"],
        refine_model=config["refine_model"],
        refine_reasoning=config["refine_reasoning"],
        driving_side=load_case_driving_side(dataset, case_id),
    )
    return assembled["connections_path"]


def stage_generate_roads(config, dataset, case_id, connections_path=None):
    from scripts.methods.generate_roads import run_generate_roads

    run_dir, _run_name = _run_dir_for(config, dataset, case_id)
    if connections_path is not None:
        skip_xodr = Path(connections_path).parent / "scenario.xodr"
    else:
        skip_xodr = scenario_xodr_path_for_model(run_dir, config["refine_model"], config["refine_reasoning"])
    if config["skip_generate_roads"]:
        BaseLLM.rule("GENERATE ROADS SKIPPED")
        BaseLLM.log("loaded: %s" % skip_xodr)
        return skip_xodr

    BaseLLM.rule("GENERATE ROADS")
    out = run_generate_roads(
        dataset=dataset,
        case_id=case_id,
        output_dir=config["output_dir"],
        model=config["refine_model"],
        reasoning=config["refine_reasoning"],
        run_esmini=config["run_esmini"],
        esmini_path=config["esmini_path"],
        verbose=config["verbose"],
        connections_path=connections_path,
    )
    return out["xodr_path"]


def stage_carla_validate(config, dataset, case_id):
    from scripts.methods.carla_validate import run_carla_validate_subprocess

    if config["skip_carla_validate"]:
        BaseLLM.rule("CARLA VALIDATE SKIPPED")
        return {"passed": None, "skipped": True}

    BaseLLM.rule("CARLA VALIDATE")
    report = run_carla_validate_subprocess(
        dataset=dataset,
        case_id=case_id,
        output_dir=config["output_dir"],
        model=config["refine_model"],
        reasoning=config["refine_reasoning"],
        spawn_vehicles=config["spawn_vehicles"],
        spawn_duration=config["spawn_duration"],
        verbose=config["verbose"],
    )
    if report.get("crashed"):
        BaseLLM.log("CARLA validation CRASHED (subprocess returncode=%s)" % report.get("subprocess_returncode"))
    elif report.get("passed"):
        BaseLLM.log("CARLA validation PASSED")
    else:
        BaseLLM.log("CARLA validation FAILED")
    report["skipped"] = False
    return report


def _path_str(path):
    return str(path) if path is not None else None


def run_one_case(config, dataset, case_id):
    case_dir, run_name = _run_dir_for(config, dataset, case_id)
    model_dir = lane_level_dir(case_dir, config["lane_model"], config["lane_reasoning"])
    refine_dir = refinement_dir(case_dir, config["refine_model"], config["refine_reasoning"])
    xodr_dir = xodr_result_dir_for_model(case_dir, config["refine_model"], config["refine_reasoning"])
    t_case = time.perf_counter()

    BaseLLM.rule("CASE %s" % run_name)
    BaseLLM.log("case_dir: %s" % case_dir)
    BaseLLM.log("models: junction=%s/%s arm=%s/%s lane=%s/%s refine=%s/%s" % (
        config["junction_model"], config["junction_reasoning"],
        config["arm_model"], config["arm_reasoning"],
        config["lane_model"], config["lane_reasoning"],
        config["refine_model"], config["refine_reasoning"],
    ))
    BaseLLM.log(
        "skip: junction=%s arm=%s lane=%s "
        "refine=%s fea_solver=%s roads=%s carla=%s"
        % (
            config["skip_junction_level"],
            config["skip_arm_level"],
            config["skip_lane_level"],
            config["skip_refinement"],
            config.get("skip_fea_solver", False),
            config["skip_generate_roads"],
            config["skip_carla_validate"],
        )
    )
    BaseLLM.log(
        "crop-and-inspect: lane=%s refine=%s"
        % (
            config["lane_crop_and_inspect"],
            config["refine_crop_and_inspect"],
        )
    )

    junctions_doc = None
    junctions_path = None
    work_items = []
    arm_result = None
    arms_path = None
    lanes_path = None
    connections_path = None
    xodr_path = None
    carla_report = {"passed": None, "skipped": True}
    metrics = None
    error = None
    failed_stage = None

    def _run_stage(stage_name, stage_fn, retry_limit=None):
        nonlocal error, failed_stage
        failed_stage = stage_name
        limit = CASE_STAGE_RETRY_LIMIT if retry_limit is None else retry_limit
        attempts = max(0, int(limit)) + 1
        for attempt in range(attempts):
            try:
                result = stage_fn()
                error = None
                return result
            except FileNotFoundError as exc:
                error = "%s: %s" % (stage_name, exc)
                print("[error] dataset=%s case_id=%s %s" % (dataset, case_id, error), flush=True)
                return None
            except Exception as exc:
                if attempt < CASE_STAGE_RETRY_LIMIT:
                    print(
                        "[retry] dataset=%s case_id=%s %s attempt %s/%s: %s"
                        % (dataset, case_id, stage_name, attempt + 2, attempts, exc),
                        flush=True,
                    )
                    continue
                error = "%s: %s" % (stage_name, exc)
                print("[error] dataset=%s case_id=%s %s" % (dataset, case_id, error), flush=True)
                return None
        return None

    def _summary_roots():
        roots = [Path(work_dir) for _jid, work_dir in work_items if work_dir is not None]
        return roots or [junction_work_dir(case_dir, 1)]

    def _write_abort_record(work_dir, stage_error, stage_name):
        """Record a flow-stopping error immediately at the artifact-root summary path."""
        root = Path(work_dir) if work_dir is not None else junction_work_dir(case_dir, 1)
        dump_json(pipeline_summary_path(root), {
            "run_name": run_name,
            "dataset": dataset,
            "case_id": str(case_id),
            "case_dir": str(case_dir),
            "error": stage_error,
            "failed_stage": stage_name,
            "fail_reason": stage_error,
            "completed_at": format_completed_at(),
        })

    def _persist_pipeline_summary():
        nonlocal refine_dir, xodr_dir
        elapsed_sec = round(time.perf_counter() - t_case, 3)
        completed_at = format_completed_at()
        if connections_path is not None:
            xodr_dir = Path(connections_path).parent
            refine_dir = xodr_dir.parent
        cost = {
            "cost_usd": None,
            "junction_level_usd": None,
            "arm_level_usd": None,
            "lane_level_usd": None,
            "refine_level_usd": None,
        }
        try:
            cost = collect_case_cost(
                case_dir,
                config["arm_model"],
                config["lane_model"],
                config["refine_model"],
                arm_reasoning=config["arm_reasoning"],
                lane_reasoning=config["lane_reasoning"],
                refine_reasoning=config["refine_reasoning"],
                junction_model=config["junction_model"],
                junction_reasoning=config["junction_reasoning"],
            )
        except Exception as exc:
            print("[error] dataset=%s case_id=%s pipeline_summary cost: %s" % (dataset, case_id, exc), flush=True)
        summary = {
            "run_name": run_name,
            "dataset": dataset,
            "case_id": str(case_id),
            "case_dir": str(case_dir),
            "model_dir": str(model_dir),
            "refine_dir": str(refine_dir),
            "junction_model": config["junction_model"],
            "junction_reasoning": config["junction_reasoning"],
            "arm_model": config["arm_model"],
            "arm_reasoning": config["arm_reasoning"],
            "lane_model": config["lane_model"],
            "lane_reasoning": config["lane_reasoning"],
            "refine_model": config["refine_model"],
            "refine_reasoning": config["refine_reasoning"],
            "junctions_path": _path_str(junctions_path),
            "junction_count": len((junctions_doc or {}).get("junctions") or []),
            "arms_path": _path_str(arms_path),
            "lanes_path": _path_str(lanes_path),
            "connections_path": _path_str(connections_path),
            "xodr_dir": str(xodr_dir),
            "xodr_path": _path_str(xodr_path),
            "carla_passed": carla_report.get("passed"),
            "carla_crashed": bool(carla_report.get("crashed")),
            "carla_skipped": bool(carla_report.get("skipped")),
            "arm_count": len((arm_result or {}).get("arms") or []),
            "elapsed_sec": elapsed_sec,
            "completed_at": completed_at,
            "cost_usd": cost["cost_usd"],
            "junction_level_usd": cost["junction_level_usd"],
            "arm_level_usd": cost["arm_level_usd"],
            "lane_level_usd": cost["lane_level_usd"],
            "refine_level_usd": cost["refine_level_usd"],
            "metrics": metrics,
            "error": error,
            "failed_stage": failed_stage,
            "fail_reason": error,
        }
        try:
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
                pipeline_error=error,
                failed_stage=failed_stage,
            )
            summary.update({
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
                "fail_reason": validation.get("fail_reason") or error,
            })
        except Exception as exc:
            print("[error] dataset=%s case_id=%s pipeline_summary validation: %s" % (dataset, case_id, exc), flush=True)
            if not summary.get("fail_reason"):
                summary["fail_reason"] = error or str(exc)
        for root in _summary_roots():
            dump_json(pipeline_summary_path(root), summary)
        BaseLLM.log("completed_at: %s" % completed_at)
        return summary

    if config["skip_junction_level"]:
        try:
            junctions_path = junctions_path_for_model(
                case_dir, config["junction_model"], config["junction_reasoning"],
            )
            junctions_doc = json.loads(junctions_path.read_text(encoding="utf-8"))
            BaseLLM.rule("JUNCTION-LEVEL SKIPPED")
            BaseLLM.log("loaded: %s" % junctions_path)
        except FileNotFoundError:
            junctions_path = None
            junctions_doc = None
            BaseLLM.rule("JUNCTION-LEVEL SKIPPED")
            BaseLLM.log("junction_level artifacts not found")
        work_items = list_junction_work_items(case_dir, junctions_doc)
        if junctions_doc is not None and not (junctions_doc.get("junctions") or []):
            error = "junction_level: junctions.json has no junctions"
            failed_stage = "junction_level"
            work_items = []
            _write_abort_record(junction_work_dir(case_dir, 1), error, failed_stage)
        else:
            if work_items == [(None, None)]:
                work_dir = junction_work_dir(case_dir, 1)
                work_dir.mkdir(parents=True, exist_ok=True)
                work_items = [(1, work_dir)]
            ensured = _run_stage(
                "junction_images",
                lambda items=work_items: _ensure_junction_crop_images(
                    config, dataset, case_id, case_dir, items, junctions_doc,
                ),
            )
            if error is None:
                work_items = ensured
                if junctions_doc is None:
                    junctions_doc = _full_image_junctions_doc(work_items)
            else:
                work_items = []
                _write_abort_record(junction_work_dir(case_dir, 1), error, failed_stage)
    else:
        result = _run_stage("junction_level", lambda: stage_junction_level(config, dataset, case_id))
        if error is None:
            junctions_doc, junctions_path = result
            work_items = list_junction_work_items(case_dir, junctions_doc)
            BaseLLM.log("junctions: %s" % ",".join(str(item[0]) for item in work_items))
        else:
            work_items = []
            _write_abort_record(junction_work_dir(case_dir, 1), error, failed_stage)

    junction_failures = []
    for jid, work_dir in work_items:
        stage_tag = "j%s/" % jid if jid is not None else ""
        if jid is not None:
            BaseLLM.rule("JUNCTION %s" % jid)
            BaseLLM.log("work_dir: %s" % work_dir)
        result = _run_stage("%sarm_level" % stage_tag, lambda d=work_dir: stage_arm_level(
            config, dataset, case_id, artifact_case_dir=d,
        ))
        if error is not None:
            junction_failures.append((failed_stage, error))
            _write_abort_record(work_dir, error, failed_stage)
            error = None
            failed_stage = None
            continue
        arm_result, arms_path = result
        result = _run_stage(
            "%slane_level" % stage_tag,
            lambda d=work_dir: stage_lane_level(
                config, dataset, case_id, artifact_case_dir=d,
            ),
            retry_limit=0,
        )
        if error is not None:
            junction_failures.append((failed_stage, error))
            _write_abort_record(work_dir, error, failed_stage)
            error = None
            failed_stage = None
            continue
        lanes_path = result
        result = _run_stage("%srefinement" % stage_tag, lambda d=work_dir: stage_refinement(
            config, dataset, case_id, artifact_case_dir=d,
        ))
        if error is not None:
            junction_failures.append((failed_stage, error))
            _write_abort_record(work_dir, error, failed_stage)
            error = None
            failed_stage = None
            continue
        connections_path = result

    if junction_failures:
        failed_stage = junction_failures[0][0]
        error = "; ".join(item[1] for item in junction_failures)

    if error is None and junctions_doc is not None and len((junctions_doc.get("junctions") or [])) > 1:
        _run_stage("layer3", lambda: stage_layer3_check(
            config, dataset, case_id, junctions_doc, work_items,
        ))

    if error is None and junctions_doc is not None and len((junctions_doc.get("junctions") or [])) > 1:
        _run_stage("scene_feasibility", lambda: stage_scene_feasibility(
            config, dataset, case_id, junctions_doc, work_items,
        ))

    if error is None and junctions_doc is not None:
        result = _run_stage("assemble_scene", lambda: stage_assemble_scene(
            config, dataset, case_id, junctions_doc,
        ))
        if error is None:
            connections_path = result

    if error is None:
        result = _run_stage("generate_roads", lambda: stage_generate_roads(
            config, dataset, case_id, connections_path=connections_path,
        ))
        if error is None:
            xodr_path = result
    if error is None:
        result = _run_stage("carla_validate", lambda: stage_carla_validate(config, dataset, case_id))
        if error is None:
            carla_report = result
    if error is None:
        failed_stage = None
    else:
        for _jid, work_dir in work_items or [(None, None)]:
            _write_abort_record(work_dir, error, failed_stage)

    # Always evaluate available arm / lane / refine artifacts, including failed cases.
    try:
        metrics = stage_evaluation(config, dataset, case_id, case_dir)
    except Exception as exc:
        eval_error = "evaluation: %s" % exc
        print("[error] dataset=%s case_id=%s %s" % (dataset, case_id, eval_error), flush=True)
        if error is None:
            error = eval_error
            failed_stage = "evaluation"
            for _jid, work_dir in work_items or [(None, None)]:
                _write_abort_record(work_dir, error, failed_stage)

    return _persist_pipeline_summary()


DATASETS = ["openaerialmap"]

MODELS = [{
        "junction-level": {
            "api": "openai",
            "provider": "openrouter",
            "model": "openai/gpt-5.6-sol",
            "reasoning": "medium",
        },
        "arm-level": {
            "api": "openai",
            "provider": "openrouter",
            "model": "openai/gpt-5.6-sol",
            "reasoning": "medium",
        },
        "lane-level": {
            "api": "openai",
            "provider": "openrouter",
            "model": "openai/gpt-5.6-sol",
            "reasoning": "medium",
        },
        "refinement": {
            "api": "openai",
            "provider": "openrouter",
            "model": "openai/gpt-5.6-sol",
            "reasoning": "medium",
        },
    }
]


def _run_case_job(run_models, dataset, case_id):
    config = make_config(run_models)
    print(
        "[run] dataset=%s case_id=%s arm=%s lane=%s refine=%s"
        % (
            dataset,
            case_id,
            config["arm_model"],
            config["lane_model"],
            config["refine_model"],
        ),
        flush=True,
    )
    return run_one_case(config, dataset, case_id)


def _report_finished_case(summary, dataset, case_id, dataset_case_ids, summaries):
    summaries.append(summary)
    if summary.get("error"):
        print(
            "[done] success=False dataset=%s case_id=%s stage=%s error=%s elapsed=%.3fs cost=%s"
            % (
                dataset,
                case_id,
                summary.get("failed_stage"),
                summary["error"],
                summary["elapsed_sec"],
                summary.get("cost_usd"),
            ),
            flush=True,
        )
    else:
        print(
            "[done] success=True dataset=%s case_id=%s error=None elapsed=%.3fs cost=%s"
            % (dataset, case_id, summary["elapsed_sec"], summary.get("cost_usd")),
            flush=True,
        )
    print(format_case_stage_metrics(case_stage_row(
        pipeline_item_from_summary(summary, output_dir=OUTPUT_DIR),
    )), flush=True)
    refresh_case_metrics_csv(
        OUTPUT_DIR, dataset_case_ids, MODELS, quiet=True,
    )


def _collect_case_id_list():
    """Return (dataset_case_ids, case_id_list) before any case starts."""
    dataset_case_ids = {}
    case_id_list = []
    for dataset in DATASETS:
        available_ids = list_case_ids_with_arms_json(dataset)
        case_ids = select_case_ids(available_ids, RANDOM_CASE_COUNT, RANDOM_CASE_SEED)
        dataset_case_ids[dataset] = available_ids
        BaseLLM.rule("DATASET %s" % dataset)
        if RANDOM_CASE_COUNT:
            BaseLLM.log(
                "cases with arms.json: %d  selected: %d (seed=%s)  ids=%s"
                % (len(available_ids), len(case_ids), RANDOM_CASE_SEED, ",".join(case_ids) or "(none)")
            )
        else:
            BaseLLM.log("cases with arms.json: %d" % len(case_ids))
        for case_id in case_ids:
            case_id_list.append((dataset, case_id))
    return dataset_case_ids, case_id_list


def _run_case_id_list(case_id_list, dataset_case_ids, summaries, *, workers):
    jobs = [
        (run_models, dataset, case_id)
        for dataset, case_id in case_id_list
        for run_models in MODELS
    ]
    if workers <= 1 or len(jobs) <= 1:
        for run_models, dataset, case_id in jobs:
            summary = _run_case_job(run_models, dataset, case_id)
            _report_finished_case(summary, dataset, case_id, dataset_case_ids, summaries)
        return
    BaseLLM.log("parallel cases: %d workers" % workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_run_case_job, run_models, dataset, case_id): (dataset, case_id)
            for run_models, dataset, case_id in jobs
        }
        for future in as_completed(futures):
            dataset, case_id = futures[future]
            try:
                summary = future.result()
            except Exception as exc:
                print(
                    "[error] dataset=%s case_id=%s worker: %s" % (dataset, case_id, exc),
                    flush=True,
                )
                continue
            _report_finished_case(summary, dataset, case_id, dataset_case_ids, summaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Batch lane-level / refinement pipeline.")
    parser.add_argument(
        "--parallel", type=int, default=CASE_PARALLELISM,
        help="How many cases to run at once (default: CASE_PARALLELISM=%s)" % CASE_PARALLELISM,
    )
    args = parser.parse_args(argv)
    workers = max(1, int(args.parallel))

    set_verbose(VERBOSE)
    summaries = []
    dataset_case_ids, case_id_list = _collect_case_id_list()
    BaseLLM.rule("CASE LIST")
    BaseLLM.log("n=%d  parallel=%d" % (len(case_id_list), workers))
    if case_id_list:
        _run_case_id_list(case_id_list, dataset_case_ids, summaries, workers=workers)

    evaluate_outputs(
        OUTPUT_DIR, dataset_case_ids, MODELS, reevaluate=False, quiet=False,
    )


if __name__ == "__main__":
    main()
