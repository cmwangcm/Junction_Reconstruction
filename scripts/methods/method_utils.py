"""Shared helpers for method entrypoints: paths, JSON I/O, case/model artifacts.

Result layout::

    {output_dir}/{dataset}/{case_id}/
        images/                 # shared processed full-scene images
        junction_level/{model}_{reasoning}/
            junctions.json          # ids, boxes, inter-junction links
            overlay.png
            crops/junction_{id}.png
        junctions/{id}/             # per-junction crop + arm/lane/refine (when junction_level ran)
            images/
            arm_level/
            lane_level/
            refinement/
            pipeline_summary.json   # sibling of arm/lane/refine; written on success or abort
        arm_level/{model}_{reasoning}/
            arms.json
        lane_level/{model}_{reasoning}/
            lanes.json              # immutable generation snapshot
            conversation.json       # immutable generation snapshot
            cost.json               # immutable generation snapshot
            evaluation/             # arm_level + lane_level eval
        refinement/{model}_{reasoning}/
            lanes.json              # refined lane reading
            cost.json               # cost including refine calls
            evaluation/             # refinement eval
            xodr_result/            # final xodr artifacts (connections / scenario.xodr)
            layer_1/conversation.json   # {round_id: {arm_id: turn}}
            layer_1/result.json
            layer_1/lanes.json          # snapshot after layer-1
            layer_2/conversation.json
            layer_2/result.json
            layer_2/lanes.json          # snapshot after layer-2 (before feasibility_solver)
            layer_fea/                  # min-edit solver after L2 fail
            layer_fea/result.json
            layer_fea/lanes.json
            layer_fea/feasibility_solver.json
            layer_fea/conversation.json # LLM rank of top-k repairs
            layer_fea/cost.json         # rank-call tokens / USD
            layer_fea/evaluation/       # GT mismatch for solver lanes
            layer_fea_scene/            # multi-junction connector-mouth min-edit (case root)
            layer_3/                    # multi-junction inter-junction check (case root)
"""

import json
import re
import shutil
from pathlib import Path

METHODS_DIR = Path(__file__).resolve().parent
LLM_KG_DIR = METHODS_DIR / "llm_kg"


def dump_json(path, obj):
    """Write ``obj`` as indented UTF-8 JSON, creating parent dirs if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json_if_exists(path):
    """Return parsed JSON, or ``None`` if the file is missing or invalid."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def list_junction_artifact_roots(case_dir):
    """Per-junction artifact dirs, or ``[case_dir]`` for the legacy single-junction layout."""
    case_dir = Path(case_dir)
    root = case_dir / "junctions"
    if not root.is_dir():
        return [case_dir]

    def _sort_key(path):
        try:
            return (0, int(path.name))
        except ValueError:
            return (1, path.name)

    dirs = sorted((path for path in root.iterdir() if path.is_dir()), key=_sort_key)
    return dirs or [case_dir]


def _path_slug(value):
    return str(value).replace("/", "_").replace("-", "_").replace(".", "_")


_MODEL_DIR_REASONINGS = ("xhigh", "minimal", "medium", "high", "low", "none")


def model_dir_name(model_name, reasoning="none"):
    """Return ``{model}_{reasoning}`` with path-unsafe chars replaced by ``_``."""
    return "%s_%s" % (_path_slug(model_name), _path_slug(reasoning or "none"))


def parse_model_dir_name(dirname):
    """Inverse of ``model_dir_name`` for ``{model_slug}_{reasoning}`` directories."""
    name = str(dirname or "")
    reasoning = "none"
    slug = name
    for token in _MODEL_DIR_REASONINGS:
        suffix = "_" + _path_slug(token)
        if name.endswith(suffix) and len(name) > len(suffix):
            slug = name[:-len(suffix)]
            reasoning = token
            break
    if not slug:
        return None, None
    return re.sub(r"(?<=\d)-(?=\d)", ".", slug.replace("_", "-")), reasoning


def resolve_output_root(output_dir):
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = LLM_KG_DIR / output_path
    return output_path


def resolve_run_dir(output_dir, run_name):
    return resolve_output_root(output_dir) / run_name


def resolve_case_dir(output_dir, dataset, case_id):
    return resolve_run_dir(output_dir, "%s/%s" % (dataset, case_id))


def junction_level_dir(case_dir, model_name, reasoning="none"):
    return Path(case_dir) / "junction_level" / model_dir_name(model_name, reasoning)


def junction_work_dir(case_dir, junction_id):
    """Per-junction artifact root: ``{case_dir}/junctions/{id}/``."""
    return Path(case_dir) / "junctions" / str(junction_id)


PIPELINE_SUMMARY_NAME = "pipeline_summary.json"


def pipeline_summary_path(artifact_root):
    """``{artifact_root}/pipeline_summary.json``, sibling of arm_level/lane_level/refinement."""
    return Path(artifact_root) / PIPELINE_SUMMARY_NAME


def list_pipeline_summary_paths(artifact_root, *, refine_dir=None, case_root=None):
    """Canonical artifact-root path first, then legacy refine_dir / case-root copies."""
    paths = []
    seen = set()

    def _add(path):
        path = Path(path)
        key = str(path)
        if key in seen:
            return
        seen.add(key)
        paths.append(path)

    _add(pipeline_summary_path(artifact_root))
    if refine_dir is not None:
        _add(Path(refine_dir) / PIPELINE_SUMMARY_NAME)
    if case_root is not None:
        _add(Path(case_root) / PIPELINE_SUMMARY_NAME)
    return paths


def junctions_path_for_model(case_dir, model_name, reasoning="none"):
    path = junction_level_dir(case_dir, model_name, reasoning) / "junctions.json"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"junction-level result not found for model {model_name!r} reasoning {reasoning!r}: {path}"
    )


def arm_level_dir(case_dir, model_name, reasoning="none"):
    return Path(case_dir) / "arm_level" / model_dir_name(model_name, reasoning)


def lane_level_dir(case_dir, model_name, reasoning="none"):
    return Path(case_dir) / "lane_level" / model_dir_name(model_name, reasoning)


def refinement_dir(case_dir, model_name, reasoning="none"):
    return Path(case_dir) / "refinement" / model_dir_name(model_name, reasoning)


def resolve_model_dir(output_dir, dataset, case_id, model_name, reasoning="none"):
    """Lane-level model dir (lane-level evaluation root)."""
    return resolve_case_dir(output_dir, dataset, case_id) / "lane_level" / model_dir_name(model_name, reasoning)


def model_run_dir(case_dir, model_name, reasoning="none"):
    """Alias for lane-level model dir (lane-level evaluation root)."""
    return lane_level_dir(case_dir, model_name, reasoning)


def unpack_model_config(model, reasoning=None):
    from scripts.methods.api_providers import resolve_provider
    from scripts.methods.llm_utils import _strip_trailing_v1_base_url
    provider = resolve_provider(model["provider"])
    if reasoning is None:
        reasoning = model.get("reasoning", "none")
    api = model["api"]
    base_url = provider["base_url"]
    if api == "anthropic":
        base_url = _strip_trailing_v1_base_url(base_url)
    return {
        "api": api,
        "provider": model["provider"],
        "source": provider["source"],
        "model_name": model["model"],
        "api_key": provider["api_key"],
        "base_url": base_url,
        "proxy": provider["proxy"],
        "reasoning": reasoning,
    }


def resolve_case_inputs(dataset="nhtsa_2023", case_id="28661"):
    from scripts.methods.llm_utils import load_scene_image
    _, image_path, run_name = load_scene_image(dataset, case_id=case_id)
    return Path(image_path).resolve(), run_name


def case_dataset_dir(dataset, case_id):
    """``dataset/{dataset}/{case_id}/`` under the repo dataset root."""
    from scripts.data_reader import DATASET_ROOT
    return DATASET_ROOT / dataset / str(case_id)


def load_case_driving_side(dataset, case_id):
    """Read ``driving_side`` from ``dataset/{dataset}/{case_id}/meta.json``."""
    from scripts.methods.connection_parser import normalize_driving_side

    meta_path = case_dataset_dir(dataset, case_id) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError("case meta.json not found: %s" % meta_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("driving_side") is None:
        raise ValueError("driving_side missing in %s" % meta_path)
    return normalize_driving_side(meta["driving_side"])


def prepare_output_dir(out_dir, *, clear=True):
    out_dir = Path(out_dir)
    if clear and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def set_verbose(verbose):
    from scripts.methods.llm_utils import BaseLLM
    prev = BaseLLM.verbose
    BaseLLM.verbose = verbose
    return prev


def _pick_unique_image(images_dir, patterns):
    for pattern in patterns:
        image_paths = sorted(images_dir.glob(pattern))
        if len(image_paths) == 1:
            return image_paths[0].resolve()
        if len(image_paths) > 1:
            raise ValueError(f"multiple images matching {pattern!r} found in: {images_dir}")
    return None


def arm_level_image_path(arms_path, compressed=True):
    """Resolve the processed scene image used by lane-level.

    Expects ``{case}/arm_level/{model}_{reasoning}/arms.json``.
    """
    from scripts.methods.image_processing import resolve_processed_image

    arms_path = Path(arms_path).resolve()
    if arms_path.parent.parent.name != "arm_level":
        raise FileNotFoundError(
            "expected arms.json under arm_level/{model}_{reasoning}/, got: %s" % arms_path
        )
    case_dir = arms_path.parent.parent.parent
    return resolve_processed_image(case_dir, compressed=compressed)


def default_arms_path(case_dir, model_name=None, reasoning="none"):
    case_dir = Path(case_dir)
    if model_name:
        return arms_path_for_model(case_dir, model_name, reasoning)
    found = sorted(case_dir.glob("arm_level/*/arms.json"))
    if len(found) == 1:
        return found[0]
    if found:
        return max(found, key=lambda path: path.stat().st_mtime)
    raise FileNotFoundError(f"arm-level result not found under: {case_dir}")


def arms_path_for_model(case_dir, model_name, reasoning="none"):
    path = arm_level_dir(case_dir, model_name, reasoning) / "arms.json"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"arm-level result not found for model {model_name!r} reasoning {reasoning!r}: {path}"
    )


def lanes_path_for_model(case_dir, model_name, reasoning="none"):
    path = lane_level_dir(case_dir, model_name, reasoning) / "lanes.json"
    if path.exists():
        return path
    raise FileNotFoundError(
        f"lane-level result not found for model {model_name!r} reasoning {reasoning!r}: {path}"
    )


def xodr_result_dir_for_model(case_dir, model_name, reasoning="none"):
    """Write target: ``{case_dir}/refinement/{model}_{reasoning}/xodr_result``."""
    return refinement_dir(case_dir, model_name, reasoning) / "xodr_result"


def _xodr_artifact_search_paths(case_dir, model_name, reasoning, filename):
    """Case-root first (merged multi-junction), then each ``junctions/{id}/``."""
    case_dir = Path(case_dir)
    seen = set()
    paths = []
    roots = [case_dir]
    roots.extend(list_junction_artifact_roots(case_dir))
    for root in roots:
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        paths.append(xodr_result_dir_for_model(root, model_name, reasoning) / filename)
    return paths


def find_xodr_artifact_for_model(case_dir, model_name, reasoning, filename):
    """Return the first existing ``filename`` under case-root or junction xodr_result dirs."""
    for path in _xodr_artifact_search_paths(case_dir, model_name, reasoning, filename):
        if path.is_file():
            return path
    return None


def connections_path_for_model(case_dir, model_name, reasoning="none"):
    path = find_xodr_artifact_for_model(case_dir, model_name, reasoning, "connections.json")
    if path is not None:
        return path
    fallback = xodr_result_dir_for_model(case_dir, model_name, reasoning) / "connections.json"
    raise FileNotFoundError(
        f"connections not found for model {model_name!r} reasoning {reasoning!r}: {fallback}"
    )


def scenario_xodr_path_for_model(case_dir, model_name, reasoning="none"):
    path = find_xodr_artifact_for_model(case_dir, model_name, reasoning, "scenario.xodr")
    if path is not None:
        return path
    fallback = xodr_result_dir_for_model(case_dir, model_name, reasoning) / "scenario.xodr"
    raise FileNotFoundError(
        f"scenario.xodr not found for model {model_name!r} reasoning {reasoning!r}: {fallback}"
    )
