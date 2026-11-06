"""LLM ranking of top-k feasibility_solver candidates.

The model sees the junction (or full-scene) image, the original lanes.json, and
each candidate's edit list only — not the full repaired lane documents.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.methods.image_utils import load_rgb_image
from scripts.methods.llm_kg.lane_level.prompts import _lane_level_body, arm_features_note
from scripts.methods.llm_kg.refinement.feasibility_solver import format_repair_edits
from scripts.methods.llm_utils import BaseLLM, to_jsonable

_RANK_TASK = (
    "You rank connectivity-preserving min-edit repairs of a junction lane reading. "
    "Every candidate already restores a feasible fill/connections. "
    "Choose the candidate whose resulting structure best matches the junction photograph "
    "under the element definitions and structural constraints below "
    "(lane counts, travel directions, painted arrows, one-way vs two-way, "
    "opposing_separator). "
    "The image is the full junction, not a single arm crop: judge each arm mouth "
    "in the photo against the original lanes.json and the candidate edits. "
    "lane.travel incoming/outgoing in lanes.json is relative to that arm's mouth "
    "(junction at the mouth, not full-image up). "
    "Do not invent edits. Pick only among the given candidate ids. "
    "Prefer a visually supported reading over a cheaper but implausible one."
)

_SCENE_RANK_TASK = (
    "You rank connectivity-preserving min-edit repairs of a multi-junction scene. "
    "Every candidate already restores a feasible fill/connections at each junction "
    "and matching lane counts / one-way vs two-way at each inter-junction road. "
    "Choose the candidate whose resulting structure best matches the full scene photograph "
    "under the element definitions and structural constraints below "
    "(lane counts, travel directions, painted arrows, one-way vs two-way, "
    "opposing_separator). "
    "The image is the full scene, not a single junction crop. Edits are tagged "
    "j{id}.arm{id}. Judge each edited mouth in the photo against the original "
    "per-junction lanes.json and the candidate edits. "
    "lane.travel incoming/outgoing is relative to that arm's mouth "
    "(junction at the mouth, not full-image up). "
    "Do not invent edits. Pick only among the given candidate ids. "
    "Prefer a visually supported reading over a cheaper but implausible one. "
    "If a connecting road is two-way at one mouth in the original reading, "
    "prefer completing the one-way far end over dropping the reverse stream."
)


def _channelized_constraint_block(original_lanes):
    """Junction-wide CHANNELIZED SLIP lines (the lane-level body is per-arm)."""
    if not isinstance(original_lanes, dict):
        return ""
    features = original_lanes.get("channelized_features")
    if not features:
        return ""
    side = original_lanes.get("driving_side")
    lines = []
    for arm in original_lanes.get("arms") or []:
        aid = arm.get("arm_id")
        note = arm_features_note(aid, features, driving_side=side)
        if note:
            lines.append(f"- CHANNELIZED SLIP (arm {aid}): {note}")
    if not lines:
        return ""
    return "\n".join(lines)


def _scene_channelized_block(original_lanes):
    junctions = (original_lanes or {}).get("junctions") or {}
    parts = []
    for jid, doc in sorted(
        junctions.items(),
        key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else str(kv[0]),
    ):
        block = _channelized_constraint_block(doc if isinstance(doc, dict) else {})
        if block:
            parts.append("Junction %s:\n%s" % (jid, block))
    return "\n".join(parts)


def rank_system_prompt(original_lanes=None, *, scene=False):
    """Ranking task plus the lane-level element definitions and constraints."""
    lanes = original_lanes if isinstance(original_lanes, dict) else {}
    if scene:
        nested = list((lanes.get("junctions") or {}).values())
        side = lanes.get("driving_side")
        features = lanes.get("channelized_features")
        if nested and isinstance(nested[0], dict):
            side = side or nested[0].get("driving_side")
            features = features or nested[0].get("channelized_features")
        body = _lane_level_body(side, channelized_features=features)
        extra = _scene_channelized_block(lanes)
        task = _SCENE_RANK_TASK
    else:
        body = _lane_level_body(
            lanes.get("driving_side"),
            channelized_features=lanes.get("channelized_features"),
        )
        extra = _channelized_constraint_block(lanes)
        task = _RANK_TASK
    parts = [task, "", body.strip()]
    if extra:
        parts.extend(["", extra])
    return "\n".join(parts).rstrip() + "\n"


RANK_SYSTEM_PROMPT = rank_system_prompt()

RANK_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["chosen_id", "ranking", "reasoning"],
    "properties": {
        "chosen_id": {
            "type": "integer",
            "minimum": 1,
            "maximum": 3,
            "description": "Id of the visually and structurally best candidate.",
        },
        "ranking": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {"type": "integer", "minimum": 1, "maximum": 3},
            "description": "Candidate ids best-to-worst.",
        },
        "reasoning": {
            "type": "string",
            "description": "Why this ranking matches the image and original reading.",
        },
    },
}


def _candidate_block(candidate):
    cid = candidate.candidate_id
    lines = format_repair_edits(candidate.edits)
    body = "\n".join(f"  {line}" for line in lines) if lines else "  (no edits)"
    return f"Candidate {cid} (min-edit score={candidate.score}):\n{body}"


def _original_lanes_for_prompt(original_lanes):
    """Keep the original reading; drop non-lanes.json extras if present."""
    if not isinstance(original_lanes, dict):
        return original_lanes
    if original_lanes.get("junctions") and isinstance(original_lanes["junctions"], dict):
        out = {}
        if "driving_side" in original_lanes:
            out["driving_side"] = original_lanes["driving_side"]
        out["junctions"] = {
            str(jid): _original_lanes_for_prompt(doc)
            for jid, doc in original_lanes["junctions"].items()
        }
        return out
    keep = (
        "source_image", "driving_side", "arms",
        "channelized_features", "through_roads",
    )
    return {key: original_lanes[key] for key in keep if key in original_lanes}


def format_rank_prompt(original_lanes, candidates, *, scene=False):
    blocks = [_candidate_block(cand) for cand in candidates]
    n = len(candidates)
    ids = ", ".join(str(cand.candidate_id) for cand in candidates)
    original = _original_lanes_for_prompt(original_lanes)
    image_line = (
        "The image is the original full scene photograph.\n"
        if scene else
        "The image is the original junction photograph.\n"
    )
    lanes_label = (
        "original per-junction lanes.json readings"
        if scene else
        "original lanes.json (the reading before repair)"
    )
    return (
        image_line
        + f"Below is the {lanes_label} and {n} "
        "feasible repair candidate(s). Each candidate lists only the edits "
        "relative to that original reading — not a full new lanes.json.\n\n"
        f"Valid candidate ids: {ids}. Return chosen_id as one of these ids, "
        "and ranking as those ids from best to worst.\n\n"
        "### Original lanes.json\n"
        f"{json.dumps(original, indent=2, ensure_ascii=False, default=str)}\n\n"
        "### Candidates (edits only)\n"
        + "\n\n".join(blocks)
    )


def _load_rank_image(image=None, image_path=None):
    if image is not None:
        return image
    if image_path:
        return load_rgb_image(image_path)
    return None


def _rank_conversation(
    *, model, system=None, skipped=None, messages=None, parsed=None,
    image_path=None, stage="feasibility_rank",
):
    payload = {
        "stage": stage,
        "model": model,
        "system": system if system is not None else RANK_SYSTEM_PROMPT,
    }
    if image_path is not None:
        payload["source_image"] = str(image_path)
    if skipped:
        payload["skipped"] = skipped
    if messages is not None:
        payload["messages"] = to_jsonable(messages)
    if parsed is not None:
        payload["result"] = parsed
    return payload


def rank_repair_candidates(
    result,
    *,
    llm,
    model,
    original_lanes,
    image=None,
    image_path=None,
    tracker=None,
    reasoning=None,
    max_tokens=2048,
):
    """Rank ``result.candidates`` with the junction image; apply the chosen repair.

    If ranking is skipped or fails, keep the cheapest candidate (score order).
    Always returns ``(result, conversation)`` so layer_fea can persist the call.
    """
    return _rank_repair_candidates_impl(
        result,
        llm=llm,
        model=model,
        original_lanes=original_lanes,
        image=image,
        image_path=image_path,
        tracker=tracker,
        reasoning=reasoning,
        max_tokens=max_tokens,
        scene=False,
    )


def rank_scene_repair_candidates(
    result,
    *,
    llm,
    model,
    original_lanes,
    image=None,
    image_path=None,
    tracker=None,
    reasoning=None,
    max_tokens=2048,
):
    """Rank scene-level ``result.candidates`` from the full scene image."""
    return _rank_repair_candidates_impl(
        result,
        llm=llm,
        model=model,
        original_lanes=original_lanes,
        image=image,
        image_path=image_path,
        tracker=tracker,
        reasoning=reasoning,
        max_tokens=max_tokens,
        scene=True,
    )


def _rank_repair_candidates_impl(
    result,
    *,
    llm,
    model,
    original_lanes,
    image=None,
    image_path=None,
    tracker=None,
    reasoning=None,
    max_tokens=2048,
    scene=False,
):
    candidates = list(result.candidates or [])
    system = rank_system_prompt(original_lanes, scene=scene)
    stage = "feasibility_rank_scene" if scene else "feasibility_rank"
    log_tag = "feasibility_scene" if scene else "feasibility"
    if not result.ok:
        return result, _rank_conversation(
            model=model, system=system, skipped="solver_failed", stage=stage,
        )
    if len(candidates) <= 1:
        return result, _rank_conversation(
            model=model,
            system=system,
            skipped="single_candidate" if candidates else "no_candidates",
            stage=stage,
        )
    if llm is None:
        return result, _rank_conversation(
            model=model, system=system, skipped="no_llm", stage=stage,
        )

    image = _load_rank_image(image=image, image_path=image_path)
    if image is None:
        BaseLLM.log("[validate:%s] no image; keep cheapest candidate" % log_tag)
        return result, _rank_conversation(
            model=model, system=system, skipped="no_image", image_path=image_path,
            stage=stage,
        )

    prompt = format_rank_prompt(original_lanes, candidates, scene=scene)
    messages = [llm.image_user_message(image, prompt)]
    BaseLLM.log("[validate:%s] LLM rank top-%s candidates" % (log_tag, len(candidates)))
    try:
        parsed, text, response = llm.call_json(
            model=model,
            system_prompt=system,
            messages=messages,
            schema=RANK_SCHEMA,
            max_tokens=max_tokens,
            reasoning_effort=reasoning,
        )
    except Exception as exc:
        BaseLLM.log("[validate:%s] LLM rank failed (%s); keep cheapest" % (log_tag, exc))
        return result, _rank_conversation(
            model=model,
            system=system,
            skipped=f"llm_error: {exc}",
            messages=messages,
            image_path=image_path,
            stage=stage,
        )

    if tracker is not None:
        tracker.record(stage, model, response)
    try:
        messages = list(messages) + [llm.assistant_message(response)]
    except Exception:
        messages = list(messages) + [{"role": "assistant", "content": text}]

    conversation = _rank_conversation(
        model=model,
        system=system,
        messages=messages,
        parsed=parsed,
        image_path=image_path,
        stage=stage,
    )

    valid_ids = {cand.candidate_id for cand in candidates}
    chosen_id = parsed.get("chosen_id")
    ranking = parsed.get("ranking") or []
    reasoning_text = parsed.get("reasoning")
    if chosen_id not in valid_ids:
        BaseLLM.log(
            f"[validate:{log_tag}] LLM chosen_id={chosen_id!r} not in {sorted(valid_ids)}; "
            "keep cheapest"
        )
        conversation["skipped"] = f"invalid_chosen_id: {chosen_id!r}"
        return result, conversation

    result.apply_choice(
        chosen_id,
        source="llm",
        reasoning=reasoning_text,
        rank_order=ranking,
    )
    BaseLLM.log(
        f"[validate:{log_tag}] LLM chose candidate {chosen_id} "
        f"(ranking={ranking})"
    )
    if reasoning_text:
        BaseLLM.log(str(reasoning_text), indent=1)
    return result, conversation


def resolve_junction_image_path(source_lanes=None, arms_path=None):
    """Prefer lanes.json source_image, else the arm-level processed scene image."""
    if isinstance(source_lanes, dict) and source_lanes.get("source_image"):
        path = Path(source_lanes["source_image"])
        if path.is_file():
            return path
    if arms_path is not None:
        from scripts.methods.method_utils import arm_level_image_path
        try:
            path = Path(arm_level_image_path(arms_path))
        except FileNotFoundError:
            return None
        if path.is_file():
            return path
    return None
