"""Per-arm LLM refine: identifier → evaluator → inspector → organizer."""

from pathlib import Path

from scripts.methods.llm_kg.lane_level import llm_gen as gen
from scripts.methods.llm_kg.lane_level.prompts import arm_features_note, arm_user_text, lane_level_prompt
from scripts.methods.llm_kg.refinement.common import (
    CHECK_CONSTRAINTS,
    LAYER3_CONSTRAINTS,
    LAYER3_MEDIAN_CONSTRAINT,
    _arm_has_full_cs_regen_issue,
    _check_reidentify_summaries,
    _layer3_constraint_for_arm,
)
from scripts.methods.llm_kg.refinement.prompts import (
    connectivity_reidentify_user_text,
    inter_junction_reidentify_user_text,
    median_through_one_way_reidentify_user_text,
    through_road_reidentify_user_text,
)
from scripts.methods.llm_utils import BaseLLM

LOCATE_BOX_MARGIN = gen.LOCATE_BOX_MARGIN


def _issue_mentions_arm(issue, arm_id):
    arm_id = int(arm_id)
    for item in issue.get("arm_id_list") or []:
        if item is not None and int(item) == arm_id:
            return True
    location = issue.get("location") or {}
    for key in ("arm", "from_arm", "to_arm"):
        value = location.get(key)
        if value is not None and int(value) == arm_id:
            return True
    details = issue.get("details") or {}
    for key in ("arm", "from_arm", "to_arm"):
        value = details.get(key)
        if value is not None and int(value) == arm_id:
            return True
    return False


def _arm_needs_constraint_reidentify(validation_error, arm_id, constraints):
    for issue in (validation_error or {}).get("issues") or []:
        if issue.get("constraint") not in constraints:
            continue
        location = issue.get("location") or {}
        if location.get("arm") is not None and int(location["arm"]) == int(arm_id):
            return True
        if _issue_mentions_arm(issue, arm_id):
            return True
    return False


def _arm_needs_layer1_reidentify(validation_error, arm_id):
    return _arm_needs_constraint_reidentify(validation_error, arm_id, CHECK_CONSTRAINTS)


def _arm_needs_layer3_reidentify(validation_error, arm_id):
    return _arm_needs_constraint_reidentify(validation_error, arm_id, LAYER3_CONSTRAINTS)


def _ensure_extended_crop(agent, *, save_dir=None):
    """Build the layer-1 junction-extended crop once and register it on the agent."""
    if agent.get("extended_image") is not None:
        image = agent["extended_image"]
    else:
        original = agent.get("original_image")
        if original is None:
            raise ValueError(
                f"arm {agent['arm']['id']}: original_image is required to build the layer-1 extended crop"
            )
        bundle = gen.run_rule_extended_crop(
            original, agent["arm"], agent["crop_dir"],
            width_ext_frac=agent.get("width_ext_frac", gen.DEFAULT_WIDTH_EXT_FRAC),
            min_width_ext_px=agent.get("min_width_ext_px"),
        )
        image = bundle["crop_image"]
        agent["extended_image"] = image
        agent["extended_crop_id"] = bundle["crop_id"]
        agent["crops"].update(bundle["crops"])
        size = image.size
        BaseLLM.log(
            f"arm {agent['arm']['id']}: layer-1 extended crop {size[0]}x{size[1]}px",
            indent=1,
        )
    if save_dir is not None:
        round_crops = Path(save_dir) / "crops"
        round_crops.mkdir(parents=True, exist_ok=True)
        crop_id = agent.get("extended_crop_id") or "view_extended"
        image.save(round_crops / f"{crop_id}.png")
    return image


def _reset_visual_conversations(agent):
    agent["evaluator_system_prompt"] = None
    agent["evaluator_messages"] = []
    agent["inspector_system_prompt"] = None
    agent["inspector_messages"] = []
    agent["organization_system_prompt"] = None
    agent["organization_messages"] = []
    agent["llm_calls"] = []
    agent["critiques"] = []


def _reidentify_arm_from_crop(
    llm, agent, *, user_text, image, crop_id, mode,
    model, tracker, arm_id=None, max_tokens=2048, reasoning=None,
    margin=LOCATE_BOX_MARGIN, max_inspect_rounds=1, crop_and_inspect=True,
    save_dir=None,
):
    """Re-run identifier → evaluator → inspector → organizer on ``image``."""
    arm = agent["arm"]
    arm_id = int(arm_id if arm_id is not None else arm["id"])
    agent["refinement_rounds"] += 1
    stage = f"refine_{agent['refinement_rounds']}_reidentify"

    prior = agent.get("result") or {}
    before = list(prior.get("cross_section") or [])
    before_opp = prior.get("opposing_separator")

    mouth_image = agent["base_image"]
    mouth_crop_id = agent["crop_id"]
    saved_crop_dir = agent["crop_dir"]
    if save_dir is not None:
        round_crops = Path(save_dir) / "crops"
        round_crops.mkdir(parents=True, exist_ok=True)
        agent["crop_dir"] = round_crops

    try:
        agent["system_prompt"] = lane_level_prompt(
            agent.get("driving_side"),
            channelized_features=agent.get("channelized_features"),
            arm_id=(agent.get("arm") or {}).get("id"),
        )
        agent["base_image"] = image
        agent["crop_id"] = crop_id
        agent["result"] = None
        agent["inspected"] = False
        agent["messages"] = [llm.image_user_message(image, user_text)]
        _reset_visual_conversations(agent)
        agent["refine_conversation_start"] = 0
        BaseLLM.log(f"arm {arm_id}: {stage} from {crop_id}", indent=1)
        gen.run_identifier(
            llm, agent, model=model, tracker=tracker,
            max_tokens=max_tokens, reasoning=reasoning,
        )
        if crop_and_inspect:
            gen.run_evaluate_and_inspect(
                llm, agent, model=model, tracker=tracker,
                max_tokens=max_tokens, reasoning=reasoning,
                margin=margin, max_inspect_rounds=max_inspect_rounds,
            )
        else:
            BaseLLM.log(f"arm {arm_id}: evaluator+inspector skipped", indent=1)
        gen.complete_arm_reading(
            llm, agent, model=model, tracker=tracker,
            max_tokens=max_tokens, reasoning=reasoning,
        )
        after = list((agent.get("result") or {}).get("cross_section") or [])
        agent["patch_events"].append({
            "stage": stage,
            "mode": mode,
            "modified": True,
            "applied": [{
                "region_id": "FULL",
                "crop_id": crop_id,
                "replace_start": 0,
                "replace_end": len(before),
                "before": before,
                "after": after,
                "opposing_separator": (agent.get("result") or {}).get("opposing_separator"),
                "before_opposing_separator": before_opp,
            }],
        })
        return agent
    finally:
        agent["base_image"] = mouth_image
        agent["crop_id"] = mouth_crop_id
        agent["crop_dir"] = saved_crop_dir


def _reidentify_arm_from_extended(
    llm, agent, *, model, tracker, validation_error=None, arm_id=None,
    max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN,
    max_inspect_rounds=1, crop_and_inspect=True, save_dir=None, lane_results=None,
):
    arm = agent["arm"]
    arm_id = int(arm_id if arm_id is not None else arm["id"])
    this_summary, peer_summary = _check_reidentify_summaries(
        validation_error, arm_id, lane_results=lane_results,
    )
    unknown = (
        "- outgoing lanes: unknown\n"
        "- incoming lanes: unknown\n"
        "- opposing traffic: unknown"
    )
    user_text = through_road_reidentify_user_text(
        this_summary or unknown, peer_summary or unknown,
        arm_features=arm_features_note(
            arm["id"],
            agent.get("channelized_features"),
            agent.get("through_roads"),
            driving_side=agent.get("driving_side"),
        ),
    )
    extended = _ensure_extended_crop(agent, save_dir=save_dir)
    return _reidentify_arm_from_crop(
        llm, agent,
        user_text=user_text,
        image=extended,
        crop_id=agent.get("extended_crop_id") or "view_extended",
        mode="layer1_reidentify",
        model=model, tracker=tracker, arm_id=arm_id,
        max_tokens=max_tokens, reasoning=reasoning, margin=margin,
        max_inspect_rounds=max_inspect_rounds, crop_and_inspect=crop_and_inspect,
        save_dir=save_dir,
    )


def _reidentify_arm_inter_junction(
    llm, agent, *, model, tracker, validation_error=None, arm_id=None,
    max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN,
    max_inspect_rounds=1, crop_and_inspect=True, save_dir=None, lane_results=None,
):
    arm = agent["arm"]
    arm_id = int(arm_id if arm_id is not None else arm["id"])
    this_summary, peer_summary = _check_reidentify_summaries(
        validation_error, arm_id, lane_results=lane_results,
        constraints=LAYER3_CONSTRAINTS,
    )
    unknown = (
        "- outgoing lanes: unknown\n"
        "- incoming lanes: unknown\n"
        "- opposing traffic: unknown"
    )
    features = arm_features_note(
        arm["id"],
        agent.get("channelized_features"),
        agent.get("through_roads"),
        driving_side=agent.get("driving_side"),
    )
    if _layer3_constraint_for_arm(validation_error, arm_id) == LAYER3_MEDIAN_CONSTRAINT:
        user_text = median_through_one_way_reidentify_user_text(
            this_summary or unknown, peer_summary or unknown, arm_features=features,
        )
    else:
        user_text = inter_junction_reidentify_user_text(
            this_summary or unknown, peer_summary or unknown, arm_features=features,
        )
    extended = _ensure_extended_crop(agent, save_dir=save_dir)
    return _reidentify_arm_from_crop(
        llm, agent,
        user_text=user_text,
        image=extended,
        crop_id=agent.get("extended_crop_id") or "view_extended",
        mode="layer3_reidentify",
        model=model, tracker=tracker, arm_id=arm_id,
        max_tokens=max_tokens, reasoning=reasoning, margin=margin,
        max_inspect_rounds=max_inspect_rounds, crop_and_inspect=crop_and_inspect,
        save_dir=save_dir,
    )


def _reidentify_arm_from_mouth(
    llm, agent, *, model, tracker, validation_error=None, arm_id=None,
    max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN,
    max_inspect_rounds=1, crop_and_inspect=True, save_dir=None, lane_results=None,
):
    from scripts.methods.llm_kg.refinement.layer2_con_fix import (
        _UNKNOWN_CONNECTIVITY_SUMMARY,
        connectivity_reidentify_summaries,
        format_connectivity_errors,
    )
    arm = agent["arm"]
    arm_id = int(arm_id if arm_id is not None else arm["id"])
    this_summary, related_summary = connectivity_reidentify_summaries(
        validation_error, arm_id, lane_results=lane_results,
    )
    user_text = connectivity_reidentify_user_text(
        this_summary or _UNKNOWN_CONNECTIVITY_SUMMARY,
        related_summary or "- (no related arm reading)",
        error_text=format_connectivity_errors(validation_error, arm_id),
        arm_features=arm_features_note(
            arm["id"],
            agent.get("channelized_features"),
            agent.get("through_roads"),
            driving_side=agent.get("driving_side"),
        ),
    )
    return _reidentify_arm_from_crop(
        llm, agent,
        user_text=user_text,
        image=agent["base_image"],
        crop_id=agent.get("crop_id") or "view_initial",
        mode="layer2_reidentify",
        model=model, tracker=tracker, arm_id=arm_id,
        max_tokens=max_tokens, reasoning=reasoning, margin=margin,
        max_inspect_rounds=max_inspect_rounds, crop_and_inspect=crop_and_inspect,
        save_dir=save_dir,
    )


def _evaluate_existing_arm(
    llm, agent, refinement_feedback, *, model, tracker, arm_id=None,
    max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN,
    max_inspect_rounds=1, crop_and_inspect=True, save_dir=None,
):
    """Re-run lane-level identification on the mouth crop with refinement feedback in the identifier input."""
    extra = (refinement_feedback or "").strip()
    user_text = arm_user_text(
        agent.get("arm"),
        agent.get("channelized_features"),
        agent.get("through_roads"),
        driving_side=agent.get("driving_side"),
    )
    if extra:
        user_text = f"{user_text}\n\n{extra}"
    return _reidentify_arm_from_crop(
        llm, agent,
        user_text=user_text,
        image=agent["base_image"],
        crop_id=agent.get("crop_id") or "view_initial",
        mode="feedback_reidentify",
        model=model, tracker=tracker, arm_id=arm_id,
        max_tokens=max_tokens, reasoning=reasoning, margin=margin,
        max_inspect_rounds=max_inspect_rounds, crop_and_inspect=crop_and_inspect,
        save_dir=save_dir,
    )


def refine_arm_agent(
    llm, agent, refinement_feedback, *, model, tracker, validation_error=None,
    arm_id=None, max_tokens=2048, reasoning=None, margin=LOCATE_BOX_MARGIN,
    max_inspect_rounds=1, crop_and_inspect=True, save_dir=None, layer=None,
    lane_results=None,
):
    """Dispatch one arm to the strategy required by its validation issues."""
    arm = agent["arm"]
    arm_id = int(arm_id if arm_id is not None else arm["id"])
    kwargs = dict(
        model=model, tracker=tracker, validation_error=validation_error, arm_id=arm_id,
        max_tokens=max_tokens, reasoning=reasoning, margin=margin,
        max_inspect_rounds=max_inspect_rounds, crop_and_inspect=crop_and_inspect,
        save_dir=save_dir, lane_results=lane_results,
    )
    if layer == "layer3" or _arm_needs_layer3_reidentify(validation_error, arm_id):
        return _reidentify_arm_inter_junction(llm, agent, **kwargs)
    if _arm_needs_layer1_reidentify(validation_error, arm_id):
        return _reidentify_arm_from_extended(llm, agent, **kwargs)
    if layer in ("fix", "fix_full_cs") or _arm_has_full_cs_regen_issue(validation_error, arm_id):
        return _reidentify_arm_from_mouth(llm, agent, **kwargs)
    return _evaluate_existing_arm(
        llm, agent, refinement_feedback,
        model=model, tracker=tracker, arm_id=arm_id,
        max_tokens=max_tokens, reasoning=reasoning, margin=margin,
        max_inspect_rounds=max_inspect_rounds, crop_and_inspect=crop_and_inspect,
        save_dir=save_dir,
    )
