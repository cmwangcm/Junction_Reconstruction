"""Shared refine round + check→refine loop. Layers call ``run_refine_loop``; entry wires layers."""

from pathlib import Path

from scripts.methods.llm_kg.lane_level import llm_gen as gen
from scripts.methods.llm_kg.refinement.arm_refine import refine_arm_agent
from scripts.methods.llm_kg.refinement.common import (
    build_refinement_feedback,
    log_validation_error,
    refinement_arm_ids,
    refinement_layer_dir,
    refinement_round_dir,
    save_layer_refine_conversation,
    save_refinement_round_dir,
)
from scripts.methods.llm_kg.refinement.config_build import build_config
from scripts.methods.llm_utils import BaseLLM


class RefineContext:
    """Runtime state for layered refinement. Data only — no function callbacks."""

    def __init__(
        self,
        *,
        llm,
        agents_by_arm,
        trackers_by_arm,
        lane_results,
        arms,
        all_arm_ids,
        channelized_features,
        through_roads,
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
    ):
        self.llm = llm
        self.agents_by_arm = agents_by_arm
        self.trackers_by_arm = trackers_by_arm
        self.lane_results = lane_results
        self.arms = arms
        self.all_arm_ids = all_arm_ids
        self.channelized_features = channelized_features
        self.through_roads = through_roads
        self.name = name
        self.model = model
        self.max_tokens = max_tokens
        self.reasoning = reasoning
        self.margin = margin
        self.crop_and_inspect = bool(crop_and_inspect)
        self.max_inspect_rounds = int(max_inspect_rounds)
        self.refinement_root = Path(refinement_root) if refinement_root is not None else None
        self.driving_side = driving_side or "left"
        self.image_path = image_path

    def rebuild_config(self):
        return build_config(
            self.arms,
            gen._lane_result_doc(self.lane_results),
            self.name,
            channelized_features=self.channelized_features,
            through_roads=self.through_roads,
            driving_side=self.driving_side,
        )


def _arm_reading_summary(lane_arm):
    if not lane_arm:
        return None
    lanes = []
    for item in lane_arm.get("cross_section") or []:
        if item.get("type") != "lane":
            continue
        lanes.append({"travel": item.get("travel"), "movements": gen._lane_movements(item) or []})
    return {"arm_id": lane_arm.get("arm_id"), "lanes": lanes}


def _refine_one_arm(ctx, arm_id, *, validation_error, layer, round_no, max_rounds, round_dir):
    """Run one arm's refine strategy and commit the result into ``ctx.lane_results``."""
    agent = ctx.agents_by_arm[arm_id]
    tracker = ctx.trackers_by_arm[arm_id]
    before = _arm_reading_summary(ctx.lane_results.get(arm_id))
    feedback = build_refinement_feedback(
        round_no, max_rounds, validation_error=validation_error, arm_id=arm_id, layer=layer,
        lane_results=ctx.lane_results if layer in ("check", "fix", "fix_full_cs", "layer3") else None,
    )
    arm_dir = (round_dir / f"arm_{arm_id}") if round_dir is not None else None
    messages = agent.get("messages") or []
    msg_start = len(messages) - 1 if messages and messages[-1].get("role") == "user" else len(messages)
    patch_start = len(agent.get("patch_events") or [])
    crop_ids_before = set(agent.get("crops") or {})
    try:
        refine_arm_agent(
            ctx.llm, agent, feedback,
            model=ctx.model, tracker=tracker,
            validation_error=validation_error, arm_id=arm_id,
            max_tokens=ctx.max_tokens, reasoning=ctx.reasoning, margin=ctx.margin,
            max_inspect_rounds=ctx.max_inspect_rounds,
            crop_and_inspect=ctx.crop_and_inspect,
            save_dir=arm_dir,
            layer=layer,
            lane_results=ctx.lane_results if layer in ("check", "fix", "fix_full_cs", "layer3") else None,
        )
        ctx.lane_results[arm_id] = agent["result"]
    finally:
        if ctx.refinement_root is not None:
            try:
                save_start = agent.pop("refine_conversation_start", msg_start)
                save_layer_refine_conversation(
                    refinement_layer_dir(ctx.refinement_root, layer),
                    round_id=round_no, arm_id=arm_id, agent=agent, model=ctx.model,
                    msg_start=save_start, patch_start=patch_start, crop_ids_before=crop_ids_before,
                )
            except Exception as exc:
                BaseLLM.log(f"[refine:{layer}] could not save conversation arm={arm_id}: {exc}")
    after = _arm_reading_summary(ctx.lane_results.get(arm_id))
    BaseLLM.log(f"[refine:{layer}] arm {arm_id} before: {(before or {}).get('lanes')}")
    BaseLLM.log(f"[refine:{layer}] arm {arm_id} after: {(after or {}).get('lanes')}")
    return True


def _apply_check_round(ctx, *, validation_error, round_no, max_rounds, round_dir):
    """Sequential through-road pair refine: A, recompute, maybe B with A_current."""
    from scripts.methods.llm_kg.refinement.layer1_tr_check import (
        pair_inconsistent,
        through_road_pairs_from_issues,
    )

    known = set(ctx.all_arm_ids)
    pairs = through_road_pairs_from_issues(validation_error.get("issues") or [])
    progressed = False
    for a_id, b_id in pairs:
        if a_id not in known:
            continue
        BaseLLM.log(f"[refine:check] pair {a_id}/{b_id}: re-identify arm {a_id} (peer={b_id} current)")
        _refine_one_arm(
            ctx, a_id,
            validation_error=validation_error, layer="check",
            round_no=round_no, max_rounds=max_rounds, round_dir=round_dir,
        )
        progressed = True
        still = pair_inconsistent(a_id, b_id, ctx.lane_results)
        if not still:
            BaseLLM.log(
                f"[refine:check] pair {a_id}/{b_id} consistent after arm {a_id}; skip arm {b_id}"
            )
            continue
        if b_id not in known:
            BaseLLM.log(
                f"[refine:check] pair {a_id}/{b_id} still inconsistent after arm {a_id}; "
                f"peer arm {b_id} not in case"
            )
            continue
        BaseLLM.log(
            f"[refine:check] pair {a_id}/{b_id} still inconsistent after arm {a_id}; "
            f"re-identify arm {b_id} using arm {a_id} current"
        )
        _refine_one_arm(
            ctx, b_id,
            validation_error=validation_error, layer="check",
            round_no=round_no, max_rounds=max_rounds, round_dir=round_dir,
        )
        still = pair_inconsistent(a_id, b_id, ctx.lane_results)
        BaseLLM.log(
            f"[refine:check] pair {a_id}/{b_id} after arm {b_id}: "
            + ("still inconsistent" if still else "consistent")
        )
    return ctx.rebuild_config(), validation_error, progressed


def apply_refinement_round(ctx, *, exc, layer, round_no, max_rounds, check=None):
    """Refine every target arm for one validation failure, then rebuild config."""
    validation_error = log_validation_error(exc, round_no=round_no, max_rounds=max_rounds, layer=layer)
    round_dir = None
    if ctx.refinement_root is not None:
        round_dir = save_refinement_round_dir(
            refinement_round_dir(ctx.refinement_root, layer, round_no),
            validation_error=validation_error,
        )
    if layer == "check":
        return _apply_check_round(
            ctx, validation_error=validation_error,
            round_no=round_no, max_rounds=max_rounds, round_dir=round_dir,
        )
    target_ids = refinement_arm_ids(exc.issues, validation_error, ctx.all_arm_ids)
    if not target_ids:
        return ctx.rebuild_config(), validation_error, False
    progressed = False
    for arm_id in target_ids:
        BaseLLM.log(f"[refine:{layer}] re-identify arm {arm_id} from mouth crop")
        _refine_one_arm(
            ctx, arm_id,
            validation_error=validation_error, layer=layer,
            round_no=round_no, max_rounds=max_rounds, round_dir=round_dir,
        )
        progressed = True
        if check is None:
            continue
        config = ctx.rebuild_config()
        if check(config) is None:
            BaseLLM.log(
                f"[refine:{layer}] consistent after arm {arm_id}; skip remaining arms"
            )
            break
    return ctx.rebuild_config(), validation_error, progressed


def run_refine_loop(ctx, config, *, layer, max_refine_rounds, check):
    """Shared check → refine → recheck loop.

    ``check(config)`` returns None on pass, or a ConnectionParseError on fail.
    Returns (config, refine_rounds, validation_error, passed).
    """
    exc = check(config)
    refine_rounds = 0
    validation_error = None
    while exc is not None and refine_rounds < max_refine_rounds:
        refine_rounds += 1
        config, validation_error, progressed = apply_refinement_round(
            ctx, exc=exc, layer=layer,
            round_no=refine_rounds, max_rounds=max_refine_rounds,
            check=check,
        )
        if not progressed:
            break
        exc = check(config)
    if exc is not None:
        validation_error = log_validation_error(exc, layer=layer)
        return config, refine_rounds, validation_error, False
    return config, refine_rounds, None, True
