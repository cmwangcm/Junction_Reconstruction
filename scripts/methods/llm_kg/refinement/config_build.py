"""Build xodr_config dicts from lane-level cross-section results."""

from scripts.methods.connection_parser import (
    arm_heading_midpoint,
    enrich_channelized_features,
    normalize_channelized_source_type,
    normalize_driving_side,
    normalize_movement_candidates,
    slip_assigned_code,
)
from scripts.methods.llm_kg.lane_level import llm_gen as gen
from scripts.methods.llm_kg.lane_level.schemas import LANE_TYPE


def _lane_config_entry(item):
    travel = item.get("travel")
    movements = normalize_movement_candidates(gen._lane_movements(item)) if travel != "outgoing" else []
    return {"travel": travel, "movements": movements}


def _apply_channelized_slip_turns(config):
    """Assign the curb-edge slip movement on each channelized from_arm (xodr_config only).

    dedicated → R/L; shared → SR/LS from arm-level channelized_features.source_type.
    lanes.json stays visual.
    """
    side = normalize_driving_side(config.get("driving_side"))
    source_by_arm = {}
    for feature in config.get("channelized_features") or []:
        if feature.get("from_arm") is None:
            continue
        source_by_arm[int(feature["from_arm"])] = feature.get("source_type")
    origins = set(source_by_arm)
    for arm in config.get("arms") or []:
        arm_id = int(arm.get("id"))
        if arm_id not in origins:
            continue
        lanes = list(arm.get("lanes") or [])
        if not lanes:
            continue
        idx = len(lanes) - 1 if side == "left" else 0
        lane = dict(lanes[idx] or {})
        if lane.get("travel") != "incoming":
            continue
        source_type = source_by_arm.get(arm_id)
        lane["movements"] = [slip_assigned_code(source_type, side)]
        lanes[idx] = lane
        arm["lanes"] = lanes
    return config


def build_config(arms, lane_result, name, channelized_features=None, through_roads=None, driving_side=None):
    """Convert accepted lane-level cross-sections into an xodr_config dict."""
    arms_by_id = {arm["id"]: arm for arm in arms}
    config_arms = []
    note_parts = []
    for lane_arm in lane_result["arms"]:
        arm = arms_by_id[lane_arm["arm_id"]]
        lanes = [_lane_config_entry(item) for item in lane_arm["cross_section"] if item.get("type") == LANE_TYPE]
        heading = arm_heading_midpoint(arm)
        config_arm = {
            "id": arm["id"],
            "heading_deg": heading,
            "width": arm["width"],
            "length": arm["length"],
            "lanes": lanes,
        }
        config_arms.append(config_arm)
    features = []
    for feature in enrich_channelized_features(channelized_features, arms, driving_side):
        if feature.get("from_arm") is None or feature.get("to_arm") is None:
            continue
        from_arm, to_arm = int(feature["from_arm"]), int(feature["to_arm"])
        features.append({
            "arm_list": [from_arm, to_arm],
            "from_arm": from_arm,
            "to_arm": to_arm,
            "source_type": normalize_channelized_source_type(feature.get("source_type")),
        })
    through_road_items = []
    for item in through_roads or []:
        arm_ids = item.get("arm_ids") or []
        if len(arm_ids) < 2:
            continue
        through_road_items.append({"arm_ids": [int(arm_ids[0]), int(arm_ids[1])]})
    return _apply_channelized_slip_turns({
        "reasoning": "Direct conversion from accepted lane-level cross-sections.",
        "notes": " | ".join(note_parts),
        "arms": config_arms,
        "channelized_features": features,
        "through_roads": through_road_items,
        "driving_side": normalize_driving_side(driving_side),
        "name": name,
    })
