"""Shared refinement helpers and structured validation errors."""

import json
from pathlib import Path

from scripts.methods.connection_parser import (
    MOVEMENT_MAP,
    canonicalize_movement_code,
    is_unknown,
)
from scripts.methods.llm_utils import BaseLLM, dump_json, to_jsonable

LAYER_FOLDERS = {
    "check": "layer_1",
    "fix": "layer_2",
    "fix_full_cs": "layer_2",
    "layer3": "layer_3",
    "fea": "layer_fea",
    "fea_scene": "layer_fea_scene",
}

_REFINE_FALLBACK = (
    "Config rejected (round {round_no}/{max_rounds}). "
    "Re-inspect the affected arm mouth span and patch only what is visible there."
)

FULL_CS_REGEN_CONSTRAINTS = frozenset({
    "incoming_lane_has_outflow",
    "outgoing_lane_has_inflow",
    "one_to_one_mapping",
})
CHECK_CONSTRAINTS = frozenset({
    "through_road_lane_balance",
    "through_road_opposing_traffic",
})
LAYER3_CONSTRAINTS = frozenset({
    "inter_junction_lane_balance",
    "inter_junction_opposing_traffic",
    "median_through_one_way",
})
LAYER3_MEDIAN_CONSTRAINT = "median_through_one_way"

_REFINE_HEADERS = {
    "check": (
        "Layer-1 through-road CHECK failed (round {round_no}/{max_rounds}; not a hard constraint). "
        "Re-identify THIS arm from the extended crop using the compact THIS/peer summary.\n"
    ),
    "fix": (
        "Layer-2 HARD connectivity constraint failed (round {round_no}/{max_rounds}). "
        "Re-identify THIS arm from the original mouth crop using the validation error "
        "and THIS/related-arm summary.\n"
    ),
    "fix_full_cs": (
        "Layer-2 HARD connectivity constraint failed (round {round_no}/{max_rounds}). "
        "Re-identify THIS arm from the original mouth crop using the validation error "
        "and THIS/related-arm summary.\n"
    ),
    "layer3": (
        "Layer-3 HARD inter-junction constraint failed (round {round_no}/{max_rounds}). "
        "Re-identify THIS arm from the extended crop using the compact THIS/far-end summary.\n"
    ),
}


def _arm_has_full_cs_regen_issue(validation_error, arm_id):
    if arm_id is None:
        return False
    arm_id = int(arm_id)
    for issue in (validation_error or {}).get("issues") or []:
        if issue.get("constraint") not in FULL_CS_REGEN_CONSTRAINTS:
            continue
        if arm_id in {int(x) for x in (issue.get("arm_id_list") or []) if x is not None}:
            return True
        location = issue.get("location") or {}
        details = issue.get("details") or {}
        for key in ("arm", "from_arm", "to_arm"):
            for src in (location, details):
                value = src.get(key)
                if value is not None and int(value) == arm_id:
                    return True
    return False


def _check_reidentify_summaries(validation_error, arm_id, *, lane_results=None, constraints=None):
    """Return (this_summary, peer_summary) for peer-consistency re-identify.

    When ``lane_results`` is provided, rebuild both summaries from the live
    readings so a later arm sees a peer that was already re-identified.
    """
    if arm_id is None:
        return "", ""
    arm_id = int(arm_id)
    allowed = constraints if constraints is not None else CHECK_CONSTRAINTS
    frozen_this, frozen_peer = "", ""
    peer_arm = None
    peer_lane = None
    peer_junction = None
    this_junction = None
    for issue in (validation_error or {}).get("issues") or []:
        if issue.get("constraint") not in allowed:
            continue
        location = issue.get("location") or {}
        if location.get("arm") is None or int(location["arm"]) != arm_id:
            continue
        details = issue.get("details") or {}
        peer_arm = details.get("peer_arm")
        peer_lane = details.get("peer_lane_result")
        peer_junction = details.get("peer_junction")
        this_junction = details.get("this_junction")
        frozen_this = details.get("this_summary") or ""
        frozen_peer = details.get("peer_summary") or ""
        if frozen_this or frozen_peer or peer_arm is not None:
            break
    if lane_results is not None:
        from scripts.methods.llm_kg.refinement.layer1_tr_check import arm_interpretation_summary
        this_summary = arm_interpretation_summary(lane_results.get(arm_id))
        if peer_lane is not None:
            return this_summary, arm_interpretation_summary(peer_lane)
        same_junction = (
            peer_junction is None
            or this_junction is None
            or int(peer_junction) == int(this_junction)
        )
        if peer_arm is not None and same_junction:
            return this_summary, arm_interpretation_summary(lane_results.get(int(peer_arm)))
        return this_summary, frozen_peer
    return frozen_this, frozen_peer


def _layer3_constraint_for_arm(validation_error, arm_id):
    if arm_id is None:
        return None
    arm_id = int(arm_id)
    found = None
    for issue in (validation_error or {}).get("issues") or []:
        constraint = issue.get("constraint")
        if constraint not in LAYER3_CONSTRAINTS:
            continue
        location = issue.get("location") or {}
        if location.get("arm") is None or int(location["arm"]) != arm_id:
            continue
        found = constraint
        if constraint == LAYER3_MEDIAN_CONSTRAINT:
            return constraint
    return found


def build_refinement_feedback(round_no, max_rounds, *, validation_error=None, arm_id=None, layer="fix", lane_results=None):
    if layer == "check":
        this_summary, peer_summary = _check_reidentify_summaries(
            validation_error, arm_id, lane_results=lane_results,
        )
        if this_summary or peer_summary:
            from scripts.methods.llm_kg.refinement.prompts import through_road_reidentify_user_text
            return through_road_reidentify_user_text(this_summary, peer_summary)
        return _REFINE_HEADERS["check"].format(round_no=round_no, max_rounds=max_rounds)
    if layer == "layer3":
        this_summary, peer_summary = _check_reidentify_summaries(
            validation_error, arm_id, lane_results=lane_results,
            constraints=LAYER3_CONSTRAINTS,
        )
        if this_summary or peer_summary:
            from scripts.methods.llm_kg.refinement.prompts import (
                inter_junction_reidentify_user_text,
                median_through_one_way_reidentify_user_text,
            )
            if _layer3_constraint_for_arm(validation_error, arm_id) == LAYER3_MEDIAN_CONSTRAINT:
                return median_through_one_way_reidentify_user_text(this_summary, peer_summary)
            return inter_junction_reidentify_user_text(this_summary, peer_summary)
        return _REFINE_HEADERS["layer3"].format(round_no=round_no, max_rounds=max_rounds)
    if layer in ("fix", "fix_full_cs"):
        from scripts.methods.llm_kg.refinement.layer2_con_fix import (
            connectivity_reidentify_summaries,
            format_connectivity_errors,
        )
        from scripts.methods.llm_kg.refinement.prompts import connectivity_reidentify_user_text
        this_summary, related_summary = connectivity_reidentify_summaries(
            validation_error, arm_id, lane_results=lane_results,
        )
        if this_summary or related_summary:
            return connectivity_reidentify_user_text(
                this_summary, related_summary,
                error_text=format_connectivity_errors(validation_error, arm_id),
            )
        return _REFINE_HEADERS["fix"].format(round_no=round_no, max_rounds=max_rounds)
    checks = list(((validation_error or {}).get("llm_feedback") or {}).get("checks") or [])
    if arm_id is not None:
        checks = [check for check in checks if check.get("arm") is not None and int(check["arm"]) == int(arm_id)]
    if not checks:
        return _REFINE_FALLBACK.format(round_no=round_no, max_rounds=max_rounds)
    header_tmpl = _REFINE_HEADERS.get(layer) or _REFINE_HEADERS["fix"]
    header = header_tmpl.format(round_no=round_no, max_rounds=max_rounds)
    keys = ("arm", "field", "reported_value", "why_recheck", "visual_evidence")
    contested = {"contested_readings": [{key: check.get(key) for key in keys} for check in checks]}
    return header + json.dumps(contested, ensure_ascii=False, indent=2)


ORDINALS = {1: '1st', 2: '2nd', 3: '3rd'}


def code_display(code):
    if code is None:
        return 'null (no arrow was read)'
    key = canonicalize_movement_code(code)
    return 'null (no arrow was read)' if is_unknown(key) or key == '' else f'"{key}"'


def ordinal(n):
    return ORDINALS.get(int(n), f'{int(n)}th')


def lane_reference(arm_id, lane_no, lane_count=None):
    total = f' of {lane_count}' if lane_count else ''
    last = str(lane_count) if lane_count else 'the last'
    return (
        f'arm {arm_id}, incoming lane {lane_no}{total} = the {ordinal(lane_no)} item with '
        f'type="lane" and travel="incoming" in that arm\'s cross_section, counting left to right. '
        f'Lane 1 is the innermost incoming lane (next to the centerline or median); lane {last} is the curb-side one.'
    )


def outgoing_reference(arm_id, count=None):
    now = '' if count is None else f' (you reported {count})'
    return f'arm {arm_id}, outgoing lane count{now} = the number of items with type="lane" and travel="outgoing" in that arm\'s cross_section.'


def incoming_reference(arm_id, count=None):
    now = '' if count is None else f' (you reported {count})'
    return f'arm {arm_id}, incoming lane count{now} = the number of items with type="lane" and travel="incoming" in that arm\'s cross_section.'


def make_issue(stage, constraint, message, *, location=None, details=None, arm_id_list=None):
    location = {k: v for k, v in (location or {}).items() if v is not None}
    details = details or {}
    resolved_arm_id_list = arm_id_list if arm_id_list is not None else infer_arm_id_list(location, details)
    issue = {'stage': stage, 'constraint': constraint, 'message': message, 'location': location, 'details': details}
    if resolved_arm_id_list:
        issue['arm_id_list'] = sorted({int(arm_id) for arm_id in resolved_arm_id_list})
    return issue


def infer_arm_id_list(location=None, details=None):
    location = location or {}
    details = details or {}
    arm_ids = set()
    for key in ('arm', 'from_arm', 'to_arm'):
        value = location.get(key)
        if value is not None:
            arm_ids.add(int(value))
        value = details.get(key)
        if value is not None:
            arm_ids.add(int(value))
    for key in ('arm_id_list', 'arm_list'):
        values = details.get(key)
        if isinstance(values, list):
            for item in values:
                if item is not None:
                    arm_ids.add(int(item))
    for slot in details.get('uncertain_slots') or []:
        if isinstance(slot, dict):
            for key in ('arm', 'from_arm', 'to_arm'):
                item = slot.get(key)
                if item is not None:
                    arm_ids.add(int(item))
    for blocked in details.get('blocking_issues') or []:
        if isinstance(blocked, dict):
            for item in blocked.get('arm_id_list') or []:
                if item is not None:
                    arm_ids.add(int(item))
            arm_ids.update(infer_arm_id_list(blocked.get('location') or {}, blocked.get('details') or {}))
    return sorted(arm_ids)


def issue_arm_ids(issues, all_arm_ids=None):
    arm_ids = set()
    for issue in issues:
        for item in issue.get('arm_id_list') or []:
            if item is not None:
                arm_ids.add(int(item))
        arm_ids.update(infer_arm_id_list(issue.get('location') or {}, issue.get('details') or {}))
    if arm_ids:
        return sorted(arm_ids)
    return sorted(all_arm_ids or [])


def index_issues_for_refinement(issues):
    by_arm = {}
    seen_by_arm = {}
    global_issues = []
    seen_global = set()

    def add_to_arm(arm_id, issue):
        key = str(int(arm_id))
        seen = seen_by_arm.setdefault(key, set())
        issue_id = id(issue)
        if issue_id in seen:
            return
        seen.add(issue_id)
        by_arm.setdefault(key, []).append(issue)

    for issue in issues:
        arm_id_list = issue.get('arm_id_list') or infer_arm_id_list(issue.get('location') or {}, issue.get('details') or {})
        if not arm_id_list:
            issue_id = id(issue)
            if issue_id not in seen_global:
                seen_global.add(issue_id)
                global_issues.append(issue)
        for arm_id in arm_id_list:
            add_to_arm(arm_id, issue)
        details = issue.get('details') or {}
        for blocked in details.get('blocking_issues') or []:
            if isinstance(blocked, dict):
                blocked_arms = blocked.get('arm_id_list') or infer_arm_id_list(blocked.get('location') or {}, blocked.get('details') or {})
                for arm_id in blocked_arms:
                    add_to_arm(arm_id, blocked)
    return {'arm_id_list': sorted(int(key) for key in by_arm), 'by_arm': by_arm, 'global': global_issues}


def _movement_name(move):
    return MOVEMENT_MAP.get(str(move), str(move))


def _fmt_unreachable_outflow(details):
    items = details.get("unreachable_movements") or []
    if not items and details.get("movement"):
        items = [{
            "movement": details["movement"],
            "target_arm": details.get("target_arm"),
            "target_outgoing": details.get("target_outgoing"),
            "why": details.get("why"),
        }]
    parts = []
    for item in items:
        letter = item.get("movement")
        if not letter:
            continue
        clause = "Reported %s has no receiving outgoing" % letter
        target = item.get("target_arm")
        outgoing = item.get("target_outgoing")
        if target is not None and int(outgoing or 0) <= 0:
            clause += " (arm %s is inbound-only)" % target
        elif target is not None:
            clause += " (arm %s)" % target
        elif item.get("why"):
            clause += " (%s)" % item["why"]
        parts.append(clause)
    return "; ".join(parts)


def _outflow_why_recheck(details, *, fallback):
    """Reported movement with no receiving outgoing."""
    text = (_fmt_unreachable_outflow(details) or fallback).rstrip(".")
    return text + "."


def _arrow_visual_evidence(arm, lane, move=None):
    if lane is not None:
        return (
            f"Painted turn arrow inside arm {arm} incoming lane {lane} between its two separators "
            f"at the junction mouth."
        )
    if move:
        return (
            f"Painted arrows implying {_movement_name(move)} on arm {arm} incoming lanes at the junction mouth."
        )
    return f"Painted arrows on arm {arm} incoming lanes at the junction mouth."


def _capacity_structure_checks(issue):
    """Arm-level sender/receiver recount checks from incoming/outgoing count deltas."""
    location = issue.get('location', {})
    details = issue.get('details', {})
    side = details.get('side')
    from_arm = details.get('from_arm', location.get('from_arm', location.get('arm') if side != 'outgoing' else None))
    to_arm = details.get('to_arm', location.get('to_arm', location.get('arm') if side == 'outgoing' else None))
    excess = details.get('excess')
    checks = []
    # Issues are already split per arm; emit one check for the issue's own arm.
    arm = location.get('arm')
    if arm is None:
        return checks
    if side == 'outgoing' or (side is None and to_arm is not None and int(arm) == int(to_arm)):
        why = f'Arm {arm} outgoing is {excess} too few' if excess is not None else f'Arm {arm} outgoing count looks too low'
        if from_arm is not None:
            why += f' relative to arm {from_arm} incoming'
        checks.append({
            'arm': int(arm),
            'field': 'cross_section',
            'reported_value': details.get('reported_value') or 'outgoing lane count may be too low',
            'why_recheck': why + '.',
            'visual_evidence': f'Re-read arm {arm} mouth cross_section (opposing_separator and all ordinary lanes).',
        })
        return checks
    why = f'Arm {arm} incoming is {excess} too many' if excess is not None else f'Arm {arm} incoming count looks too high'
    if to_arm is not None:
        why += f' relative to arm {to_arm} outgoing'
    checks.append({
        'arm': int(arm),
        'field': 'cross_section',
        'reported_value': details.get('reported_value') or 'incoming lane count may be too high',
        'why_recheck': why + '.',
        'visual_evidence': f'Re-read arm {arm} mouth cross_section (opposing_separator and all ordinary lanes).',
    })
    return checks


def translate_issue_for_llm(issue):
    constraint = issue.get('constraint', 'unknown_constraint')
    location = issue.get('location', {})
    details = issue.get('details', {})
    arm = location.get('arm')
    lane = location.get('lane')
    move = details.get('movement')
    arrow = code_display(details['current_code']) if details.get('current_code') is not None else None
    field = location.get('field')
    if field is None and lane is not None:
        field = f'incoming_lanes[{lane}].arrow'
    elif field is None and constraint in ('lane_ordering', 'movement_contiguity', 'arm_no_valid_assignment', 'incoming_lane_has_outflow'):
        field = 'incoming_lanes'
    elif field is None and constraint == 'outgoing_lane_has_inflow':
        field = 'outgoing_lanes'
    elif field is None and constraint == 'movement_reachability':
        field = f'incoming_lanes.{move or "movement"}'
    elif field is None:
        field = constraint

    if details.get('current_code') is not None:
        reported_value = arrow
    elif details.get('reported_value') is not None:
        reported_value = details['reported_value']
    elif constraint == 'lane_presence_verification_required' and details.get('current_value') is not None:
        cur = details.get('current_value')
        nxt = details.get('consistent_value')
        edge = details.get('edge')
        if nxt is not None and cur is not None and nxt < cur:
            reported_value = f'{cur} mouth lanes LTR; drop {edge or "edge"} lane' + (f' #{lane}' if lane is not None else '')
        elif nxt is not None and cur is not None and nxt > cur:
            reported_value = f'{cur} mouth lanes LTR; add unresolved {edge or "edge"} lane'
        else:
            reported_value = cur
    elif details.get('current_value') is not None:
        reported_value = details['current_value']
    elif details.get('lane_codes') is not None:
        reported_value = details['lane_codes']
    else:
        reported_value = None

    if constraint == 'lane_presence_verification_required' and lane is not None:
        # Physical LTR mouth index (0-based in lanes[i]); not incoming-side numbering.
        field = f'lanes[{int(lane) - 1}]'

    soft_recount = (
        f'On arm {arm} at the junction mouth, re-count ordinary lanes and opposing_separator '
        f'(left of cut=outgoing, right=incoming).'
    )
    templates = {
        'through_road_lane_balance': {
            'why_recheck': (
                f'Through-road peer lane counts differ'
                + (f' vs arm {details.get("peer_arm")}' if details.get("peer_arm") is not None else '')
                + '; re-identify THIS arm from the extended image.'
            ),
            'visual_evidence': soft_recount,
        },
        'through_road_opposing_traffic': {
            'why_recheck': (
                f'Through-road opposing traffic disagrees with peer'
                + (f' arm {details.get("peer_arm")}' if details.get("peer_arm") is not None else '')
                + ' (one-way vs two-way); re-identify THIS arm from the extended image.'
            ),
            'visual_evidence': (
                f'On arm {arm}: decide whether opposing traffic exists, then set '
                f'opposing_separator.element_id (or null) with reasoning. '
                f'Do not start from lane counts.'
            ),
        },
        'inter_junction_lane_balance': {
            'why_recheck': (
                f'HARD: connecting-road far-end lane counts must match'
                + (f' vs junction {details.get("peer_junction")} arm {details.get("peer_arm")}'
                   if details.get("peer_arm") is not None else '')
                + '; re-identify THIS arm from the extended image as the other mouth '
                'of the same roadway.'
            ),
            'visual_evidence': soft_recount,
        },
        'inter_junction_opposing_traffic': {
            'why_recheck': (
                f'HARD: connecting-road opposing traffic must match the far-end arm'
                + (f' (junction {details.get("peer_junction")} arm {details.get("peer_arm")})'
                   if details.get("peer_arm") is not None else '')
                + ' (one-way vs two-way); re-identify THIS arm from the extended image.'
            ),
            'visual_evidence': (
                f'On arm {arm}: this is the same roadway as the far-end mouth, so '
                f'one-way vs two-way must agree. Set opposing_separator.element_id '
                f'(or null) with reasoning. Do not start from lane counts. '
                f'If THIS mouth is currently one-way and the far end is two-way, '
                f'look for the missing reverse travel rather than assuming the '
                f'far end should become one-way.'
            ),
        },
        'median_through_one_way': {
            'why_recheck': (
                'This through carriageway is median-linked to another junction '
                'and must be one-way so the two carriageways can merge into one '
                'two-way divided road'
                + (
                    f' (peer junction {details.get("peer_junction")} arm {details.get("peer_arm")})'
                    if details.get("peer_arm") is not None else ''
                )
                + '; re-identify THIS arm from the extended image.'
            ),
            'visual_evidence': (
                f'On arm {arm}: this is one carriageway of a divided road. '
                f'There must be no opposing traffic. Set opposing_separator.element_id '
                f'to null and one_way_travel to incoming or outgoing. '
                f'Do not start from lane counts.'
            ),
        },
        'arrow_verification_required': {
            'why_recheck': 'This reported arrow is among the readings that make the junction infeasible under current topology and connectivity assumptions.',
            'visual_evidence': f'Painted turn arrow inside arm {arm} incoming lane {lane} between its two separators at the junction mouth; confirm whether {arrow} is actually marked there.',
        },
        'outgoing_count_verification_required': {
            'why_recheck': 'This outgoing lane count is among the readings that make the junction infeasible under current topology and connectivity assumptions.',
            'visual_evidence': f'Number of distinct travel="outgoing" lanes on arm {arm} at the junction mouth; use pavement edges and longitudinal separators, not stop lines or crosswalks.',
        },
        'incoming_count_verification_required': {
            'why_recheck': 'This incoming lane count is among the readings that make the junction infeasible under current topology and connectivity assumptions.',
            'visual_evidence': f'Number of distinct travel="incoming" lanes on arm {arm} at the junction mouth, and the outgoing/incoming boundary (median, double_solid_line, or edge).',
        },
        'travel_verification_required': {
            'why_recheck': 'This fixed outgoing/incoming travel label is among the readings that make the junction infeasible under current topology and connectivity assumptions.',
            'visual_evidence': (
                f'On arm {arm} at the junction mouth, re-check lane.travel from opposing_separator: '
                f'left of the cut is travel=outgoing, right is travel=incoming. '
                f'One-way arms have opposing_separator.element_id null — shared painted orientation up=incoming, down=outgoing; keep every lane the same. '
                f'cue.orientation is visual only — do not rewrite it to match travel.'
            ),
        },
        'lane_presence_verification_required': {
            'why_recheck': issue.get('message') or (
                'The number or presence of ordinary mouth lanes on this arm is among the readings that make the junction infeasible under current topology and connectivity assumptions.'
            ),
            'visual_evidence': (
                f'On arm {arm} at the junction mouth, re-count ordinary lanes LTR using pavement edges and longitudinal separators. '
                f'Focus on the contested {details.get("edge") or "edge"} mouth lane'
                + (f' (LTR #{lane})' if lane is not None else '')
                + '. If that strip is NOT a real ordinary travel lane, DELETE it from cross_section '
                f'(and remove separators that only bounded that false lane). '
                f'If a real lane is missing at that edge, INSERT an ordinary lane with travel=incoming or travel=outgoing '
                f'(from opposing_separator). Do not keep an extra lane just to preserve a previous count.'
            ),
        },
        'not_repairable_at_lane_level': {
            'why_recheck': 'No lane-level reading change restores a consistent layout; the conflict appears to be arm-level topology.',
            'visual_evidence': 'Arm count, outward headings, and which roadways meet at the junction (arm-level geometry), not ordinary lane arrows.',
        },
        'lane_ordering': {
            'why_recheck': 'Incoming arrows are not ordered left -> straight -> right, which usually means an arrow was assigned to the wrong lane.',
            'visual_evidence': f'Which painted arrow lies inside each incoming lane on arm {arm}, left to right at the junction mouth.',
        },
        'movement_contiguity': {
            'why_recheck': (
                'No legal assignment of unread incoming lanes keeps each turn direction in a '
                'contiguous block, which usually means an arrow was assigned to the wrong lane.'
            ),
            'visual_evidence': f'Painted arrows in the intervening incoming lanes on arm {arm} at the junction mouth.',
        },
        'movement_reachability': {
            'why_recheck': _outflow_why_recheck(
                details,
                fallback=(
                    f'An arrow implies a {_movement_name(move)} movement, but no arm of this junction '
                    f'can receive it under fixed headings.'
                ),
            ),
            'visual_evidence': _arrow_visual_evidence(arm, lane, move),
        },
        'unfillable_n_lane': {
            'why_recheck': 'This lane has no readable arrow and no legal movement can be assigned under current assumptions.',
            'visual_evidence': f'Whether arm {arm} incoming lane {lane} is a real travel lane (vs buffer/median/island), and outgoing counts on neighbouring arms.',
        },
        'arm_no_valid_assignment': {
            'why_recheck': _outflow_why_recheck(
                details,
                fallback='No arrow assignment for this arm satisfies lane-connectivity rules under current assumptions.',
            ),
            'visual_evidence': _arrow_visual_evidence(arm, lane, details.get('movement')),
        },
        'no_feasible_fill_solution': {
            'why_recheck': 'Unread arrows cannot be filled into any consistent layout, so a reported lane or arrow is likely wrong.',
            'visual_evidence': 'The previously reported ordinary lanes and painted arrows at each suspect arm mouth.',
        },
        'outgoing_lane_has_inflow': {
            'why_recheck': 'This arm\'s outgoing count looks too high relative to feeding incoming under legal assignments.',
            'visual_evidence': f'travel="outgoing" items on arm {arm} at the junction mouth, and the opposing_separator boundary.',
        },
        'incoming_lane_has_outflow': {
            'why_recheck': 'This arm\'s incoming count looks too high relative to receiving outgoing under legal assignments.',
            'visual_evidence': f'Ordinary incoming lanes on arm {arm} at the junction mouth and the opposing_separator boundary.',
        },
        'io_completeness': {
            'why_recheck': 'No legal movement assignment keeps two incoming lanes of the same arm off the same destination lane.',
            'visual_evidence': f'Full ordinary cross_section of arm {arm} at the junction mouth: opposing_separator, lane count, and painted arrows.',
        },
        'one_to_one_mapping': {
            'why_recheck': (
                'Two incoming lanes of the same arm would share one destination lane. '
                'Incoming from different arms may merge; this arm cannot.'
            ),
            'visual_evidence': f'Full ordinary cross_section of arm {arm} at the junction mouth: opposing_separator, lane count, and painted arrows.',
        },
        'receiving_capacity': {
            'why_recheck': issue.get('message') or 'Incoming/outgoing counts for a source→target pair disagree.',
            'visual_evidence': f'Re-read arm {arm} mouth cross_section (opposing_separator and all ordinary lanes).',
        },
        'right_turn_pair': {
            'why_recheck': 'A channelized feature is not between a valid curb-side slip-turn arm pair.',
            'visual_evidence': 'Whether a separated slip lane physically connects two arms as an early curb-side turn.',
        },
        'channelized_forbids_slip_turn': {
            'why_recheck': (
                'This arm feeds a channelized slip lane, so only the curb-edge incoming '
                'lane may carry the slip turn; other incoming markings of that turn double-count the slip.'
            ),
            'visual_evidence': (
                f'Whether arm {arm} incoming lane {lane} is a main-road lane (must not be the slip turn) or '
                f'the curb-edge slip-fed incoming lane (slip turn only there).'
            ),
        },
        'channelized_forbids_right_turn': {
            'why_recheck': (
                'This arm feeds a channelized right-turn slip lane, so only the rightmost incoming '
                'lane may be R; other incoming R markings double-count the slip.'
            ),
            'visual_evidence': (
                f'Whether arm {arm} incoming lane {lane} is a main-road lane (must not be R) or '
                f'the rightmost slip-fed incoming lane (R only there).'
            ),
        },
        'channelized_from_arm_slip_travel': {
            'why_recheck': (
                'This arm feeds a channelized slip lane, so the curb-edge ordinary lane '
                'must be travel=incoming (the slip-fed lane), not outgoing.'
            ),
            'visual_evidence': (
                f'On arm {arm}, re-check the curb-edge ordinary mouth lane: it should be the '
                f'incoming slip-fed lane beside the channelized island.'
            ),
        },
        'channelized_from_arm_rightmost_travel': {
            'why_recheck': (
                'This arm feeds a channelized right-turn slip lane, so the rightmost ordinary lane '
                'must be travel=incoming (the slip-fed R lane), not outgoing.'
            ),
            'visual_evidence': (
                f'On arm {arm}, re-check the rightmost ordinary mouth lane: it should be the '
                f'incoming slip-fed lane beside the channelized island.'
            ),
        },
    }
    chosen = templates.get(constraint, {
        'why_recheck': issue.get('message', 'This reading conflicts with current topology/connectivity assumptions.'),
        'visual_evidence': 'The ordinary cross_section at the junction mouth for the affected arm: separators, lane count, and painted arrows.',
    })
    return {
        'arm': arm,
        'field': field,
        'reported_value': reported_value,
        'why_recheck': chosen['why_recheck'],
        'visual_evidence': chosen['visual_evidence'],
    }


def _inbound_only_target_arms(details):
    items = list((details or {}).get("unreachable_movements") or [])
    if not items and (details or {}).get("target_arm") is not None:
        items = [details]
    arms = []
    seen = set()
    for item in items:
        if item.get("target_arm") is None:
            continue
        if int(item.get("target_outgoing") or 0) > 0:
            continue
        arm = int(item["target_arm"])
        if arm in seen:
            continue
        seen.add(arm)
        arms.append(arm)
    return arms


def _unreachable_is_inbound_only(details):
    return bool(_inbound_only_target_arms(details))


def _inbound_only_structure_checks(arm):
    """When a movement has no receiver because an arm is inbound-only, re-read travel and cut."""
    arm = int(arm)
    return [
        {
            "arm": arm,
            "field": "lanes.travel",
            "reported_value": "inbound-only",
            "why_recheck": (
                f"A reported movement has no receiving outgoing because an arm is inbound-only. "
                f"Re-check lane.travel on arm {arm} — it may not be one-way incoming. "
                f"Do not only re-read the arrow."
            ),
            "visual_evidence": (
                f"On arm {arm} at the junction mouth, re-check lane.travel from opposing_separator. "
                f"An up-facing mouth arrow on only one side is not one-way incoming — "
                f"outgoing often has no mouth arrow (up=incoming, down=outgoing)."
            ),
        },
        {
            "arm": arm,
            "field": "opposing_separator",
            "reported_value": None,
            "why_recheck": (
                f"Arm {arm} is treated as inbound-only (no opposing separator). "
                f"Re-check whether a two-way cut exists at the mouth."
            ),
            "visual_evidence": (
                f"On arm {arm}, decide whether a two-way cut exists "
                f"(yellow / double solid / median between motor strips) and set "
                f"opposing_separator.element_id (or null). School/BUS facing is not enough for one-way."
            ),
        },
    ]


def _dedupe_checks(checks):
    seen = set()
    out = []
    for check in checks:
        key = (check.get("arm"), check.get("field"))
        if key in seen:
            continue
        seen.add(key)
        out.append(check)
    return out


def checks_for_issue(issue):
    if issue.get('constraint') in ('receiving_capacity', 'one_to_one_mapping'):
        expanded = _capacity_structure_checks(issue)
        if expanded:
            return expanded
    translated = translate_issue_for_llm(issue)
    # Prefer concrete lane targeting when the issue already names a lane.
    if translated.get('arm') is not None and issue.get('location', {}).get('lane') is not None:
        field = issue.get('location', {}).get('field')
        if field:
            translated['field'] = field
    checks = [translated]
    if (
        issue.get("constraint") in ("arm_no_valid_assignment", "movement_reachability")
        and translated.get("arm") is not None
        and _unreachable_is_inbound_only(issue.get("details") or {})
    ):
        checks.extend(_inbound_only_structure_checks(translated["arm"]))
    return checks


def build_llm_feedback(title, issues, *, repair=None, violations=None):
    checks = _dedupe_checks(
        check for issue in issues for check in checks_for_issue(issue)
    )
    refinement = index_issues_for_refinement(issues)
    target_arm_ids = sorted({int(check['arm']) for check in checks if check.get('arm') is not None})
    if target_arm_ids:
        refinement['arm_id_list'] = target_arm_ids
    feedback = {
        'title': title,
        'issues': issues,
        'refinement': refinement,
        'llm_feedback': {'checks': checks},
        'target_arm_ids': target_arm_ids,
    }
    if repair is not None:
        feedback['repair'] = repair
        feedback['escalate_to_arm_level'] = repair.get('diagnosis') == 'not_repairable_at_lane_level'
    if violations:
        feedback['violations'] = violations
    return feedback


class ConnectionParseError(ValueError):
    def __init__(self, issues, title='Validation failed', *, repair=None, violations=None):
        self.issues = list(issues)
        self.title = title
        self.repair = repair
        self.violations = list(violations or [])
        super().__init__(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    def to_dict(self):
        return build_llm_feedback(self.title, self.issues, repair=self.repair, violations=self.violations)


def fail(stage, constraint, message, *, location=None, details=None, arm_id_list=None):
    return ConnectionParseError(
        [make_issue(stage, constraint, message, location=location, details=details, arm_id_list=arm_id_list)],
        title=f'[{stage}] {constraint}',
    )


def layer_result(*, passed, config, refine_rounds=0, validation_error=None, filled_config=None, connections=None, extra=None):
    out = {
        "passed": bool(passed),
        "config": config,
        "refine_rounds": int(refine_rounds),
        "validation_error": validation_error,
        "filled_config": filled_config,
        "connections": connections,
    }
    if extra:
        out.update(extra)
    return out


def log_issues(issues, *, indent=1, limit=8):
    for issue in issues[:limit]:
        loc = ", ".join(f"{k}={v}" for k, v in (issue.get("location") or {}).items()) or "global"
        BaseLLM.log(f"[{issue.get('stage')}/{issue.get('constraint')}] {loc}: {issue.get('message')}", indent=indent)
    if len(issues) > limit:
        BaseLLM.log(f"... and {len(issues) - limit} more issue(s)", indent=indent)


def log_repair_diagnosis(repair, *, indent=1):
    """Print fill_search repair candidates: each set of edits that restores feasibility."""
    from scripts.methods.llm_kg.refinement.fill_solver import format_repair_diagnosis
    for line in format_repair_diagnosis(repair):
        BaseLLM.log(line, indent=indent)


def log_validation_error(exc, *, round_no=None, max_rounds=None, layer=None):
    validation_error = exc.to_dict() if hasattr(exc, "to_dict") else {"issues": getattr(exc, "issues", [])}
    layer_tag = f"{layer}" if layer else "validate"
    round_tag = f" {round_no}/{max_rounds}" if round_no is not None else ""
    issues = validation_error.get("issues") or []
    targets = validation_error.get("target_arm_ids") or []
    BaseLLM.log(
        f"[validate:{layer_tag}]{round_tag} {validation_error.get('title') or 'failed'} "
        f"(issues={len(issues)}, arms={targets})"
    )
    repair = validation_error.get("repair") or getattr(exc, "repair", None)
    if repair:
        log_repair_diagnosis(repair, indent=1)
    log_issues(issues, indent=1)
    return validation_error


def target_arm_ids_from_feedback(validation_error, issues, all_arm_ids):
    known = set(all_arm_ids)
    target = validation_error.get("target_arm_ids") or ((validation_error.get("refinement") or {}).get("arm_id_list"))
    if target:
        return [int(arm_id) for arm_id in target if int(arm_id) in known]
    return [arm_id for arm_id in issue_arm_ids(issues, None) if arm_id in known]


def refinement_arm_ids(issues, validation_error, all_arm_ids):
    arm_ids = target_arm_ids_from_feedback(validation_error, issues, all_arm_ids)
    if not arm_ids:
        BaseLLM.log("[refinement] no target arms; skip", indent=1)
    return arm_ids


def refinement_layer_dir(refinement_root, layer):
    return Path(refinement_root) / LAYER_FOLDERS[layer]


def refinement_round_dir(refinement_root, layer, round_no):
    return refinement_layer_dir(refinement_root, layer) / f"round_{round_no}"


def save_refinement_round_dir(round_dir, *, validation_error):
    """Create the round directory and write validation_error.json."""
    round_dir = Path(round_dir)
    round_dir.mkdir(parents=True, exist_ok=True)
    dump_json(round_dir / "validation_error.json", validation_error)
    return round_dir


def lanes_doc_from_ctx(ctx):
    """Current arm cross-sections as a lanes document."""
    from scripts.methods.llm_kg.lane_level.llm_gen import _lane_result_doc

    doc = dict(_lane_result_doc(ctx.lane_results or {}))
    doc["channelized_features"] = ctx.channelized_features
    doc["through_roads"] = ctx.through_roads
    doc["driving_side"] = getattr(ctx, "driving_side", None)
    image_path = getattr(ctx, "image_path", None)
    if image_path is not None:
        doc["source_image"] = str(image_path)
    return doc


def save_layer_result(refinement_root, layer, *, passed, refine_rounds=0, validation_error=None, lanes_doc=None, extra=None):
    """Write ``layer_*/result.json`` (and ``lanes.json`` when provided) for metrics."""
    if refinement_root is None:
        return None
    layer_dir = refinement_layer_dir(refinement_root, layer)
    layer_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": bool(passed),
        "refine_rounds": int(refine_rounds),
        "validation_error": validation_error,
    }
    if extra:
        payload.update(extra)
    dump_json(layer_dir / "result.json", payload)
    if lanes_doc is not None:
        dump_json(layer_dir / "lanes.json", lanes_doc)
    return payload


def _arm_refine_conversation_entry(agent, *, model, msg_start, patch_start, crop_ids_before):
    return list(agent.get("llm_calls") or [])


def save_layer_refine_conversation(layer_dir, *, round_id, arm_id, agent, model, msg_start, patch_start, crop_ids_before):
    """Upsert one arm turn into ``layer_dir/conversation.json`` keyed by round_id then arm_id."""
    layer_dir = Path(layer_dir)
    layer_dir.mkdir(parents=True, exist_ok=True)
    path = layer_dir / "conversation.json"
    doc = {}
    if path.is_file():
        doc = json.loads(path.read_text(encoding="utf-8"))
    round_key = str(round_id)
    arm_key = str(arm_id)
    round_doc = doc.setdefault(round_key, {})
    round_doc[arm_key] = _arm_refine_conversation_entry(
        agent, model=model, msg_start=msg_start, patch_start=patch_start, crop_ids_before=crop_ids_before,
    )
    dump_json(path, doc)


def save_refine_artifacts(save_dir, loop_out):
    fill_path = save_dir / "xodr_config_fill.json"
    conns_path = save_dir / "connections.json"
    valerr_path = save_dir / "validation_error.json"
    accepted = loop_out["accepted"]
    refine_rounds = loop_out["refine_rounds"]
    suffix = f" after {refine_rounds} round(s)" if refine_rounds else ""
    if accepted:
        dump_json(fill_path, loop_out["filled_config"])
        dump_json(conns_path, loop_out["connections"])
        BaseLLM.log(f"[validate] PASS{suffix}; saved {fill_path.name}, {conns_path.name}")
    else:
        dump_json(valerr_path, loop_out["validation_error"])
        BaseLLM.log(f"[validate] FAIL{suffix}; saved {valerr_path.name}")
    return fill_path, conns_path, valerr_path, accepted
