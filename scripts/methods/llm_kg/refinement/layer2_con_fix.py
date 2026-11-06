"""Layer 2: HARD connectivity constraint + sequential mouth-crop re-identification.

Runs only when neither layer-1 nor the original lane-level reading is
connection-valid; in that case the L1 result is kept as the L2 starting point.
Connectivity must hold: some legal fill gives every incoming an outflow and
every outgoing an inflow. Same-arm incoming lanes cannot share a destination
lane; incoming from different arms may merge onto one outgoing. Remaining
violations fail the pipeline.
Failing arms are re-identified from the original mouth crop using the lane-level
flow (system prompt, initial inspect, resolution inspect). The user message
includes the validation error plus a compact THIS/related-arm summary. The new
reading fully replaces the old one.

Local arrow/order/channelized checks are not part of this layer; fill_solver
still imports those helpers for its own candidate validation.
"""

from collections import defaultdict

from scripts.methods.connection_parser import (
    DEFAULT_DIAGNOSIS_LEAF_BUDGET,
    DEFAULT_DIAGNOSIS_NODE_BUDGET,
    DEFAULT_PER_ARM_CAP,
    INCOMING_LANE_1_IS_LEFTMOST,
    ORDER,
    allowed_movement_codes,
    build_connections_output,
    canonical_code,
    channelized_origin_arms,
    channelized_source_of,
    channelized_turn_name,
    collect_channelized_features,
    entry_letters,
    find_target_arm,
    is_contiguous,
    is_unknown,
    legal_outflow_letters,
    letters_to_code,
    is_valid_movement_code,
    load_junction_from_dict,
    movement_codes,
    movement_has_outflow,
    normalize_driving_side,
    resolve_channelized_links,
    slip_assigned_code,
    slip_edge,
    slip_turn_letter,
)
from scripts.methods.llm_kg.refinement.common import (
    ConnectionParseError,
    code_display,
    dump_json,
    lanes_doc_from_ctx,
    layer_result,
    make_issue,
    refinement_layer_dir,
    save_layer_result,
)
from scripts.methods.llm_kg.refinement.loop import run_refine_loop
from scripts.methods.llm_utils import BaseLLM, CostTracker

HARD_IO_PER_ARM_CAP = DEFAULT_PER_ARM_CAP
HARD_IO_NODE_BUDGET = DEFAULT_DIAGNOSIS_NODE_BUDGET
HARD_IO_LEAF_BUDGET = DEFAULT_DIAGNOSIS_LEAF_BUDGET

_UNKNOWN_CONNECTIVITY_SUMMARY = (
    "- outgoing lanes: unknown\n"
    "- incoming lanes: unknown\n"
    "- opposing traffic: unknown\n"
    "- lanes LTR: unknown"
)


def _lane_arrow_code(item):
    letters = item.get("movements") if item else None
    if not letters:
        for cue in (item or {}).get("cues") or []:
            if cue.get("kind") == "arrow":
                letters = cue.get("movements") or []
                break
    if not letters:
        return "none"
    if isinstance(letters, (list, tuple)):
        return "".join(str(letter) for letter in letters) or "none"
    return str(letters) or "none"


def _lane_travel_label(item):
    travel = (item or {}).get("travel")
    if travel in ("incoming", "outgoing"):
        return travel
    if travel:
        return str(travel)
    return "travel=null"


def arm_connectivity_summary(lane_arm):
    """Compact THIS/related interpretation: IO counts plus each lane's arrow."""
    from scripts.methods.llm_kg.refinement.layer1_tr_check import (
        _has_opposing_traffic,
        arm_io_counts,
    )
    if not lane_arm:
        return _UNKNOWN_CONNECTIVITY_SUMMARY
    lanes = [item for item in (lane_arm.get("cross_section") or []) if item.get("type") == "lane"]
    if not lanes and lane_arm.get("lanes"):
        lanes = list(lane_arm.get("lanes") or [])
    out_n, in_n = arm_io_counts(lanes)
    opp = "yes" if _has_opposing_traffic(lane_arm) else "no"
    if lanes:
        lane_lines = "\n".join(
            f"  {idx}. {_lane_travel_label(item)}, arrow: {_lane_arrow_code(item)}"
            for idx, item in enumerate(lanes, start=1)
        )
    else:
        lane_lines = "  (none)"
    return (
        f"- outgoing lanes: {out_n}\n"
        f"- incoming lanes: {in_n}\n"
        f"- opposing traffic: {opp}\n"
        f"- lanes LTR:\n{lane_lines}"
    )


def related_arm_ids_for_issue(issue, this_arm=None):
    ids = []
    seen = set()
    this_arm = int(this_arm) if this_arm is not None else None

    def _add(value):
        if value is None:
            return
        arm_id = int(value)
        if this_arm is not None and arm_id == this_arm:
            return
        if arm_id in seen:
            return
        seen.add(arm_id)
        ids.append(arm_id)

    loc = issue.get("location") or {}
    details = issue.get("details") or {}
    for src in (loc, details):
        for key in ("from_arm", "to_arm", "peer_arm"):
            _add(src.get(key))
    for value in issue.get("arm_id_list") or details.get("arm_id_list") or []:
        _add(value)
    return ids


def format_related_arm_summaries(related):
    if not related:
        return "- (no related arm reading)"
    blocks = []
    for arm_id, summary in related:
        blocks.append(f"arm {arm_id}:\n{summary}")
    return "\n\n".join(blocks)


def _issue_arm_ids(issue):
    ids = set()
    loc = issue.get("location") or {}
    details = issue.get("details") or {}
    for src in (loc, details):
        for key in ("arm", "from_arm", "to_arm"):
            if src.get(key) is not None:
                ids.add(int(src[key]))
    for value in issue.get("arm_id_list") or details.get("arm_id_list") or []:
        if value is not None:
            ids.add(int(value))
    return ids


def format_connectivity_errors(validation_error, arm_id=None):
    """Compact validation-error text for the layer-2 re-identify user message."""
    issues = list((validation_error or {}).get("issues") or [])
    this_issues = []
    other_issues = []
    for issue in issues:
        arms = _issue_arm_ids(issue)
        if arm_id is not None and arms and int(arm_id) in arms:
            this_issues.append(issue)
        else:
            other_issues.append(issue)
    lines = []
    seen = set()
    for issue in this_issues + other_issues:
        constraint = issue.get("constraint") or "unknown"
        message = (issue.get("message") or "").rstrip(".")
        loc = issue.get("location") or {}
        arm = loc.get("arm")
        prefix = f"arm {arm} " if arm is not None else ""
        line = f"- {prefix}{constraint}: {message}." if message else f"- {prefix}{constraint}."
        if line in seen:
            continue
        seen.add(line)
        lines.append(line)
    return "\n".join(lines) if lines else "- (none)"


def connectivity_reidentify_summaries(validation_error, arm_id, *, lane_results=None):
    """Return (this_summary, related_text) for layer-2 re-identify.

    When ``lane_results`` is provided, rebuild from live readings so a later
    arm sees related arms that were already re-identified.
    """
    if arm_id is None:
        return "", ""
    arm_id = int(arm_id)
    frozen_this, frozen_related = "", []
    matched = None
    for issue in (validation_error or {}).get("issues") or []:
        location = issue.get("location") or {}
        details = issue.get("details") or {}
        loc_arm = location.get("arm")
        mentions = loc_arm is not None and int(loc_arm) == arm_id
        if not mentions:
            mentions = arm_id in {int(x) for x in (issue.get("arm_id_list") or []) if x is not None}
        if not mentions:
            continue
        matched = issue
        frozen_this = details.get("this_summary") or ""
        frozen_related = list(details.get("related_summaries") or [])
        break
    if lane_results is not None:
        this_summary = arm_connectivity_summary(lane_results.get(arm_id))
        pair_ids = related_arm_ids_for_issue(matched or {}, arm_id)
        related = []
        seen = set()
        for other_id in list(pair_ids) + sorted(int(x) for x in lane_results):
            if other_id == arm_id or other_id in seen:
                continue
            seen.add(other_id)
            related.append((other_id, arm_connectivity_summary(lane_results.get(other_id))))
        return this_summary, format_related_arm_summaries(related)
    related = []
    for item in frozen_related:
        if isinstance(item, dict):
            related.append((item.get("arm"), item.get("summary") or _UNKNOWN_CONNECTIVITY_SUMMARY))
        else:
            related.append((None, str(item)))
    return frozen_this, format_related_arm_summaries(related)


def _annotate_connectivity_summaries(issues, lane_results):
    lane_results = lane_results or {}
    for issue in issues or []:
        loc = dict(issue.get("location") or {})
        details = dict(issue.get("details") or {})
        arm_id = loc.get("arm")
        if arm_id is not None:
            details["this_summary"] = arm_connectivity_summary(lane_results.get(int(arm_id)))
        related = []
        pair_ids = related_arm_ids_for_issue(issue, arm_id)
        seen = set()
        for other_id in list(pair_ids) + sorted(int(x) for x in lane_results):
            if arm_id is not None and int(other_id) == int(arm_id):
                continue
            if other_id in seen:
                continue
            seen.add(other_id)
            related.append({
                "arm": other_id,
                "summary": arm_connectivity_summary(lane_results.get(other_id)),
            })
        details["related_summaries"] = related
        issue["details"] = details


def _left_to_right(letters_per_lane):
    return letters_per_lane if INCOMING_LANE_1_IS_LEFTMOST else list(reversed(letters_per_lane))


def _open_lane_codes(entry, *, allow_lr):
    """Legal movement codes for an unread / multi-candidate incoming slot."""
    allowed = list(movement_codes(allow_lr=allow_lr))
    if is_unknown(entry) or entry is None or entry == []:
        return allowed
    if isinstance(entry, list):
        opts = []
        for raw in entry:
            code = canonical_code(raw)
            if is_unknown(code):
                continue
            if is_valid_movement_code(entry_letters(code), allow_lr=allow_lr) and code not in opts:
                opts.append(code)
        return opts or allowed
    code = canonical_code(entry)
    if is_unknown(code):
        return allowed
    if is_valid_movement_code(entry_letters(code), allow_lr=allow_lr):
        return [code]
    return allowed


def _letter_rows_are_contiguous(letter_rows):
    indices = defaultdict(list)
    for idx, letters in enumerate(letter_rows, start=1):
        for move in letters or ():
            indices[move].append(idx)
    return all((not ids) or is_contiguous(ids) for ids in indices.values())


def _contiguity_broken_after_fill(lanes, *, allow_lr, incoming_entries=None):
    """True iff every LTR-ordered fill of open lanes is non-contiguous.

    Open slots are enumerated from ``incoming_entries`` (unknown → all legal
    codes; multi-candidate → those codes). A single contiguous ordered fill
    is enough to pass. If no ordered fill exists, ordering/validity handles
    it — this is not a contiguity failure.
    """
    n = len(lanes)
    entries = list(incoming_entries or [])
    if len(entries) != n:
        entries = [None] * n
    choices = []
    for i, letters in enumerate(lanes):
        if letters:
            if not is_valid_movement_code(letters, allow_lr=allow_lr):
                return False
            choices.append([set(letters)])
        else:
            codes = _open_lane_codes(entries[i], allow_lr=allow_lr)
            if not codes:
                return False
            choices.append([entry_letters(code) for code in codes])
    ltr_index = list(range(n)) if INCOMING_LANE_1_IS_LEFTMOST else list(range(n - 1, -1, -1))
    chosen = [None] * n
    found_ordered = False

    def recurse(t, prev_low, prev_high):
        nonlocal found_ordered
        if t == len(ltr_index):
            found_ordered = True
            return _letter_rows_are_contiguous(chosen)
        pos = ltr_index[t]
        for letters in choices[pos]:
            low = min(ORDER[c] for c in letters)
            high = max(ORDER[c] for c in letters)
            if low < prev_low or high < prev_high:
                continue
            chosen[pos] = letters
            if recurse(t + 1, low, high):
                return True
        return False

    if recurse(0, -1, -1):
        return False
    return found_ordered


def _slip_lane_reading(arm, driving_side):
    """Ordinary mouth lane that feeds the channelized slip (LTR edge)."""
    if not arm.lane_readings:
        return None
    if normalize_driving_side(driving_side) == 'left':
        return arm.lane_readings[-1]
    return arm.lane_readings[0]


def _source_type_for_arm(raw, arm_id, channelized=None):
    """shared/dedicated from the arm-level channelized feature."""
    return channelized_source_of(channelized, arm_id)


def _force_channelized_from_arm_slip(spec, marking, raw, arm_id, channelized=None):
    """Slip code on spec, marking, and xodr_config. dedicated → R/L; shared → SR/LS."""
    side = normalize_driving_side(getattr(spec, 'driving_side', None))
    code = slip_assigned_code(_source_type_for_arm(raw, arm_id, channelized), side)
    letters = entry_letters(code)
    arm = spec.arm_map[arm_id]
    slip_reading = _slip_lane_reading(arm, side)
    slip_reading.movements = [code]
    arm.refresh_incoming_from_readings()
    if marking.get(arm_id):
        if side == 'left':
            marking[arm_id][-1] = set(letters)
        else:
            marking[arm_id][0] = set(letters)
    for arm_data in (raw or {}).get('arms') or []:
        if int(arm_data.get('id')) != int(arm_id):
            continue
        lanes = arm_data.get('lanes')
        if isinstance(lanes, list) and lanes:
            idx = -1 if side == 'left' else 0
            lanes[idx] = dict(lanes[idx] or {})
            lanes[idx]['travel'] = 'incoming'
            lanes[idx]['movements'] = [code]
        incoming = arm_data.get('incoming_lanes')
        if isinstance(incoming, list) and incoming:
            idx = -1 if side == 'left' else 0
            incoming[idx] = code
        break


def hard_channelized_from_arm_issues(spec, marking, stage, channelized=None, raw=None):
    """Channelized from_arm: curb-edge must be incoming (assign slip turn); other incoming forbid it."""
    issues = []
    side = normalize_driving_side(getattr(spec, 'driving_side', None))
    edge = slip_edge(side)
    letter = slip_turn_letter(side)
    turn_name = channelized_turn_name(side)
    origins = channelized_origin_arms(channelized)
    for arm_id in sorted(origins):
        arm = spec.arm_map.get(int(arm_id))
        if arm is None or not arm.lane_readings:
            continue
        slip_reading = _slip_lane_reading(arm, side)
        if slip_reading.travel == 'outgoing':
            issues.append(make_issue(
                stage, 'channelized_from_arm_slip_travel',
                (
                    f'Arm {arm_id} feeds a channelized {turn_name} slip lane, but its {edge} '
                    f'ordinary lane is outgoing; the slip-fed lane must be the {edge} incoming.'
                ),
                location={'arm': int(arm_id), 'field': 'cross_section'},
                details={
                    'side': edge,
                    'travel': 'outgoing',
                    'slip_turn': letter,
                    'reported_value': f'{edge} ordinary lane is outgoing',
                },
                arm_id_list=[int(arm_id)],
            ))
            continue
        if slip_reading.travel != 'incoming':
            issues.append(make_issue(
                stage, 'channelized_from_arm_slip_travel',
                (
                    f'Arm {arm_id} feeds a channelized {turn_name} slip lane, but its {edge} '
                    f'ordinary lane travel is unresolved; it must be incoming (slip-fed {letter} lane).'
                ),
                location={'arm': int(arm_id), 'field': 'cross_section'},
                details={
                    'side': edge,
                    'travel': slip_reading.travel,
                    'slip_turn': letter,
                    'reported_value': f'{edge} ordinary lane travel unresolved',
                },
                arm_id_list=[int(arm_id)],
            ))
            continue
        if not arm.incoming_lanes:
            continue
        _force_channelized_from_arm_slip(spec, marking, raw, int(arm_id), channelized=channelized)
        lanes = marking.get(int(arm_id)) or []
        slip_idx = len(lanes) if side == 'left' else 1
        for idx, letters in enumerate(lanes, start=1):
            if idx == slip_idx:
                continue
            if letters and letter in letters:
                issues.append(make_issue(
                    stage, 'channelized_forbids_slip_turn',
                    (
                        f'Arm {arm_id} incoming lane #{idx} contains {letter}, but the channelized slip '
                        f'already uses the {edge} incoming lane.'
                    ),
                    location={'arm': int(arm_id), 'lane': idx},
                    details={
                        'current_code': letters_to_code(letters),
                        'lane_count': len(lanes),
                        'slip_turn': letter,
                        'reported_value': letters_to_code(letters),
                    },
                    arm_id_list=[int(arm_id)],
                ))
    return issues


def hard_lane_constraint_issues(spec, marking, stage, channelized=None, raw=None):
    """Per-arm structural issues only; no reachability/capacity/connectivity.

    Channelized from_arm slip checks are handled by ``hard_channelized_from_arm_issues``.
    """
    issues = []
    for arm_id, lanes in marking.items():
        allow_lr = spec.arm_allows_lr(arm_id)
        prev_low = prev_high = -1
        for letters in _left_to_right(lanes):
            if not letters:
                continue
            if not is_valid_movement_code(letters, allow_lr=allow_lr):
                issues.append(make_issue(
                    stage, 'movement_code_validity',
                    f'Arm {arm_id} has an illegal movement code.',
                    location={'arm': arm_id},
                    details={'code': letters_to_code(letters), 'allowed': allowed_movement_codes(allow_lr=allow_lr)},
                ))
                continue
            low = min((ORDER[c] for c in letters))
            high = max((ORDER[c] for c in letters))
            if low < prev_low or high < prev_high:
                issues.append(make_issue(
                    stage, 'lane_ordering',
                    f'Arm {arm_id} lanes are not ordered left->straight->right.',
                    location={'arm': arm_id},
                    details={'lane_codes': [code_display(letters_to_code(x)) for x in lanes], 'lane_count': len(lanes)},
                ))
                break
            prev_low, prev_high = low, high
        arm = (spec.arm_map or {}).get(int(arm_id)) if spec is not None else None
        incoming = list(arm.incoming_lanes) if arm is not None else None
        if _contiguity_broken_after_fill(lanes, allow_lr=allow_lr, incoming_entries=incoming):
            indices = defaultdict(list)
            for idx, letters in enumerate(lanes, start=1):
                for move in letters or ():
                    indices[move].append(idx)
            issues.append(make_issue(
                stage, 'movement_contiguity',
                f'Arm {arm_id} has no legal fill of unread lanes that keeps movements contiguous.',
                location={'arm': arm_id},
                details={
                    'lane_codes': [code_display(letters_to_code(x)) for x in lanes],
                    'assigned': {move: sorted(ids) for move, ids in indices.items()},
                    'lane_count': len(lanes),
                },
            ))
    return issues


def _arm_movement_outflow_diag(spec, arm, *, outgoing_override=None):
    """Reported incoming arrows vs which L/S/R still have receiving outgoing."""
    reported = []
    unreachable = []
    seen = set()
    for idx, lane in enumerate(arm.incoming_lanes, start=1):
        letters = entry_letters(lane)
        if not letters:
            continue
        reported.append({"lane": idx, "code": letters_to_code(letters)})
        for move in sorted(letters, key=lambda letter: ORDER[letter]):
            if move in seen or movement_has_outflow(
                spec, arm.id, move, outgoing_override=outgoing_override,
            ):
                continue
            seen.add(move)
            target_id = find_target_arm(spec, arm.id, move)
            item = {"movement": move, "lane": idx}
            if target_id is None:
                item["why"] = "no destination arm in that heading"
            else:
                out_n = (
                    outgoing_override.get(target_id)
                    if outgoing_override is not None
                    else spec.arm_map[target_id].outgoing_lanes
                )
                item["target_arm"] = int(target_id)
                item["target_outgoing"] = int(out_n or 0)
                item["why"] = "arm %s has no outgoing lane" % target_id
            unreachable.append(item)
    return reported, unreachable, legal_outflow_letters(
        spec, arm.id, outgoing_override=outgoing_override,
    )


def _arm_unfillable_issues(spec, arm, stage, *, outgoing_override=None):
    """When no travel-cut × movement fill exists, emit inflow/outflow correspondence issues."""
    reported, unreachable, legal = _arm_movement_outflow_diag(
        spec, arm, outgoing_override=outgoing_override,
    )
    details = {
        "lane_count": arm.incoming_lane_count,
        "exhaustive": True,
        "reported_value": "incoming lane(s) have no legal outflow",
        "reported_arrows": reported,
        "unreachable_movements": unreachable,
        "legal_outflow": legal,
    }
    location = {"arm": arm.id, "field": "incoming_lanes"}
    if unreachable:
        details["movement"] = unreachable[0]["movement"]
    issues = [make_issue(
        stage, "incoming_lane_has_outflow",
        "Arm %s incoming lane(s) have no legal outflow under current headings/outgoing counts." % arm.id,
        location=location,
        details=details,
    )]
    seen_targets = set()
    for item in unreachable:
        target = item.get("target_arm")
        if target is None or int(item.get("target_outgoing") or 0) > 0:
            continue
        target = int(target)
        if target in seen_targets:
            continue
        seen_targets.add(target)
        issues.append(make_issue(
            stage, "outgoing_lane_has_inflow",
            "Arm %s has no outgoing to receive inflow from arm %s." % (target, arm.id),
            location={"arm": target, "from_arm": arm.id, "field": "outgoing_lanes"},
            details={
                "from_arm": int(arm.id),
                "to_arm": target,
                "reported_value": "outgoing lane(s) have no legal inflow",
                "unreachable_movements": unreachable,
            },
        ))
    return issues


def _one_to_one_pair_issues(spec, marking, stage):
    """Same-arm incoming demand vs destination outgoing count (no same-arm merge)."""
    issues = []
    pair_demand = {}
    for arm_id, lanes in (marking or {}).items():
        counts = defaultdict(int)
        for letters in lanes:
            for move in letters or ():
                counts[move] += 1
        for move, count in counts.items():
            target_id = find_target_arm(spec, arm_id, move)
            if target_id is None:
                continue
            capacity = int(spec.arm_map[target_id].outgoing_lanes)
            if count <= capacity:
                continue
            key = (int(arm_id), int(target_id))
            prev = pair_demand.get(key)
            if prev is None or int(count) - capacity > prev[0] - prev[1]:
                pair_demand[key] = (int(count), capacity)
    for (from_arm, to_arm), (incoming_n, outgoing_n) in sorted(pair_demand.items()):
        excess = int(incoming_n) - int(outgoing_n)
        details = {
            'from_arm': int(from_arm),
            'to_arm': int(to_arm),
            'incoming_count': int(incoming_n),
            'outgoing_count': int(outgoing_n),
            'excess': excess,
            'reported_value': 'incoming/outgoing lane counts disagree for this arm pair',
        }
        message = (
            f'same-arm merge forbidden: {incoming_n} incoming lanes on arm '
            f'{from_arm} cannot share {outgoing_n} outgoing on arm {to_arm}.'
        )
        issues.append(make_issue(
            stage, 'one_to_one_mapping', message,
            location={
                'arm': int(from_arm), 'from_arm': from_arm, 'to_arm': to_arm,
                'field': 'incoming_lanes',
            },
            details={**details, 'side': 'incoming'},
        ))
        issues.append(make_issue(
            stage, 'one_to_one_mapping', message,
            location={
                'arm': int(to_arm), 'from_arm': from_arm, 'to_arm': to_arm,
                'field': 'outgoing_lanes',
            },
            details={**details, 'side': 'outgoing'},
        ))
    return issues


def _retag_connectivity_issues(issues, stage):
    """Add refine fields and split one-to-one mapping onto both arms."""
    out = []
    for issue in issues:
        constraint = issue.get("constraint")
        loc = dict(issue.get("location") or {})
        details = dict(issue.get("details") or {})
        message = issue.get("message") or ""
        if constraint == "incoming_lane_has_outflow":
            loc.setdefault("field", "incoming_lanes")
            details.setdefault("reported_value", "incoming lane(s) have no legal outflow")
            out.append(make_issue(stage, constraint, message, location=loc, details=details))
        elif constraint == "outgoing_lane_has_inflow":
            loc.setdefault("field", "outgoing_lanes")
            details.setdefault("reported_value", "outgoing lane(s) have no legal inflow")
            out.append(make_issue(stage, constraint, message, location=loc, details=details))
        elif constraint == "one_to_one_mapping":
            from_arm = loc.get("from_arm")
            to_arm = loc.get("to_arm")
            details.setdefault(
                "reported_value",
                "incoming/outgoing lane counts disagree for this arm pair",
            )
            pair = {"from_arm": from_arm, "to_arm": to_arm}
            if from_arm is not None:
                out.append(make_issue(
                    stage, constraint, message,
                    location={
                        "arm": int(from_arm), "from_arm": from_arm, "to_arm": to_arm,
                        "field": "incoming_lanes",
                    },
                    details={**details, **pair, "side": "incoming"},
                ))
            if to_arm is not None:
                out.append(make_issue(
                    stage, constraint, message,
                    location={
                        "arm": int(to_arm), "from_arm": from_arm, "to_arm": to_arm,
                        "field": "outgoing_lanes",
                    },
                    details={**details, **pair, "side": "outgoing"},
                ))
        else:
            out.append(issue)
    return out


def _budget_connectivity_issues(spec, stage, *, budget_hit):
    arm_ids = [arm.id for arm in spec.arms]
    details = {
        "exhaustive": not budget_hit,
        "budget_hit": budget_hit,
    }
    return [
        make_issue(
            stage, "incoming_lane_has_outflow",
            "No legal movement assignment gives every incoming lane an outflow "
            f"(search budget hit={budget_hit}).",
            details={**details, "reported_value": "incoming lane(s) have no legal outflow"},
            arm_id_list=arm_ids,
        ),
        make_issue(
            stage, "outgoing_lane_has_inflow",
            "No legal movement assignment gives every outgoing lane an inflow "
            f"(search budget hit={budget_hit}).",
            details={**details, "reported_value": "outgoing lane(s) have no legal inflow"},
            arm_id_list=arm_ids,
        ),
    ]


def connectivity_issues(spec, features, stage, *, channelized=None, per_arm_cap=None, node_budget=None, leaf_budget=None):
    """Existential check: some legal fill gives every incoming an outflow and every outgoing an inflow.

    Unknown movements / travel cuts are enumerated. Pass if any assignment satisfies
    ``validate_connectivity`` (same-arm no shared destination lane,
    incoming_lane_has_outflow, outgoing_lane_has_inflow). Cross-arm merge is allowed.
    """
    from scripts.methods.llm_kg.refinement.fill_solver import (
        CostWeights,
        _apply_outgoing,
        _restore_outgoing,
        enumerate_arm_solutions,
        filled_from_codes,
        marking_from_filled,
        max_outgoing_for_arm,
        validate_connectivity,
    )
    per_arm_cap = HARD_IO_PER_ARM_CAP if per_arm_cap is None else int(per_arm_cap)
    node_budget = HARD_IO_NODE_BUDGET if node_budget is None else int(node_budget)
    leaf_budget = HARD_IO_LEAF_BUDGET if leaf_budget is None else int(leaf_budget)
    channelized_template = channelized or {}
    outgoing_override = {arm.id: max_outgoing_for_arm(arm) for arm in spec.arms}
    weights = CostWeights()
    arm_solutions = {}
    issues = []
    for arm in spec.arms:
        sols, _capped = enumerate_arm_solutions(
            spec, arm, weights, per_arm_cap, channelized_template, outgoing_override=outgoing_override,
        )
        if not sols:
            issues.extend(_arm_unfillable_issues(
                spec, arm, stage, outgoing_override=outgoing_override,
            ))
        else:
            arm_solutions[arm.id] = sols
    if issues:
        return issues
    arms_sorted = sorted(arm_solutions, key=lambda aid: (len(arm_solutions[aid]), aid))
    best_fail_issues = None
    stats = {'nodes': 0, 'leaves': 0, 'budget_hit': False}

    def evaluate_leaf(codes_by_arm, outgoing_by_arm):
        nonlocal best_fail_issues
        saved = _apply_outgoing(spec, outgoing_by_arm)
        cand = None
        try:
            resolved = resolve_channelized_links(spec, features)
            marking = marking_from_filled(filled_from_codes(codes_by_arm))
            cand = _one_to_one_pair_issues(spec, marking, stage)
            if not cand:
                output = build_connections_output(spec, marking, resolved)
                validate_connectivity(output, stage)
                return True
        except ConnectionParseError as exc:
            cand = _retag_connectivity_issues(list(exc.issues), stage)
        except ValueError as exc:
            if 'same-arm merge' not in str(exc) and 'many-to-one' not in str(exc):
                raise
            cand = cand or [
                make_issue(
                    stage, 'one_to_one_mapping', str(exc),
                    details={'reported_value': 'incoming/outgoing lane counts disagree for this arm pair'},
                    arm_id_list=[arm.id for arm in spec.arms],
                )
            ]
        finally:
            _restore_outgoing(spec, saved)
        if cand and (best_fail_issues is None or len(cand) < len(best_fail_issues)):
            best_fail_issues = cand
        return False

    def dfs(k, codes_by_arm, outgoing_by_arm):
        if stats['budget_hit']:
            return False
        stats['nodes'] += 1
        if stats['nodes'] > node_budget:
            stats['budget_hit'] = True
            return False
        if k == len(arms_sorted):
            stats['leaves'] += 1
            if stats['leaves'] > leaf_budget:
                stats['budget_hit'] = True
                return False
            return evaluate_leaf(codes_by_arm, outgoing_by_arm)
        arm_id = arms_sorted[k]
        for sol in arm_solutions[arm_id]:
            codes_by_arm[arm_id] = sol.codes
            outgoing_by_arm[arm_id] = sol.outgoing_lanes
            if dfs(k + 1, codes_by_arm, outgoing_by_arm):
                return True
            if stats['budget_hit']:
                codes_by_arm.pop(arm_id, None)
                outgoing_by_arm.pop(arm_id, None)
                return False
        codes_by_arm.pop(arm_id, None)
        outgoing_by_arm.pop(arm_id, None)
        return False

    if dfs(0, {}, {}):
        return []
    if best_fail_issues:
        return best_fail_issues
    return _budget_connectivity_issues(spec, stage, budget_hit=stats['budget_hit'])


def validate_connectivity_constraints(raw):
    spec = load_junction_from_dict(raw)
    features = collect_channelized_features(raw)
    channelized = resolve_channelized_links(spec, features)
    issues = connectivity_issues(spec, features, 'stage_fix_local', channelized=channelized)
    if issues:
        raise ConnectionParseError(issues, '[stage_fix_local] connectivity check failed')
    return spec


def validate_hard_local_constraints(raw):
    """Backward-compatible alias: layer 2 now only checks connectivity."""
    return validate_connectivity_constraints(raw)


def check(config, name, *, lane_results=None):
    try:
        validate_hard_local_constraints({**config, "name": name})
        return None
    except ConnectionParseError as exc:
        _annotate_connectivity_summaries(exc.issues, lane_results)
        return exc
    except Exception as exc:
        return ConnectionParseError(
            [make_issue("stage_fix_local", "unexpected_error", f"Unexpected error during connectivity validation: {exc}")],
            title="[stage_fix_local] unexpected error",
        )


def _lanes_doc_from_results(lane_results, *, channelized_features=None, through_roads=None, driving_side=None, extra=None):
    from scripts.methods.llm_kg.lane_level.llm_gen import _lane_result_doc

    doc = dict(_lane_result_doc(lane_results))
    doc["channelized_features"] = channelized_features
    doc["through_roads"] = through_roads
    doc["driving_side"] = driving_side
    if extra:
        for key in ("source_image",):
            if key in extra:
                doc[key] = extra[key]
    return doc


def _persist_feasibility_layer(
    refinement_root, *, passed, validation_error, lanes_doc, repair,
    rank_cost=None, rank_conversation=None,
):
    save_layer_result(
        refinement_root, "fea",
        passed=bool(passed), refine_rounds=0,
        validation_error=None if passed else validation_error,
        lanes_doc=lanes_doc,
        extra={"feasibility_repair": repair.to_dict()} if repair is not None else None,
    )
    if refinement_root is None:
        return
    fea_dir = refinement_layer_dir(refinement_root, "fea")
    if repair is not None:
        dump_json(fea_dir / "feasibility_solver.json", repair.to_dict())
    dump_json(fea_dir / "cost.json", rank_cost if rank_cost is not None else {"steps": [], "totals": {}})
    dump_json(fea_dir / "conversation.json", rank_conversation if rank_conversation is not None else {})


def run_feasibility_repair(
    lane_results,
    *,
    arms,
    channelized_features=None,
    through_roads=None,
    name=None,
    driving_side=None,
    refinement_root=None,
    validation_error=None,
    agents_by_arm=None,
    source_lanes=None,
    persist=True,
    skip_noop=False,
    llm=None,
    model=None,
    reasoning=None,
    max_tokens=2048,
    image=None,
    image_path=None,
    tracker=None,
):
    """Min-edit repair. On success, update ``lane_results`` and write ``layer_fea``.

    Rank up to 3 cheapest repairs when an LLM and image are available.
    ``skip_noop`` leaves an already feasible reading unwritten.
    """
    from scripts.methods.llm_kg.refinement.feasibility_rank import rank_repair_candidates
    from scripts.methods.llm_kg.refinement.feasibility_solver import (
        format_repair_edits,
        solve_junction,
    )

    if source_lanes is None:
        source_lanes = _lanes_doc_from_results(
            lane_results,
            channelized_features=channelized_features,
            through_roads=through_roads,
            driving_side=driving_side,
        )

    BaseLLM.log("[validate:feasibility] start")
    rank_tracker = CostTracker(source=getattr(tracker, "source", None), silent=True)
    rank_conversation = None
    repair = solve_junction(
        lane_results,
        arms=arms,
        channelized_features=channelized_features,
        through_roads=through_roads,
        name=name,
        driving_side=driving_side,
    )
    if repair.ok and (repair.candidates or []):
        BaseLLM.log(
            f"[validate:feasibility] {len(repair.candidates)} candidate(s); "
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
        repair, rank_conversation = rank_repair_candidates(
            repair,
            llm=llm,
            model=model,
            original_lanes=source_lanes,
            image=image,
            image_path=image_path,
            tracker=rank_tracker,
            reasoning=reasoning,
            max_tokens=max_tokens,
        )

    if not repair.ok:
        BaseLLM.log(
            "[validate:feasibility] FAIL"
            + (f" ({repair.note})" if repair.note else "")
            + f" trials={repair.trials}"
        )
    else:
        for arm_id, result in (repair.lane_results or {}).items():
            lane_results[arm_id] = result
            if agents_by_arm is not None and arm_id in agents_by_arm:
                agents_by_arm[arm_id]["result"] = result
        BaseLLM.log(
            f"[validate:feasibility] chosen score={repair.score} "
            f"({len(repair.edits)} edit(s); source={repair.rank_source or 'score'}; "
            f"trials={repair.trials})"
        )
        for line in format_repair_edits(repair.edits):
            BaseLLM.log(line, indent=1)

    is_noop = bool(repair.ok and not repair.edits)
    if persist and not (skip_noop and is_noop):
        out_lanes = (
            _lanes_doc_from_results(
                lane_results,
                channelized_features=channelized_features,
                through_roads=through_roads,
                driving_side=driving_side,
                extra=source_lanes,
            )
            if repair.ok else source_lanes
        )
        rank_cost = rank_tracker.summary()
        if rank_conversation is None:
            rank_conversation = {
                "stage": "feasibility_rank",
                "model": model,
                "skipped": "solver_failed" if not repair.ok else "no_candidates",
            }
        if image_path and "source_image" not in rank_conversation:
            rank_conversation["source_image"] = str(image_path)
        elif isinstance(source_lanes, dict) and source_lanes.get("source_image"):
            rank_conversation.setdefault("source_image", str(source_lanes["source_image"]))
        _persist_feasibility_layer(
            refinement_root,
            passed=bool(repair.ok),
            validation_error=validation_error,
            lanes_doc=out_lanes,
            repair=repair,
            rank_cost=rank_cost,
            rank_conversation=rank_conversation,
        )
        if rank_tracker.steps:
            rank_tracker.print_summary()
    return repair


def _run_feasibility_solver(ctx, config, *, validation_error=None, source_lanes=None):
    """Last-resort min-edit repair after layer-2 still fails."""
    image = None
    for agent in (ctx.agents_by_arm or {}).values():
        image = agent.get("original_image")
        if image is not None:
            break
    tracker = None
    if ctx.trackers_by_arm:
        tracker = next(iter(ctx.trackers_by_arm.values()), None)
    return run_feasibility_repair(
        ctx.lane_results,
        arms=ctx.arms,
        channelized_features=ctx.channelized_features,
        through_roads=ctx.through_roads,
        name=ctx.name,
        driving_side=(config or {}).get("driving_side") or getattr(ctx, "driving_side", None),
        refinement_root=ctx.refinement_root,
        validation_error=validation_error,
        agents_by_arm=ctx.agents_by_arm,
        source_lanes=source_lanes,
        llm=ctx.llm,
        model=ctx.model,
        reasoning=ctx.reasoning,
        max_tokens=min(int(ctx.max_tokens or 2048), 4096),
        image=image,
        image_path=getattr(ctx, "image_path", None),
        tracker=tracker,
    )


def run(ctx, config, *, max_refine_rounds=2, skip_fea_solver=False):
    BaseLLM.log("[validate:fix] start")

    def _check(cfg):
        return check(cfg, ctx.name, lane_results=ctx.lane_results or {})

    config, refine_rounds, validation_error, passed = run_refine_loop(
        ctx, config, layer="fix", max_refine_rounds=max_refine_rounds, check=_check,
    )
    if passed:
        config = ctx.rebuild_config()
    l2_lanes = lanes_doc_from_ctx(ctx)
    save_layer_result(
        ctx.refinement_root, "fix",
        passed=passed, refine_rounds=refine_rounds,
        validation_error=None if passed else validation_error,
        lanes_doc=l2_lanes,
    )

    feasibility_repair = None
    if not passed:
        BaseLLM.log(f"[validate:fix] FAIL after {refine_rounds} round(s)")
        if skip_fea_solver:
            BaseLLM.log("[validate:feasibility] skipped (skip_fea_solver=True)")
            extra = None
        else:
            feasibility_repair = _run_feasibility_solver(
                ctx, config, validation_error=validation_error, source_lanes=l2_lanes,
            )
            solver_ok = bool(feasibility_repair and feasibility_repair.ok)
            if solver_ok:
                config = ctx.rebuild_config()
                passed = True
                validation_error = None
            extra = {"feasibility_repair": feasibility_repair.to_dict()} if feasibility_repair is not None else None
        out = layer_result(
            passed=passed, config=config, refine_rounds=refine_rounds,
            validation_error=validation_error, extra=extra,
        )
    else:
        BaseLLM.log("[validate:fix] PASS" + (f" after {refine_rounds} round(s)" if refine_rounds else ""))
        out = layer_result(passed=True, config=config, refine_rounds=refine_rounds, validation_error=None)
    return out
