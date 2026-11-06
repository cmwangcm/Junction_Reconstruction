"""Fill optimization solver: search, diagnose, and emit filled config + connections."""

import copy
import itertools
import math
import random
from collections import Counter, defaultdict

from scripts.methods.connection_parser import (
    DEFAULT_DIAGNOSIS_LEAF_BUDGET,
    DEFAULT_DIAGNOSIS_NODE_BUDGET,
    DEFAULT_MAX_REPAIR_CANDIDATES,
    DEFAULT_MAX_REPAIR_SIZE,
    DEFAULT_MAX_REPAIR_TRIALS,
    DEFAULT_PER_ARM_CAP,
    DEFAULT_SEARCH_BUDGET,
    DEFAULT_SEED,
    INCOMING_LANE_1_IS_LEFTMOST,
    ORDER,
    UNKNOWN,
    VALID_MOVEMENTS,
    assigned_travel_at,
    build_connections_output,
    canonical_code,
    channelized_origin_arms,
    collect_channelized_features,
    entry_letters,
    explicit_marking,
    find_target_arm,
    has_unknown_lanes,
    is_contiguous,
    is_unknown,
    is_valid_movement_code,
    load_junction_from_dict,
    load_raw_and_spec,
    legal_outflow_letters,
    max_outgoing_for_arm as _max_outgoing_for_arm,
    movement_has_outflow,
    normalize_driving_side,
    normalize_movement_candidates,
    resolve_channelized_links,
    slip_from_lane_id,
    slip_assigned_code,
    slip_turn_letter,
    channelized_source_of,
    clip_movement_code_to_outflow,
    valid_travel_cuts,
)
from scripts.methods.llm_kg.refinement.common import (
    ConnectionParseError,
    code_display,
    fail,
    lane_reference,
    make_issue,
    incoming_reference,
    outgoing_reference,
)
from scripts.methods.llm_kg.refinement.layer2_con_fix import (
    _left_to_right,
    hard_channelized_from_arm_issues,
    hard_lane_constraint_issues,
)

def lane_constraint_issues(spec, marking, stage, channelized=None, raw=None):
    issues = hard_channelized_from_arm_issues(
        spec, marking, stage, channelized=channelized, raw=raw,
    )
    issues.extend(hard_lane_constraint_issues(spec, marking, stage, channelized=channelized, raw=raw))
    for arm_id, lanes in marking.items():
        counts = defaultdict(int)
        for letters in lanes:
            for move in letters or ():
                counts[move] += 1
        for move, count in counts.items():
            target_id = find_target_arm(spec, arm_id, move)
            if target_id is None or spec.arm_map[target_id].outgoing_lanes <= 0:
                details = {
                    "movement": move,
                    "lane_count": len(lanes),
                    "legal_outflow": legal_outflow_letters(spec, arm_id),
                }
                if target_id is None:
                    details["why"] = "no destination arm in that heading"
                else:
                    details["target_arm"] = int(target_id)
                    details["target_outgoing"] = int(spec.arm_map[target_id].outgoing_lanes)
                    details["why"] = "arm %s has no outgoing lane" % target_id
                issues.append(make_issue(
                    stage, "movement_reachability",
                    f"Arm {arm_id} movement {move} has no reachable target arm.",
                    location={"arm": arm_id},
                    details=details,
                ))
                continue
            capacity = spec.arm_map[target_id].outgoing_lanes
            if count > capacity:
                excess = int(count) - int(capacity)
                issues.append(make_issue(
                    stage, 'receiving_capacity',
                    f'Arm {arm_id} incoming is {excess} too many.',
                    location={'arm': arm_id, 'to_arm': target_id, 'field': 'cross_section'},
                    details={
                        'from_arm': int(arm_id),
                        'to_arm': int(target_id),
                        'incoming_count': int(count),
                        'outgoing_count': int(capacity),
                        'excess': excess,
                        'side': 'incoming',
                        'reported_value': 'incoming/outgoing lane counts disagree for this arm pair',
                    },
                ))
                issues.append(make_issue(
                    stage, 'receiving_capacity',
                    f'Arm {target_id} outgoing is {excess} too few.',
                    location={'arm': target_id, 'from_arm': arm_id, 'field': 'cross_section'},
                    details={
                        'from_arm': int(arm_id),
                        'to_arm': int(target_id),
                        'incoming_count': int(count),
                        'outgoing_count': int(capacity),
                        'excess': excess,
                        'side': 'outgoing',
                        'reported_value': 'incoming/outgoing lane counts disagree for this arm pair',
                    },
                ))
    return issues


def validate_lane_constraints(spec, marking, stage, require_resolved, channelized=None):
    issues = []
    if require_resolved:
        for arm in spec.arms:
            for idx, letters in enumerate(marking.get(arm.id, []), start=1):
                if not letters:
                    issues.append(make_issue(stage, 'resolved_required', f'Arm {arm.id} lane #{idx} is still unresolved (N).', location={'arm': arm.id, 'lane': idx}))
    issues.extend(lane_constraint_issues(spec, marking, stage, channelized=channelized))
    if issues:
        raise ConnectionParseError(issues, f'[{stage}] lane constraint check failed')


def validate_connectivity(output, stage='stage4_connectivity'):
    """Check IO completeness and same-arm destination uniqueness.

    Two incoming lanes of the same from_arm must not share a to_lane_id.
    Incoming from different arms may merge onto the same outgoing lane.
    Every incoming still needs an outflow and every outgoing an inflow.
    """
    roads = {int(r['id']): r for r in output.get('roads', [])}
    issues = []

    def add(constraint, message, location=None, details=None):
        issues.append(make_issue(stage, constraint, message, location=location, details=details))

    if not roads:
        raise fail(stage, 'roads_exist', 'No roads found in connections output.')
    used_from = defaultdict(set)
    used_to = defaultdict(set)
    for junction in output.get('junctions', []):
        for idx, conn in enumerate(junction.get('connections', [])):
            loc = {'connection_index': idx, 'from_arm': conn.get('from_arm'), 'to_arm': conn.get('to_arm')}
            try:
                from_arm = int(conn['from_arm'])
                to_arm = int(conn['to_arm'])
            except (TypeError, ValueError, KeyError):
                add('connection_arm_ids_valid', 'from_arm/to_arm is missing or invalid.', loc)
                continue
            if from_arm not in roads or to_arm not in roads:
                add('arm_exists', 'Connection references a missing arm.', loc)
                continue
            lane_links = conn.get('lane_links')
            if not isinstance(lane_links, list) or not lane_links:
                add('lane_links_non_empty', 'lane_links is empty.', loc)
                continue
            incoming = int(roads[from_arm].get('incoming_lanes', 0))
            outgoing = int(roads[to_arm].get('outgoing_lanes', 0))
            seen_to_lane = {}
            for link in lane_links:
                try:
                    f = int(link['from_lane_id'])
                    t = int(link['to_lane_id'])
                except (TypeError, ValueError, KeyError):
                    add('lane_link_ids_valid', 'lane_link id is missing or invalid.', loc, {'link': link})
                    continue
                if not 1 <= f <= incoming:
                    add('from_lane_id_range', f'from_lane_id={f} out of range 1..{incoming}.', {**loc, 'lane': f}, {'range': [1, incoming]})
                    continue
                if not 1 <= t <= outgoing:
                    add('to_lane_id_range', f'to_lane_id={t} out of range 1..{outgoing}.', {**loc, 'lane': t}, {'range': [1, outgoing]})
                    continue
                prior = seen_to_lane.get(t)
                if prior is not None:
                    add(
                        'one_to_one_mapping',
                        f'same-arm merge forbidden: arm {from_arm} incoming lanes {prior} and {f} both target arm {to_arm} to_lane_id={t}.',
                        {**loc, 'to_arm': to_arm},
                        {'to_lane_id': t, 'from_lane_ids': [prior, f]},
                    )
                    continue
                seen_to_lane[t] = f
                used_from[from_arm].add(f)
                used_to[to_arm].add(t)
    for road_id, road in roads.items():
        missing_from = sorted(set(range(1, int(road.get('incoming_lanes', 0)) + 1)) - used_from[road_id])
        missing_to = sorted(set(range(1, int(road.get('outgoing_lanes', 0)) + 1)) - used_to[road_id])
        if missing_from:
            add('incoming_lane_has_outflow', f'Road {road_id} incoming lane(s) with no outflow: {missing_from}.', {'arm': road_id}, {'lanes': missing_from})
        if missing_to:
            add('outgoing_lane_has_inflow', f'Road {road_id} outgoing lane(s) with no inflow: {missing_to}.', {'arm': road_id}, {'lanes': missing_to})
    if issues:
        raise ConnectionParseError(issues, f'[{stage}] connectivity check failed')


def travel_flip_penalty(lane_readings, cut, weights, driving_side=None):
    penalty = 0.0
    side = normalize_driving_side(driving_side)
    for i, reading in enumerate(lane_readings):
        if reading.travel is not None:
            continue
        assigned = assigned_travel_at(i, cut, side)
        if assigned == 'outgoing' and reading.movements:
            penalty += float(weights.travel_flip)
    return penalty


def rank_penalty(codes, lane_options, weights):
    penalty = 0.0
    for code, opt in zip(codes, lane_options):
        if not opt.get('ranked'):
            continue
        try:
            idx = list(opt['options']).index(code)
        except ValueError:
            idx = 0
        penalty += float(weights.rank) * idx
    return penalty


def max_outgoing_for_arm(arm):
    return _max_outgoing_for_arm(arm)


def _spec_driving_side(spec):
    return normalize_driving_side(getattr(spec, 'driving_side', None))


def _is_slip_lane_index(lane_index, n_lanes, driving_side=None):
    slip_id = slip_from_lane_id(n_lanes, driving_side)
    return slip_id is not None and int(lane_index) == int(slip_id)


def lane_possibilities(spec, arm, channelized=None, *, outgoing_override=None, lane_index=None, n_lanes=None):
    # Channelized from_arm: curb-edge incoming is reserved slip turn; other lanes forbid it.
    side = _spec_driving_side(spec)
    letter = slip_turn_letter(side)
    if arm.id in channelized_origin_arms(channelized) and lane_index is not None and n_lanes is not None:
        if _is_slip_lane_index(lane_index, n_lanes, side):
            return [slip_assigned_code(channelized_source_of(channelized, arm.id), side)]
        forbidden = {letter}
    else:
        forbidden = set()
    moves = sorted((m for m in VALID_MOVEMENTS if m not in forbidden and movement_has_outflow(spec, arm.id, m, outgoing_override=outgoing_override)), key=lambda m: ORDER[m])
    allow_lr = spec.arm_allows_lr(arm.id)
    codes = []
    for size in range(1, len(moves) + 1):
        for combo in itertools.combinations(moves, size):
            if is_valid_movement_code(set(combo), allow_lr=allow_lr):
                codes.append(''.join(combo))
    return codes


def lane_option_from_entry(spec, arm, entry, channelized=None, *, outgoing_override=None, lane_index=None, n_lanes=None):
    allow_lr = spec.arm_allows_lr(arm.id)
    side = _spec_driving_side(spec)
    letter = slip_turn_letter(side)
    if arm.id in channelized_origin_arms(channelized) and lane_index is not None and n_lanes is not None and _is_slip_lane_index(lane_index, n_lanes, side):
        return {
            'kind': 'fixed',
            'code': slip_assigned_code(channelized_source_of(channelized, arm.id), side),
            'ranked': False,
        }
    if is_unknown(entry) or entry is None or entry == []:
        choices = lane_possibilities(
            spec, arm, channelized, outgoing_override=outgoing_override,
            lane_index=lane_index, n_lanes=n_lanes,
        )
        return {'kind': 'choice', 'options': choices, 'ranked': False}
    if isinstance(entry, list):
        options = []
        for raw in entry:
            code = canonical_code(raw)
            if is_unknown(code):
                continue
            if not is_valid_movement_code(entry_letters(code), allow_lr=allow_lr):
                continue
            if arm.id in channelized_origin_arms(channelized) and letter in entry_letters(code):
                continue
            clipped = clip_movement_code_to_outflow(
                spec, arm.id, code, outgoing_override=outgoing_override, allow_lr=allow_lr,
            )
            if clipped and clipped not in options:
                options.append(clipped)
        if not options:
            choices = lane_possibilities(
                spec, arm, channelized, outgoing_override=outgoing_override,
                lane_index=lane_index, n_lanes=n_lanes,
            )
            return {'kind': 'choice', 'options': choices, 'ranked': False}
        if len(options) == 1:
            return {'kind': 'fixed', 'code': options[0], 'ranked': False}
        return {'kind': 'choice', 'options': options, 'ranked': True}
    code = canonical_code(entry)
    if is_unknown(code):
        choices = lane_possibilities(
            spec, arm, channelized, outgoing_override=outgoing_override,
            lane_index=lane_index, n_lanes=n_lanes,
        )
        return {'kind': 'choice', 'options': choices, 'ranked': False}
    if arm.id in channelized_origin_arms(channelized) and letter in entry_letters(code):
        choices = lane_possibilities(
            spec, arm, channelized, outgoing_override=outgoing_override,
            lane_index=lane_index, n_lanes=n_lanes,
        )
        return {'kind': 'choice', 'options': choices, 'ranked': False}
    clipped = clip_movement_code_to_outflow(
        spec, arm.id, code, outgoing_override=outgoing_override, allow_lr=allow_lr,
    )
    if clipped is None:
        choices = lane_possibilities(
            spec, arm, channelized, outgoing_override=outgoing_override,
            lane_index=lane_index, n_lanes=n_lanes,
        )
        return {'kind': 'choice', 'options': choices, 'ranked': False}
    return {'kind': 'fixed', 'code': clipped, 'ranked': False}


def marking_from_filled(filled):
    return {arm_id: [lane['letters'] for lane in lanes] for arm_id, lanes in filled.items()}


def uncertain_slots(options):
    return [(arm_id, idx, opt['options']) for arm_id, lane_options in options.items() for idx, opt in enumerate(lane_options) if opt['kind'] == 'choice']


def filled_from_codes(codes_by_arm):
    return {arm_id: [{'display': code, 'letters': entry_letters(code)} for code in codes] for arm_id, codes in codes_by_arm.items()}



class CostWeights:
    coverage = 2.0
    share = 1.0
    outer = 2.0
    rank = 5.0
    travel_flip = 0.25

    def __init__(self, coverage=None, share=None, outer=None, rank=None, travel_flip=None):
        self.coverage = self.coverage if coverage is None else coverage
        self.share = self.share if share is None else share
        self.outer = self.outer if outer is None else outer
        self.rank = self.rank if rank is None else rank
        self.travel_flip = self.travel_flip if travel_flip is None else travel_flip

    def to_dict(self):
        return {'coverage': self.coverage, 'share': self.share, 'outer': self.outer, 'rank': self.rank, 'travel_flip': self.travel_flip}


def servable_movements(spec, arm, channelized=None):
    moves = {m for m in VALID_MOVEMENTS if movement_has_outflow(spec, arm.id, m)}
    if arm.id in channelized_origin_arms(channelized):
        moves.discard(slip_turn_letter(_spec_driving_side(spec)))
    return moves

def arm_cost_components(spec, arm, codes, channelized=None):
    letters = [entry_letters(c) for c in codes]
    comp = {'coverage': 0.0, 'share': 0.0, 'outer': 0.0}
    arm_letters = set().union(*letters) if letters else set()
    comp['coverage'] = float(len(servable_movements(spec, arm, channelized) - arm_letters))
    for lane_letters in letters:
        comp['share'] += max(0, len(lane_letters) - 1)
    ordered = _left_to_right(letters)
    if 'L' in arm_letters and ordered and ('L' not in ordered[0]):
        comp['outer'] += 1
    if 'R' in arm_letters and ordered and ('R' not in ordered[-1]):
        comp['outer'] += 1
    return comp

def weighted_arm_cost(comp, w):
    return w.coverage * comp['coverage'] + w.share * comp['share'] + w.outer * comp['outer']

def arm_cost(spec, arm, codes, w, channelized=None):
    return weighted_arm_cost(arm_cost_components(spec, arm, codes, channelized), w)

def total_cost(spec, codes_by_arm, w, channelized=None):
    return sum((arm_cost(spec, spec.arm_map[aid], codes, w, channelized) for aid, codes in codes_by_arm.items()))

def cost_breakdown(spec, codes_by_arm, w, channelized=None):
    agg = {'coverage': 0.0, 'share': 0.0, 'outer': 0.0}
    for aid, codes in codes_by_arm.items():
        comp = arm_cost_components(spec, spec.arm_map[aid], codes, channelized)
        for key in agg:
            agg[key] += comp[key]
    breakdown = {'coverage': round(w.coverage * agg['coverage'], 6), 'share': round(w.share * agg['share'], 6), 'outer': round(w.outer * agg['outer'], 6)}
    breakdown['total'] = round(sum(breakdown.values()), 6)
    return breakdown

class ArmSolution:

    def __init__(self, cost, codes, outgoing_lanes, lane_options=None):
        self.cost = cost
        self.codes = codes
        self.outgoing_lanes = int(outgoing_lanes)
        self.lane_options = lane_options or []

def movement_groups_are_contiguous(letters):
    for move in VALID_MOVEMENTS:
        ids = [idx for idx, lane_letters in enumerate(letters, start=1) if move in lane_letters]
        if ids and (not is_contiguous(ids)):
            return False
    return True

def _arm_per_arm_feasible(spec, arm, letters, channelized=None, *, outgoing_override=None):
    side = _spec_driving_side(spec)
    letter = slip_turn_letter(side)
    if arm.id in channelized_origin_arms(channelized):
        if not letters:
            return False
        slip_letters = letters[-1] if side == 'left' else letters[0]
        if letter not in (slip_letters or ()):
            return False
        if channelized_source_of(channelized, arm.id) == 'shared' and 'S' not in (slip_letters or ()):
            return False
        other = letters[:-1] if side == 'left' else letters[1:]
        if any(letter in (lane_letters or ()) for lane_letters in other):
            return False
    if not movement_groups_are_contiguous(letters):
        return False
    counts = Counter()
    for lane_letters in letters:
        for move in lane_letters:
            counts[move] += 1
    for move, count in counts.items():
        target_id = find_target_arm(spec, arm.id, move)
        if target_id is None:
            return False
        capacity = outgoing_override.get(target_id) if outgoing_override is not None else spec.arm_map[target_id].outgoing_lanes
        if capacity is None or capacity <= 0:
            return False
        if count > capacity + 1:
            return False
    return True

def _enumerate_movement_solutions(spec, arm, lane_options, weights, cap, channelized, *, outgoing_lanes, outgoing_override, cut):
    n = len(lane_options)
    candidates_by_pos = [[opt['code']] if opt['kind'] == 'fixed' else list(opt['options']) for opt in lane_options]
    if any(not opts for opts in candidates_by_pos):
        return ([], False)
    lr_positions = list(range(n)) if INCOMING_LANE_1_IS_LEFTMOST else list(range(n - 1, -1, -1))
    solutions = []
    capped = False
    chosen = [None] * n
    flip = travel_flip_penalty(arm.lane_readings, cut, weights, driving_side=_spec_driving_side(spec))

    def recurse(t, prev_low, prev_high):
        nonlocal capped
        if capped:
            return
        if t == len(lr_positions):
            codes = tuple(chosen)
            letters = [entry_letters(c) for c in codes]
            if _arm_per_arm_feasible(spec, arm, letters, channelized, outgoing_override=outgoing_override):
                base = arm_cost(spec, arm, codes, weights, channelized)
                cost = base + rank_penalty(codes, lane_options, weights) + flip
                solutions.append(ArmSolution(cost, codes, outgoing_lanes, lane_options))
                if len(solutions) >= cap:
                    capped = True
            return
        pos = lr_positions[t]
        for code in candidates_by_pos[pos]:
            letters = entry_letters(code)
            low = min((ORDER[c] for c in letters))
            high = max((ORDER[c] for c in letters))
            if low < prev_low or high < prev_high:
                continue
            chosen[pos] = code
            recurse(t + 1, low, high)
            chosen[pos] = None
            if capped:
                return
    recurse(0, -1, -1)
    return (solutions, capped)

def enumerate_arm_solutions(spec, arm, weights, cap, channelized=None, *, outgoing_override=None):
    """All per-arm-feasible (travel cut × movement) assignments, cost-sorted. Returns (solutions, capped)."""
    solutions = []
    capped = False
    side = _spec_driving_side(spec)
    saved_cut = arm.outgoing_lanes if side == 'left' else len(arm.lane_readings) - arm.outgoing_lanes
    for cut in valid_travel_cuts(arm.lane_readings, driving_side=side):
        arm.apply_cut(cut)
        try:
            lane_options = []
            empty = False
            for idx, entry in enumerate(arm.incoming_lanes, start=1):
                opt = lane_option_from_entry(
                    spec, arm, entry, channelized, outgoing_override=outgoing_override,
                    lane_index=idx, n_lanes=len(arm.incoming_lanes),
                )
                if opt['kind'] == 'choice' and not opt['options']:
                    empty = True
                    break
                lane_options.append(opt)
            if empty:
                continue
            sols, hit_cap = _enumerate_movement_solutions(spec, arm, lane_options, weights, cap - len(solutions), channelized, outgoing_lanes=arm.outgoing_lanes, outgoing_override=outgoing_override, cut=cut)
            solutions.extend(sols)
            if hit_cap or len(solutions) >= cap:
                capped = True
                solutions = solutions[:cap]
                break
        finally:
            arm.apply_cut(saved_cut)
    solutions.sort(key=lambda s: s.cost)
    return (solutions, capped)


def build_fill_failure_error(*, options, representative_issues):
    details = {'uncertain_slots': [{'arm': a, 'lane': lane + 1, 'options': choices} for a, lane, choices in uncertain_slots(options)], 'blocking_issues': representative_issues[:5]}
    return ConnectionParseError([make_issue('stage2_fill_search', 'no_feasible_fill_solution', 'No assignment of N lanes can satisfy all lane-connectivity constraints.', details=details)], 'The current xodr_config cannot be converted into a feasible connection layout.')


class FillSolution:
    def __init__(self, codes_by_arm, filled, cost, breakdown, stats, outgoing_by_arm=None):
        self.codes_by_arm = codes_by_arm
        self.filled = filled
        self.cost = cost
        self.breakdown = breakdown
        self.stats = stats
        self.outgoing_by_arm = outgoing_by_arm or {}


def _apply_outgoing(spec, outgoing_by_arm):
    side = _spec_driving_side(spec)
    saved = {}
    for arm in spec.arms:
        if side == 'left':
            saved[arm.id] = arm.outgoing_lanes
            arm.apply_cut(int(outgoing_by_arm[arm.id]))
        else:
            saved[arm.id] = len(arm.lane_readings) - arm.outgoing_lanes
            arm.apply_cut(len(arm.lane_readings) - int(outgoing_by_arm[arm.id]))
    return saved


def _restore_outgoing(spec, saved):
    for arm in spec.arms:
        arm.apply_cut(saved[arm.id])


def solve_fill_optimization(spec, features, channelized_template, *, weights, rng, node_budget, leaf_budget, per_arm_cap):
    """Minimum-cost feasible travel-split × N-fill via branch-and-bound."""
    outgoing_override = {arm.id: max_outgoing_for_arm(arm) for arm in spec.arms}
    arm_solutions = {}
    any_capped = False
    infeasible_arms = []
    for arm in spec.arms:
        sols, capped = enumerate_arm_solutions(spec, arm, weights, per_arm_cap, channelized_template, outgoing_override=outgoing_override)
        any_capped = any_capped or capped
        if not sols:
            infeasible_arms.append(make_issue('stage2_fill_search', 'arm_no_valid_assignment', f'Arm {arm.id} has no lane assignment satisfying its per-arm constraints.', location={'arm': arm.id}))
        else:
            rng.shuffle(sols)
            sols.sort(key=lambda s: s.cost)
            arm_solutions[arm.id] = sols
    if infeasible_arms:
        raise ConnectionParseError(infeasible_arms, '[stage2_fill_search] lane-level infeasibility')
    arms_sorted = sorted(arm_solutions, key=lambda aid: (len(arm_solutions[aid]), aid))
    k_total = len(arms_sorted)
    min_tail = [0.0] * (k_total + 1)
    for k in range(k_total - 1, -1, -1):
        min_tail[k] = arm_solutions[arms_sorted[k]][0].cost + min_tail[k + 1]
    best = {'cost': math.inf, 'codes': None, 'filled': None, 'outgoing': None}
    representative_issues = []
    stats = {'nodes': 0, 'leaves': 0, 'accepted': 0, 'budget_hit': False}
    options_snapshot = {arm.id: (arm_solutions[arm.id][0].lane_options if arm_solutions[arm.id] else []) for arm in spec.arms}

    def evaluate_leaf(codes_by_arm, outgoing_by_arm, acc_cost):
        nonlocal representative_issues
        saved = _apply_outgoing(spec, outgoing_by_arm)
        try:
            channelized = resolve_channelized_links(spec, features)
            filled = filled_from_codes(codes_by_arm)
            marking = marking_from_filled(filled)
            validate_lane_constraints(spec, marking, 'stage2_fill_candidate', True, channelized=channelized)
            output = build_connections_output(spec, marking, channelized)
            validate_connectivity(output, 'stage2_fill_candidate_connectivity')
        except ConnectionParseError as exc:
            if not representative_issues:
                representative_issues = exc.issues[:5]
            return
        except ValueError as exc:
            if 'same-arm merge' not in str(exc) and 'many-to-one' not in str(exc):
                raise
            if not representative_issues:
                representative_issues = [make_issue(
                    'stage2_fill_candidate', 'one_to_one_mapping', str(exc),
                    details={'reported_value': 'same-arm incoming lanes would share a destination lane'},
                )]
            return
        finally:
            _restore_outgoing(spec, saved)
        stats['accepted'] += 1
        if acc_cost < best['cost']:
            best['cost'] = acc_cost
            best['codes'] = dict(codes_by_arm)
            best['filled'] = filled_from_codes(codes_by_arm)
            best['outgoing'] = dict(outgoing_by_arm)

    def dfs(k, acc_cost, codes_by_arm, outgoing_by_arm):
        if stats['budget_hit']:
            return
        stats['nodes'] += 1
        if stats['nodes'] > node_budget:
            stats['budget_hit'] = True
            return
        if k == k_total:
            stats['leaves'] += 1
            if stats['leaves'] > leaf_budget:
                stats['budget_hit'] = True
                return
            evaluate_leaf(codes_by_arm, outgoing_by_arm, acc_cost)
            return
        arm_id = arms_sorted[k]
        for sol in arm_solutions[arm_id]:
            if acc_cost + sol.cost + min_tail[k + 1] >= best['cost']:
                break
            codes_by_arm[arm_id] = sol.codes
            outgoing_by_arm[arm_id] = sol.outgoing_lanes
            dfs(k + 1, acc_cost + sol.cost, codes_by_arm, outgoing_by_arm)
            if stats['budget_hit']:
                codes_by_arm.pop(arm_id, None)
                outgoing_by_arm.pop(arm_id, None)
                return
        codes_by_arm.pop(arm_id, None)
        outgoing_by_arm.pop(arm_id, None)
    dfs(0, 0.0, {}, {})
    exhausted = not stats['budget_hit'] and (not any_capped)
    if best['codes'] is None:
        raise build_fill_failure_error(options=options_snapshot, representative_issues=representative_issues)
    stats['exhausted_full_space'] = exhausted
    return FillSolution(codes_by_arm=best['codes'], filled=best['filled'], cost=best['cost'], breakdown={'total': round(best['cost'], 6)}, stats=stats, outgoing_by_arm=best['outgoing'])

def build_filled_config(raw, spec, filled, outgoing_by_arm=None):
    """Write solved incoming/outgoing onto config, preserving mouth LTR order for driving_side.

    left:  [outgoing × out_n][incoming …]
    right: [incoming …][outgoing × out_n]
    """
    filled_raw = copy.deepcopy(raw)
    side = _spec_driving_side(spec)
    for idx, arm in enumerate(spec.arms):
        record = filled_raw['arms'][idx]
        in_lanes = filled[arm.id]
        record['incoming_lanes'] = [lane['display'] for lane in in_lanes]
        out_n = int(outgoing_by_arm[arm.id] if outgoing_by_arm is not None else arm.outgoing_lanes)
        record['outgoing_lanes'] = out_n
        if record.get('lanes') is not None:
            incoming_entries = [
                {
                    'travel': 'incoming',
                    'movements': [] if is_unknown(lane['display']) else [lane['display']],
                }
                for lane in in_lanes
            ]
            outgoing_entries = [{'travel': 'outgoing', 'movements': []} for _ in range(out_n)]
            if side == 'left':
                record['lanes'] = outgoing_entries + incoming_entries
            else:
                record['lanes'] = incoming_entries + outgoing_entries
    return filled_raw



class ArrowEdit:
    def __init__(self, arm, lane, lane_count, from_code, *, lane_pos=None, lane_ref=None):
        self.arm = arm
        self.lane = lane
        self.lane_count = lane_count
        self.from_code = from_code
        self.lane_pos = lane_pos if lane_pos is not None else lane - 1
        self._lane_ref = lane_ref

    @property
    def kind(self):
        return 'arrow'

    def apply(self, arm_record):
        if arm_record.get('lanes') is not None:
            arm_record['lanes'][self.lane_pos]['movements'] = []
            return
        arm_record['incoming_lanes'][self.lane - 1] = UNKNOWN

    def to_dict(self):
        ref = self._lane_ref or lane_reference(self.arm, self.lane, self.lane_count)
        return {'kind': 'arrow', 'arm': self.arm, 'lane': self.lane, 'lane_count': self.lane_count, 'current_code': self.from_code, 'lane_ref': ref}


class TravelRelaxEdit:
    def __init__(self, arm, lane_pos, from_travel):
        self.arm = arm
        self.lane_pos = int(lane_pos)
        self.from_travel = from_travel

    @property
    def kind(self):
        return 'count'

    @property
    def field(self):
        return 'lanes.travel'

    @property
    def from_value(self):
        return self.from_travel

    @property
    def to_value(self):
        return None

    @property
    def drop_lane(self):
        return None

    @property
    def disputed_lane(self):
        return self.lane_pos + 1

    def apply(self, arm_record):
        arm_record['lanes'][self.lane_pos]['travel'] = None

    def to_dict(self):
        return {'kind': 'count', 'arm': self.arm, 'field': 'lanes.travel', 'reported': self.from_travel, 'consistent_value': None, 'disputed_lane': self.disputed_lane, 'lane_ref': f'arm {self.arm}, lanes[{self.lane_pos}] travel={self.from_travel}'}


class CountEdit:
    def __init__(self, arm, field, from_value, to_value, drop_lane=None, insert_pos=None, driving_side=None):
        self.arm = arm
        self.field = field
        self.from_value = from_value
        self.to_value = to_value
        self.drop_lane = drop_lane
        self.insert_pos = insert_pos
        self.driving_side = normalize_driving_side(driving_side)

    @property
    def kind(self):
        return 'count'

    def apply(self, arm_record):
        side = self.driving_side
        if self.field == 'outgoing_lanes':
            arm_record['outgoing_lanes'] = self.to_value
            if arm_record.get('lanes') is not None and self.to_value > self.from_value:
                entry = {'travel': 'outgoing', 'movements': []}
                if side == 'left':
                    arm_record['lanes'].insert(0, entry)
                else:
                    arm_record['lanes'].append(entry)
            elif arm_record.get('lanes') is not None and self.to_value < self.from_value:
                lanes = arm_record['lanes']
                if self.drop_lane is not None:
                    idx = max(0, min(len(lanes) - 1, self.drop_lane - 1))
                    if lanes[idx].get('travel') == 'outgoing':
                        lanes.pop(idx)
                        return
                if side == 'left':
                    for i, lane in enumerate(lanes):
                        if lane.get('travel') == 'outgoing':
                            lanes.pop(i)
                            break
                else:
                    for i in range(len(lanes) - 1, -1, -1):
                        if lanes[i].get('travel') == 'outgoing':
                            lanes.pop(i)
                            break
            return
        if arm_record.get('lanes') is not None:
            lanes = list(arm_record['lanes'])
            if self.field == 'lanes' and self.to_value > self.from_value:
                idx = 0 if self.insert_pos is None else max(0, min(len(lanes), int(self.insert_pos)))
                lanes.insert(idx, {'travel': None, 'movements': []})
            elif self.to_value > self.from_value:
                entry = {'travel': 'incoming', 'movements': []}
                if side == 'left':
                    lanes.append(entry)
                else:
                    lanes.insert(0, entry)
            elif self.drop_lane is not None:
                idx = max(0, min(len(lanes) - 1, self.drop_lane - 1))
                lanes.pop(idx)
            arm_record['lanes'] = lanes
            return
        lanes = list(arm_record['incoming_lanes'])
        if self.to_value > self.from_value:
            if side == 'left':
                lanes.append(UNKNOWN)
            else:
                lanes.insert(0, UNKNOWN)
        elif self.drop_lane is not None:
            lanes.pop(self.drop_lane - 1)
        arm_record['incoming_lanes'] = lanes

    @property
    def disputed_lane(self):
        if self.drop_lane is not None:
            return self.drop_lane
        if self.insert_pos is not None:
            return int(self.insert_pos) + 1
        return max(1, max(self.from_value, self.to_value))

    def to_dict(self):
        if self.field == 'outgoing_lanes':
            ref = outgoing_reference(self.arm, self.from_value)
        elif self.field == 'lanes' and self.to_value > self.from_value:
            edge = 'leftmost' if self.insert_pos == 0 else 'rightmost'
            ref = f'arm {self.arm}, add unresolved {edge} mouth lane ({self.from_value} -> {self.to_value}; travel left for solver)'
        elif self.field == 'lanes':
            edge = 'leftmost' if self.drop_lane == 1 else 'rightmost'
            ref = f'arm {self.arm}, drop {edge} mouth lane {self.disputed_lane} of {self.from_value} (LTR; travel not distinguished)'
        else:
            ref = incoming_reference(self.arm, self.from_value)
        out = {'kind': 'count', 'arm': self.arm, 'field': self.field, 'reported': self.from_value, 'consistent_value': self.to_value, 'disputed_lane': self.disputed_lane, 'lane_ref': ref}
        if self.insert_pos is not None:
            out['insert_pos'] = self.insert_pos
        return out


def _arm_records(raw):
    return {int(arm.get('id', i + 1)): arm for i, arm in enumerate(raw.get('arms', []))}


def apply_edits(raw, edits):
    patched = copy.deepcopy(raw)
    records = _arm_records(patched)
    for edit in edits:
        record = records.get(int(edit.arm))
        if record is not None:
            edit.apply(record)
    return patched


def relax_all_arrows(raw):
    relaxed = copy.deepcopy(raw)
    for arm in relaxed.get('arms', []):
        if arm.get('lanes') is not None:
            for lane in arm['lanes']:
                lane['movements'] = []
            continue
        arm['incoming_lanes'] = [UNKNOWN for _ in arm.get('incoming_lanes') or []]
    return relaxed


def marked_lanes(raw):
    edits = []
    for arm_id, record in _arm_records(raw).items():
        if record.get('lanes') is not None:
            lanes = record['lanes']
            for idx, lane in enumerate(lanes):
                if lane.get('travel') == 'outgoing':
                    continue
                movements = normalize_movement_candidates(lane.get('movements'))
                if not movements:
                    continue
                edits.append(ArrowEdit(arm=arm_id, lane=idx + 1, lane_count=len(lanes), from_code=movements[0], lane_pos=idx, lane_ref=f'arm {arm_id}, lanes[{idx}] (LTR mouth lane; travel may be unresolved)'))
            continue
        lanes = record.get('incoming_lanes') or []
        for idx, code in enumerate(lanes, start=1):
            if not is_unknown(code):
                edits.append(ArrowEdit(arm=arm_id, lane=idx, lane_count=len(lanes), from_code=canonical_code(code)))
    return edits


def count_edit_candidates(raw):
    edits = []
    side = normalize_driving_side((raw or {}).get('driving_side'))
    for arm_id, record in _arm_records(raw).items():
        if record.get('lanes') is not None:
            lanes = record['lanes']
            for idx, lane in enumerate(lanes):
                travel = lane.get('travel')
                if travel in ('incoming', 'outgoing'):
                    edits.append(TravelRelaxEdit(arm=arm_id, lane_pos=idx, from_travel=travel))
            n = len(lanes)
            # Physical lane add/drop: edge-only (LTR leftmost/rightmost). Added lanes are
            # travel=null so the solver chooses incoming vs outgoing via travel cut.
            if n > 1:
                edits.append(CountEdit(arm=arm_id, field='lanes', from_value=n, to_value=n - 1, drop_lane=1, driving_side=side))
                edits.append(CountEdit(arm=arm_id, field='lanes', from_value=n, to_value=n - 1, drop_lane=n, driving_side=side))
            edits.append(CountEdit(arm=arm_id, field='lanes', from_value=n, to_value=n + 1, insert_pos=0, driving_side=side))
            edits.append(CountEdit(arm=arm_id, field='lanes', from_value=n, to_value=n + 1, insert_pos=n, driving_side=side))
            continue
        out = max(0, int(record.get('outgoing_lanes', 0)))
        lanes = record.get('incoming_lanes') or []
        n_in = len(lanes)
        edits.append(CountEdit(arm=arm_id, field='outgoing_lanes', from_value=out, to_value=out + 1, driving_side=side))
        if out > 0:
            edits.append(CountEdit(arm=arm_id, field='outgoing_lanes', from_value=out, to_value=out - 1, driving_side=side))
        edits.append(CountEdit(arm=arm_id, field='incoming_lanes', from_value=n_in, to_value=n_in + 1, driving_side=side))
        if n_in >= 1:
            for pos in range(1, n_in + 1):
                edits.append(CountEdit(arm=arm_id, field='incoming_lanes', from_value=n_in, to_value=n_in - 1, drop_lane=pos, driving_side=side))
    return edits


class _RepairProbe:
    def __init__(self, *, weights, seed, per_arm_cap, max_trials):
        self.kwargs = dict(weights=weights, seed=seed, per_arm_cap=per_arm_cap, node_budget=DEFAULT_DIAGNOSIS_NODE_BUDGET, leaf_budget=DEFAULT_DIAGNOSIS_LEAF_BUDGET)
        self.max_trials = int(max_trials)
        self.trials = 0

    @property
    def exhausted(self):
        return self.trials >= self.max_trials

    def solve(self, raw):
        if self.exhausted:
            return None
        self.trials += 1
        try:
            _filled, _connections, solution = _run_pipeline(raw, return_solution=True, **self.kwargs)
            return solution
        except ConnectionParseError:
            return None
        except Exception:
            return None

    def feasible(self, raw):
        return self.solve(raw) is not None


def _repairs_of_size(raw, candidates, size, probe, max_candidates):
    found = []
    for combo in itertools.combinations(candidates, size):
        if probe.exhausted:
            break
        if probe.feasible(apply_edits(raw, combo)):
            found.append(combo)
            if len(found) >= max_candidates:
                break
    return found


def _mixed_repairs_of_size(raw, arrow_pool, count_pool, size, probe, max_candidates):
    """Size-k repairs that mix arrow edits with count/travel/lane-presence edits."""
    if size < 2 or not arrow_pool or not count_pool:
        return []
    found = []
    for n_arrow in range(1, size):
        n_count = size - n_arrow
        if n_count > len(count_pool) or n_arrow > len(arrow_pool):
            continue
        for arrow_combo in itertools.combinations(arrow_pool, n_arrow):
            if probe.exhausted or len(found) >= max_candidates:
                return found
            for count_combo in itertools.combinations(count_pool, n_count):
                if probe.exhausted or len(found) >= max_candidates:
                    return found
                combo = arrow_combo + count_combo
                if probe.feasible(apply_edits(raw, combo)):
                    found.append(combo)
    return found


def _minimal_repairs(raw, arrow_pool, count_pool, probe, *, max_size, max_candidates):
    for size in range(1, max_size + 1):
        arrow_repairs = _repairs_of_size(raw, arrow_pool, size, probe, max_candidates)
        count_repairs = _repairs_of_size(raw, count_pool, size, probe, max_candidates)
        mixed_repairs = _mixed_repairs_of_size(raw, arrow_pool, count_pool, size, probe, max_candidates)
        repairs = arrow_repairs + count_repairs + mixed_repairs
        if repairs:
            return (repairs, size)
    return ([], None)


def _diagnosis_label(repairs):
    kinds = {edit.kind for combo in repairs for edit in combo}
    if kinds == {'arrow'}:
        return 'arrow_misread'
    if kinds == {'count'}:
        return 'lane_count_misread'
    return 'arrow_or_lane_count_misread'


def diagnose_infeasibility(raw, *, weights, seed, per_arm_cap, max_repair_size=DEFAULT_MAX_REPAIR_SIZE, max_trials=DEFAULT_MAX_REPAIR_TRIALS, max_candidates=DEFAULT_MAX_REPAIR_CANDIDATES):
    """Localise infeasible config to minimal conflicting reports (hypotheses for image re-check)."""
    probe = _RepairProbe(weights=weights, seed=seed, per_arm_cap=per_arm_cap, max_trials=max_trials)
    arrow_pool = marked_lanes(raw)
    count_pool = count_edit_candidates(raw)
    arrows_relaxed_feasible = probe.feasible(relax_all_arrows(raw))
    repair = {'all_arrows_relaxed_feasible': arrows_relaxed_feasible, 'candidates': [], 'minimal_size': None, 'search_exhausted': False}
    repairs, size = _minimal_repairs(raw, arrow_pool, count_pool, probe, max_size=max_repair_size, max_candidates=max_candidates)
    if repairs:
        candidates = [{'edits': [edit.to_dict() for edit in combo]} for combo in repairs]
        repair.update(diagnosis=_diagnosis_label(repairs), minimal_size=size, candidates=candidates)
    elif arrows_relaxed_feasible:
        repair.update(diagnosis='arrow_misread', minimal_size=None, note=f'No repair of size <= {max_repair_size} was found, but feasibility is restored once every arrow is dropped under the current topology and connectivity assumptions. The conflict involves the reported arrows, possibly spread over more lanes than the search covers.', candidates=[{'edits': [edit.to_dict() for edit in arrow_pool]}])
    else:
        relaxed = relax_all_arrows(raw)
        mixed = []
        mixed_size = None
        for size in range(1, max_repair_size + 1):
            mixed = _repairs_of_size(relaxed, count_pool, size, probe, max_candidates)
            if mixed:
                mixed_size = size
                break
        if mixed:
            repair.update(diagnosis='lane_count_misread', minimal_size=mixed_size, note='Under the current assumptions, this lane count only becomes feasible once the arrows on the affected arm(s) are also treated as unknown, so re-read the whole cross-section there, not just the count.', candidates=[{'edits': [edit.to_dict() for edit in combo]} for combo in mixed])
        else:
            repair.update(diagnosis='not_repairable_at_lane_level', note=f'No edit set of size <= {max_repair_size} to any arrow or lane count restores feasibility, and no arrow assignment works even when all arrows are dropped. The inconsistency lies in the arm-level topology (arm count or headings), which the lane-level stage cannot fix.')
    repair['search_exhausted'] = probe.exhausted
    repair['trials'] = probe.trials
    return repair


def _format_repair_edit(edit):
    kind = edit.get("kind")
    arm = edit.get("arm")
    if kind == "arrow":
        return f"arm {arm} incoming lane {edit.get('lane')}: relax arrow {code_display(edit.get('current_code'))} -> N"
    field = edit.get("field")
    reported = edit.get("reported")
    consistent = edit.get("consistent_value")
    disputed = edit.get("disputed_lane")
    if field == "lanes.travel":
        return f"arm {arm} lane {disputed}: relax travel {reported!r} -> None (re-search travel cut)"
    if field == "lanes":
        if consistent is not None and reported is not None and consistent > reported:
            edge = "leftmost" if edit.get("insert_pos") == 0 else "rightmost"
            return f"arm {arm}: add unresolved {edge} mouth lane ({reported} -> {consistent}; travel=null for solver)"
        edge = "leftmost" if disputed == 1 else "rightmost"
        return f"arm {arm}: drop {edge} mouth lane {disputed} ({reported} -> {consistent} lanes LTR)"
    if field == "outgoing_lanes":
        drop = f", drop outgoing lane {disputed}" if disputed is not None and consistent is not None and consistent < reported else ""
        return f"arm {arm}: outgoing_lanes {reported} -> {consistent}{drop}"
    if field == "incoming_lanes":
        drop = f", drop incoming lane {disputed}" if disputed is not None and consistent is not None and consistent < reported else ""
        return f"arm {arm}: incoming_lanes {reported} -> {consistent}{drop}"
    return f"arm {arm}: {field} {reported} -> {consistent}"


def format_repair_diagnosis(repair):
    """Human-readable lines explaining fill_search failure + minimal feasibility-restoring edits."""
    if not repair:
        return []
    lines = []
    diagnosis = repair.get("diagnosis")
    size = repair.get("minimal_size")
    trials = repair.get("trials")
    exhausted = repair.get("search_exhausted")
    lines.append(
        f"fill_search FAILED -> stage5_repair diagnosis={diagnosis!r}, "
        f"minimal_edit_size={size}, trials={trials}, search_exhausted={exhausted}"
    )
    if repair.get("note"):
        lines.append(f"note: {repair['note']}")
    if diagnosis == "not_repairable_at_lane_level":
        lines.append("no lane-level edit restores feasibility; escalate to arm-level geometry")
        return lines
    candidates = repair.get("candidates") or []
    if not candidates:
        lines.append("no concrete repair candidates found")
        return lines
    lines.append(
        f"meaning: each candidate below is a MINIMAL set of reported readings that, if relaxed/changed as shown, "
        f"makes fill+connectivity FEASIBLE under current topology (not proven unique true cause; re-check on image)."
    )
    if size is not None:
        lines.append(f"found {len(candidates)} candidate repair(s) of size {size}:")
    else:
        lines.append(f"found {len(candidates)} candidate repair(s):")
    for rank, candidate in enumerate(candidates, start=1):
        edits = candidate.get("edits") or []
        lines.append(f"candidate {rank}: apply ALL of the following ({len(edits)} edit(s)) -> feasible")
        for edit in edits:
            lines.append(f"  - {_format_repair_edit(edit)}")
    return lines


def build_repair_issues(repair):
    diagnosis = repair.get('diagnosis')
    issues = []
    if diagnosis == 'not_repairable_at_lane_level':
        return [make_issue('stage5_repair', 'not_repairable_at_lane_level', repair.get('note', 'No lane-level change can make this junction consistent.'), details={'repair': repair})]
    seen = {}
    for rank, candidate in enumerate(repair.get('candidates') or []):
        for edit in candidate.get('edits', []):
            key = (edit['kind'], edit['arm'], edit.get('lane'), edit.get('disputed_lane'), edit.get('field'), edit.get('consistent_value'), edit.get('insert_pos'))
            if key not in seen:
                seen[key] = {**edit, 'candidate_rank': rank}
    for edit in seen.values():
        arm_id = int(edit['arm'])
        if edit['kind'] == 'arrow':
            details = {'current_code': edit['current_code'], 'lane_count': edit.get('lane_count'), 'candidate_rank': edit['candidate_rank'], 'lane_ref': edit['lane_ref']}
            issues.append(make_issue('stage5_repair', 'arrow_verification_required', f"Arm {arm_id} incoming lane {edit['lane']}: reported arrow {code_display(details['current_code'])} is a potential visual misread under current topology/connectivity assumptions.", location={'arm': arm_id, 'lane': int(edit['lane'])}, details=details, arm_id_list=[arm_id]))
        else:
            field = edit.get('field')
            if field == 'lanes.travel':
                constraint = 'travel_verification_required'
                message = f"Arm {arm_id} {field}={edit['reported']} is a potential visual misread under current topology/connectivity assumptions."
            elif field == 'lanes':
                constraint = 'lane_presence_verification_required'
                if edit.get('consistent_value') is not None and edit.get('reported') is not None and edit['consistent_value'] > edit['reported']:
                    edge = 'leftmost' if edit.get('insert_pos') == 0 else 'rightmost'
                    message = f"Arm {arm_id} may be missing a {edge} mouth lane (add unresolved travel=null lane; {edit['reported']} -> {edit['consistent_value']})."
                else:
                    edge = 'leftmost' if edit.get('disputed_lane') == 1 else 'rightmost'
                    message = f"Arm {arm_id} {edge} mouth lane {edit['disputed_lane']} (of {edit['reported']} LTR) may be an extra lane; dropping it restores feasibility under current topology/connectivity assumptions."
            elif field == 'outgoing_lanes':
                constraint = 'outgoing_count_verification_required'
                message = f"Arm {arm_id} {field}={edit['reported']} is a potential visual misread under current topology/connectivity assumptions."
            else:
                constraint = 'incoming_count_verification_required'
                message = f"Arm {arm_id} {field}={edit['reported']} is a potential visual misread under current topology/connectivity assumptions."
            issues.append(make_issue(
                'stage5_repair', constraint, message,
                location={'arm': arm_id, 'field': field, 'lane': edit.get('disputed_lane')},
                details={
                    'current_value': edit['reported'],
                    'consistent_value': edit.get('consistent_value'),
                    'lanes': [edit['disputed_lane']],
                    'candidate_rank': edit['candidate_rank'],
                    'lane_ref': edit['lane_ref'],
                    'insert_pos': edit.get('insert_pos'),
                    'edge': (
                        'leftmost' if (
                            (edit.get('insert_pos') == 0) or (edit.get('disputed_lane') == 1 and edit.get('consistent_value') is not None and edit.get('reported') is not None and edit['consistent_value'] < edit['reported'])
                        ) else 'rightmost'
                    ),
                },
                arm_id_list=[arm_id],
            ))
    return issues


def _localised_error(raw, exc, *, weights, seed, per_arm_cap):
    if any((issue.get('constraint') in (
        'channelized_forbids_slip_turn',
        'channelized_from_arm_slip_travel',
        'channelized_forbids_right_turn',
        'channelized_from_arm_rightmost_travel',
    ) for issue in exc.issues)):
        return exc
    try:
        repair = diagnose_infeasibility(raw, weights=weights, seed=seed, per_arm_cap=per_arm_cap)
    except Exception as diag_exc:
        return ConnectionParseError(exc.issues, exc.title, repair={'diagnosis': 'diagnosis_failed', 'error': str(diag_exc)}, violations=exc.issues)
    issues = build_repair_issues(repair)
    if not issues:
        return ConnectionParseError(exc.issues, exc.title, repair=repair, violations=exc.issues)
    titles = {'arrow_misread': '[stage5_repair] a reported arrow conflicts with topology/connectivity assumptions', 'arrow_or_lane_count_misread': '[stage5_repair] a reported arrow or lane count conflicts with topology/connectivity assumptions', 'lane_count_misread': '[stage5_repair] a reported lane count conflicts with topology/connectivity assumptions', 'not_repairable_at_lane_level': '[stage5_repair] not fixable at the lane level - escalate to arm level'}
    return ConnectionParseError(issues, titles.get(repair.get('diagnosis'), exc.title), repair=repair, violations=exc.issues)


def _run_pipeline(raw, *, weights, seed, node_budget, leaf_budget, per_arm_cap, return_solution=False):
    """Validate, fill, and connect."""
    spec = load_junction_from_dict(raw)
    features = collect_channelized_features(raw)
    channelized = resolve_channelized_links(spec, features)
    if has_unknown_lanes(spec):
        solution = solve_fill_optimization(spec, features, channelized, weights=weights, rng=random.Random(seed), node_budget=node_budget, leaf_budget=leaf_budget, per_arm_cap=per_arm_cap)
        saved = _apply_outgoing(spec, solution.outgoing_by_arm)
        try:
            channelized = resolve_channelized_links(spec, features)
            filled_config = build_filled_config(raw, spec, solution.filled, solution.outgoing_by_arm)
            output = build_connections_output(spec, marking_from_filled(solution.filled), channelized)
        finally:
            _restore_outgoing(spec, saved)
        if return_solution:
            return (filled_config, output, solution)
        return (filled_config, output)
    validate_lane_constraints(spec, explicit_marking(spec), 'stage1_pre_solve', require_resolved=False, channelized=channelized)
    marking = explicit_marking(spec)
    validate_lane_constraints(spec, marking, 'stage1_pre_solve_explicit', require_resolved=True, channelized=channelized)
    output = build_connections_output(spec, marking, channelized)
    validate_connectivity(output, 'stage1_pre_solve_connectivity')
    filled_config = build_filled_config(raw, spec, filled_from_codes({arm.id: tuple(canonical_code(l) for l in arm.incoming_lanes) for arm in spec.arms}), {arm.id: arm.outgoing_lanes for arm in spec.arms})
    if return_solution:
        codes_by_arm = {arm.id: tuple((canonical_code(l) for l in arm.incoming_lanes)) for arm in spec.arms}
        solution = FillSolution(codes_by_arm=codes_by_arm, filled=filled_from_codes(codes_by_arm), cost=total_cost(spec, codes_by_arm, weights, channelized), breakdown=cost_breakdown(spec, codes_by_arm, weights, channelized), stats={'no_unknown_lanes': True}, outgoing_by_arm={arm.id: arm.outgoing_lanes for arm in spec.arms})
        return (filled_config, output, solution)
    return (filled_config, output)


def process(source, seed=DEFAULT_SEED, search_budget=DEFAULT_SEARCH_BUDGET, weights=None, per_arm_cap=DEFAULT_PER_ARM_CAP, return_solution=False, diagnose=True):
    """Resolve N lanes; on failure localise to minimal conflicting readings when diagnose=True."""
    weights = weights or CostWeights()
    if isinstance(source, dict):
        raw = copy.deepcopy(source)
    else:
        raw, _ = load_raw_and_spec(str(source))
    node_budget = int(search_budget)
    leaf_budget = int(search_budget)
    try:
        return _run_pipeline(raw, weights=weights, seed=seed, node_budget=node_budget, leaf_budget=leaf_budget, per_arm_cap=per_arm_cap, return_solution=return_solution)
    except ConnectionParseError as exc:
        if not diagnose:
            raise
        raise _localised_error(raw, exc, weights=weights, seed=seed, per_arm_cap=per_arm_cap) from None

def parse_filled_config_and_connections(source, *, seed=DEFAULT_SEED, search_budget=DEFAULT_SEARCH_BUDGET, weights=None):
    return process(source, seed=seed, search_budget=search_budget, weights=weights)


def parse_connections_from_file(source, *, seed=DEFAULT_SEED, search_budget=DEFAULT_SEARCH_BUDGET, weights=None):
    _filled_config, connections_output = process(source, seed=seed, search_budget=search_budget, weights=weights)
    return connections_output


def solve(config, *, name=None, seed=None, search_budget=None, weights=None, diagnose=True, return_solution=False):
    raw = {**config}
    if name is not None:
        raw["name"] = name
    kwargs = {"diagnose": diagnose, "return_solution": return_solution}
    if seed is not None:
        kwargs["seed"] = seed
    if search_budget is not None:
        kwargs["search_budget"] = search_budget
    if weights is not None:
        kwargs["weights"] = weights
    return process(raw, **kwargs)


def _lane_movements_list(lane):
    return list(normalize_movement_candidates(lane.get("movements")))


def _arm_outgoing_count(arm):
    if arm.get("outgoing_lanes") is not None:
        return int(arm["outgoing_lanes"])
    lanes = arm.get("lanes") or []
    return sum(1 for lane in lanes if lane.get("travel") == "outgoing")


def summarize_fill_adjustments(config, filled_config):
    """Human-readable list of travel/movement/outgoing changes from fill search."""
    before_by_id = {int(arm["id"]): arm for arm in (config or {}).get("arms") or []}
    after_by_id = {int(arm["id"]): arm for arm in (filled_config or {}).get("arms") or []}
    lines = []
    for arm_id in sorted(before_by_id):
        before_arm = before_by_id[arm_id]
        after_arm = after_by_id.get(arm_id)
        if after_arm is None:
            lines.append(f"arm {arm_id}: missing from filled_config")
            continue
        before_out = _arm_outgoing_count(before_arm)
        after_out = _arm_outgoing_count(after_arm)
        if before_out != after_out:
            lines.append(f"arm {arm_id}: outgoing_lanes {before_out} -> {after_out} (travel cut adjusted)")
        before_lanes = before_arm.get("lanes") or []
        after_lanes = after_arm.get("lanes") or []
        n = max(len(before_lanes), len(after_lanes))
        for idx in range(n):
            if idx >= len(before_lanes):
                lines.append(f"arm {arm_id} lane {idx + 1}: added in filled_config")
                continue
            if idx >= len(after_lanes):
                lines.append(f"arm {arm_id} lane {idx + 1}: removed in filled_config")
                continue
            before_lane = before_lanes[idx]
            after_lane = after_lanes[idx]
            parts = []
            before_travel = before_lane.get("travel")
            after_travel = after_lane.get("travel")
            if before_travel != after_travel:
                parts.append(f"travel {before_travel!r} -> {after_travel!r}")
            before_mov = _lane_movements_list(before_lane)
            after_mov = _lane_movements_list(after_lane)
            if before_mov != after_mov:
                if not before_mov and after_mov:
                    parts.append(f"filled unknown N -> {after_mov[0]}")
                elif len(before_mov) > 1 and after_mov:
                    parts.append(f"chose {after_mov[0]} from candidates {before_mov}")
                elif before_mov and not after_mov and after_travel == "outgoing":
                    parts.append(f"cleared movements {before_mov} (now outgoing)")
                else:
                    parts.append(f"movements {before_mov} -> {after_mov}")
            if parts:
                lines.append(f"arm {arm_id} lane {idx + 1}: " + "; ".join(parts))
    return lines


def try_solve(config, *, name=None, seed=None, search_budget=None, weights=None, diagnose=True):
    try:
        filled_config, connections = solve(
            config, name=name, seed=seed, search_budget=search_budget,
            weights=weights, diagnose=diagnose, return_solution=False,
        )
        return filled_config, connections, None
    except ConnectionParseError as exc:
        return None, None, exc
    except Exception as exc:
        synthetic = ConnectionParseError(
            [make_issue("stage_validate", "unexpected_error", f"Unexpected error while solving fill/connections: {exc}")],
            title="[stage_validate] unexpected error during fill/connections",
        )
        return None, None, synthetic
