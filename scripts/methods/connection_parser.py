import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.methods.method_utils import dump_json
from scripts.methods.llm_kg.lane_level.schemas import MOVEMENT_CODES, movement_codes

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

MOVEMENT_MAP = {'L': 'left', 'S': 'straight', 'R': 'right'}
VALID_MOVEMENTS = set(MOVEMENT_MAP)
UNKNOWN = 'N'
ORDER = {'L': 0, 'S': 1, 'R': 2}
DEFAULT_ARM_LENGTH = 100.0
DEFAULT_JUNCTION_RADIUS = 50.0
DEFAULT_CHANNELIZED_LENGTH = 25
DEFAULT_SEED = 42
DEFAULT_SEARCH_BUDGET = 200_000
DEFAULT_PER_ARM_CAP = 20_000
STRAIGHT_THROUGH_TOLERANCE = 30.0
INCOMING_LANE_1_IS_LEFTMOST = True
DEFAULT_DRIVING_SIDE = 'left'
DEFAULT_MAX_REPAIR_SIZE = 2
DEFAULT_MAX_REPAIR_TRIALS = 600
DEFAULT_MAX_REPAIR_CANDIDATES = 6
DEFAULT_DIAGNOSIS_NODE_BUDGET = 50_000
DEFAULT_DIAGNOSIS_LEAF_BUDGET = 20_000
REVISION_INSTRUCTION = "Re-inspect only the arms and lanes named as suspects, directly from the image, then resubmit the full cross_section for each of them. The validator only flags potential visual misreads under current topology/connectivity assumptions; it does not prescribe what value to write. Report only what is visible in the image. If the image still matches your prior reading, keep it and say so in `uncertainty`. Do not change arm headings or junction geometry - they are fixed."

# --------------------------------------------------------------------------- #
# Driving side (left = LHD/RHT mouth out|in; right = RHD/LHT mouth in|out)
# --------------------------------------------------------------------------- #

def normalize_driving_side(value):
    if value is None or value == '':
        return DEFAULT_DRIVING_SIDE
    key = str(value).lower().strip()
    if key in ('left', 'l', 'lhd', 'left_hand_drive'):
        return 'left'
    if key in ('right', 'r', 'rhd', 'right_hand_drive'):
        return 'right'
    raise ValueError("invalid driving_side %r; expected 'left' or 'right'" % (value,))


def travel_block_order(driving_side=None):
    """LTR mouth travel blocks: left -> (outgoing, incoming); right -> (incoming, outgoing)."""
    if normalize_driving_side(driving_side) == 'left':
        return ('outgoing', 'incoming')
    return ('incoming', 'outgoing')


def assigned_travel_at(index, cut, driving_side=None):
    """Travel assigned to LTR index under cut (first-block length)."""
    first, second = travel_block_order(driving_side)
    return first if index < cut else second


def slip_turn_letter(driving_side=None):
    """Channelized slip / curb-side turn: R for left (RHT), L for right (LHT)."""
    return 'R' if normalize_driving_side(driving_side) == 'left' else 'L'


def inner_turn_letter(driving_side=None):
    """Center-side turn onto innermost outgoing: L for left (RHT), R for right (LHT)."""
    return 'L' if normalize_driving_side(driving_side) == 'left' else 'R'


def normalize_channelized_source_type(value):
    kind = str(value or '').strip().lower()
    return kind if kind in ('shared', 'dedicated') else 'dedicated'


def slip_assigned_code(source_type, driving_side=None):
    """xodr movement on the curb-edge slip lane: dedicated=R/L, shared=SR/LS."""
    letter = slip_turn_letter(driving_side)
    if normalize_channelized_source_type(source_type) == 'shared':
        return 'SR' if letter == 'R' else 'LS'
    return letter


def channelized_source_of(channelized, from_arm):
    """Resolve shared/dedicated from arm-level channelized feature source_type."""
    for (fa, _to_arm), info in (channelized or {}).items():
        if int(fa) == int(from_arm):
            return normalize_channelized_source_type((info or {}).get('source_type'))
    return 'dedicated'


def slip_edge(driving_side=None):
    """LTR mouth edge of the slip-fed incoming lane: rightmost (left) or leftmost (right)."""
    return 'rightmost' if normalize_driving_side(driving_side) == 'left' else 'leftmost'


def slip_to_edge(driving_side=None):
    """LTR mouth edge of the receiving outgoing merge: leftmost (left/RHT) or rightmost (right/LHT)."""
    return 'leftmost' if normalize_driving_side(driving_side) == 'left' else 'rightmost'


def slip_from_lane_id(incoming_lane_count, driving_side=None):
    """Incoming lane_id for the curb-side slip (1 = leftmost of incoming block)."""
    n = int(incoming_lane_count)
    if n <= 0:
        return None
    if normalize_driving_side(driving_side) == 'left':
        return n
    return 1


def slip_to_lane_id(outgoing_lane_count, driving_side=None):
    """Outgoing lane_id that receives the curb-side slip (1 = leftmost of outgoing block)."""
    n = int(outgoing_lane_count)
    if n <= 0:
        return None
    if normalize_driving_side(driving_side) == 'left':
        return 1
    return n


def odr_left_right_lanes(outgoing, incoming, driving_side=None):
    """OpenDRIVE left/right lane counts when the road reference points toward the junction."""
    out_n, in_n = int(outgoing or 0), int(incoming or 0)
    if normalize_driving_side(driving_side) == 'left':
        return out_n, in_n
    return in_n, out_n


def cut_rule_text(driving_side=None):
    if normalize_driving_side(driving_side) == 'left':
        return (
            'every lane left of that cut must have travel=outgoing, and every lane '
            'right of it travel=incoming'
        )
    return (
        'every lane left of that cut must have travel=incoming, and every lane '
        'right of it travel=outgoing'
    )


def channelized_turn_name(driving_side=None):
    return 'right-turn' if normalize_driving_side(driving_side) == 'left' else 'left-turn'


def channelized_turn_direction(driving_side=None):
    """Geometric turn class for a channelized slip: 'right' (left/RHT) or 'left' (right/LHT)."""
    return 'right' if normalize_driving_side(driving_side) == 'left' else 'left'


def arm_heading_midpoint(arm):
    """Best outward heading for an arm dict (explicit heading_deg or min/max midpoint)."""
    if not isinstance(arm, dict):
        return None
    if arm.get('heading_deg') is not None:
        return float(arm['heading_deg']) % 360.0
    if arm.get('heading_deg_min') is None or arm.get('heading_deg_max') is None:
        return None
    hmin = float(arm['heading_deg_min']) % 360.0
    hmax = float(arm['heading_deg_max']) % 360.0
    span = (hmax - hmin) % 360.0
    return (hmin + span / 2.0) % 360.0


def _channelized_turn_score(from_heading, to_heading, turn):
    """Lower is closer to an ideal left/right turn from from_heading into to_heading."""
    incoming = normalize_angle(float(from_heading) + 180.0)
    target_diff = {'left': 90.0, 'right': 270.0}[turn]
    diff = angle_diff_ccw(incoming, to_heading)
    return abs(diff - target_diff)


def resolve_channelized_endpoints(arm_a, arm_b, heading_a, heading_b, driving_side=None):
    """Orient an unordered arm pair into (from_arm, to_arm) for the channelized slip turn."""
    turn = channelized_turn_direction(driving_side)
    a, b = int(arm_a), int(arm_b)
    ha, hb = float(heading_a), float(heading_b)
    ab = classify_relative_direction(ha, hb)
    ba = classify_relative_direction(hb, ha)
    if ab == turn and ba != turn:
        return a, b
    if ba == turn and ab != turn:
        return b, a
    if _channelized_turn_score(ha, hb, turn) <= _channelized_turn_score(hb, ha, turn):
        return a, b
    return b, a


def enrich_channelized_features(features, arms, driving_side=None):
    """Fill from_arm/to_arm from arm_list + headings when missing (arm-level emits unordered pairs)."""
    side = normalize_driving_side(driving_side)
    headings = {}
    for arm in arms or []:
        if not isinstance(arm, dict) or arm.get('id') is None:
            continue
        heading = arm_heading_midpoint(arm)
        if heading is not None:
            headings[int(arm['id'])] = heading
    out = []
    for feature in features or []:
        if not isinstance(feature, dict):
            continue
        item = dict(feature)
        from_arm, to_arm = item.get('from_arm'), item.get('to_arm')
        if from_arm is not None and to_arm is not None:
            from_arm, to_arm = int(from_arm), int(to_arm)
            item['from_arm'] = from_arm
            item['to_arm'] = to_arm
            item['arm_list'] = [from_arm, to_arm]
        else:
            arm_list = item.get('arm_list') or []
            if len(arm_list) == 2:
                a, b = int(arm_list[0]), int(arm_list[1])
                if a in headings and b in headings:
                    from_arm, to_arm = resolve_channelized_endpoints(a, b, headings[a], headings[b], side)
                    item['from_arm'] = from_arm
                    item['to_arm'] = to_arm
                    item['arm_list'] = [from_arm, to_arm]
        kind = str(item.get('source_type') or '').strip().lower()
        item['source_type'] = kind if kind in ('shared', 'dedicated') else 'dedicated'
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# Specs, geometry, movement codes (shared)
# --------------------------------------------------------------------------- #

def fail(stage, constraint, message, *, location=None, details=None, arm_id_list=None):
    """Lazy bridge to refinement.common (used while loading ArmSpec)."""
    from scripts.methods.llm_kg.refinement.common import fail as _fail
    return _fail(stage, constraint, message, location=location, details=details, arm_id_list=arm_id_list)

def is_unknown(value):
    if value is None:
        return True
    if isinstance(value, list):
        return len(value) == 0
    return isinstance(value, str) and value.upper() in (UNKNOWN, 'NULL', 'NONE', 'UNDEFINED', '')

def entry_letters(value):
    if is_unknown(value):
        return set()
    if isinstance(value, list):
        if not value:
            return set()
        return entry_letters(value[0])
    code = canonicalize_movement_code(value)
    if is_unknown(code):
        return set()
    return set(code)

def canonical_code(value):
    if isinstance(value, list):
        if not value:
            return UNKNOWN
        return canonical_code(value[0])
    code = canonicalize_movement_code(value)
    return UNKNOWN if is_unknown(code) else code

def letters_to_code(letters):
    return ''.join(sorted(letters, key=lambda c: ORDER[c])) if letters else UNKNOWN

def is_lr_only(letters):
    return bool(letters) and ({'L', 'R'} <= letters) and ('S' not in letters)


def is_valid_movement_code(letters, *, allow_lr=False):
    if not letters or not (letters <= VALID_MOVEMENTS):
        return False
    code = letters_to_code(letters)
    if code not in MOVEMENT_CODES:
        return False
    if is_lr_only(letters):
        return allow_lr
    return True


def allowed_movement_codes(*, allow_lr=False):
    return [UNKNOWN] + movement_codes(allow_lr=allow_lr)


def canonicalize_movement_code(code):
    """Uppercase a movement code; null-like markers become N."""
    if code is None:
        return UNKNOWN
    key = str(code).upper().strip()
    if key in ('', UNKNOWN, 'NULL', 'NONE', 'UNDEFINED'):
        return UNKNOWN
    return key


def normalize_movement_candidates(value):
    """Normalize a movements field into a ranked list of distinct codes (empty = unknown)."""
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    ordered = []
    for item in items:
        if item is None:
            continue
        code = canonicalize_movement_code(item)
        if is_unknown(code):
            continue
        if code not in ordered:
            ordered.append(code)
    return ordered


class LaneReading:
    """One LTR mouth lane: optional fixed travel; movements only for incoming (or travel-null)."""

    def __init__(self, travel, movements):
        if travel not in (None, 'incoming', 'outgoing'):
            raise ValueError(f'invalid travel {travel!r}')
        self.travel = travel
        self.movements = [] if travel == 'outgoing' else list(movements or [])


def parse_lane_reading(item):
    if not isinstance(item, dict):
        raise fail('lane', 'lane_object', 'Each lanes[] entry must be an object.')
    travel = item.get('travel')
    if travel is not None:
        travel = str(travel).lower()
        if travel not in ('incoming', 'outgoing'):
            raise fail('lane', 'travel_value', f"lanes[].travel must be incoming, outgoing, or null (got {item.get('travel')!r}).")
    movements = normalize_movement_candidates(item.get('movements'))
    if travel == 'outgoing':
        movements = []
    return LaneReading(travel, movements)


def movement_entry_from_reading(reading):
    """Encode an incoming-slot reading for option building: N / code / candidate list."""
    if reading.travel == 'outgoing' or not reading.movements:
        return UNKNOWN
    if len(reading.movements) == 1:
        return reading.movements[0]
    return list(reading.movements)


def valid_travel_cuts(lane_readings, driving_side=None):
    """Cut k = length of the first LTR travel block; respect fixed travel.

    left:  [0:k]=outgoing, [k:]=incoming
    right: [0:k]=incoming, [k:]=outgoing
    """
    side = normalize_driving_side(driving_side)
    n = len(lane_readings)
    cuts = []
    for k in range(n + 1):
        ok = True
        for i, reading in enumerate(lane_readings):
            assigned = assigned_travel_at(i, k, side)
            if reading.travel is not None and reading.travel != assigned:
                ok = False
                break
        if ok:
            cuts.append(k)
    return cuts


def through_road_arm_ids(data):
    ids = set()
    for a, b in through_road_pairs(data):
        ids.add(a)
        ids.add(b)
    return ids


def through_road_pairs(data):
    """Declared through-road arm pairs from the lane/config document."""
    pairs = []
    seen = set()
    for item in (data or {}).get("through_roads") or []:
        ids = [int(arm_id) for arm_id in (item.get("arm_ids") or []) if arm_id is not None]
        if len(ids) < 2:
            continue
        key = (min(ids[0], ids[1]), max(ids[0], ids[1]))
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def through_peer_id(spec, arm_id):
    arm_id = int(arm_id)
    for left, right in getattr(spec, "through_road_pairs", ()) or ():
        if left == arm_id:
            return right
        if right == arm_id:
            return left
    return None

def normalize_angle(angle):
    return angle % 360.0

def angle_diff_ccw(from_angle, to_angle):
    return (normalize_angle(to_angle) - normalize_angle(from_angle)) % 360.0

def is_contiguous(ids):
    return all((b - a == 1 for a, b in zip(ids, ids[1:])))

class ArmSpec:

    def __init__(self, arm_id, heading_deg, *, lanes=None, incoming_lanes=None, outgoing_lanes=None, allow_lr=False, driving_side=None):
        self.id = int(arm_id)
        self.heading_deg = float(heading_deg)
        self.length = DEFAULT_ARM_LENGTH
        self.allow_lr = bool(allow_lr)
        self.driving_side = normalize_driving_side(driving_side)
        if lanes is not None:
            self.lane_readings = [parse_lane_reading(item) for item in lanes]
        else:
            out_n = max(0, int(outgoing_lanes or 0))
            out_readings = [LaneReading('outgoing', []) for _ in range(out_n)]
            in_readings = []
            for lane in incoming_lanes or []:
                if is_unknown(lane) or lane is None:
                    candidates = []
                elif isinstance(lane, list):
                    candidates = normalize_movement_candidates(lane)
                else:
                    candidates = normalize_movement_candidates([lane])
                in_readings.append(LaneReading('incoming', candidates))
            if self.driving_side == 'left':
                self.lane_readings = out_readings + in_readings
            else:
                self.lane_readings = in_readings + out_readings
        for idx, reading in enumerate(self.lane_readings, start=1):
            for code in reading.movements:
                if not is_valid_movement_code(entry_letters(code), allow_lr=self.allow_lr):
                    raise fail('lane', 'movement_code_validity', f"Arm {self.id} lane #{idx} has an invalid movement '{code}'.", location={'arm': self.id, 'lane': idx}, details={'value': code, 'allowed': allowed_movement_codes(allow_lr=self.allow_lr)}, arm_id_list=[self.id])
        cuts = valid_travel_cuts(self.lane_readings, driving_side=self.driving_side)
        if not cuts:
            order = travel_block_order(self.driving_side)
            raise fail('lane', 'travel_ordering', f'Arm {self.id} travel assignments violate {order[0]}-left-of-{order[1]}.', location={'arm': self.id}, arm_id_list=[self.id])
        self.apply_cut(cuts[0])

    def apply_cut(self, cut):
        cut = int(cut)
        n = len(self.lane_readings)
        if self.driving_side == 'left':
            self.outgoing_lanes = cut
            self.incoming_lanes = [movement_entry_from_reading(r) for r in self.lane_readings[cut:]]
        else:
            self.outgoing_lanes = n - cut
            self.incoming_lanes = [movement_entry_from_reading(r) for r in self.lane_readings[:cut]]

    def refresh_incoming_from_readings(self):
        """Rebuild incoming_lanes from lane_readings under the current cut layout."""
        if self.driving_side == 'left':
            self.incoming_lanes = [movement_entry_from_reading(r) for r in self.lane_readings[self.outgoing_lanes:]]
        else:
            cut = len(self.lane_readings) - int(self.outgoing_lanes)
            self.incoming_lanes = [movement_entry_from_reading(r) for r in self.lane_readings[:cut]]

    @property
    def travel_uncertain(self):
        return any(reading.travel is None for reading in self.lane_readings)

    @property
    def incoming_lane_count(self):
        return len(self.incoming_lanes)

    def needs_search(self):
        if self.travel_uncertain:
            return True
        for entry in self.incoming_lanes:
            if is_unknown(entry):
                return True
            if isinstance(entry, list) and len(entry) > 1:
                return True
        return False

    def to_road_dict(self):
        return {'id': self.id, 'length': self.length, 'incoming_lanes': self.incoming_lane_count, 'outgoing_lanes': self.outgoing_lanes, 'heading_deg': self.heading_deg}

class JunctionSpec:

    def __init__(self, arms, junction_type='unknown', through_road_arm_ids=None, driving_side=None, through_road_pairs=None):
        self.type = str(junction_type)
        self.junction_radius = DEFAULT_JUNCTION_RADIUS
        self.arms = arms
        self.arm_map = {arm.id: arm for arm in arms}
        self.through_road_pairs = [
            (int(a), int(b)) for a, b in (through_road_pairs or [])
        ]
        ids = set(through_road_arm_ids or ())
        for a, b in self.through_road_pairs:
            ids.add(a)
            ids.add(b)
        self.through_road_arm_ids = ids
        self.driving_side = normalize_driving_side(driving_side)
        for arm in self.arms:
            arm.driving_side = self.driving_side

    def arm_allows_lr(self, arm_id):
        return int(arm_id) not in self.through_road_arm_ids

def load_junction_from_dict(data, driving_side=None):
    if not isinstance(data, dict):
        raise fail('lane', 'top_level_object', 'Top-level JSON must be an object.')
    arms_data = data.get('arms')
    if not isinstance(arms_data, list) or not arms_data:
        raise fail('lane', 'arms_required', "'arms' must be a non-empty list of lane-level arm records.")
    side = normalize_driving_side(driving_side if driving_side is not None else data.get('driving_side'))
    through_ids = through_road_arm_ids(data)
    through_pairs = through_road_pairs(data)
    arms = []
    for i, arm_data in enumerate(arms_data):
        if not isinstance(arm_data, dict):
            raise fail('lane', 'arm_object', f'arms[{i}] must be an object.', details={'index': i})
        arm_id = arm_data.get('id', i + 1)
        heading = arm_data.get('heading_deg', arm_data.get('angle', 0.0))
        allow_lr = int(arm_id) not in through_ids
        if arm_data.get('lanes') is not None:
            if not isinstance(arm_data['lanes'], list):
                raise fail('lane', 'lanes_list', f'arms[{i}].lanes must be a list.', location={'arm': arm_id}, details={'index': i}, arm_id_list=[int(arm_id)])
            arms.append(ArmSpec(arm_id=arm_id, heading_deg=heading, lanes=arm_data['lanes'], allow_lr=allow_lr, driving_side=side))
            continue
        incoming_lanes = arm_data.get('incoming_lanes', [])
        outgoing_lanes = arm_data.get('outgoing_lanes', 0)
        if not isinstance(incoming_lanes, list):
            raise fail('lane', 'incoming_lanes_list', f'arms[{i}].incoming_lanes must be a list.', location={'arm': arm_id}, details={'index': i}, arm_id_list=[int(arm_id)])
        arms.append(ArmSpec(arm_id=arm_id, heading_deg=heading, incoming_lanes=incoming_lanes, outgoing_lanes=outgoing_lanes, allow_lr=allow_lr, driving_side=side))
    return JunctionSpec(
        junction_type=data.get('type', 'unknown'),
        arms=arms,
        through_road_arm_ids=through_ids,
        through_road_pairs=through_pairs,
        driving_side=side,
    )

def load_raw_and_spec(json_path):
    path = Path(json_path)
    if not path.is_file():
        raise fail('schema', 'input_file', f'JSON file not found: {path}', location={'field': 'input'})
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise fail('schema', 'invalid_json', f"Invalid JSON in '{path}': {exc}", location={'field': 'input'})
    return (data, load_junction_from_dict(data))

def classify_relative_direction(from_angle, to_angle):
    diff = angle_diff_ccw(normalize_angle(from_angle + 180.0), to_angle)
    if diff <= STRAIGHT_THROUGH_TOLERANCE or diff >= 360.0 - STRAIGHT_THROUGH_TOLERANCE:
        return 'straight'
    if abs(diff - 180.0) <= STRAIGHT_THROUGH_TOLERANCE:
        return 'uturn'
    return 'left' if diff < 180.0 else 'right'


def straight_deflection_deg(from_angle, to_angle):
    """Smallest absolute deflection from a through/straight path (0° = perfect straight)."""
    diff = angle_diff_ccw(normalize_angle(from_angle + 180.0), to_angle)
    return min(diff, 360.0 - diff)


def find_target_arm(spec, from_arm_id, movement):
    from_arm = spec.arm_map[from_arm_id]
    direction = MOVEMENT_MAP[movement]
    candidates = [
        arm for arm in spec.arms
        if arm.id != from_arm_id and classify_relative_direction(from_arm.heading_deg, arm.heading_deg) == direction
    ]
    # S is only a geometric through, or the declared through-road peer when the
    # pair sits just outside the straight tolerance. Never invent S toward a
    # T-stem or any other non-through arm.
    if not candidates and direction == 'straight':
        peer_id = through_peer_id(spec, from_arm_id)
        peer = spec.arm_map.get(peer_id) if peer_id is not None else None
        if (
            peer is not None
            and classify_relative_direction(from_arm.heading_deg, peer.heading_deg) != 'uturn'
        ):
            candidates = [peer]
    if len(candidates) <= 1:
        return candidates[0].id if candidates else None
    incoming = normalize_angle(from_arm.heading_deg + 180.0)
    target_diff = {'left': 90.0, 'right': 270.0}.get(direction)

    def score(arm):
        diff = angle_diff_ccw(incoming, arm.heading_deg)
        return abs(diff - target_diff) if target_diff is not None else min(diff, 360.0 - diff)
    return min(candidates, key=score).id

def movement_has_outflow(spec, arm_id, movement, *, outgoing_override=None):
    target_id = find_target_arm(spec, arm_id, movement)
    if target_id is None:
        return False
    if outgoing_override is not None:
        return outgoing_override.get(target_id, 0) > 0
    return spec.arm_map[target_id].outgoing_lanes > 0


def legal_outflow_letters(spec, arm_id, *, outgoing_override=None):
    """L/S/R letters that currently have a receiving outgoing arm."""
    return [
        move for move in ("L", "S", "R")
        if movement_has_outflow(spec, arm_id, move, outgoing_override=outgoing_override)
    ]


def clip_movement_code_to_outflow(spec, arm_id, code, *, outgoing_override=None, allow_lr=None):
    """Drop geometrically illegal letters from a movement code.

    Returns the clipped canonical code, or None if nothing legal remains.
    """
    if allow_lr is None:
        allow_lr = spec.arm_allows_lr(arm_id)
    kept = [
        move for move in entry_letters(code)
        if movement_has_outflow(spec, arm_id, move, outgoing_override=outgoing_override)
    ]
    if not kept or not is_valid_movement_code(set(kept), allow_lr=allow_lr):
        return None
    return "".join(sorted(set(kept), key=lambda move: ORDER[move]))


def max_outgoing_for_arm(arm, driving_side=None):
    side = normalize_driving_side(driving_side if driving_side is not None else getattr(arm, 'driving_side', None))
    cuts = valid_travel_cuts(arm.lane_readings, driving_side=side)
    if not cuts:
        return 0
    n = len(arm.lane_readings)
    if side == 'left':
        return max(cuts)
    return max(n - k for k in cuts)

def collect_channelized_features(data):
    """Collect channelized slips; resolve unordered arm_list via headings when needed.

    ``source_type`` comes only from arm-level ``channelized_features`` (shared/dedicated).
    """
    side = normalize_driving_side(data.get('driving_side'))
    enriched = enrich_channelized_features(data.get('channelized_features') or [], data.get('arms') or [], side)
    features = []
    for item in enriched:
        if item.get('from_arm') is None or item.get('to_arm') is None:
            continue
        from_arm, to_arm = int(item['from_arm']), int(item['to_arm'])
        features.append({
            'arm_list': [from_arm, to_arm],
            'from_arm': from_arm,
            'to_arm': to_arm,
            'split_length': DEFAULT_CHANNELIZED_LENGTH,
            'merge_length': DEFAULT_CHANNELIZED_LENGTH,
            'source_type': normalize_channelized_source_type(item.get('source_type')),
        })
    return features

def resolve_channelized_links(spec, features):
    """Turn each channelized feature into a curb-side slip onto an existing outgoing.

    left (RHT): origin rightmost incoming = R onto destination leftmost outgoing.
    right (LHT): origin leftmost incoming = L onto destination rightmost outgoing.
    Destination outgoing count is unchanged; the slip merges into that curb-edge lane.
    """
    side = normalize_driving_side(getattr(spec, 'driving_side', None))
    resolved = {}
    for feature in features:
        from_arm, to_arm = feature['from_arm'], feature['to_arm']
        from_spec = spec.arm_map.get(from_arm)
        to_spec = spec.arm_map.get(to_arm)
        if from_spec is None or to_spec is None:
            continue
        from_lane = slip_from_lane_id(int(from_spec.incoming_lane_count), side)
        to_lane = slip_to_lane_id(int(to_spec.outgoing_lanes), side)
        if from_lane is None or to_lane is None:
            continue
        resolved[from_arm, to_arm] = {
            'from_lane_id': from_lane,
            'to_lane_id': to_lane,
            'split_length': feature['split_length'],
            'merge_length': feature['merge_length'],
            'slip_turn': slip_turn_letter(side),
            'source_type': normalize_channelized_source_type(feature.get('source_type')),
        }
    return resolved


def channelized_origin_arms(channelized):
    return {from_arm for from_arm, _to_arm in (channelized or {})}

def channelized_target_arms(channelized):
    return {to_arm for _from_arm, to_arm in (channelized or {})}


def channelized_from_lane_ids(channelized):
    """Map origin arm_id -> set of reserved slip from_lane_id values."""
    out = {}
    for (from_arm, _to_arm), info in (channelized or {}).items():
        out.setdefault(int(from_arm), set()).add(int(info['from_lane_id']))
    return out


def explicit_marking(spec):
    marking = {}
    for arm in spec.arms:
        letters = []
        for lane in arm.incoming_lanes:
            if is_unknown(lane) or (isinstance(lane, list) and len(lane) != 1):
                letters.append(None)
            elif isinstance(lane, list):
                letters.append(entry_letters(lane[0]))
            else:
                letters.append(entry_letters(lane))
        marking[arm.id] = letters
    return marking

def has_unknown_lanes(spec):
    """True when travel or movements still need search."""
    return any(arm.needs_search() for arm in spec.arms)


# --------------------------------------------------------------------------- #
# connections.json generation
# --------------------------------------------------------------------------- #

def map_movement_lanes(movement, in_lane_ids, outgoing_count, driving_side=None, *, geometric_direction=None):
    """Map one source arm's movement onto distinct outgoing lanes on the target.

    to_lane_id 1 is innermost (OpenDRIVE |id|=1); N is outermost. Center-side
    turn (L for left/RHT, R for right/LHT) packs low ids; curb-side turn
    (R for left, L for right) packs high ids. Two incoming lanes of the same
    arm must not share a destination lane. ``build_junction_connections`` then
    spreads links from different arms onto unused outgoing lanes before
    allowing a cross-arm merge.

    ``geometric_direction`` is the actual from→to heading class. When a reported
    ``S`` was snapped onto a skewed left/right destination (no arm inside the
    straight tolerance), pack like that turn so it does not collide with true
    through/center packing on the same destination arm.
    """
    n = len(in_lane_ids)
    if n > outgoing_count:
        raise ValueError(
            f'same-arm merge forbidden: {n} incoming lanes into {outgoing_count} outgoing'
        )
    effective = movement
    if movement == 'S' and geometric_direction in ('left', 'right'):
        effective = 'L' if geometric_direction == 'left' else 'R'
    curb_turn = slip_turn_letter(driving_side)
    if effective == curb_turn:
        out = list(range(outgoing_count - n + 1, outgoing_count + 1))
    else:
        out = list(range(1, n + 1))
    return list(zip(in_lane_ids, out)), (effective == curb_turn), effective


def _outgoing_preference(n, *, curb):
    """1 = innermost outgoing. Outer/curb-side turns prefer high ids."""
    ids = list(range(1, int(n) + 1))
    if not ids:
        return ids
    if curb:
        return list(reversed(ids))
    return ids


def _pack_rank(item, driving_side=None):
    """Assign inner turns first, then through, then outer/curb-side turns."""
    if item.get('channelized_link') or item.get('curb'):
        return 2
    if item.get('effective') == inner_turn_letter(driving_side):
        return 0
    return 1


def _spread_links_across_arms(links, spec, driving_side=None):
    """Prefer unused outgoing lanes; only then merge across different from_arms.

    Inner (center-side) turns claim low ids first so a left turn under left-side
    traffic lands on the innermost outgoing rather than yielding it to through.
    """
    side = normalize_driving_side(driving_side)
    by_dest = defaultdict(list)
    for (from_arm, to_arm), items in links.items():
        for item in items:
            by_dest[int(to_arm)].append((int(from_arm), item))
    for to_arm, claims in by_dest.items():
        n = int(spec.arm_map[to_arm].outgoing_lanes)
        occupied = set()
        used_by_arm = defaultdict(set)
        claims = sorted(
            claims,
            key=lambda row: (
                _pack_rank(row[1], side),
                row[0],
                int(row[1]['from_lane_id']),
            ),
        )
        for from_arm, item in claims:
            curb = bool(item.get('channelized_link') or item.get('curb'))
            order = _outgoing_preference(n, curb=curb)
            chosen = next((t for t in order if t not in occupied), None)
            if chosen is None:
                chosen = next((t for t in order if t not in used_by_arm[from_arm]), None)
            if chosen is None:
                raise ValueError(
                    f'same-arm merge forbidden: arm {from_arm} has no free outgoing '
                    f'on arm {to_arm}'
                )
            item['to_lane_id'] = chosen
            occupied.add(chosen)
            used_by_arm[from_arm].add(chosen)


def build_junction_connections(spec, marking, channelized=None):
    channelized = channelized or {}
    side = normalize_driving_side(getattr(spec, 'driving_side', None))
    reserved_from = channelized_from_lane_ids(channelized)
    curb_turn = slip_turn_letter(side)
    links = defaultdict(list)
    for arm in spec.arms:
        lanes_by_move = defaultdict(list)
        for lane_idx, letters in enumerate(marking[arm.id], start=1):
            moves = set(letters or ())
            if lane_idx in reserved_from.get(arm.id, ()):
                # Same slip lane: dedicated packs nothing extra (R/L is the island link);
                # shared SR/LS still packs S on this lane.
                moves.discard(curb_turn)
            for move in sorted(moves, key=lambda m: ORDER[m]):
                lanes_by_move[move].append(lane_idx)
        for move, in_lane_ids in lanes_by_move.items():
            target_id = find_target_arm(spec, arm.id, move)
            if target_id is None or spec.arm_map[target_id].outgoing_lanes <= 0:
                continue
            to_arm = spec.arm_map[target_id]
            geom = classify_relative_direction(arm.heading_deg, to_arm.heading_deg)
            pairs, curb, effective = map_movement_lanes(
                move, in_lane_ids, to_arm.outgoing_lanes, driving_side=side, geometric_direction=geom,
            )
            for from_id, to_id in pairs:
                links[arm.id, target_id].append({
                    'from_lane_id': from_id,
                    'to_lane_id': to_id,
                    'channelized_link': False,
                    'curb': curb,
                    'effective': effective,
                })
    for (from_arm_id, to_arm_id), info in channelized.items():
        links[from_arm_id, to_arm_id].append({
            'from_lane_id': info['from_lane_id'],
            'to_lane_id': info['to_lane_id'],
            'channelized_link': True,
            'curb': True,
            'effective': curb_turn,
        })
    _spread_links_across_arms(links, spec, side)
    connections = []
    for (from_arm_id, to_arm_id), lane_links in links.items():
        lane_links = sorted(lane_links, key=lambda x: x['from_lane_id'])
        connection = {
            'from_arm': from_arm_id,
            'to_arm': to_arm_id,
            'lane_links': [
                {'from_lane_id': link['from_lane_id'], 'to_lane_id': link['to_lane_id']}
                for link in lane_links
            ],
        }
        info = channelized.get((from_arm_id, to_arm_id))
        if info is not None:
            connection.update({
                'channelized': True,
                'split_length': info['split_length'],
                'merge_length': info['merge_length'],
            })
        connections.append(connection)
    return sorted(connections, key=lambda c: (c['from_arm'], c['to_arm']))

def road_dict_for_output(arm, channelized=None):
    # Slip uses an existing curb-edge incoming on from_arm and an existing
    # curb-edge outgoing on to_arm; do not inflate either count.
    return arm.to_road_dict()

def build_connections_output(spec, marking, channelized=None):
    side = normalize_driving_side(getattr(spec, 'driving_side', None))
    return {
        'driving_side': side,
        'roads': [road_dict_for_output(arm, channelized) for arm in spec.arms],
        'connectors': [],
        'junctions': [{'id': 0, 'radius': spec.junction_radius, 'connections': build_junction_connections(spec, marking, channelized)}],
    }


def build_connections_from_config(source, driving_side=None):
    """Build connections.json from a fully resolved config (no N / no fill search).

    Unknown travel or movement candidates are rejected; run refinement fill first.
    """
    if isinstance(source, dict):
        raw = source
    else:
        raw, _ = load_raw_and_spec(str(source))
    if driving_side is not None:
        raw = dict(raw)
        raw['driving_side'] = normalize_driving_side(driving_side)
    spec = load_junction_from_dict(raw)
    if has_unknown_lanes(spec):
        raise fail(
            'connections',
            'unresolved_lanes',
            'Config still has unknown travel/movements; run refinement fill to produce xodr_config_fill.json first.',
        )
    features = collect_channelized_features(raw)
    channelized = resolve_channelized_links(spec, features)
    marking = explicit_marking(spec)
    return build_connections_output(spec, marking, channelized)

def build_argparser():
    parser = argparse.ArgumentParser(description='Build connections.json from a resolved xodr_config / xodr_config_fill.json (no fill search).')
    parser.add_argument('-i', '--input', default='../samples/2022/25729/xodr_config_fill.json')
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('--driving-side', default=None, choices=['left', 'right'], help="Override driving_side (default: value in JSON or 'left').")
    return parser

def main():
    args = build_argparser().parse_args()
    input_path = Path(args.input)
    connections_path = Path(args.output) if args.output else input_path.with_name('connections.json')
    try:
        connections_output = build_connections_from_config(str(input_path), driving_side=args.driving_side)
    except Exception as exc:
        # Prefer structured ConnectionParseError JSON when available.
        to_dict = getattr(exc, 'to_dict', None)
        if callable(to_dict):
            print(json.dumps(to_dict(), ensure_ascii=False, indent=2))
        else:
            print(f'[ERROR] {exc}')
        raise SystemExit(2) from exc
    dump_json(connections_path, connections_output)
    print(json.dumps(connections_output, indent=2, ensure_ascii=False))
    print(f'\n[INFO] Connections -> {connections_path}')

if __name__ == '__main__':
    main()
