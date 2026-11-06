"""Minimum-edit structural repair for lane-level junction results.

Allowed operators:
  1. add/drop an incoming or outgoing motor lane at a travel-block edge;
     a one-way arm (no cut) may grow a reverse-travel lane, which also
     creates the opposing_separator;
  2. rewrite an incoming-lane arrow: only clear a valued code to N;
  3. move the opposing-traffic separator, which re-derives lane travel directions;
  4. delete the opposing-traffic separator and unify lane travel direction.

Search objective (Scheme B):
    Minimize atomic structural changes, plus a heavier charge for rewriting an
    observed incoming arrow. Each inserted/deleted cross-section element, changed
    lane travel, and changed opposing_separator counts as 1. An incoming-arrow
    rewrite costs ``ARROW_REWRITE_COST`` (4): more than a single lane add/drop (2),
    equal to replacing that lane by add-then-drop, so dummy-lane workarounds do
    not win on score (fewer operators still prefer rewrite when scores tie).
    Cue orientation is not edited.

    Search expands one-way reverse-creating operators first (move the cut onto
    an existing separator, or add a reverse-travel lane). Cheap add/drop on
    already two-way arms is deferred until every arm has both travels; otherwise
    the budget is spent before those combinations are reached. If that pass
    finds fewer than ``top_k`` repairs, a second unrestricted pass fills any
    remaining slots at the same minimum score.

    ``solve_junction`` returns up to ``DEFAULT_TOP_K`` (3) cheapest distinct
    feasible repairs (same travel/arrow layout is one candidate, even if the
    dropped lane was the outer vs cut-side edge). An LLM then ranks those
    candidates from the junction photograph, the original lanes.json, and each
    candidate's edit list.

    ``solve_scene`` is the multi-junction counterpart: the same operators and
    score, keyed by ``(junction_id, arm_id)``, with an oracle of per-junction
    ``try_solve`` plus Layer-3 connector-end equalities. Enumerated search
    skips arrow rewrites (they cannot fix mouth counts). When a structural
    mouth edit already satisfies the pair oracle but ``try_solve`` fails
    only because a locked incoming arrow starves a new stream, the probe
    clears the fewest such arrows to N and charges ``ARROW_REWRITE_COST``.
    A two-way ``connecting_road`` mouth is sticky: search will not drop its
    last incoming or outgoing lane (or delete the cut) to match a one-way
    far end; the one-way mouth must grow the missing reverse travel.

Feasibility objective:
  A candidate repair only needs to admit at least one feasible connectivity
  completion. Connectivity quality/preferences are optimized later by the normal
  connectivity-completion stage; they do not affect the minimum-edit objective.
"""

from __future__ import annotations

import copy
import heapq
import itertools
from dataclasses import dataclass, field, replace

from scripts.methods.connection_parser import (
    UNKNOWN,
    canonical_code,
    collect_channelized_features,
    entry_letters,
    is_unknown,
    load_junction_from_dict,
    normalize_driving_side,
    resolve_channelized_links,
    travel_block_order,
)
from scripts.methods.llm_kg.lane_level import llm_gen as gen
from scripts.methods.llm_kg.lane_level.schemas import (
    AREA_TYPES,
    LANE_TYPE,
    LINE_TYPES,
    SEPARATOR_TYPES,
    TRAVELS,
)
from scripts.methods.llm_kg.lane_level.structure import (
    _opposing_cut_index,
    validate_arm_result,
)
from scripts.methods.llm_kg.refinement.config_build import build_config
from scripts.methods.llm_kg.refinement.fill_solver import try_solve
from scripts.methods.llm_kg.refinement.layer2_con_fix import (
    hard_channelized_from_arm_issues,
    hard_lane_constraint_issues,
)


DEFAULT_MAX_TRIALS = 1000
SCENE_MAX_TRIALS = 1500
DEFAULT_MAX_SCORE = 60
DEFAULT_TOP_K = 3
_SOLVER_NOTE = "feasibility_solver"
_SYNTHETIC_EVIDENCE = "constraint_repair"
# Heavier than one lane add/drop (2); equal to add-then-drop replacing that lane (4).
ARROW_REWRITE_COST = 4.0


@dataclass
class RepairCandidate:
    candidate_id: int
    score: float
    edits: list
    lane_results: dict
    config: dict | None = None

    def summary_dict(self):
        return {
            "id": int(self.candidate_id),
            "score": self.score,
            "edits": list(self.edits),
        }


@dataclass
class RepairResult:
    ok: bool
    score: float = 0.0
    edits: list = field(default_factory=list)
    lane_results: dict | None = None
    config: dict | None = None
    trials: int = 0
    search_exhausted: bool = False
    frontier_min_score: float | None = None
    note: str | None = None
    candidates: list = field(default_factory=list)
    chosen_id: int | None = None
    rank_source: str | None = None
    rank_reasoning: str | None = None
    rank_order: list | None = None

    def to_dict(self):
        return {
            "ok": bool(self.ok),
            "score": self.score,
            "edits": list(self.edits),
            "trials": int(self.trials),
            "search_exhausted": bool(self.search_exhausted),
            "frontier_min_score": self.frontier_min_score,
            "note": self.note,
            "candidates": [
                cand.summary_dict() if isinstance(cand, RepairCandidate) else cand
                for cand in (self.candidates or [])
            ],
            "chosen_id": self.chosen_id,
            "rank_source": self.rank_source,
            "rank_reasoning": self.rank_reasoning,
            "rank_order": list(self.rank_order) if self.rank_order else None,
        }

    def apply_choice(self, chosen_id, *, source="score", reasoning=None, rank_order=None):
        if not self.candidates:
            return self
        by_id = {int(cand.candidate_id): cand for cand in self.candidates}
        cand = None
        try:
            if chosen_id is not None:
                cand = by_id.get(int(chosen_id))
        except (TypeError, ValueError):
            cand = None
        if cand is None:
            cand = self.candidates[0]
            chosen_id = cand.candidate_id
            source = "score"
            reasoning = (reasoning or "") + " [invalid chosen_id; fell back to cheapest]"
        self.ok = True
        self.score = cand.score
        self.edits = list(cand.edits)
        self.lane_results = cand.lane_results
        self.config = cand.config
        self.chosen_id = int(chosen_id)
        self.rank_source = source
        self.rank_reasoning = reasoning
        self.rank_order = list(rank_order) if rank_order else None
        return self


@dataclass
class Operator:
    op_id: str
    arm_id: int
    kind: str
    payload: dict
    # Filled after the operator is applied. Cost is not pre-assigned because one
    # operator may change multiple structural attributes.
    score: float = 0.0
    junction_id: int | None = None

    def to_dict(self):
        payload = {
            "arm_id": int(self.arm_id),
            "kind": self.kind,
            "score": self.score,
            "id": self.op_id,
            **self.payload,
        }
        if self.junction_id is not None:
            payload["junction_id"] = int(self.junction_id)
        return payload


# --------------------------------------------------------------------------- #
# Basic helpers
# --------------------------------------------------------------------------- #


def is_structurally_valid(arm_result, *, driving_side=None):
    return not validate_arm_result(arm_result, driving_side=driving_side)


def format_repair_edits(edits):
    lines = []
    for edit in edits or []:
        arm = edit.get("arm_id")
        kind = edit.get("kind")
        score = edit.get("score")
        loc = ""
        if edit.get("junction_id") is not None:
            loc = "j%s." % edit["junction_id"]
        if kind in ("drop_incoming_lane", "drop_outgoing_lane"):
            text = (
                f"{loc}arm {arm}: {kind} edge={edit.get('edge')} "
                f"cs={edit.get('cs_index')} +{score}"
            )
        elif kind in ("add_incoming_lane", "add_outgoing_lane"):
            text = f"{loc}arm {arm}: {kind} edge={edit.get('edge')} +{score}"
        elif kind == "move_cut":
            text = (
                f"{loc}arm {arm}: move_cut {edit.get('from_eid')!r} -> "
                f"{edit.get('to_eid')!r} +{score}"
            )
        elif kind == "delete_cut":
            text = f"{loc}arm {arm}: delete_cut unify={edit.get('unify_travel')!r} +{score}"
        elif kind == "rewrite_arrow":
            text = (
                f"{loc}arm {arm}: rewrite_arrow lane_cs={edit.get('cs_index')} "
                f"{edit.get('from_code')!r} -> {edit.get('to_code')!r} +{score}"
            )
        else:
            text = f"{loc}arm {arm}: {kind} +{score}"

        atomic = edit.get("atomic_changes") or []
        if atomic:
            text += " [" + "; ".join(str(x) for x in atomic) + "]"
        lines.append(text)
    return lines


def _copy_arm(arm):
    return copy.deepcopy(arm)


def _cs(arm):
    return list((arm or {}).get("cross_section") or [])


def _opp(arm):
    opp = (arm or {}).get("opposing_separator")
    return dict(opp) if isinstance(opp, dict) else {"element_id": None, "reasoning": ""}


def _write_opposing(arm, *, element_id, one_way_travel, note):
    """Keep element_id and one_way_travel mutually exclusive."""
    opp = _opp(arm)
    arm["opposing_separator"] = {
        **opp,
        "element_id": element_id,
        "one_way_travel": one_way_travel,
        "reasoning": ((opp.get("reasoning") or "") + f" [{note}]").strip(),
    }
    return arm


def _inferred_one_way_travel(cs):
    travels = {item.get("travel") for item in cs or [] if _is_lane(item)}
    travels.discard(None)
    if len(travels) == 1:
        travel = next(iter(travels))
        if travel in TRAVELS:
            return travel
    return None


def _cut_idx(arm):
    idx, eid = _opposing_cut_index(_cs(arm), _opp(arm))
    return idx, eid


def _is_separator(item):
    return (item or {}).get("type") in SEPARATOR_TYPES


def _is_lane(item):
    return (item or {}).get("type") == LANE_TYPE


def _is_eligible_cut(item):
    item_type = (item or {}).get("type")
    if not (item or {}).get("element_id"):
        return False
    return item_type in LINE_TYPES or item_type == "median"


def _has_arrow_cue(item):
    return any(cue.get("kind") == "arrow" for cue in (item or {}).get("cues") or [])


def _incoming_arrow_code(item):
    return canonical_code(gen._lane_movements(item))


def _set_arrow_code(item, code):
    """Rewrite arrow movements only; keep orientation."""
    if not _has_arrow_cue(item):
        return None
    code = canonical_code(code)
    letters = [] if is_unknown(code) else list(code)
    out = copy.deepcopy(item)
    cues = []
    for cue in out.get("cues") or []:
        cue_copy = copy.deepcopy(cue)
        if cue_copy.get("kind") == "arrow":
            cue_copy["movements"] = letters
            note = f"[{_SOLVER_NOTE} rewrite_arrow {code}]"
            prev = (cue_copy.get("reasoning") or "").strip()
            cue_copy["reasoning"] = f"{prev} {note}".strip() if prev else note
        cues.append(cue_copy)
    out["cues"] = cues
    if "movements" in out:
        out["movements"] = letters
    return out


def _through_road_arm_ids(through_roads):
    ids = set()
    for road in through_roads or []:
        for value in road.get("arm_ids") or []:
            if value is not None:
                ids.add(int(value))
    return ids


def _arm_allows_lr(arm_id, through_roads):
    return int(arm_id) not in _through_road_arm_ids(through_roads)


def _eligible_cuts(cs):
    return [
        (idx, str(item.get("element_id")))
        for idx, item in enumerate(cs or [])
        if _is_eligible_cut(item)
    ]


def _lane_cs_indices(cs):
    return [idx for idx, item in enumerate(cs or []) if _is_lane(item)]


def _lanes_of_travel(cs, travel):
    return [
        idx
        for idx, item in enumerate(cs or [])
        if _is_lane(item) and item.get("travel") == travel
    ]


def _next_id(cs, *, field="element_id"):
    generation = 0
    order = 0
    for item in cs or []:
        raw = item.get(field)
        if not raw:
            continue
        parts = str(raw).split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            generation = max(generation, int(parts[0]))
            order = max(order, int(parts[1]))
    if order == 0 and generation == 0:
        return f"{generation + 1}-1"
    return f"{max(generation, 1)}-{order + 1}"


def _new_element_id(cs):
    return _next_id(cs, field="element_id")


# --------------------------------------------------------------------------- #
# Structural transformations
# --------------------------------------------------------------------------- #


def _derive_travels(cs, cut_idx, driving_side):
    """Re-derive travel from a cut without modifying any visual cue."""
    left_travel, right_travel = travel_block_order(driving_side)
    out = []
    for idx, item in enumerate(cs or []):
        if not _is_lane(item):
            out.append(copy.deepcopy(item))
            continue
        entry = copy.deepcopy(item)
        entry["travel"] = left_travel if idx < cut_idx else right_travel
        out.append(entry)
    return out


def _unify_travels(cs, travel):
    """Set all lane travels to one direction without modifying visual cues."""
    out = []
    for item in cs or []:
        if not _is_lane(item):
            out.append(copy.deepcopy(item))
            continue
        entry = copy.deepcopy(item)
        entry["travel"] = travel
        out.append(entry)
    return out


def _sync_opposing(arm):
    """Drop a dangling opposing_separator if its referenced element was removed."""
    cs = _cs(arm)
    opp = _opp(arm)
    eid = opp.get("element_id")
    if not eid:
        return arm
    if any(str(item.get("element_id") or "") == str(eid) for item in cs):
        return arm

    arm = dict(arm)
    arm["cross_section"] = cs
    return _write_opposing(
        arm,
        element_id=None,
        one_way_travel=_inferred_one_way_travel(cs),
        note=f"{_SOLVER_NOTE} cut removed",
    )


def _drop_lane_at(cs, lane_idx, block):
    """Remove a block-edge lane and the separator toward the rest of the block.

    A one-lane travel block has no same-direction divider. The adjacent
    separator is the opposing cut (left of a right-of-cut lane, right of a
    left-of-cut lane). Look both sides so drop_incoming and drop_outgoing
    both work on a 1+1 two-way mouth.
    """
    if len(block) == 1:
        left, right = lane_idx - 1, lane_idx + 1
        if left >= 0 and _is_separator(cs[left]):
            return cs[:left] + cs[lane_idx + 1 :]
        if right < len(cs) and _is_separator(cs[right]):
            return cs[:lane_idx] + cs[right + 1 :]
        return None
    if lane_idx == block[0]:
        sep_idx = lane_idx + 1
        if sep_idx >= len(cs) or not _is_separator(cs[sep_idx]):
            return None
        return cs[:lane_idx] + cs[sep_idx + 1 :]

    sep_idx = lane_idx - 1
    if sep_idx < 0 or not _is_separator(cs[sep_idx]):
        return None
    return cs[:sep_idx] + cs[lane_idx + 1 :]


def _new_separator(cs, crop_id):
    # This element is structurally inferred by the repair solver, not from pixels.
    return {
        "type": "unmarked_line",
        "color": None,
        "element_id": _new_element_id(cs),
        "evidence": _SYNTHETIC_EVIDENCE,
        "crop_id": crop_id,
    }


def _new_lane(travel, crop_id):
    return {
        "type": LANE_TYPE,
        "travel": travel,
        "evidence": _SYNTHETIC_EVIDENCE,
        "cues": [],
        "crop_id": crop_id,
    }


def _insert_lane_before(cs, idx, travel, crop_id):
    """Insert a lane before idx, adding a separator next to another area."""
    lane = _new_lane(travel, crop_id)
    prev = cs[idx - 1] if idx > 0 else None
    nxt = cs[idx] if idx < len(cs) else None

    if prev and prev.get("type") in AREA_TYPES:
        sep = _new_separator(cs, crop_id)
        return cs[:idx] + [sep, lane] + cs[idx:]
    if nxt and nxt.get("type") in AREA_TYPES:
        sep = _new_separator(cs, crop_id)
        return cs[:idx] + [lane, sep] + cs[idx:]
    return cs[:idx] + [lane] + cs[idx:]


def _insert_lane_after(cs, idx, travel, crop_id):
    return _insert_lane_before(cs, idx + 1, travel, crop_id)


def _block_edges(indices, *, cut_idx, travel, driving_side):
    """Map editable edge name to lane cross-section index for a travel block."""
    if not indices:
        return {}

    left, right = indices[0], indices[-1]
    if cut_idx is None:
        return {"left": left, "right": right}

    first, _second = travel_block_order(driving_side)
    if travel == first:
        return {"outer": left, "cut": right}
    return {"cut": left, "outer": right}


def _one_way_reverse_insert(cs, *, travel, driving_side, crop_id):
    """Grow a reverse-travel lane on a one-way arm; return (cs, new cut eid).

    Always inserts a new separator between the existing travel block and the
    reverse lane; that separator becomes ``opposing_separator``.
    """
    other = "outgoing" if travel == "incoming" else "incoming"
    existing = _lanes_of_travel(cs, other)
    if not existing or _lanes_of_travel(cs, travel):
        return None, None
    first, second = travel_block_order(driving_side)
    lane = _new_lane(travel, crop_id)
    sep = _new_separator(cs, crop_id)
    cut_eid = str(sep["element_id"])
    if travel == second:
        idx = existing[-1] + 1
        new_cs = list(cs[:idx]) + [sep, lane] + list(cs[idx:])
    elif travel == first:
        idx = existing[0]
        new_cs = list(cs[:idx]) + [lane, sep] + list(cs[idx:])
    else:
        return None, None
    cut_idx = next(
        i
        for i, item in enumerate(new_cs)
        if str(item.get("element_id") or "") == cut_eid
    )
    return _derive_travels(new_cs, cut_idx, driving_side), cut_eid


def _one_way_arm_ids(lane_results, editable=None):
    """Arms that still have only one travel direction (no opposing cut)."""
    ids = []
    for arm_id, arm in (lane_results or {}).items():
        if editable is not None and int(arm_id) not in editable:
            continue
        cut_idx, _ = _cut_idx(arm)
        if cut_idx is not None:
            continue
        cs = _cs(arm)
        has_in = bool(_lanes_of_travel(cs, "incoming"))
        has_out = bool(_lanes_of_travel(cs, "outgoing"))
        if has_in ^ has_out:
            ids.append(int(arm_id))
    return ids


def _creates_reverse_travel(op, arm_id, one_way_ids):
    """Whether this operator can give a one-way arm an opposing stream."""
    if int(arm_id) not in one_way_ids:
        return False
    if op.kind == "move_cut":
        return True
    if op.kind in ("add_incoming_lane", "add_outgoing_lane"):
        return op.payload.get("edge") == "reverse"
    return False


def _add_insert(cs, *, travel, edge, cut_idx, block, driving_side, crop_id):
    """Insert a lane at a travel-block edge; empty blocks insert at the cut."""
    if not block:
        if cut_idx is None:
            return None
        first, _ = travel_block_order(driving_side)
        if travel == first:
            return _insert_lane_before(cs, cut_idx, travel, crop_id)
        insert_at = cut_idx + 1
        if insert_at >= len(cs):
            return None
        return _insert_lane_before(cs, insert_at, travel, crop_id)

    left, right = block[0], block[-1]
    if cut_idx is None:
        if edge == "left":
            return _insert_lane_before(cs, left, travel, crop_id)
        if edge == "right":
            return _insert_lane_after(cs, right, travel, crop_id)
        return None

    first, _ = travel_block_order(driving_side)
    on_left = travel == first
    if edge == "outer":
        if on_left:
            return _insert_lane_before(cs, left, travel, crop_id)
        return _insert_lane_after(cs, right, travel, crop_id)
    if edge == "cut":
        if on_left:
            return _insert_lane_after(cs, right, travel, crop_id)
        return _insert_lane_before(cs, left, travel, crop_id)
    return None


# --------------------------------------------------------------------------- #
# Atomic structural diff (Scheme B)
# --------------------------------------------------------------------------- #


def _item_label(item):
    item = item or {}
    item_type = item.get("type")
    if _is_lane(item):
        return f"lane({item.get('travel')})"
    eid = item.get("element_id")
    return f"{item_type}({eid})" if eid else str(item_type)


def _same_nonlane_item(a, b):
    """Whether two non-lane items represent the same structural element."""
    if (a or {}).get("type") != (b or {}).get("type"):
        return False
    a_eid = (a or {}).get("element_id")
    b_eid = (b or {}).get("element_id")
    if a_eid is not None or b_eid is not None:
        return str(a_eid or "") == str(b_eid or "")
    return True


def _substitution_cost(a, b):
    """Atomic substitution cost used by cross-section edit distance."""
    if _is_lane(a) and _is_lane(b):
        return 0 if a.get("travel") == b.get("travel") else 1
    if not _is_lane(a) and not _is_lane(b) and _same_nonlane_item(a, b):
        return 0
    # A structural replacement equals one deletion plus one insertion.
    return 2


def _cross_section_diff(old_cs, new_cs):
    """Return minimum atomic edit count and readable changes between two CS lists.

    Atomic edits:
      * insert one cross-section element: 1
      * delete one cross-section element: 1
      * change one matched lane's travel: 1

    Visual cue/reasoning differences are deliberately ignored.
    """
    old_cs = list(old_cs or [])
    new_cs = list(new_cs or [])
    n, m = len(old_cs), len(new_cs)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = (i - 1, 0, "delete")
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = (0, j - 1, "insert")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = _substitution_cost(old_cs[i - 1], new_cs[j - 1])
            candidates = [
                (dp[i - 1][j - 1] + sub_cost, 0, (i - 1, j - 1, "match_or_sub")),
                (dp[i - 1][j] + 1, 1, (i - 1, j, "delete")),
                (dp[i][j - 1] + 1, 2, (i, j - 1, "insert")),
            ]
            cost, _tie, prev = min(candidates, key=lambda x: (x[0], x[1]))
            dp[i][j] = cost
            back[i][j] = prev

    changes = []
    i, j = n, m
    while i > 0 or j > 0:
        prev_i, prev_j, action = back[i][j]
        if action == "delete":
            old_item = old_cs[i - 1]
            changes.append(f"delete cs[{i - 1}] {_item_label(old_item)}")
        elif action == "insert":
            new_item = new_cs[j - 1]
            changes.append(f"insert cs[{j - 1}] {_item_label(new_item)}")
        else:
            old_item = old_cs[i - 1]
            new_item = new_cs[j - 1]
            sub_cost = _substitution_cost(old_item, new_item)
            if sub_cost == 1 and _is_lane(old_item) and _is_lane(new_item):
                changes.append(
                    f"travel cs[{i - 1}] {old_item.get('travel')} -> {new_item.get('travel')}"
                )
            elif sub_cost == 2:
                # Express replacement as two atomic changes to match its cost.
                changes.append(f"delete cs[{i - 1}] {_item_label(old_item)}")
                changes.append(f"insert cs[{j - 1}] {_item_label(new_item)}")
        i, j = prev_i, prev_j

    changes.reverse()
    return int(dp[n][m]), changes


def _arm_structural_diff(before, after):
    """Count actual structural changes caused by one operator."""
    cs_cost, changes = _cross_section_diff(_cs(before), _cs(after))

    before_cut = _opp(before).get("element_id")
    after_cut = _opp(after).get("element_id")
    before_cut = None if before_cut in (None, "") else str(before_cut)
    after_cut = None if after_cut in (None, "") else str(after_cut)

    cut_cost = 0
    if before_cut != after_cut:
        cut_cost = 1
        changes.append(f"opposing_separator {before_cut!r} -> {after_cut!r}")

    return float(cs_cost + cut_cost), changes


# --------------------------------------------------------------------------- #
# Solver-only structural view
# --------------------------------------------------------------------------- #


def _solver_view_arm(arm):
    """Create a downstream-compatible view without altering stored observations.

    Arrow observations on lanes interpreted as outgoing are ignored only in
    this temporary view because movement arrows constrain incoming-lane movement,
    not the existence of the observed pixels themselves.
    """
    out = _copy_arm(arm)
    cs = []

    for item in _cs(out):
        entry = copy.deepcopy(item)

        if _is_lane(entry) and entry.get("travel") == "outgoing":
            cues = []
            for cue in entry.get("cues") or []:
                cue_copy = copy.deepcopy(cue)
                if cue_copy.get("kind") == "arrow":
                    cue_copy["movements"] = []
                cues.append(cue_copy)
            entry["cues"] = cues

        cs.append(entry)

    for item in cs:
        if not _is_lane(item):
            continue
        cues = list(item.get("cues") or [])
        item["cues"] = cues

    out["cross_section"] = cs
    return out


def _solver_view_results(lane_results):
    return {int(arm_id): _solver_view_arm(arm) for arm_id, arm in lane_results.items()}


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


def apply_operator(arm, op, *, driving_side):
    """Apply one operator. Arrow rewrite changes movements only, not orientation."""
    arm = _copy_arm(arm)
    cs = _cs(arm)
    crop_id = arm.get("crop_id") or ""
    kind = op.kind
    payload = op.payload

    if kind == "rewrite_arrow":
        idx = int(payload["cs_index"])
        if idx >= len(cs):
            return None
        item = cs[idx]
        if not _is_lane(item) or item.get("travel") != "incoming":
            return None
        updated = _set_arrow_code(item, payload.get("to_code"))
        if updated is None:
            return None
        cs = list(cs)
        cs[idx] = updated
        arm["cross_section"] = cs
        return arm

    if kind in ("drop_incoming_lane", "drop_outgoing_lane"):
        travel = "incoming" if kind == "drop_incoming_lane" else "outgoing"
        lane_idx = int(payload["cs_index"])
        block = _lanes_of_travel(cs, travel)
        if lane_idx not in block or len(_lane_cs_indices(cs)) <= 1:
            return None
        new_cs = _drop_lane_at(cs, lane_idx, block)
        if new_cs is None:
            return None
        arm["cross_section"] = new_cs
        return _sync_opposing(arm)

    if kind in ("add_incoming_lane", "add_outgoing_lane"):
        travel = "incoming" if kind == "add_incoming_lane" else "outgoing"
        cut_idx, _ = _cut_idx(arm)
        block = _lanes_of_travel(cs, travel)
        if not block and cut_idx is None:
            new_cs, cut_eid = _one_way_reverse_insert(
                cs, travel=travel, driving_side=driving_side, crop_id=crop_id,
            )
            if new_cs is None or not cut_eid:
                return None
            arm["cross_section"] = new_cs
            return _write_opposing(
                arm,
                element_id=cut_eid,
                one_way_travel=None,
                note=f"{_SOLVER_NOTE} add reverse {travel}",
            )
        new_cs = _add_insert(
            cs,
            travel=travel,
            edge=payload["edge"],
            cut_idx=cut_idx,
            block=block,
            driving_side=driving_side,
            crop_id=crop_id,
        )
        if new_cs is None:
            return None
        arm["cross_section"] = new_cs
        return arm

    if kind == "move_cut":
        to_eid = str(payload["to_eid"])
        found = next(
            (
                idx
                for idx, item in enumerate(cs)
                if str(item.get("element_id") or "") == to_eid
            ),
            None,
        )
        if found is None:
            return None
        arm["cross_section"] = _derive_travels(cs, found, driving_side)
        return _write_opposing(
            arm,
            element_id=to_eid,
            one_way_travel=None,
            note=f"{_SOLVER_NOTE} move_cut",
        )

    if kind == "delete_cut":
        travel = payload["unify_travel"]
        arm["cross_section"] = _unify_travels(cs, travel)
        return _write_opposing(
            arm,
            element_id=None,
            one_way_travel=travel,
            note=f"{_SOLVER_NOTE} delete_cut",
        )

    return None


def enumerate_operators(arm, *, driving_side, through_roads=None):
    """Enumerate structural and incoming-arrow candidates from the current state.

    No path-history lock is used. Cycles and duplicate states are handled by the
    Dijkstra best-cost table, so operator availability depends only on the current
    structure.
    """
    arm_id = int(arm.get("arm_id"))
    cs = _cs(arm)
    cut_idx, cur_eid = _cut_idx(arm)
    ops = []

    # Rewrite a valued incoming arrow to N.
    for idx, item in enumerate(cs):
        if not _is_lane(item) or item.get("travel") != "incoming":
            continue
        if not _has_arrow_cue(item):
            continue
        from_code = _incoming_arrow_code(item)
        if is_unknown(from_code):
            continue
        ops.append(
            Operator(
                op_id=f"a{arm_id}:{idx}:{from_code}->{UNKNOWN}",
                arm_id=arm_id,
                kind="rewrite_arrow",
                payload={
                    "cs_index": idx,
                    "from_code": from_code,
                    "to_code": UNKNOWN,
                },
            )
        )

    # --- add / drop lanes ---
    for travel in TRAVELS:
        block = _lanes_of_travel(cs, travel)
        kind_drop = "drop_incoming_lane" if travel == "incoming" else "drop_outgoing_lane"
        kind_add = "add_incoming_lane" if travel == "incoming" else "add_outgoing_lane"
        edges = _block_edges(
            block,
            cut_idx=cut_idx,
            travel=travel,
            driving_side=driving_side,
        )

        seen_drop = set()
        for edge, lane_idx in edges.items():
            if lane_idx not in seen_drop and len(_lane_cs_indices(cs)) > 1:
                seen_drop.add(lane_idx)
                ops.append(
                    Operator(
                        op_id=f"d{arm_id}:{travel}:{edge}:{lane_idx}",
                        arm_id=arm_id,
                        kind=kind_drop,
                        payload={
                            "edge": edge,
                            "cs_index": lane_idx,
                            "travel": travel,
                        },
                    )
                )

            ops.append(
                Operator(
                    op_id=f"n{arm_id}:{travel}:{edge}",
                    arm_id=arm_id,
                    kind=kind_add,
                    payload={"edge": edge, "travel": travel},
                )
            )

        if not block and cut_idx is not None:
            ops.append(
                Operator(
                    op_id=f"n{arm_id}:{travel}:cut:empty",
                    arm_id=arm_id,
                    kind=kind_add,
                    payload={"edge": "cut", "travel": travel},
                )
            )
        elif not block and cut_idx is None:
            other = "outgoing" if travel == "incoming" else "incoming"
            if _lanes_of_travel(cs, other):
                ops.append(
                    Operator(
                        op_id=f"n{arm_id}:{travel}:reverse",
                        arm_id=arm_id,
                        kind=kind_add,
                        payload={"edge": "reverse", "travel": travel},
                    )
                )

    # --- move / delete opposing separator ---
    eligible = _eligible_cuts(cs)
    cur = str(cur_eid) if cur_eid else None

    for _idx, eid in eligible:
        if eid == cur:
            continue
        ops.append(
            Operator(
                op_id=f"m{arm_id}:{cur}->{eid}",
                arm_id=arm_id,
                kind="move_cut",
                payload={"from_eid": cur, "to_eid": eid},
            )
        )

    if cur is not None:
        for travel in TRAVELS:
            ops.append(
                Operator(
                    op_id=f"x{arm_id}:{travel}",
                    arm_id=arm_id,
                    kind="delete_cut",
                    payload={"unify_travel": travel, "from_eid": cur},
                )
            )

    return sorted(ops, key=lambda op: op.op_id)


# --------------------------------------------------------------------------- #
# Constraints + feasibility probe
# --------------------------------------------------------------------------- #


def _structure_ok(lane_results, driving_side):
    solver_results = _solver_view_results(lane_results)
    return all(
        is_structurally_valid(arm, driving_side=driving_side)
        for arm in solver_results.values()
    )


def _build_config(
    lane_results,
    *,
    arms,
    channelized_features,
    through_roads,
    name,
    driving_side,
):
    solver_results = _solver_view_results(lane_results)
    return build_config(
        arms,
        gen._lane_result_doc(solver_results),
        name,
        channelized_features=channelized_features,
        through_roads=through_roads,
        driving_side=driving_side,
    )


def _arrow_layout_ok(
    lane_results,
    *,
    arms,
    channelized_features,
    through_roads,
    name,
    driving_side,
):
    try:
        config = _build_config(
            lane_results,
            arms=arms,
            channelized_features=channelized_features,
            through_roads=through_roads,
            name=name,
            driving_side=driving_side,
        )
        spec = load_junction_from_dict(config)
    except Exception:
        return False

    features = collect_channelized_features(config)
    try:
        channelized = resolve_channelized_links(spec, features)
    except Exception:
        channelized = {}

    marking = {
        arm.id: [entry_letters(lane) for lane in arm.incoming_lanes]
        for arm in spec.arms
    }

    issues = hard_channelized_from_arm_issues(
        spec,
        marking,
        "feasibility",
        channelized=channelized,
        raw=config,
    )
    issues.extend(
        hard_lane_constraint_issues(
            spec,
            marking,
            "feasibility",
            channelized=channelized,
            raw=config,
        )
    )
    return not issues


# --------------------------------------------------------------------------- #
# Search state
# --------------------------------------------------------------------------- #


def _fingerprint(lane_results):
    """Travel/arrow layout fingerprint; separator identity is ignored.

    Dropping the outer vs cut-side lane of the same block leaves the same
    outgoing/incoming counts and arrow codes, so those repairs are one candidate.
    """
    parts = []
    for arm_id in sorted(lane_results):
        arm = lane_results[arm_id]
        tokens = []
        for item in _cs(arm):
            if _is_lane(item):
                if item.get("travel") == "incoming":
                    tokens.append(f"L:{item.get('travel')}:{_incoming_arrow_code(item)}")
                else:
                    tokens.append(f"L:{item.get('travel')}")
            else:
                tokens.append(str(item.get("type") or ""))
        parts.append(f"{arm_id}:{'/'.join(tokens)}")
    return "|".join(parts)


def _int_lane_results(lane_results):
    return {
        int(arm_id): copy.deepcopy(arm)
        for arm_id, arm in (lane_results or {}).items()
    }


def _result_from_candidates(candidates, *, trials, exhausted=False, frontier_min_score=None, note=None):
    if not candidates:
        return RepairResult(
            ok=False,
            score=0.0,
            edits=[],
            lane_results=None,
            config=None,
            trials=trials,
            search_exhausted=exhausted,
            frontier_min_score=frontier_min_score,
            note=note,
            candidates=[],
        )
    result = RepairResult(
        ok=True,
        trials=trials,
        search_exhausted=exhausted,
        frontier_min_score=frontier_min_score,
        note=note,
        candidates=list(candidates),
    )
    return result.apply_choice(candidates[0].candidate_id, source="score")


def solve_junction(
    lane_results,
    *,
    arms,
    channelized_features=None,
    through_roads=None,
    name=None,
    driving_side=None,
    max_trials=DEFAULT_MAX_TRIALS,
    max_score=DEFAULT_MAX_SCORE,
    editable_arm_ids=None,
    top_k=DEFAULT_TOP_K,
):
    """Find the cheapest feasible repairs (up to ``top_k`` distinct states).

    The queue is ordered only by atomic structural edit count. Connectivity
    completion is used as a feasibility witness via ``try_solve``; its own
    preference objective does not contribute to the repair score.

    ``editable_arm_ids`` can restrict repair to arms implicated by Layer-2.
    Passing None keeps backward-compatible behavior and allows all arms.

    Global minimum is guaranteed only within the explored operator/search budget.
    """
    side = normalize_driving_side(driving_side)
    start = _int_lane_results(lane_results)
    editable = (
        None
        if editable_arm_ids is None
        else {int(arm_id) for arm_id in editable_arm_ids}
    )
    top_k = max(1, int(top_k or DEFAULT_TOP_K))

    kwargs = dict(
        arms=arms,
        channelized_features=channelized_features,
        through_roads=through_roads,
        name=name,
        driving_side=side,
    )

    def probe(current):
        if not _structure_ok(current, side):
            return False, None
        if not _arrow_layout_ok(current, **kwargs):
            return False, None

        config = _build_config(current, **kwargs)
        _filled, _conns, exc = try_solve(config, name=name, diagnose=False)
        if exc is None:
            return True, config
        return False, config

    # Probe the original structure first. No repair is needed if at least one
    # feasible connectivity completion already exists.
    ok, config = probe(start)
    if ok:
        cand = RepairCandidate(
            candidate_id=1, score=0.0, edits=[], lane_results=start, config=config,
        )
        return _result_from_candidates([cand], trials=1)

    def dijkstra(*, focus_one_way, cap_score, cap_k, seen_fps, trial_budget):
        heap = []
        states = []
        best_cost = {}
        seq = 0
        local_trials = 0
        exhausted = False
        frontier_min_score = None
        found = []

        start_fp = _fingerprint(start)
        states.append((start, ()))
        best_cost[start_fp] = 0.0
        heapq.heappush(heap, (0.0, 0, "", seq))

        while heap:
            score, n_ops, _tie, state_id = heapq.heappop(heap)
            current, edits = states[state_id]
            fp = _fingerprint(current)

            if score > best_cost.get(fp, float("inf")):
                continue

            if score > cap_score:
                exhausted = True
                frontier_min_score = score
                break

            if found and score > found[0].score:
                frontier_min_score = score
                break

            if state_id != 0:
                local_trials += 1
                ok, config = probe(current)
                if ok:
                    if fp not in seen_fps and (
                        not found or score == found[0].score
                    ):
                        found.append(
                            RepairCandidate(
                                candidate_id=len(found) + 1,
                                score=score,
                                edits=[op.to_dict() for op in edits],
                                lane_results=copy.deepcopy(current),
                                config=config,
                            )
                        )
                    if len(found) >= cap_k:
                        break
                    continue

                if local_trials >= trial_budget:
                    exhausted = True
                    frontier_min_score = heap[0][0] if heap else None
                    break

            if not _structure_ok(current, side):
                continue

            one_way_ids = set(_one_way_arm_ids(current, editable))

            for arm_id, arm in current.items():
                if editable is not None and int(arm_id) not in editable:
                    continue

                for op in enumerate_operators(
                    arm, driving_side=side, through_roads=through_roads
                ):
                    if (
                        focus_one_way
                        and one_way_ids
                        and not _creates_reverse_travel(op, arm_id, one_way_ids)
                    ):
                        continue
                    patched = apply_operator(arm, op, driving_side=side)
                    if patched is None:
                        continue

                    if op.kind == "rewrite_arrow":
                        from_code = op.payload.get("from_code")
                        to_code = op.payload.get("to_code")
                        step_cost = ARROW_REWRITE_COST
                        atomic_changes = [
                            f"arrow cs[{op.payload.get('cs_index')}] "
                            f"{from_code!r} -> {to_code!r}"
                        ]
                    elif op.payload.get("edge") == "reverse":
                        # Lane + required new cut: charge like a normal
                        # lane add (2), not 3, so reverse growth competes
                        # with add/drop.
                        step_cost = 2.0
                        atomic_changes = [
                            f"reverse-add {op.payload.get('travel')} "
                            "+ opposing_separator"
                        ]
                    else:
                        step_cost, atomic_changes = _arm_structural_diff(
                            arm, patched
                        )
                        if step_cost <= 0:
                            continue

                    next_score = score + step_cost
                    if next_score > cap_score:
                        continue

                    nxt = dict(current)
                    nxt[arm_id] = patched
                    if not _structure_ok(nxt, side):
                        continue

                    nfp = _fingerprint(nxt)
                    if next_score >= best_cost.get(nfp, float("inf")):
                        continue

                    realized_op = replace(
                        op,
                        score=step_cost,
                        payload={
                            **op.payload,
                            "atomic_changes": atomic_changes,
                        },
                    )

                    best_cost[nfp] = next_score
                    seq += 1
                    states.append((nxt, edits + (realized_op,)))
                    heapq.heappush(
                        heap,
                        (next_score, n_ops + 1, op.op_id, seq),
                    )

        if frontier_min_score is None and heap:
            frontier_min_score = heap[0][0]
        return found, local_trials, exhausted, frontier_min_score

    # One-way arms first: otherwise cheap add/drop on two-way arms exhausts
    # the budget before reverse-travel combinations are reached.
    candidates, t1, exhausted, frontier_min_score = dijkstra(
        focus_one_way=True,
        cap_score=max_score,
        cap_k=top_k,
        seen_fps=set(),
        trial_budget=max_trials,
    )
    trials = 1 + t1

    # Second pass adds the one-way add/drop repairs the first pass skipped.
    if len(candidates) < top_k:
        seen = {_fingerprint(cand.lane_results) for cand in candidates}
        more, t2, exh2, fr2 = dijkstra(
            focus_one_way=False,
            cap_score=candidates[0].score if candidates else max_score,
            cap_k=top_k - len(candidates),
            seen_fps=seen,
            trial_budget=max(0, max_trials - t1),
        )
        trials += t2
        exhausted = exhausted or exh2
        if fr2 is not None:
            frontier_min_score = fr2
        candidates.extend(more)
        for idx, cand in enumerate(candidates, start=1):
            cand.candidate_id = idx

    if candidates:
        note = None
        if len(candidates) < top_k and exhausted:
            note = (
                f"collected {len(candidates)}/{top_k} feasible repairs before "
                f"search budget; max_trials={max_trials} max_score={max_score}"
            )
        return _result_from_candidates(
            candidates, trials=trials, exhausted=exhausted,
            frontier_min_score=frontier_min_score, note=note,
        )

    if exhausted:
        note = (
            f"no feasible repair found before search budget was exhausted; "
            f"max_trials={max_trials}, max_score={max_score}"
        )
    else:
        note = (
            f"no feasible repair exists in the explored operator space within "
            f"max_score={max_score}"
        )

    return RepairResult(
        ok=False,
        score=0.0,
        edits=[],
        lane_results=None,
        config=None,
        trials=trials,
        search_exhausted=exhausted,
        frontier_min_score=frontier_min_score,
        note=note,
        candidates=[],
    )


def _copy_scene_state(state):
    return {
        int(jid): {
            int(arm_id): copy.deepcopy(arm)
            for arm_id, arm in (arms or {}).items()
        }
        for jid, arms in (state or {}).items()
    }


def _scene_fingerprint(state):
    return "||".join(
        "%s:%s" % (jid, _fingerprint(state[jid]))
        for jid in sorted(state)
    )


def _connector_arm_keys(pairs):
    keys = set()
    for item in pairs or []:
        left, right = item.get("left") or {}, item.get("right") or {}
        if left.get("junction") is not None and left.get("arm") is not None:
            keys.add((int(left["junction"]), int(left["arm"])))
        if right.get("junction") is not None and right.get("arm") is not None:
            keys.add((int(right["junction"]), int(right["arm"])))
    return keys


def _arm_travels(arm):
    cs = _cs(arm)
    return {travel for travel in TRAVELS if _lanes_of_travel(cs, travel)}


def _connecting_road_mouth_keys(pairs):
    """Driving-connector mouths only; median throughs may still collapse to one-way."""
    from scripts.methods.llm_kg.refinement.layer3_jun_fix import (
        KIND_CONNECTING_ROAD,
        _pair_kind,
    )

    keys = set()
    for item in pairs or []:
        if _pair_kind(item) != KIND_CONNECTING_ROAD:
            continue
        for end in (item.get("left") or {}, item.get("right") or {}):
            if end.get("junction") is None or end.get("arm") is None:
                continue
            keys.add((int(end["junction"]), int(end["arm"])))
    return keys


def _collapses_two_way_connecting_mouth(jid, arm_id, arm, patched, connecting_keys):
    """Forbid matching a one-way far end by deleting the reverse stream.

    Dropping the last incoming/outgoing lane on a two-way connecting-road
    mouth is cheaper than reverse-adding at the one-way end plus unlocking
    arrows, but it also wipes through connectivity on that carriageway.
    """
    if (int(jid), int(arm_id)) not in connecting_keys:
        return False
    return len(_arm_travels(arm)) == 2 and len(_arm_travels(patched)) < 2


def _violating_connector_keys(state, pairs):
    from scripts.methods.llm_kg.refinement.layer3_jun_fix import item_inconsistent

    keys = set()
    for item in pairs or []:
        left, right = item["left"], item["right"]
        if item_inconsistent(item, state):
            keys.add((int(left["junction"]), int(left["arm"])))
            keys.add((int(right["junction"]), int(right["arm"])))
    return keys


def _scene_pairs_ok(state, pairs):
    from scripts.methods.llm_kg.refinement.layer3_jun_fix import item_inconsistent

    for item in pairs or []:
        if item_inconsistent(item, state):
            return False
    return True


def _scene_one_way_keys(state, editable_keys):
    keys = set()
    for jid, arm_id in editable_keys or ():
        arms = state.get(int(jid)) or {}
        if int(arm_id) not in arms:
            continue
        if int(arm_id) in _one_way_arm_ids({int(arm_id): arms[int(arm_id)]}, {int(arm_id)}):
            keys.add((int(jid), int(arm_id)))
    return keys


def _incoming_valued_arrow_slots(lane_results):
    """(arm_id, cs_index, code) for incoming lanes with a valued arrow."""
    slots = []
    for arm_id in sorted(lane_results or {}):
        arm = lane_results[arm_id]
        for idx, item in enumerate(_cs(arm)):
            if not _is_lane(item) or item.get("travel") != "incoming":
                continue
            if not _has_arrow_cue(item):
                continue
            code = _incoming_arrow_code(item)
            if is_unknown(code):
                continue
            slots.append((int(arm_id), int(idx), code))
    return slots


def _clear_incoming_arrows(lane_results, chosen):
    """Set the chosen incoming arrows to N. Returns (results, operators) or (None, None)."""
    out = _int_lane_results(lane_results)
    ops = []
    for arm_id, idx, code in chosen:
        arm = out.get(int(arm_id))
        if arm is None:
            return None, None
        cs = _cs(arm)
        if idx >= len(cs):
            return None, None
        updated = _set_arrow_code(cs[idx], UNKNOWN)
        if updated is None:
            return None, None
        cs[idx] = updated
        arm["cross_section"] = cs
        out[int(arm_id)] = arm
        ops.append(
            Operator(
                op_id=f"a{arm_id}:{idx}:{code}->{UNKNOWN}",
                arm_id=int(arm_id),
                kind="rewrite_arrow",
                payload={
                    "cs_index": int(idx),
                    "from_code": code,
                    "to_code": UNKNOWN,
                    "atomic_changes": [
                        f"arrow cs[{idx}] {code!r} -> {UNKNOWN!r}"
                    ],
                },
                score=ARROW_REWRITE_COST,
            )
        )
    return out, ops


def _fill_by_clearing_arrows(jid, lane_results, probe_junction, *, max_size=2):
    """If locked incoming arrows are the only fill blocker, clear the fewest to N."""
    slots = _incoming_valued_arrow_slots(lane_results)
    if not slots:
        return None
    combos = []
    for size in range(1, min(max_size, len(slots)) + 1):
        combos.extend(itertools.combinations(slots, size))
    combos.append(tuple(slots))
    seen = set()
    for combo in combos:
        key = tuple(combo)
        if key in seen:
            continue
        seen.add(key)
        patched, ops = _clear_incoming_arrows(lane_results, combo)
        if patched is None:
            continue
        ok, config = probe_junction(jid, patched)
        if ok:
            return patched, [replace(op, junction_id=int(jid)) for op in ops], config
    return None


def solve_scene(
    junctions,
    pairs,
    *,
    driving_side=None,
    max_trials=SCENE_MAX_TRIALS,
    max_score=DEFAULT_MAX_SCORE,
    top_k=DEFAULT_TOP_K,
):
    """Joint min-edit repair across junctions so connector mouths stay consistent.

    ``junctions`` maps junction id -> ``{lane_results, arms, channelized_features,
    through_roads, name}``. ``pairs`` are Layer-3 connector ends from
    ``locate_inter_junction_pairs``. Oracle: each junction ``try_solve``s and
    every pair satisfies Layer-3 HARD constraints: connecting-road lane-balance /
    opposing-traffic agreement, and median-linked through arms one-way.

    Single-junction callers should not invoke this (no-op at the pipeline).
    Default ``max_trials`` is ``SCENE_MAX_TRIALS`` (1500). Search starts at the
    inconsistent connector mouths, then other connector arms, then the remaining
    arms of those junctions (a compensating edit on a non-connector arm).
    Pair-ok structural states that fail only on locked incoming arrows are
    completed by clearing those arrows to N rather than burning the budget on
    dummy lane adds. Two-way connecting-road mouths are not collapsed to
    one-way to satisfy the pair oracle.
    """
    if len(junctions or {}) <= 1:
        return RepairResult(
            ok=True, score=0.0, edits=[], trials=0, note="single_junction",
            candidates=[],
        )

    side = normalize_driving_side(driving_side)
    junctions = {
        int(jid): spec or {}
        for jid, spec in (junctions or {}).items()
    }
    start = {
        int(jid): _int_lane_results(spec.get("lane_results"))
        for jid, spec in junctions.items()
    }
    top_k = max(1, int(top_k or DEFAULT_TOP_K))
    connector_keys = _connector_arm_keys(pairs)
    connecting_mouths = _connecting_road_mouth_keys(pairs)
    violating_keys = _violating_connector_keys(start, pairs)
    local_ok_cache = {}

    def _junction_kwargs(jid):
        spec = junctions[int(jid)]
        return dict(
            arms=spec.get("arms"),
            channelized_features=spec.get("channelized_features"),
            through_roads=spec.get("through_roads"),
            name=spec.get("name") or "j%s" % jid,
            driving_side=side,
        )

    def probe_junction(jid, lane_results):
        fp = _fingerprint(lane_results)
        cached = local_ok_cache.get((jid, fp))
        if cached is not None:
            return cached
        kwargs = _junction_kwargs(jid)
        if not _structure_ok(lane_results, side):
            local_ok_cache[(jid, fp)] = (False, None)
            return False, None
        if not _arrow_layout_ok(lane_results, **kwargs):
            local_ok_cache[(jid, fp)] = (False, None)
            return False, None
        config = _build_config(lane_results, **kwargs)
        _filled, _conns, exc = try_solve(config, name=kwargs["name"], diagnose=False)
        ok = exc is None
        local_ok_cache[(jid, fp)] = (ok, config if ok else None)
        return ok, config if ok else None

    def probe(state):
        configs = {}
        failed = []
        for jid, lane_results in state.items():
            ok, config = probe_junction(jid, lane_results)
            if ok:
                configs[jid] = config
            else:
                failed.append(jid)
        if failed:
            if not _scene_pairs_ok(state, pairs):
                return False, None, state, ()
            completed = _copy_scene_state(state)
            extra = []
            for jid in failed:
                relaxed = _fill_by_clearing_arrows(
                    jid, completed[jid], probe_junction,
                )
                if relaxed is None:
                    return False, None, state, ()
                patched, ops, config = relaxed
                completed[jid] = patched
                extra.extend(ops)
                configs[jid] = config
            return True, configs, completed, tuple(extra)
        if not _scene_pairs_ok(state, pairs):
            return False, configs, state, ()
        return True, configs, state, ()

    ok, config, completed, extra_ops = probe(start)
    if ok:
        extra_cost = sum(float(op.score or 0.0) for op in extra_ops)
        cand = RepairCandidate(
            candidate_id=1, score=extra_cost, edits=[op.to_dict() for op in extra_ops],
            lane_results=_copy_scene_state(completed), config=config,
        )
        return _result_from_candidates([cand], trials=1)

    def dijkstra(*, focus_one_way, editable_keys, cap_score, cap_k, seen_fps, trial_budget):
        heap = []
        states = []
        best_cost = {}
        seq = 0
        local_trials = 0
        exhausted = False
        frontier_min_score = None
        found = []

        start_fp = _scene_fingerprint(start)
        states.append((start, ()))
        best_cost[start_fp] = 0.0
        heapq.heappush(heap, (0.0, 0, "", seq))

        while heap:
            score, n_ops, _tie, state_id = heapq.heappop(heap)
            current, edits = states[state_id]
            fp = _scene_fingerprint(current)

            if score > best_cost.get(fp, float("inf")):
                continue
            if score > cap_score:
                exhausted = True
                frontier_min_score = score
                break
            if found and score > found[0].score:
                frontier_min_score = score
                break

            if state_id != 0:
                local_trials += 1
                ok, config, completed, extra_ops = probe(current)
                if ok:
                    extra_cost = sum(float(op.score or 0.0) for op in extra_ops)
                    cand_score = score + extra_cost
                    cand_fp = _scene_fingerprint(completed)
                    if cand_fp not in seen_fps and (
                        not found or cand_score <= found[0].score
                    ):
                        if found and cand_score < found[0].score:
                            found.clear()
                        found.append(
                            RepairCandidate(
                                candidate_id=len(found) + 1,
                                score=cand_score,
                                edits=[op.to_dict() for op in edits]
                                + [op.to_dict() for op in extra_ops],
                                lane_results=_copy_scene_state(completed),
                                config=config,
                            )
                        )
                        seen_fps.add(cand_fp)
                    if len(found) >= cap_k and score >= found[0].score:
                        break
                    continue
                if local_trials >= trial_budget:
                    exhausted = True
                    frontier_min_score = heap[0][0] if heap else None
                    break
                if found and score >= found[0].score:
                    continue

            one_way_keys = _scene_one_way_keys(current, editable_keys)
            for jid, arm_id in sorted(editable_keys or ()):
                arms = current.get(jid) or {}
                arm = arms.get(arm_id)
                if arm is None:
                    continue
                through_roads = (junctions.get(jid) or {}).get("through_roads")
                for op in enumerate_operators(
                    arm, driving_side=side, through_roads=through_roads,
                ):
                    op = replace(op, junction_id=jid)
                    if op.kind == "rewrite_arrow":
                        # Pair oracle is counts / opposing traffic; arrow
                        # rewrites cannot repair a mouth mismatch and crowd
                        # out the structural combinations that can.
                        continue
                    if (
                        focus_one_way
                        and (jid, arm_id) in one_way_keys
                        and not _creates_reverse_travel(op, arm_id, {arm_id})
                    ):
                        continue
                    patched = apply_operator(arm, op, driving_side=side)
                    if patched is None:
                        continue
                    if op.kind == "rewrite_arrow":
                        from_code = op.payload.get("from_code")
                        to_code = op.payload.get("to_code")
                        step_cost = ARROW_REWRITE_COST
                        atomic_changes = [
                            f"arrow cs[{op.payload.get('cs_index')}] "
                            f"{from_code!r} -> {to_code!r}"
                        ]
                    elif op.payload.get("edge") == "reverse":
                        step_cost = 2.0
                        atomic_changes = [
                            f"reverse-add {op.payload.get('travel')} "
                            "+ opposing_separator"
                        ]
                    else:
                        step_cost, atomic_changes = _arm_structural_diff(arm, patched)
                        if step_cost <= 0:
                            continue
                    next_score = score + step_cost
                    if next_score > cap_score:
                        continue
                    nxt = {
                        other: dict(other_arms)
                        for other, other_arms in current.items()
                    }
                    nxt[jid] = dict(arms)
                    nxt[jid][arm_id] = patched
                    if _collapses_two_way_connecting_mouth(
                        jid, arm_id, arm, patched, connecting_mouths,
                    ):
                        continue
                    if not _structure_ok(nxt[jid], side):
                        continue
                    nfp = _scene_fingerprint(nxt)
                    if next_score >= best_cost.get(nfp, float("inf")):
                        continue
                    realized_op = replace(
                        op,
                        score=step_cost,
                        payload={**op.payload, "atomic_changes": atomic_changes},
                    )
                    best_cost[nfp] = next_score
                    seq += 1
                    states.append((nxt, edits + (realized_op,)))
                    heapq.heappush(heap, (next_score, n_ops + 1, op.op_id, seq))

        if frontier_min_score is None and heap:
            frontier_min_score = heap[0][0]
        return found, local_trials, exhausted, frontier_min_score

    # Inconsistent mouths first. Expanding to every arm of those junctions
    # up front burns the trial budget on already-valid median throughs.
    editable_first = violating_keys or connector_keys
    candidates, t1, exhausted, frontier_min_score = dijkstra(
        focus_one_way=True,
        editable_keys=editable_first,
        cap_score=max_score,
        cap_k=top_k,
        seen_fps=set(),
        trial_budget=max_trials,
    )
    trials = 1 + t1

    search_keys = editable_first
    remaining = max(0, max_trials - t1)
    if not candidates and remaining and connector_keys and connector_keys != editable_first:
        search_keys = connector_keys
        more, t2, exh2, fr2 = dijkstra(
            focus_one_way=True,
            editable_keys=connector_keys,
            cap_score=max_score,
            cap_k=top_k,
            seen_fps=set(),
            trial_budget=remaining,
        )
        trials += t2
        exhausted = exhausted or exh2
        if fr2 is not None:
            frontier_min_score = fr2
        candidates.extend(more)
        remaining = max(0, max_trials - (trials - 1))

    if not candidates and remaining:
        connector_jids = {jid for jid, _arm in (violating_keys or connector_keys)} or set(start)
        all_keys = {
            (jid, arm_id)
            for jid, arms in start.items()
            for arm_id in arms
            if jid in connector_jids
        }
        if all_keys and all_keys != search_keys:
            search_keys = all_keys
            more, t_all, exh_all, fr_all = dijkstra(
                focus_one_way=True,
                editable_keys=all_keys,
                cap_score=max_score,
                cap_k=top_k,
                seen_fps=set(),
                trial_budget=max(0, max_trials - trials + 1),
            )
            trials += t_all
            exhausted = exhausted or exh_all
            if fr_all is not None:
                frontier_min_score = fr_all
            candidates.extend(more)

    if len(candidates) < top_k:
        seen = {_scene_fingerprint(cand.lane_results) for cand in candidates}
        more, t3, exh3, fr3 = dijkstra(
            focus_one_way=False,
            editable_keys=search_keys or connector_keys or editable_first,
            cap_score=candidates[0].score if candidates else max_score,
            cap_k=top_k - len(candidates),
            seen_fps=seen,
            trial_budget=max(0, max_trials - trials + 1),
        )
        trials += t3
        exhausted = exhausted or exh3
        if fr3 is not None:
            frontier_min_score = fr3
        candidates.extend(more)
        for idx, cand in enumerate(candidates, start=1):
            cand.candidate_id = idx

    if candidates:
        min_score = min(float(cand.score) for cand in candidates)
        candidates = [
            cand for cand in candidates if float(cand.score) == min_score
        ][:top_k]
        for idx, cand in enumerate(candidates, start=1):
            cand.candidate_id = idx
        note = None
        if len(candidates) < top_k and exhausted:
            note = (
                f"collected {len(candidates)}/{top_k} scene repairs before "
                f"search budget; max_trials={max_trials} max_score={max_score}"
            )
        return _result_from_candidates(
            candidates, trials=trials, exhausted=exhausted,
            frontier_min_score=frontier_min_score, note=note,
        )

    if exhausted:
        note = (
            f"no feasible scene repair found before search budget was exhausted; "
            f"max_trials={max_trials}, max_score={max_score}"
        )
    else:
        note = (
            f"no feasible scene repair exists in the explored operator space "
            f"within max_score={max_score}"
        )
    return RepairResult(
        ok=False,
        score=0.0,
        edits=[],
        lane_results=None,
        config=None,
        trials=trials,
        search_exhausted=exhausted,
        frontier_min_score=frontier_min_score,
        note=note,
        candidates=[],
    )