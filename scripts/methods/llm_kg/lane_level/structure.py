"""Hard structural checks for mouth cross_section readings.

Enforced after every LLM generate / patch / modify:
1. boundary–area–separator alternation (boundaries only at the two ends)
2. unique code-assigned element_id on every non-lane item (after visual is finalized)
3. painted lines report color white or yellow; unmarked_line and edge have color null
4. opposing_separator one-way vs cut consistency (check_organization)
5. travel vs opposing_separator for the driving side (one-way when cut is null)
   — organization fills travel from the cut / one_way_travel
   — check_travel only scores lanes that already have travel
6. incoming-arrow order (L before S before R among incoming-facing arrows)
   — by travel when check_travel=True, else by orientation when requested
"""

from scripts.methods.connection_parser import cut_rule_text, normalize_driving_side, travel_block_order
from scripts.methods.llm_kg.lane_level.schemas import (
    AREA_TYPES,
    BOUNDARY_TYPES,
    LANE_TYPE,
    LINE_COLORS,
    LINE_TYPES,
    PAINTED_LINE_TYPES,
    SEPARATOR_TYPES,
    TRAVELS,
    UNPAINTED_SEPARATOR_TYPES,
)

STRUCTURE_RETRY_LIMIT = 1


class StructureValidationError(ValueError):
    """Raised when cross_section structure is invalid after the allowed retry."""

    def __init__(self, message, *, issues=None, arm_id=None):
        super().__init__(message)
        self.issues = list(issues or [])
        self.arm_id = arm_id


def _item_kind(item):
    item_type = (item or {}).get("type")
    if item_type in BOUNDARY_TYPES:
        return "boundary"
    if item_type in SEPARATOR_TYPES:
        return "separator"
    if item_type in AREA_TYPES:
        return "area"
    return None


def alternation_issues(cross_section):
    """Return human-readable issues if boundary/separator/area alternation is violated."""
    items = list(cross_section or [])
    issues = []
    if not items:
        return ["cross_section is empty; need boundary, area, ..., boundary"]
    kinds = []
    for idx, item in enumerate(items):
        kind = _item_kind(item)
        if kind is None:
            issues.append(
                f"index {idx}: unknown type {item.get('type')!r}; "
                f"expected boundary ({', '.join(BOUNDARY_TYPES)}), "
                f"separator ({', '.join(SEPARATOR_TYPES)}), or "
                f"area ({', '.join(AREA_TYPES)})"
            )
            kinds.append(None)
        else:
            kinds.append(kind)
    if kinds[0] != "boundary":
        issues.append(
            f"index 0: must start with a boundary, got type={items[0].get('type')!r}"
        )
    if kinds[-1] != "boundary":
        issues.append(
            f"index {len(items) - 1}: must end with a boundary, "
            f"got type={items[-1].get('type')!r}"
        )
    n_boundary = sum(1 for kind in kinds if kind == "boundary")
    if n_boundary != 2:
        issues.append(
            f"cross_section has {n_boundary} boundary item(s); "
            "exactly two are required (first and last)"
        )
    if len(items) % 2 == 0:
        issues.append(
            f"cross_section length {len(items)} is even; "
            "valid pattern has odd length (boundary, area, separator, ..., area, boundary)"
        )
    for idx in range(1, max(len(items) - 1, 1)):
        if kinds[idx] == "boundary":
            issues.append(
                f"index {idx}: boundary is only allowed at the two ends; "
                "internal physical cuts use type=edge"
            )
    for idx in range(len(items) - 1):
        left, right = kinds[idx], kinds[idx + 1]
        if left is None or right is None:
            continue
        left_type = items[idx].get("type")
        right_type = items[idx + 1].get("type")
        if left in ("boundary", "separator") and right in ("boundary", "separator"):
            issues.append(
                f"indices {idx}-{idx + 1}: adjacent {left}+{right} "
                f"{left_type!r}+{right_type!r}; insert an area "
                f"(lane/median/buffer/bicycle/parking) between them"
            )
        elif left == "area" and right == "area":
            issues.append(
                f"indices {idx}-{idx + 1}: adjacent areas "
                f"{left_type!r}+{right_type!r}; insert a separator (edge/line) between them"
            )
        elif left in ("boundary", "separator") and right != "area":
            issues.append(
                f"index {idx}: {left} {left_type!r} must be followed by an area, "
                f"got {right_type!r}"
            )
        elif left == "area" and right not in ("boundary", "separator"):
            issues.append(
                f"index {idx}: area {left_type!r} must be followed by a boundary or separator, "
                f"got {right_type!r}"
            )
    return issues


def element_id_issues(cross_section):
    """Non-lane items must have unique element_id; lanes must not."""
    issues = []
    seen = {}
    for idx, item in enumerate(cross_section or []):
        item_type = item.get("type")
        eid = item.get("element_id")
        if item_type == LANE_TYPE:
            if eid:
                issues.append(f"index {idx}: lane must not have element_id")
            continue
        if not eid:
            issues.append(f"index {idx}: {item_type!r} missing element_id")
            continue
        eid = str(eid)
        if eid in seen:
            issues.append(
                f"index {idx}: duplicate element_id={eid!r} (also index {seen[eid]})"
            )
        else:
            seen[eid] = idx
    return issues


def _opposing_cut_index(cross_section, opposing_separator):
    if not isinstance(opposing_separator, dict):
        return None, None
    element_id = opposing_separator.get("element_id")
    if not element_id:
        return None, None
    eid = str(element_id)
    for idx, item in enumerate(cross_section or []):
        if str(item.get("element_id") or "") == eid:
            return idx, eid
    return None, eid


_ORIENTATION_TO_TRAVEL = {"up": "incoming", "down": "outgoing"}
_CUE_TRAVEL_KINDS = ("arrow", "road_marking", "vehicle_heading")


def _cue_travels(item, kind):
    votes = []
    for cue in (item or {}).get("cues") or []:
        if cue.get("kind") != kind:
            continue
        travel = _ORIENTATION_TO_TRAVEL.get(cue.get("orientation"))
        if travel in TRAVELS:
            votes.append(travel)
    return votes


def _majority_travel(votes):
    incoming = sum(1 for travel in votes if travel == "incoming")
    outgoing = sum(1 for travel in votes if travel == "outgoing")
    if incoming > outgoing:
        return "incoming"
    if outgoing > incoming:
        return "outgoing"
    return None


def lane_travel_from_cues(item):
    """Vote lane.travel from cues: arrow, then road_marking, then vehicle_heading."""
    if (item or {}).get("type") != LANE_TYPE:
        return None
    for kind in _CUE_TRAVEL_KINDS:
        votes = _cue_travels(item, kind)
        if votes:
            return _majority_travel(votes)
    return None


def fill_lane_travels_from_cues(cross_section):
    """Write travel only on lanes that have a directional cue vote; leave the rest unset."""
    updated = []
    for item in cross_section or []:
        entry = dict(item)
        if entry.get("type") == LANE_TYPE:
            travel = lane_travel_from_cues(entry)
            if travel:
                entry["travel"] = travel
            else:
                entry.pop("travel", None)
        updated.append(entry)
    return updated


def _one_way_travel_issues(cross_section, one_way_travel=None):
    """When opposing cut is null, filled lane.travel values must agree with one_way_travel."""
    issues = []
    lane_travels = []
    for idx, item in enumerate(cross_section or []):
        if item.get("type") != LANE_TYPE:
            continue
        travel = item.get("travel")
        if travel not in TRAVELS:
            continue
        lane_travels.append((idx, travel))
        if one_way_travel in TRAVELS and travel != one_way_travel:
            issues.append(
                f"index {idx}: one-way arm (opposing_separator.element_id is null) "
                f"requires travel={one_way_travel!r}, got {travel!r}"
            )
    unique = {travel for _, travel in lane_travels}
    if len(unique) > 1:
        incoming = [idx for idx, travel in lane_travels if travel == "incoming"]
        outgoing = [idx for idx, travel in lane_travels if travel == "outgoing"]
        issues.append(
            "one-way arm (opposing_separator.element_id is null) requires filled "
            f"lane.travel values to agree; mixed incoming (indices {incoming}) and "
            f"outgoing (indices {outgoing})"
        )
    return issues


def painted_line_color_issues(cross_section):
    """Painted lines: white/yellow. unmarked_line and edge: color null (no paint)."""
    issues = []
    for idx, item in enumerate(cross_section or []):
        item_type = item.get("type")
        color = item.get("color")
        if item_type in PAINTED_LINE_TYPES:
            if color not in LINE_COLORS:
                issues.append(
                    f"index {idx}: painted {item_type} color must be "
                    f"'white' or 'yellow', got {color!r}"
                )
        elif item_type in UNPAINTED_SEPARATOR_TYPES:
            if color is not None:
                issues.append(
                    f"index {idx}: {item_type} has no paint; color must be null, got {color!r}"
                )
    return issues


_LETTER_RANK = {"L": 0, "S": 1, "R": 2}


def _top_arrow(item):
    for cue in (item or {}).get("cues") or []:
        if cue.get("kind") != "arrow":
            continue
        movements = cue.get("movements") or []
        if not movements:
            continue
        return str(movements[0]), cue
    return None, None


def _code_letter_span(code):
    ranks = [_LETTER_RANK[ch] for ch in str(code or "") if ch in _LETTER_RANK]
    if not ranks:
        return None
    return min(ranks), max(ranks)


def incoming_arrow_order_issues(cross_section, *, by="travel"):
    """L-before-S-before-R among incoming-facing arrows (one top code per lane)."""
    rows = []
    for idx, item in enumerate(cross_section or []):
        if item.get("type") != LANE_TYPE:
            continue
        code, cue = _top_arrow(item)
        if code is None:
            continue
        if by == "orientation":
            if (cue or {}).get("orientation") != "up":
                continue
        elif item.get("travel") != "incoming":
            continue
        span = _code_letter_span(code)
        if span is None:
            continue
        rows.append((idx, code, span))
    issues = []
    for i in range(len(rows) - 1):
        idx_a, code_a, (min_a, max_a) = rows[i]
        idx_b, code_b, (min_b, max_b) = rows[i + 1]
        if min_a <= min_b and max_a <= max_b:
            continue
        issues.append(
            f"INCOMING-ARROW ORDER: lane index {idx_a} code {code_a!r} is not left-of "
            f"lane index {idx_b} code {code_b!r}; left-to-right incoming-facing arrows "
            "must go L before S before R"
        )
    return issues


def fill_lane_travels(cross_section, opposing_separator, *, driving_side=None, one_way_travel=None):
    """Write every lane.travel from the opposing cut. one_way_travel is used when the cut is null."""
    updated = [dict(item) for item in (cross_section or [])]
    cut_idx, eid = _opposing_cut_index(updated, opposing_separator)
    if eid is not None and cut_idx is None:
        return updated, [
            "opposing_separator.element_id=%r is not present on any "
            "non-lane cross_section item" % eid
        ]
    if eid is None:
        travel = one_way_travel
        if travel not in TRAVELS and isinstance(opposing_separator, dict):
            travel = opposing_separator.get("one_way_travel")
        if travel not in TRAVELS:
            return updated, [
                "one-way arm (opposing_separator.element_id is null) requires "
                "one_way_travel incoming or outgoing"
            ]
        for item in updated:
            if item.get("type") == LANE_TYPE:
                item["travel"] = travel
        return updated, []
    left_travel, right_travel = travel_block_order(driving_side)
    for idx, item in enumerate(updated):
        if item.get("type") != LANE_TYPE:
            continue
        item["travel"] = left_travel if idx < cut_idx else right_travel
    return updated, []


def one_way_direction_issues(opposing_separator):
    """one_way_travel is required iff the opposing cut is null."""
    if not isinstance(opposing_separator, dict):
        return []
    eid = opposing_separator.get("element_id")
    travel = opposing_separator.get("one_way_travel")
    if eid:
        if travel is not None:
            return [
                "two-way arm (opposing_separator.element_id is set) must leave "
                "one_way_travel null"
            ]
        return []
    if travel not in TRAVELS:
        return [
            "one-way arm (opposing_separator.element_id is null) requires "
            "one_way_travel incoming or outgoing"
        ]
    return []


def _motor_lane_counts_about(cross_section, cut_idx):
    left = right = 0
    for idx, item in enumerate(cross_section or []):
        if item.get("type") != LANE_TYPE:
            continue
        if idx < cut_idx:
            left += 1
        elif idx > cut_idx:
            right += 1
    return left, right


def opposing_travel_issues(cross_section, opposing_separator, driving_side=None):
    """Check filled lane.travel vs opposing cut: one-way when null, LTR blocks when set."""
    cut_idx, eid = _opposing_cut_index(cross_section, opposing_separator)
    if eid is None:
        one_way = None
        if isinstance(opposing_separator, dict):
            one_way = opposing_separator.get("one_way_travel")
        return _one_way_travel_issues(cross_section, one_way)
    if cut_idx is None:
        return [
            f"opposing_separator.element_id={eid!r} is not present on any "
            "non-lane cross_section item"
        ]
    side = normalize_driving_side(driving_side)
    left_travel, right_travel = travel_block_order(side)
    issues = []
    cut_item = (cross_section or [])[cut_idx]
    cut_type = cut_item.get("type")
    if cut_type not in LINE_TYPES and cut_type != "median":
        issues.append(
            f"opposing_separator.element_id={eid!r} points to type={cut_type!r}; "
            "cut must be a line or median (never boundary/edge/buffer/bicycle/parking)"
        )
    left_n, right_n = _motor_lane_counts_about(cross_section, cut_idx)
    if left_n == 0 or right_n == 0:
        issues.append(
            f"opposing_separator.element_id={eid!r} has motor lanes on only one side "
            f"(left={left_n}, right={right_n}); cite a cut with motor lanes on both "
            "sides, or set element_id=null (one-way)"
        )
    for idx, item in enumerate(cross_section or []):
        if item.get("type") != LANE_TYPE:
            continue
        travel = item.get("travel")
        if travel not in TRAVELS:
            continue
        if idx < cut_idx:
            expected = left_travel
            side_name = "left"
        elif idx > cut_idx:
            expected = right_travel
            side_name = "right"
        else:
            continue
        if travel != expected:
            issues.append(
                f"index {idx}: lane {side_name} of opposing cut {eid!r} "
                f"must have travel={expected!r} (driving_side={side}), got {travel!r}"
            )
    return issues


def _area_items(cross_section):
    return [item for item in (cross_section or []) if item.get("type") in AREA_TYPES]


def _lane_items(cross_section):
    return [item for item in (cross_section or []) if item.get("type") == LANE_TYPE]


def _type_seq(items):
    return tuple((item or {}).get("type") for item in (items or []))


def _with_prior_element_id(prior_item, new_item):
    """Keep a stable element_id on non-lane areas when the new item omitted it."""
    if (new_item or {}).get("type") == LANE_TYPE:
        return new_item
    if new_item.get("element_id") or not (prior_item or {}).get("element_id"):
        return new_item
    return {**new_item, "element_id": prior_item["element_id"]}


def _rebuild_from_prior(prior, replacements, *, kind):
    out = []
    idx = 0
    for item in prior:
        take = item.get("type") == LANE_TYPE if kind == "lanes" else item.get("type") in AREA_TYPES
        if take:
            out.append(_with_prior_element_id(item, replacements[idx]))
            idx += 1
        else:
            out.append(item)
    return out


def overlay_on_prior_skeleton(prior_cs, new_cs):
    """Merge a partial/invalid modify onto the prior mouth layout.

    Used only after the new sequence fails alternation. The prior skeleton
    (boundaries, separators, and any non-lane areas not present in the new
    dump) is kept. New items supply updated lane/area fields.

    Same layout: same number of motor lanes. If the new dump also repeats the
    prior non-lane area types, those areas are updated too; if it omits them,
    they stay from the prior reading. Extra non-lane areas that were not in
    the prior reading are not invented — return None so the model must emit
    a complete valid sequence.
    """
    prior = list(prior_cs or [])
    new_lanes = _lane_items(new_cs)
    prior_lanes = _lane_items(prior)
    if not prior_lanes or len(prior_lanes) != len(new_lanes):
        return None
    new_areas = _area_items(new_cs)
    prior_areas = _area_items(prior)
    new_non_lane = [item for item in new_areas if item.get("type") != LANE_TYPE]
    prior_non_lane = [item for item in prior_areas if item.get("type") != LANE_TYPE]
    if _type_seq(new_areas) == _type_seq(prior_areas):
        return _rebuild_from_prior(prior, new_areas, kind="areas")
    if not new_non_lane or _type_seq(new_non_lane) == _type_seq(prior_non_lane):
        return _rebuild_from_prior(prior, new_lanes, kind="lanes")
    return None


def validate_cross_section_structure(cross_section, opposing_separator=None, *, driving_side=None, check_travel=True, check_organization=None, check_arrow_order=None, check_element_ids=None):
    """Return all structure issues (empty list means valid)."""
    if check_organization is None:
        check_organization = check_travel
    if check_arrow_order is None:
        check_arrow_order = "travel" if check_travel else None
    if check_element_ids is None:
        check_element_ids = check_organization
    issues = []
    issues.extend(alternation_issues(cross_section))
    if check_element_ids:
        issues.extend(element_id_issues(cross_section))
    issues.extend(painted_line_color_issues(cross_section))
    if check_organization:
        issues.extend(one_way_direction_issues(opposing_separator))
    if check_travel:
        issues.extend(opposing_travel_issues(cross_section, opposing_separator, driving_side))
    if check_arrow_order:
        issues.extend(incoming_arrow_order_issues(cross_section, by=check_arrow_order))
    return issues


def validate_arm_result(result, *, driving_side=None, check_travel=True, check_organization=None, check_arrow_order=None, check_element_ids=None):
    """Validate a normalized arm result dict."""
    result = result or {}
    return validate_cross_section_structure(
        result.get("cross_section"),
        result.get("opposing_separator"),
        driving_side=driving_side if driving_side is not None else result.get("driving_side"),
        check_travel=check_travel,
        check_organization=check_organization,
        check_arrow_order=check_arrow_order,
        check_element_ids=check_element_ids,
    )


def format_organization_retry_text(issues, *, driving_side=None):
    """Retry text when organization resubmits opposing_separator and arrow_updates."""
    side = normalize_driving_side(driving_side)
    bullet = "\n".join(f"- {issue}" for issue in (issues or []) if issue)
    return (
        "ORGANIZATION VALIDATION FAILED — resubmit opposing_separator and arrow_updates.\n\n"
        "Do not output cross_section or lane.travel. "
        "arrow_updates may only rewrite movements on existing arrow cues, "
        "and only when this crop's painted heads support the new code.\n\n"
        "Violations:\n"
        f"{bullet or '- (unspecified)'}\n\n"
        "Hard rules:\n"
        "1. opposing_separator.element_id must cite a median or line already in the "
        "given visual cross_section that has at least one motor lane on both sides, "
        "or be null.\n"
        "2. When element_id is non-null: "
        f"{cut_rule_text(side)}. Leave one_way_travel null.\n"
        "3. When element_id is null, set one_way_travel to incoming or outgoing. "
        "Use one-way only when the mouth is a single shared stream. "
        "A center double line, median, or channelization buffer can be the cut "
        "only when motor lanes exist on both sides, even if no vehicles are visible. "
        "Code fills every lane.travel from opposing_separator.\n"
        "4. INCOMING-ARROW ORDER: after travel is filled, incoming-lane arrows "
        "must go L before S before R (combined codes count by their L/S/R components). "
        "Prefer a structural median or center-line cut; re-read the painted heads "
        "and rewrite only codes the image supports, instead of moving that cut.\n\n"
        "Return opposing_separator and arrow_updates for this same request."
    )


def format_structure_retry_text(
    issues, *, driving_side=None, context="reading", prior_cross_section=None,
    check_travel=True, check_organization=None,
):
    """User-turn / tool-error text asking the model to regenerate once."""
    if check_organization is None:
        check_organization = check_travel
    side = normalize_driving_side(driving_side)
    bullet = "\n".join(f"- {issue}" for issue in (issues or []) if issue)
    prior_note = ""
    if prior_cross_section:
        types = [item.get("type") for item in prior_cross_section]
        prior_note = (
            "The prior sequence is the skeleton to copy unless you are adding or removing "
            "a cut or area. Re-emit it in full (boundaries, separators, and areas) and "
            f"change only the fields you intend to revise: {types}\n\n"
        )
    if check_organization:
        extra_rules = (
            "2. opposing_separator.element_id must cite a median or line already in "
            "the given visual cross_section (code-assigned) that has a motor lane "
            "on both sides, or be null.\n"
            "3. When opposing_separator.element_id is non-null: "
            f"{cut_rule_text(side)}. "
            "When opposing_separator.element_id is null (one-way arm), set "
            "one_way_travel to incoming or outgoing. Do not assign lane.travel; "
            "code fills it from opposing_separator.\n"
            "4. INCOMING-ARROW ORDER: among travel=incoming lanes with a readable "
            "arrow, top confidence codes left-to-right must go L before S "
            "before R (combined codes count by their L/S/R components). "
            "Re-read painted heads on the crop and rewrite only codes the image supports; "
            "do not move a structural median or center-line cut.\n\n"
        )
    else:
        extra_rules = (
            "2. Do not output element_id; code assigns them later. "
            "This is visual identification only: do not output opposing_separator "
            "or lane.travel.\n\n"
        )
    return (
        "STRUCTURE VALIDATION FAILED — resubmit a corrected "
        f"{context}.\n\n"
        "Violations:\n"
        f"{bullet or '- (unspecified)'}\n\n"
        f"{prior_note}"
        "Hard rules to satisfy:\n"
        "1. Alternation: left-to-right must be boundary, area, separator, area, ..., area, boundary "
        "(odd length; start and end with a boundary; internal cuts are separators; "
        "no two boundaries/separators adjacent; no two areas adjacent). "
        "Boundaries = boundary. Separators = edge / solid_line / double_solid_line / "
        "broken_line / unmarked_line (edge is internal only). "
        "Painted lines (solid / double solid / broken) include color white or yellow; "
        "unmarked_line and edge have color null. "
        "Areas = lane / median / buffer / bicycle / parking.\n"
        f"{extra_rules}"
        "Return only the corrected schema object for this same request."
    )


def raise_structure_error(issues, *, arm_id=None, stage="structure"):
    arm = f"arm {arm_id}: " if arm_id is not None else ""
    detail = "; ".join(issues) if issues else "invalid cross_section structure"
    raise StructureValidationError(
        f"{arm}{stage} validation failed after {STRUCTURE_RETRY_LIMIT} retry: {detail}",
        issues=issues,
        arm_id=arm_id,
    )
