import json

from scripts.methods.connection_parser import (
    channelized_turn_name,
    cut_rule_text,
    normalize_driving_side,
    slip_to_edge,
)
from scripts.methods.llm_kg.lane_level.schemas import (
    CRITIC_KINDS,
    PAINTED_LINE_TYPES,
    UNPAINTED_SEPARATOR_TYPES,
)


def arm_features_note(arm_id, channelized_features=None, through_roads=None, driving_side=None):
    """Return arm-specific structural facts already established upstream."""
    if arm_id is None:
        return ""
    aid = int(arm_id)
    side = normalize_driving_side(driving_side)
    turn_name = channelized_turn_name(side)
    to_edge = slip_to_edge(side)
    notes = []
    for feature in channelized_features or []:
        if not isinstance(feature, dict):
            continue
        from_arm, to_arm = feature.get("from_arm"), feature.get("to_arm")
        if from_arm is None or to_arm is None:
            continue
        from_arm, to_arm = int(from_arm), int(to_arm)
        if aid == from_arm:
            notes.append(
                "This arm is from_arm of a channelized %s island onto arm %s."
                % (turn_name, to_arm)
            )
        if aid == to_arm:
            notes.append(
                "This arm is to_arm of a channelized %s island from arm %s: "
                "the %s motor lane must be travel=outgoing."
                % (turn_name, from_arm, to_edge)
            )
    return " ".join(notes)


def _structural_constraints_block(driving_side=None, channelized_features=None, arm_id=None):
    side = normalize_driving_side(driving_side)
    cut_text = cut_rule_text(side)
    note = arm_features_note(arm_id, channelized_features, driving_side=driving_side)
    slip_rule = f"- CHANNELIZED SLIP: {note}\n" if note else ""
    return f"""\
## Structural constraints
The junction is at the crop top. Incoming travel is toward the crop top; outgoing travel is toward the crop bottom. Cue orientation is an observation and must not be changed. If a reported arrow.movements does not match the painted heads on this crop, rewrite it via arrow_updates from the image; do not invent or drop a head just to satisfy a constraint, and do not add or remove cues.

Rules:
- OPPOSING CUT: set element_id when the mouth has two motor streams, judged from structure and directional cues together. A center yellow/white double line, median, or channelization buffer between two carriageways is structural support for a cut only if at least one motor lane lies on each side of that element. Directional cues, when present, should agree with that cut after travel is filled; incoming vehicles queued only on the approach side of a center cut still support two-way. Do not refuse a two-sided structural cut just because one side has no car or arrow.
- LEFT/RIGHT CUT: when opposing_separator.element_id is non-null, code assigns travel so that {cut_text}. Leave one_way_travel null. Choose the cut so the assignment matches the structural split and any directional cues.
- ONE-WAY: when opposing_separator.element_id is null, set one_way_travel to incoming or outgoing; code assigns that travel to every motor lane. Use one-way only when the mouth is a single shared stream: no structural opposing split and no opposite-direction cues. Do not infer one-way merely because this is a junction mouth, because no vehicles are visible, or because only incoming vehicles are seen.
- INCOMING-ARROW ORDER: among arrows on incoming lanes, left-to-right movement codes must progress L before S before R; combined codes are interpreted by their L/S/R components. Prefer a structural median or center-line cut. If the order fails after that cut, re-read the painted heads on this crop and rewrite only codes the image supports; do not move the cut and do not invent a code that the marking does not show. Never invent or remove a lane merely to satisfy this rule.
{slip_rule}"""


def _compose_prompt(*parts):
    return "\n\n".join(part.strip() for part in parts if part and str(part).strip())


def _task_block(text):
    return f"## Task\n{text}"


ONTOLOGY_BLOCK = """\
## Cross-section ontology
The mouth is the road cross-section nearest the junction, immediately upstream of the stop line or equivalent entry line. Only elements that intersect this mouth line belong in the cross-section.

The sequence is:
boundary, area, separator, area, ..., separator, area, boundary.

### Boundary
- boundary: one of the two outer physical edges of the represented roadway envelope.
- Sidewalk, plaza, or pedestrian-only space beyond the roadway curb is outside the cross-section.

### Separator
An internal longitudinal cut between adjacent areas:
- solid_line: one continuous painted line; color white or yellow.
- double_solid_line: two close continuous painted lines represented as one separator; color white or yellow.
- broken_line: one dashed/broken painted line; color white or yellow.
- unmarked_line: an inferred lane separator with no visible paint; use only when two distinct adjacent travel strips are otherwise well supported. Width alone is insufficient.
- edge: an internal physical edge such as a curb/median/island/buffer edge; color null.

Only longitudinal cuts are separators.

### Area
The strip between adjacent boundaries/separators:
- lane: motor-vehicle travel strip; it must be wide enough for a car.
- median: non-driving longitudinal strip separating roadway uses or directions.
- bicycle: dedicated bicycle travel strip. It must be much narrower than the adjacent motor lane or show a marking that clearly reads as a bicycle. 
- parking: dedicated parking/standing strip. Typically has painted stall markings, or many vehicles standing along the strip rather than moving through it.
- buffer: other clearly non-motor strip, such as a gore, channelization strip, or shoulder-like space.

Classify areas by roadway function and strip width, not temporary occupancy. A vehicle on a non-lane strip does not make it a lane; a bicycle/motorcycle in a motor lane does not make it a bicycle strip.
"""

LANE_CUES_BLOCK = """\
## Lane cues
Lane cues are directional observations on a motor lane. A cue is only the marking or heading that determines that lane's travel direction, not every marking or vehicle.
- arrow: pavement-painted lane-use turn arrow belonging to this lane. Combined codes (LS, SR, LSR, LR) mean one marking with multiple painted heads; L/R is a single hooked turn. Strongest travel cue. If the marking is partly covered, still report its true heads from visible fragments and continuation; do not omit it or drop a head just because it is occluded.
- road_marking: in-lane painted word or symbol other than a turn arrow. Identify what the marking is and, when no arrow is available, use that reading and how it is oriented to judge travel direction.
- vehicle_heading: image-relative heading of a visible car, van, bus, or truck. Two-wheelers do not count. Used only when that lane has no arrow (or equivalent directional marking).

Arrow outranks road_marking, which outranks vehicle_heading. An arrow that already gives travel direction makes other cues on that lane redundant.

### Cue orientation
orientation is image-relative only:
- up = toward crop top
- down = toward crop bottom
"""

OCCLUSION_BLOCK = """\
## Occlusion
Areas, separators, and lane cues may be locally hidden by vehicles, trees, or shadows. Infer each element from visible fragments and the strip's continuation; a covered segment is not a missing area, unmarked paint, or a missing cue. A partly covered arrow is still an arrow: report its true orientation and full movement code from what remains visible.
"""

ORGANIZER_ONTOLOGY_BLOCK = """\
## Vocabulary
The given visual identification is the left-to-right mouth sequence. Do not add, remove, or retype elements.

- lane: motor-vehicle travel strip. Only motor lanes count when judging opposite-direction streams.
- median is a non-driving longitudinal area; it is an opposing cut only when motor lanes exist on both sides.
- solid_line / double_solid_line / broken_line / unmarked_line are longitudinal cuts.
- edge is a physical internal cut.

Cue orientation is image-relative: up = toward the junction (incoming), down = toward the arm (outgoing).
"""

OPPOSING_SEPARATOR_BLOCK = """\
## Opposing separator
The unique cut between opposite-direction motor streams on this arm.
element_id must cite a median or line already in the given visual cross_section that has a motor lane on both sides, or be null (then set one_way_travel).
"""

IDENTIFIER_ROLE = (
    "Visual identification: read this arm crop as the global context."
)

IDENTIFIER_TASK = (
    "Report the left-to-right mouth sequence and the core cues that determine travel direction. "
    "If several vehicles in the same lane agree, report one."
)

EVALUATOR_ROLE = """\
Act as an independent visual judge of the proposed cross-section.
Do not regenerate an alternative cross-section.
Do not assume the proposal is correct merely because it is coherent or structurally valid.

Actively search for material visual evidence that could falsify the proposed interpretation,
especially omitted or merged motor lanes, false lane splits, unsupported separators,
and cue misreads.

Compare the proposal against the crop in both directions:
- every reported element should be supported by the image;
- every clearly supported roadway strip or longitudinal cut should be explained by the proposal.

Priority: motor-lane count and each reported cue first.
A cue is an arrow, road_marking, or vehicle_heading; check kind, orientation (up/down),
and for an arrow the painted heads. 

If multiple observations are consequences of the same structural problem,
report the structural problem rather than several lower-level critiques.
"""

INSPECTOR_ROLE = (
    "Inspect a local span independently. "
    "The evaluator's hypothesis is only a challenge and may be rejected. "
    "confirmed=true only when this crop itself shows the evidence the challenge needs; "
    "otherwise confirmed=false and empty items. "
    "Always output reasoning, including when confirmed=false. "
    "Do not edit items outside the given span. "
    "Internal physical cuts are type=edge, never boundary."
)

ORGANIZER_ROLE = (
    "Organization: set opposing_separator and any image-grounded arrow_updates "
    "needed to satisfy the structural constraints."
)


def evaluator_kinds_block():
    kinds = ", ".join(CRITIC_KINDS)
    return (
        f"Allowed critic kinds: {kinds}.\n"
        "- possible_missing_split: one reported area may contain multiple distinct areas because one or more longitudinal splits are missing. Challenge only if a clearly visible painted or physical cut and two distinct motor-width strips are both in the image. \n"
        "- possible_extra_split: adjacent reported lanes may actually be one strip (a false extra separator). Box both lanes and the disputed separator.\n"
        "- possible_wrong_separator: a reported separator type or color does not match the pixels. Tight box on that separator.\n"
        "- possible_cue_misread: an arrow, road_marking, or vehicle_heading was misidentified — wrong kind, orientation (up/down), or arrow code; reported where none exists; or a visible cue was omitted. An omitted cue is added; do not drop other kinds. Tight box on that cue.\n"
        "- possible_wrong_area_type: lane / parking / bicycle / median / buffer looks misclassified. Challenge only with clear visual evidence for the new type, not resemblance or occupancy alone. Tight box on that area.\n\n"
        "For each critic give the contested span [replace_start, replace_end), "
        "the box described above, visual_anchor, and inspect_content. "
        "One critic, one problem. "
        "Return an empty critiques list only when you find no material reason to challenge the proposal."
    )


def lane_level_prompt(driving_side=None, channelized_features=None, arm_id=None):
    return _compose_prompt(
        IDENTIFIER_ROLE,
        _task_block(IDENTIFIER_TASK),
        ONTOLOGY_BLOCK,
        LANE_CUES_BLOCK,
        OCCLUSION_BLOCK,
    )


def evaluator_system_prompt():
    return _compose_prompt(
        EVALUATOR_ROLE,
        ONTOLOGY_BLOCK,
        LANE_CUES_BLOCK,
        OCCLUSION_BLOCK,
        evaluator_kinds_block(),
    )


def inspector_system_prompt():
    return _compose_prompt(
        INSPECTOR_ROLE,
        ONTOLOGY_BLOCK,
        LANE_CUES_BLOCK,
        OCCLUSION_BLOCK,
    )


def organization_system_prompt(driving_side=None, channelized_features=None, arm_id=None):
    return _compose_prompt(
        ORGANIZER_ROLE,
        ORGANIZER_ONTOLOGY_BLOCK,
        OPPOSING_SEPARATOR_BLOCK,
        _structural_constraints_block(
            driving_side,
            channelized_features=channelized_features,
            arm_id=arm_id,
        ),
    )


def _lane_level_body(
    driving_side=None, channelized_features=None, arm_id=None, *,
    include_opposing_separator=True, task=None,
):
    """Full definition stack for ranking / callers that need every module."""
    parts = [
        _task_block(task or IDENTIFIER_TASK),
        ONTOLOGY_BLOCK,
        LANE_CUES_BLOCK,
        OCCLUSION_BLOCK,
    ]
    if include_opposing_separator:
        parts.append(OPPOSING_SEPARATOR_BLOCK)
        parts.append(
            _structural_constraints_block(
                driving_side,
                channelized_features=channelized_features,
                arm_id=arm_id,
            )
        )
    return _compose_prompt(*parts)


ORGANIZATION_TEXT = (
    "Return opposing_separator and any image-grounded arrow_updates for this crop "
    "and the visual identification below.\n\n"
    "{violations_block}"
    "Visual identification:\n{reading}"
)


LANE_LEVEL_PROMPT = lane_level_prompt("left")

ARM_USER_TEXT = (
    "The junction is at the crop top and the arm extends toward the crop bottom."
)

EVALUATOR_USER_TEXT = (
    "Evaluate this visual identification against the arm crop.\n"
    "Visual identification:\n{reading}"
)

INSPECT_CHALLENGE_LINE = (
    "evaluator_challenge={inspect_content}. visual_anchor={visual_anchor}.\n"
)

INSPECT_PATCH_KIND_TEXT = {
    "possible_missing_split": (
        "Decide whether this span is one area or several with the missing separator(s). "
        "confirmed=true only if the cut is clearly painted or physical and both strips "
        "are distinct motor lanes; if the cut is ambiguous, keep one area. "
        + INSPECT_CHALLENGE_LINE
    ),
    "possible_extra_split": (
        "Decide whether adjacent reported lanes in this span are one motor strip or two. "
        "This crop must show both strips and the disputed separator. "
        + INSPECT_CHALLENGE_LINE
    ),
    "possible_wrong_separator": (
        "Decide the correct separator type and color in this span. "
        "This crop must show that separator. "
        + INSPECT_CHALLENGE_LINE
    ),
    "possible_cue_misread": (
        "Decide the correct cue(s) on this span: kind (arrow / road_marking / "
        "vehicle_heading), orientation, and arrow movements if the cue is an arrow. "
        "Orientation is image-relative (up=crop top, down=crop bottom) and must match "
        "the way the arrow, marking, or vehicle points on this crop. "
        "This crop must show the contested cue. "
        "Keep other reported cues on this lane unless this crop shows they are wrong; "
        "adding a missing cue does not replace a different kind. "
        + INSPECT_CHALLENGE_LINE
    ),
    "possible_wrong_area_type": (
        "Decide the correct area type in this span. "
        "This crop must show the contested strip. "
        + INSPECT_CHALLENGE_LINE
    ),
}

INSPECT_PATCH_FALLBACK_TEXT = (
    "Decide the correct local decomposition of this span. "
    + INSPECT_CHALLENGE_LINE
)

INSPECT_CONFLICT_GROUP_TEXT = (
    "Overlapping critiques form one conflict group. "
    "Read the entire union span as one local decomposition.\n"
)

INSPECT_PATCH_FOOTER = (
    "Copy region_id={region_id}. Keep kind={kind}. "
    "Replacement items must themselves alternate validly and splice into the mouth sequence. "
    "Use type=boundary only if this span already includes a mouth-end boundary.\n"
)


def through_road_arm_ids(through_roads):
    ids = set()
    for item in through_roads or []:
        for arm_id in item.get("arm_ids") or []:
            ids.add(int(arm_id))
    return ids


def arm_on_through_road(arm_id, through_roads):
    return int(arm_id) in through_road_arm_ids(through_roads)


def arm_user_text(arm, channelized_features=None, through_roads=None, driving_side=None):
    return ARM_USER_TEXT


def organization_text(
    cross_section, *, opposing_separator=None, issues=None,
    channelized_features=None, through_roads=None, driving_side=None, arm_id=None,
):
    issue_lines = [issue for issue in (issues or []) if issue]
    violations_block = (
        "Violations:\n" + "\n".join(f"- {issue}" for issue in issue_lines) + "\n\n"
        if issue_lines else
        ""
    )
    return ORGANIZATION_TEXT.format(
        violations_block=violations_block,
        reading=json.dumps(
            _prompt_reading(
                cross_section, opposing_separator,
                hide_travel=True, hide_opposing=True, with_index=True,
            ),
            ensure_ascii=False,
            indent=2,
        ),
    )


def evaluator_text(cross_section):
    return EVALUATOR_USER_TEXT.format(
        reading=json.dumps(
            _prompt_reading(cross_section, None, hide_travel=True, hide_opposing=True),
            ensure_ascii=False,
            indent=2,
        ),
    )


def _prompt_cue(cue):
    """Prompt-facing cue without prior free-form reasoning that could anchor a later pass."""
    entry = {
        "kind": cue.get("kind"),
        "orientation": cue.get("orientation"),
    }
    if cue.get("kind") == "arrow":
        entry["movements"] = cue.get("movements")
    return entry


def _prompt_item(item, idx=None, *, hide_travel=False):
    """Prompt-facing structural item without prior evidence/reasoning/crop metadata."""
    item_type = item.get("type")
    if item_type == "lane":
        entry = {"type": "lane"}
        if not hide_travel:
            entry["travel"] = item.get("travel")
        entry["cues"] = [_prompt_cue(cue) for cue in (item.get("cues") or [])]
    else:
        entry = {"type": item_type}
        if item.get("element_id"):
            entry["element_id"] = item.get("element_id")
        if item_type in PAINTED_LINE_TYPES or item_type in UNPAINTED_SEPARATOR_TYPES:
            entry["color"] = item.get("color")
    if idx is not None:
        entry = {"index": idx, **entry}
    return entry


def _prompt_cross_section(cross_section, *, hide_travel=False, with_index=False):
    return [
        _prompt_item(item, idx if with_index else None, hide_travel=hide_travel)
        for idx, item in enumerate(cross_section or [])
    ]


def _prompt_opposing_separator(opposing_separator):
    if opposing_separator is None:
        return None
    if isinstance(opposing_separator, dict):
        return {
            "element_id": opposing_separator.get("element_id"),
            "one_way_travel": opposing_separator.get("one_way_travel"),
        }
    return opposing_separator


def _prompt_reading(cross_section, opposing_separator=None, *, hide_travel=False, hide_opposing=False, with_index=False):
    reading = {
        "cross_section": _prompt_cross_section(
            cross_section, hide_travel=hide_travel, with_index=with_index,
        ),
    }
    if not hide_opposing:
        reading["opposing_separator"] = _prompt_opposing_separator(opposing_separator)
    return reading


def _span_summaries(cross_section, spans, *, hide_travel=True):
    summaries = []
    for span in spans or []:
        start = int(span["replace_start"])
        end = int(span["replace_end"])
        items = [
            _prompt_item(cross_section[idx], idx, hide_travel=hide_travel)
            for idx in range(start, end)
        ]
        entry = {
            "region_id": span.get("region_id"),
            "kind": span.get("kind"),
            "items": items,
        }
        if span.get("inspect_content"):
            entry["inspect_content"] = span["inspect_content"]
        summaries.append(entry)
    return summaries


def inspect_patch_text(cross_section, crop):
    kind = crop.get("kind") or ""
    members = crop.get("members") or []
    fields = dict(
        region_id=crop.get("region_id") or "",
        kind=kind,
        visual_anchor=crop.get("visual_anchor") or "",
        inspect_content=crop.get("inspect_content") or crop.get("note") or "",
    )
    footer = INSPECT_PATCH_FOOTER.format(**fields)
    union = _span_summaries(cross_section, [crop], hide_travel=True)
    if len(members) > 1:
        member_spans = _span_summaries(cross_section, members, hide_travel=True)
        return (
            INSPECT_CONFLICT_GROUP_TEXT
            + footer
            + "Union span:\n"
            + json.dumps(union, ensure_ascii=False, indent=2)
            + "\nMember critiques:\n"
            + json.dumps(member_spans, ensure_ascii=False, indent=2)
        )
    body = INSPECT_PATCH_KIND_TEXT.get(kind) or INSPECT_PATCH_FALLBACK_TEXT
    return (
        body.format(**fields)
        + footer
        + "Contested span:\n"
        + json.dumps(union, ensure_ascii=False, indent=2)
    )
