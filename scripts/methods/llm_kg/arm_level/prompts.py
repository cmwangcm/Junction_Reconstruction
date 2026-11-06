from scripts.methods.connection_parser import channelized_turn_name, normalize_driving_side

FULL_DIAGRAM_USER_TEXT = (
    "This is the full top-down image of a single road junction."
)

_JUNCTION_DEFINITION = """\
## Junction
Definition: In this top-down image, the junction is the motor-vehicle pavement region where two or more roadways connect or interact, allowing traffic to transition between them. Its boundary is where an independent roadway arm begins to join, cross, merge, diverge, or otherwise interact with another roadway. Visible cues may include outer edges opening or tapering, lane lines breaking or merging, changes in lane organization, or separate roadway pavements beginning to join.
"""

_ARM_DEFINITION = """\
## Arms
Definition: An arm is one roadway branch extending outward from the junction in a distinct direction and forming one junction-side roadway connection. A divided roadway with opposite-direction carriageways separated by a median, island, verge, or other separator is still treated as ONE arm when those carriageways belong to the same branch and participate in the same junction connection. A carriageway that only passes beside the junction behind a continuous median or separator, without direct access to the joining road, does not belong to this junction arm.
"""

_JUNCTION_CENTER = """\
- JUNCTION CENTER: One coarse [x, y] reference point in full-image normalized coordinates [0, 1] (x rightward, y downward), placed within the junction region near the middle of the roadway interaction area. It is used only as a shared reference for arm localization and does not need to be the exact geometric center or lie on any arm centerline.
"""

_ARM_MOUTH_REFINE = """\
- TARGET ARM: This crop may show an entire through-road continuing through the junction into the upper half. Estimate geometry only for the arm in the LOWER half of the crop (the approach that runs downward from the junction). Do not report the opposite through-road arm in the upper half, or any other arm, sidewalk, plaza, or road in the padded crop.
- MOUTH: mouth_edges are two [x, y] points in this crop's normalized coordinates [0, 1]. They are the two outer boundary points of the target arm's motor-vehicle pavement that actually participates in this junction connection, at the same junction-entry cross-section. Prefer the cross-section indicated by a visible stop line or equivalent stopping/entry cue; if no such cue is clear, use the closest visually supported junction-entry cross-section where the arm transitions into the junction region. Do not cross a median, island, or other physical separator unless the roadway on the far side also connects to the current junction. Internal lane lines or separators between participating carriageways must not be used as mouth edges. The two points should span across the connected arm rather than along it. Do not place either point on sidewalk, plaza, overpass, another road, or gray padding.
"""

_ARM_HEADING = """\
- HEADING: outward centerline heading of this arm, away from the junction along the approach. 0°=right/East, 90°=up/North, 180°=left/West, 270°=down/South; angles increase counterclockwise.
"""

_ARM_HEADING_PRECISE = """\
- HEADING: report heading_offset_deg only — the signed counterclockwise tilt of this arm's motor centerline from the crop's downward axis. Measure painted lane lines or parallel curbs running along the arm. The crop is already rotated to a prior heading: 0 if those lines run straight down (junction at top). Positive = the centerline goes down-right; negative = down-left. Do not report an absolute image heading.
"""

_ARM_LENGTH_REFINE = """\
- LENGTH: length is the distance from the mouth center (the midpoint of mouth_edges) outward along the target arm centerline to the farthest visible extent of this arm within the original image or crop, reported as a fraction of the crop's height.
"""

_ARM_SECTION = (
    _JUNCTION_DEFINITION
    + "\n"
    + _ARM_DEFINITION
    + "\nRules:\n"
    + _JUNCTION_CENTER
    + _ARM_HEADING
)

_ARM_SECTION_REFINE = (
    _JUNCTION_DEFINITION
    + "\n"
    + _ARM_DEFINITION
    + "\nRules:\n"
    + _ARM_MOUTH_REFINE
    + _ARM_HEADING_PRECISE
    + _ARM_LENGTH_REFINE
)


def through_roads_section():
    return """\
## Through roads
Definition: A through road is one continuous straight roadway that passes through the junction. Its two ends are two distinct arms. A junction may have multiple such groups, for example two crossing through roads. Each arm belongs to at most one group.

Rules:
- CONTINUITY: aligned pavement, shared corridor, markings, or carriageway alignment through the junction.
"""


def channelized_islands_section(driving_side=None):
    turn_name = channelized_turn_name(normalize_driving_side(driving_side))
    return """\
## Channelized islands
Definition: A channelized feature is a dedicated early {turn_name} connector between two arms, separated from the ordinary junction path by a raised/curbed island or a dedicated painted/hatched channelizing area. Traffic uses it to turn {turn_name} early without entering the ordinary junction lanes. 

Rules:
- DEFAULT EMPTY: omit unless every criterion below is clearly satisfied. When uncertain, return [].
- ISLAND: a clearly identifiable channelizing separator or non-driving area near a junction corner between the ordinary junction path and an outer motor-vehicle path. It may be raised/curbed or painted/hatched/striped. A single edge line, ordinary curb, corner boundary, or isolated linework does not qualify.
- SEPARATE LANE: a distinct motor-vehicle path of ordinary carriageway pavement and width passes outside the channelizing area, while the ordinary junction path lies on its other side. Do not infer a channelized feature from a curved roadway edge, wide corner geometry, or hatched corner alone.
- CONNECTION: the outer path provides an early {turn_name} connection between the two adjacent arms. Clear fork or merge markings are not required when the channelizing area and motor-vehicle path visibly establish the connection.
""".format(turn_name=turn_name)


_ARM_LEVEL_INTRO = """\
Identify every connectable motor-vehicle roadway arm that belongs to the core road network of a top-down junction image; decide whether any through roads or channelized islands exist. Ignore peripheral minor roads. Report one junction_center for the whole image. For each arm report only heading_deg — do not report a seed, anchor, mouth edges, width, or length.

"""


_ARM_REFINE_BODY = """\
Identify this one arm's two mouth_edges, heading_offset_deg, and length from the provided heading-aligned crop. 

""" + _ARM_SECTION_REFINE


def arm_level_prompt(driving_side=None):
    side = normalize_driving_side(driving_side)
    return (
        _ARM_LEVEL_INTRO
        + _ARM_SECTION
        + "\n"
        + through_roads_section()
        + "\n"
        + channelized_islands_section(side)
    )


def arm_refine_prompt():
    return _ARM_REFINE_BODY


def arm_crop_user_text(arm_id):
    return (
        "This crop is for roadway arm %s in the LOWER half of the image. "
        % arm_id
    )


ARM_LEVEL_PROMPT = arm_level_prompt('left')
ARM_REFINE_PROMPT = arm_refine_prompt()