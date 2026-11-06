CANDIDATE_USER_TEXT = (
    "This is the full top-down aerial image of a road network. "
    "Locate every distinct motor-vehicle junction candidate by the center of its shared meeting area. "
    "Do not determine junction boxes or inter-junction connections."
)

CANDIDATE_PROMPT = """\
Locate every plausible motor-vehicle road-junction candidate in this full top-down aerial image.

## Junction candidate
A candidate is one local region that may contain one shared motor-vehicle connection/conflict area:
an intersection, T/Y junction, roundabout, merge, or diverge.

Each physical shared area is exactly one candidate. Its CENTER is the approximate midpoint of that
shared meeting pavement — the place where the arms join.

A second point on an approaching arm, mouth, crosswalk, stop line, or another overlapping view of
the same meeting area is the same candidate, not a new one. Two nearby roadway openings that belong
to the same shared area are the same candidate.

## Task
This stage is high-recall localization of distinct shared areas. False positives such as a parking
apron are acceptable; they are checked in a later local crop. Duplicate coverage of one physical
junction is not acceptable.

Use motor-vehicle pavement continuity, curbs, lane markings, stop lines, visible vehicles, and
roadway openings as evidence.

CENTER is full-image normalized [x, y]; x increases rightward and y increases downward.
This stage does not output a junction box or inter-junction connections.
"""

CANDIDATE_DEDUP_USER_TEXT = (
    "Each physical shared motor-vehicle connection/conflict area must appear exactly once. "
    "If two or more of your candidates cover the same junction — for example one center on an "
    "approach arm and another on the meeting pavement, or two overlapping views of the same T/Y — "
    "keep a single candidate and place its center on that shared meeting pavement. "
    "Keep candidates that belong to distinct nearby junctions with separate shared areas. "
    "Return the merged candidate list."
)

VERIFY_USER_TEXT_TEMPLATE = """\
This is a fixed-size square crop around candidate junction {candidate_id}.
The seed center is near crop-normalized [{cx:.4f}, {cy:.4f}]. That seed is approximate and may lie
on an approach arm rather than on the shared meeting pavement.
Independently decide from this crop whether a motor-vehicle junction associated with the seed is present.
If it does, place the refined center on the shared meeting pavement in THIS CROP's normalized coordinates.
"""

VERIFY_PROMPT = """\
Verify whether this local top-down crop contains a motor-vehicle road junction associated with the
indicated candidate.

## Junction
Definition: A junction is one shared motor-vehicle connection/conflict area where at least three
roadway arms physically meet, merge, or diverge. Ordinary intersections, T/Y junctions, roundabouts,
and genuine motor-vehicle merges/diverges are junctions.

## Arm
Definition: An arm is one persistent road-network roadway that leaves the shared meeting pavement as
its own continuing street corridor. It must remain an independent motor-vehicle roadway as it extends
away from the meeting area, with persistent roadway edges or curbs and ordinary street carriageway
geometry. Opposite directions of one physical approach are one arm. A narrow local street can be an
arm only when it clearly continues as such a road-network corridor.

Do not count a driveway, building/service access, parking or courtyard entrance, or property-access
alley as an arm merely because it is paved, vehicle-width, or physically reachable by a car. If a
narrow corridor mainly enters or serves adjacent property and does not clearly continue as an
independent road-network roadway, reject it as an arm.

## Seed location
The supplied candidate center is only an approximate seed. The shared meeting area may lie elsewhere
in this crop, including toward an edge, when the seed landed on an approach arm of that same junction.

## Decision
Trace the road-network roadway away from each apparent opening before counting it as an arm. Confirm
only when a shared meeting area of three or more persistent road-network arms is visibly supported
and belongs to the seeded candidate, including when the seed sits on one of those arms. Then place
CENTER on that shared meeting pavement.

STATUS is confirmed when that junction is present; otherwise not_junction. When the possible third
arm could instead be a driveway, property access, courtyard/parking entrance, or short service alley
and its continuation as a road-network roadway is not clearly visible, use not_junction.

Use only visible evidence in this crop; do not assume motor-vehicle use from width or pavement alone.

## Output
- STATUS: "confirmed" when the crop contains that junction; otherwise "not_junction".
- CENTER: if confirmed, the geometric center of the shared motor-vehicle meeting area in THIS crop.
  If rejected, keep the center near the seed; it will be ignored.
- EVIDENCE: briefly explain the visible reason.
- UNCERTAINTY: remaining ambiguity, or an empty string.

The final junction box is constructed later from the refined center. All coordinates are normalized
[x, y] in THIS SQUARE CROP, with x rightward and y downward.
"""

CONNECTION_PROMPT = """\
Determine direct links among the confirmed junctions in this full top-down image.
Use only the supplied junction IDs and centers.

A direct link is two confirmed junctions joined by one connecting_road or one median whose
joining element is a single span from one shared meeting area to the other.

- connecting_road: a continuous motor-vehicle carriageway between the two shared meeting areas.
- median: a non-driving strip (median, island, or verge) between the carriageways that hold the
  two junctions. Longitudinal stagger along that strip is included.

Report every such pair with its connection_type.
"""


def build_connection_user_text(junctions):
    lines = [
        "This is the original full top-down aerial image.",
        "The following junctions have already been locally verified and are fixed:",
    ]
    for junction in junctions:
        cx, cy = junction["center"]
        lines.append(
            f'- junction {int(junction["id"])} center = '
            f'[{float(cx):.6f}, {float(cy):.6f}]'
        )
    lines.append(
        "Using the full image and only these fixed centers, report every directly joined pair "
        "and its connection_type."
    )
    return "\n".join(lines)