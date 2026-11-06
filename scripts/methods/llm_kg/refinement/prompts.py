"""Prompts for lane-level refinement (layer1 re-identify / layer2 fix)."""


THROUGH_ROAD_REIDENTIFY_TEXT = (
    "Layer-1 through-road CHECK (not a hard constraint).\n"
    "A mismatch here is only a consistency check between THIS arm and its through-road "
    "peer. It does not force THIS arm to change.\n"
    "\n"
    "Current THIS-arm interpretation:\n"
    "{this_summary}\n"
    "\n"
    "Current peer-arm interpretation:\n"
    "{peer_summary}\n"
    "\n"
    "These two current interpretations are inconsistent with a through-road pair "
    "(opposing traffic and/or lane counts). Either arm may have been misidentified, "
    "or this check may be a false alarm.\n"
    "\n"
    "Re-identify THIS arm from the extended image. This pass is visual identification "
    "only: report the left-to-right mouth sequence and lane cues. Do not identify "
    "opposing_separator or lane.travel. Use the peer result only as supporting context. "
    "Report what the pixels show. Do not rewrite THIS arm just to make the check pass "
    "if the image does not support it."
)

INTER_JUNCTION_REIDENTIFY_TEXT = (
    "Layer-3 HARD constraint: connecting-road mouths must match.\n"
    "THIS arm and the far-end arm are the two mouths of one connecting roadway "
    "between two junctions. They must have the same one-way vs two-way structure, "
    "and complementary motor-lane counts: THIS outgoing equals the far-end incoming, "
    "and THIS incoming equals the far-end outgoing. Incoming/outgoing are relative "
    "to each mouth's own junction, so a one-way connector is incoming at one end "
    "and outgoing at the other — never incoming at both. Scene feasibility will "
    "later enforce this as a hard constraint, so an inconsistent reading cannot stand.\n"
    "\n"
    "Current THIS-arm interpretation:\n"
    "{this_summary}\n"
    "\n"
    "Current far-end arm interpretation (other junction):\n"
    "{peer_summary}\n"
    "\n"
    "These two current interpretations of the same roadway are inconsistent "
    "(opposing traffic and/or lane counts). Re-identify THIS arm from the extended "
    "image as the other mouth of that same road. This pass is visual identification "
    "only: report the left-to-right mouth sequence and lane cues. Do not identify "
    "opposing_separator or lane.travel. Use the far-end result as supporting context "
    "for the same roadway, but invert travel relative to THIS junction. "
    "If the far-end mouth is two-way and THIS mouth is one-way, complete THIS "
    "mouth to the complementary two-way cross-section when the pixels support it. "
    "Do not collapse a two-way mouth to one-way merely to copy a one-way far end "
    "unless THIS image clearly shows a single travel direction. "
    "Prefer a visually supported complementary reading over copying the far end's "
    "incoming/outgoing labels."
)

MEDIAN_THROUGH_ONE_WAY_REIDENTIFY_TEXT = (
    "Layer-3 HARD constraint: median-linked through carriageway must be one-way.\n"
    "THIS arm is the through roadway of a junction that is joined to another "
    "junction by a median (non-driving strip). The two junctions sit on opposite "
    "carriageways of one divided road. Those two one-way carriageways will later "
    "be merged into a single two-way road, with the median occupying the center "
    "lane. Therefore THIS arm must have no opposing traffic.\n"
    "\n"
    "Current THIS-arm interpretation:\n"
    "{this_summary}\n"
    "\n"
    "Current opposite through-arm interpretation (same junction):\n"
    "{peer_summary}\n"
    "\n"
    "If THIS arm currently has opposing traffic, it was misread as two-way. "
    "Re-identify THIS arm from the extended image as one carriageway of a "
    "divided road: all motor lanes travel the same way. The peer through-arm "
    "should be the complementary one-way end of the same carriageway "
    "(one incoming-only, the other outgoing-only). Report what the pixels show."
)

CONNECTIVITY_REIDENTIFY_TEXT = (
    "Layer-2 HARD connectivity constraint.\n"
    "Every incoming lane must have an outflow and every outgoing lane must have an "
    "inflow. Two incoming lanes of the same arm must not share one destination "
    "lane; incoming lanes from different arms may merge onto the same outgoing "
    "lane. The current junction violates this constraint, so at least one involved "
    "arm may be misread.\n"
    "\n"
    "Validation error:\n"
    "{error_text}\n"
    "\n"
    "Current THIS-arm interpretation:\n"
    "{this_summary}\n"
    "\n"
    "Related-arm interpretations:\n"
    "{related_summary}\n"
    "\n"
    "Re-examine THIS arm from the original mouth crop. This pass is visual identification "
    "only: report the left-to-right mouth sequence and lane cues. Do not identify "
    "opposing_separator or lane.travel. Use the validation error and related-arm results "
    "only to locate a possible misread, including an omitted motor lane or incorrect "
    "outer boundary. When the error involves a missing incoming or outgoing stream, "
    "explicitly check both outer sides of the current cross-section for an omitted motor "
    "lane before accepting the outer boundaries. Change THIS arm only if its pixels "
    "support a different interpretation; otherwise keep the current reading."
)


def through_road_reidentify_user_text(this_summary, peer_summary, *, arm_features=""):
    return THROUGH_ROAD_REIDENTIFY_TEXT.format(
        this_summary=this_summary,
        peer_summary=peer_summary,
    )


def inter_junction_reidentify_user_text(this_summary, peer_summary, *, arm_features=""):
    return INTER_JUNCTION_REIDENTIFY_TEXT.format(
        this_summary=this_summary,
        peer_summary=peer_summary,
    )


def median_through_one_way_reidentify_user_text(this_summary, peer_summary, *, arm_features=""):
    return MEDIAN_THROUGH_ONE_WAY_REIDENTIFY_TEXT.format(
        this_summary=this_summary,
        peer_summary=peer_summary,
    )


def connectivity_reidentify_user_text(this_summary, related_summary, *, error_text="", arm_features=""):
    return CONNECTIVITY_REIDENTIFY_TEXT.format(
        this_summary=this_summary,
        related_summary=related_summary,
        error_text=error_text or "- (none)",
    )
