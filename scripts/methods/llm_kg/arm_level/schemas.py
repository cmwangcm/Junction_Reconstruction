NORMALIZED_XY_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
}

THROUGH_ROAD_SCHEMA = {
    "type": "object",
    "description": "One continuous roadway through the junction, as two opposite arms.",
    "properties": {
        "arm_ids": {
            "type": "array",
            "description": "Exactly two distinct endpoint arm IDs of this through road.",
            "items": {"type": "integer"},
        },
        "evidence": {
            "type": "string",
            "description": "One sentence naming the visible continuity through the junction.",
        },
    },
    "required": ["arm_ids", "evidence"],
    "additionalProperties": False,
}

CHANNELIZED_FEATURE_SCHEMA = {
    "type": "object",
    "description": "One dedicated early-turn connector between two adjacent arms.",
    "properties": {
        "arm_list": {
            "type": "array",
            "description": (
                "Exactly two distinct endpoint arm IDs; order does not matter. "
                "The pipeline orients the pair from headings."
            ),
            "items": {"type": "integer"},
        },
        "evidence": {
            "type": "string",
            "description": "One sentence naming the separator, outer motor path, and the same two arm IDs.",
        },
        "uncertainty": {
            "type": "string",
            "description": "One sentence on any ambiguity in the separator, outer path, or endpoint pair.",
        },
    },
    "required": ["arm_list", "evidence", "uncertainty"],
    "additionalProperties": False,
}

_HEADING_DEG_DESCRIPTION = (
    "Outward centerline heading, away from the junction along this arm. "
    "0°=right/East, 90°=up/North, 180°=left/West, 270°=down/South; counterclockwise."
)

_MOUTH_EDGES_CROP_DESCRIPTION = (
    "Exactly two [x, y] points in this crop's normalized coordinates [0, 1] "
    "(x=0 left, x=1 right, y=0 top, y=1 bottom). One point on each outer "
    "boundary of the target-arm motor-vehicle pavement that actually participates "
    "in the current junction connection, at the same junction-entry cross-section. "
    "Do not cross a median, island, or other physical separator unless the roadway "
    "on the far side also connects to the current junction. Prefer a visible stop "
    "line or equivalent stopping/entry cue; if none is clear, use the closest "
    "visually supported junction-entry cross-section."
)

_MOUTH_EDGES_FULL_IMAGE_DESCRIPTION = (
    "Exactly two [x, y] points in this full image's normalized coordinates [0, 1] "
    "(x=0 left, x=1 right, y=0 top, y=1 bottom). One point on each outer "
    "boundary of this arm's motor-vehicle pavement that actually participates "
    "in the current junction connection, at the same junction-entry cross-section. "
    "Do not cross a median, island, or other physical separator unless the roadway "
    "on the far side also connects to the current junction. Prefer a visible stop "
    "line or equivalent stopping/entry cue; if none is clear, use the closest "
    "visually supported junction-entry cross-section."
)

_MOUTH_EDGES_DESCRIPTION = _MOUTH_EDGES_FULL_IMAGE_DESCRIPTION

ARM_LOCALIZE_SCHEMA = {
    "type": "object",
    "description": "One roadway arm: heading_deg only.",
    "properties": {
        "id": {
            "type": "integer",
            "description": "Unique arm identifier. IDs carry no ordering meaning.",
        },
        "heading_deg": {
            "type": "number",
            "description": _HEADING_DEG_DESCRIPTION,
        },
        "evidence": {
            "type": "string",
            "description": "One sentence naming the visible roadway evidence for this arm.",
        },
        "uncertainty": {
            "type": "string",
            "description": "One sentence on any ambiguity in the arm identity or heading.",
        },
    },
    "required": ["id", "heading_deg", "evidence", "uncertainty"],
    "additionalProperties": False,
}

# Full-image mouth geometry (e2e and other one-shot readers).
ARM_SCHEMA = {
    "type": "object",
    "description": "One roadway arm with mouth_edges, heading_deg, and length.",
    "properties": {
        "id": ARM_LOCALIZE_SCHEMA["properties"]["id"],
        "heading_deg": {
            "type": "number",
            "description": _HEADING_DEG_DESCRIPTION,
        },
        "length": {
            "type": "number",
            "description": (
                "Distance from the mouth center (midpoint of mouth_edges) outward along "
                "this arm's centerline to its farthest visible extent, as a fraction of "
                "the image side."
            ),
        },
        "mouth_edges": {
            "type": "array",
            "items": NORMALIZED_XY_SCHEMA,
            "description": _MOUTH_EDGES_DESCRIPTION,
        },
        "evidence": ARM_LOCALIZE_SCHEMA["properties"]["evidence"],
        "uncertainty": {
            "type": "string",
            "description": "One sentence on any ambiguity in the arm geometry.",
        },
    },
    "required": ["id", "heading_deg", "length", "mouth_edges", "evidence", "uncertainty"],
    "additionalProperties": False,
}

ARM_REFINE_SCHEMA = {
    "type": "object",
    "description": "LOWER-half target arm in this crop: mouth_edges, heading_offset_deg, and length.",
    "properties": {
        "mouth_edges": {
            "type": "array",
            "items": NORMALIZED_XY_SCHEMA,
            "description": (
                "Exactly two [x, y] points in this crop's normalized coordinates [0, 1] "
                "(x=0 left, x=1 right, y=0 top, y=1 bottom). One point on each outermost "
                "persistent boundary of the complete target-arm mouth at the same "
                "junction-entry cross-section, including a channelized slip lane that "
                "belongs to this arm. Prefer a visible stop line or equivalent stopping/"
                "entry cue to locate this cross-section; if none is clear, use the closest "
                "visually supported junction-entry cross-section."
            ),
        },
        "heading_offset_deg": {
            "type": "number",
            "description": (
                "Signed counterclockwise tilt of this arm's motor centerline from the crop's "
                "downward axis. 0 if the centerline runs straight down; positive = down-right; "
                "negative = down-left. Not an absolute image heading."
            ),
        },
        "length": {
            "type": "number",
            "description": (
                "Distance from the mouth center (midpoint of mouth_edges) outward along "
                "the target arm centerline to its farthest visible extent in the original "
                "image or crop, as a fraction of this crop's height."
            ),
        },
        "evidence": ARM_LOCALIZE_SCHEMA["properties"]["evidence"],
        "uncertainty": {
            "type": "string",
            "description": "One sentence on any ambiguity in the refined mouth or heading.",
        },
    },
    "required": ["mouth_edges", "heading_offset_deg", "length", "evidence", "uncertainty"],
    "additionalProperties": False,
}

ARM_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "junction_center": {
            **NORMALIZED_XY_SCHEMA,
            "description": (
                "One [x, y] point in full-image normalized coordinates [0, 1] "
                "(x rightward, y downward) at the shared junction meeting area "
                "where the arms connect."
            ),
        },
        "through_roads": {
            "type": "array",
            "description": "Through-road groups; [] if none.",
            "items": THROUGH_ROAD_SCHEMA,
        },
        "through_roads_evidence": {
            "type": "string",
            "description": "If through_roads is empty, say why; otherwise summarize the grouping.",
        },
        "channelized_features": {
            "type": "array",
            "description": "Channelized features; default [].",
            "items": CHANNELIZED_FEATURE_SCHEMA,
        },
        "channelized_features_evidence": {
            "type": "string",
            "description": "If channelized_features is empty, say why; otherwise summarize the evidence.",
        },
        "arms": {
            "type": "array",
            "description": "Accepted arms in any order.",
            "items": ARM_LOCALIZE_SCHEMA,
        },
    },
    "required": [
        "junction_center",
        "through_roads",
        "through_roads_evidence",
        "channelized_features",
        "channelized_features_evidence",
        "arms",
    ],
    "additionalProperties": False,
}