NORMALIZED_XY_SCHEMA = {
    "type": "array",
    "items": {"type": "number"},
    "minItems": 2,
    "maxItems": 2,
}

CANDIDATE_SCHEMA = {
    "type": "object",
    "description": "One plausible junction candidate in the full scene.",
    "properties": {
        "id": {
            "type": "integer",
            "description": "Unique candidate identifier. IDs carry no ordering meaning.",
        },
        "center": {
            **NORMALIZED_XY_SCHEMA,
            "description": "Approximate candidate center in full-image normalized coordinates.",
        },
        "evidence": {
            "type": "string",
            "description": "Brief visible reason this location is a plausible motor-vehicle junction candidate.",
        },
    },
    "required": ["id", "center", "evidence"],
    "additionalProperties": False,
}

JUNCTION_CANDIDATE_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": CANDIDATE_SCHEMA,
            "minItems": 1,
            "description": "All plausible junction candidate centers in the full image.",
        },
    },
    "required": ["candidates"],
    "additionalProperties": False,
}

JUNCTION_VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["confirmed", "not_junction"],
            "description": (
                "Whether the indicated local candidate is a true road-network junction. "
                "Confirm only when at least three persistent road-network arms are visible; "
                "driveways and property/service accesses do not count as arms."
            ),
        },
        "center": {
            **NORMALIZED_XY_SCHEMA,
            "description": (
                "If confirmed, the shared meeting-area center in crop-normalized coordinates; "
                "it may move away from the seed when the seed landed on an approach arm. "
                "If rejected, keep it near the indicated candidate location; it will be ignored."
            ),
        },
        "evidence": {
            "type": "string",
            "description": (
                "Brief visible reason for confirming or rejecting the candidate, including whether "
                "each counted branch visibly continues as a persistent road-network roadway rather "
                "than a driveway, parking/courtyard entrance, or property/service access."
            ),
        },
        "uncertainty": {
            "type": "string",
            "description": "Brief remaining ambiguity, or an empty string if none.",
        },
    },
    "required": ["status", "center", "evidence", "uncertainty"],
    "additionalProperties": False,
}

CONNECTION_TYPE_ROAD = "connecting_road"
CONNECTION_TYPE_MEDIAN = "median"
CONNECTION_TYPES = (CONNECTION_TYPE_ROAD, CONNECTION_TYPE_MEDIAN)

NON_DRIVING_LANE_TYPES = ("restricted", "median", "none")
DEFAULT_MEDIAN_LANE_TYPE = "restricted"


def normalize_connection_type(value):
    """Map a connection label to connecting_road (default) or median.

    Missing/unknown values keep the historical driving-connector behavior.
    """
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("median", "median_link", "parallel", "parallel_median"):
        return CONNECTION_TYPE_MEDIAN
    return CONNECTION_TYPE_ROAD


def normalize_lane_type(value, *, connection_type=None):
    """Lane type written onto a non-driving connector; driving connectors ignore this."""
    if normalize_connection_type(connection_type) != CONNECTION_TYPE_MEDIAN:
        return None
    key = str(value or "").strip().lower()
    if key in ("median", "restricted"):
        return DEFAULT_MEDIAN_LANE_TYPE
    if key in NON_DRIVING_LANE_TYPES:
        return key
    return DEFAULT_MEDIAN_LANE_TYPE


JUNCTION_CONNECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "junction_ids": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 2,
            "maxItems": 2,
            "description": "The two confirmed junction IDs of one direct link.",
        },
        "connection_type": {
            "type": "string",
            "enum": list(CONNECTION_TYPES),
            "description": (
                "connecting_road: a continuous motor-vehicle carriageway between the two "
                "shared meeting areas. median: a non-driving strip between the "
                "carriageways that hold the two junctions."
            ),
        },
        "evidence": {
            "type": "string",
            "description": "Visible reason for this link and its connection_type.",
        },
    },
    "required": ["junction_ids", "connection_type", "evidence"],
    "additionalProperties": False,
}

JUNCTION_CONNECTION_LEVEL_SCHEMA = {
    "type": "object",
    "properties": {
        "connections": {
            "type": "array",
            "items": JUNCTION_CONNECTION_SCHEMA,
            "description": "All direct links among the fixed confirmed junctions.",
        },
        "connections_evidence": {
            "type": "string",
            "description": "Short summary of the connection decision.",
        },
    },
    "required": ["connections", "connections_evidence"],
    "additionalProperties": False,
}