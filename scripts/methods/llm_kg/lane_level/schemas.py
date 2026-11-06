BOUNDARY_TYPE = "boundary"
BOUNDARY_TYPES = [BOUNDARY_TYPE]
LINE_TYPES = ["solid_line", "double_solid_line", "broken_line", "unmarked_line"]
PAINTED_LINE_TYPES = ["solid_line", "double_solid_line", "broken_line"]
LINE_COLORS = ["white", "yellow"]
UNPAINTED_SEPARATOR_TYPES = ["unmarked_line", "edge"]
SEPARATOR_TYPES = LINE_TYPES + ["edge"]
LANE_TYPE = "lane"
NON_LANE_AREA_TYPES = ["median", "buffer", "bicycle", "parking"]
AREA_TYPES = [LANE_TYPE] + NON_LANE_AREA_TYPES
NON_LANE_ITEM_TYPES = BOUNDARY_TYPES + SEPARATOR_TYPES + NON_LANE_AREA_TYPES

ORIENTATIONS = ["up", "down"]
TRAVELS = ["incoming", "outgoing"]
MOVEMENT_CODES = ["L", "R", "S", "LS", "SR", "LSR", "LR"]

CRITIC_KINDS = [
    "possible_missing_split",
    "possible_extra_split",
    "possible_wrong_separator",
    "possible_cue_misread",
    "possible_wrong_area_type",
]


def movement_codes(*, allow_lr):
    if allow_lr:
        return list(MOVEMENT_CODES)
    return [code for code in MOVEMENT_CODES if code != "LR"]


def movement_code_description(*, allow_lr):
    """Schema text: each code is one painted marking; letters are its heads."""
    parts = [
        "One code per painted marking, ranked by descending confidence. Empty only if no "
        "painted arrow is present. A partly covered arrow still needs its true heads, "
        "inferred from visible fragments and the marking's continuation. "
        "A code names the painted heads on that marking, not the shaft. "
        "L: left-turn head only. R: right-turn head only. S: through head only. "
        "LS: left-turn head plus through head. SR: through head plus right-turn head. "
        "LSR: left-turn, through, and right-turn heads on one marking."
    ]
    if allow_lr:
        parts.append(" LR: left-turn and right-turn heads, no through head.")
    return "".join(parts)


def _nullable_string_enum(values, *, description):
    """Nullable string enum accepted by Anthropic structured output."""
    return {
        "anyOf": [
            {"type": "string", "enum": list(values)},
            {"type": "null"},
        ],
        "description": description,
    }


def _orientation_field():
    return {
        "type": "string",
        "enum": list(ORIENTATIONS),
        "description": "Image-relative cue orientation: up=crop top, down=crop bottom.",
    }


def _evidence_field():
    return {"type": "string", "description": "Brief visual evidence."}


def _reasoning_field():
    return {"type": "string", "description": "Brief visual reason."}


def arrow_cue_schema(*, allow_lr):
    codes = movement_codes(allow_lr=allow_lr)
    return {
        "type": "object",
        "description": "Pavement-painted lane-use turn arrow associated with this motor lane; exclude trajectory/path arrows, crash annotations, overlays, labels, and other non-pavement graphics.",
        "properties": {
            "kind": {"type": "string", "enum": ["arrow"]},
            "orientation": _orientation_field(),
            "movements": {
                "type": "array",
                "items": {"type": "string", "enum": codes},
                "description": movement_code_description(allow_lr=allow_lr),
            },
            "reasoning": _reasoning_field(),
        },
        "required": ["kind", "orientation", "movements", "reasoning"],
        "additionalProperties": False,
    }


def road_marking_cue_schema():
    return {
        "type": "object",
        "description": "Visible in-lane pavement word or symbol other than a turn arrow; identify what the marking is.",
        "properties": {
            "kind": {"type": "string", "enum": ["road_marking"]},
            "orientation": _orientation_field(),
            "reasoning": _reasoning_field(),
        },
        "required": ["kind", "orientation", "reasoning"],
        "additionalProperties": False,
    }


def vehicle_heading_cue_schema():
    return {
        "type": "object",
        "description": "Image-relative heading of a visible car/van/bus/truck; exclude two-wheelers.",
        "properties": {
            "kind": {"type": "string", "enum": ["vehicle_heading"]},
            "orientation": _orientation_field(),
            "reasoning": _reasoning_field(),
        },
        "required": ["kind", "orientation", "reasoning"],
        "additionalProperties": False,
    }


def cue_schema(*, allow_lr):
    return {
        "description": "One lane cue: arrow, road_marking, or vehicle_heading.",
        "anyOf": [
            arrow_cue_schema(allow_lr=allow_lr),
            road_marking_cue_schema(),
            vehicle_heading_cue_schema(),
        ],
    }


def painted_line_separator_item_schema():
    return {
        "type": "object",
        "description": "Painted longitudinal line.",
        "properties": {
            "type": {"type": "string", "enum": PAINTED_LINE_TYPES},
            "color": {"type": "string", "enum": LINE_COLORS, "description": "white or yellow."},
            "evidence": _evidence_field(),
        },
        "required": ["type", "color", "evidence"],
        "additionalProperties": False,
    }


def _unpainted_color_field():
    return {"type": "null", "description": "Always null (no paint)."}


def unmarked_line_separator_item_schema():
    return {
        "type": "object",
        "description": "Inferred longitudinal separator with no paint.",
        "properties": {
            "type": {"type": "string", "enum": ["unmarked_line"]},
            "color": _unpainted_color_field(),
            "evidence": _evidence_field(),
        },
        "required": ["type", "color", "evidence"],
        "additionalProperties": False,
    }


def line_separator_item_schema():
    return {
        "anyOf": [
            painted_line_separator_item_schema(),
            unmarked_line_separator_item_schema(),
        ],
        "description": "Longitudinal line separator.",
    }


def boundary_item_schema():
    return {
        "type": "object",
        "description": "Outer physical edge of the mouth.",
        "properties": {
            "type": {"type": "string", "enum": BOUNDARY_TYPES},
            "evidence": _evidence_field(),
        },
        "required": ["type", "evidence"],
        "additionalProperties": False,
    }


def edge_separator_item_schema():
    return {
        "type": "object",
        "description": "Internal physical edge.",
        "properties": {
            "type": {"type": "string", "enum": ["edge"]},
            "color": _unpainted_color_field(),
            "evidence": _evidence_field(),
        },
        "required": ["type", "color", "evidence"],
        "additionalProperties": False,
    }


def separator_item_schema():
    return {
        "anyOf": [
            painted_line_separator_item_schema(),
            unmarked_line_separator_item_schema(),
            edge_separator_item_schema(),
        ],
        "description": "Longitudinal separator.",
    }


def lane_item_schema(*, allow_lr, require_travel=True):
    properties = {
        "type": {"type": "string", "enum": [LANE_TYPE]},
        "cues": {
            "type": "array",
            "items": cue_schema(allow_lr=allow_lr),
            "description": "Cues on this lane. Empty if none.",
        },
    }
    required = ["type", "cues"]
    if require_travel:
        properties["travel"] = {
            "type": "string",
            "enum": list(TRAVELS),
            "description": "Traffic direction: incoming=toward crop top/junction; outgoing=toward crop bottom.",
        }
        required = ["type", "travel", "cues"]
    return {
        "type": "object",
        "description": "Motor-vehicle travel lane.",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _area_item_schema(area_type, description):
    return {
        "type": "object",
        "description": description,
        "properties": {
            "type": {"type": "string", "enum": [area_type]},
            "evidence": _evidence_field(),
        },
        "required": ["type", "evidence"],
        "additionalProperties": False,
    }


def median_item_schema():
    return _area_item_schema("median", "Non-driving strip between travel uses.")


def buffer_item_schema():
    return _area_item_schema("buffer", "Non-motor strip that is not median, bicycle, or parking.")


def bicycle_item_schema():
    return _area_item_schema("bicycle", "Dedicated bicycle travel strip.")


def parking_item_schema():
    return _area_item_schema("parking", "Dedicated parking or standing strip.")


def non_lane_area_item_schema():
    return {
        "anyOf": [
            median_item_schema(),
            buffer_item_schema(),
            bicycle_item_schema(),
            parking_item_schema(),
        ],
        "description": "Non-motor strip: median, buffer, bicycle, or parking.",
    }


def opposing_separator_schema():
    return {
        "type": "object",
        "description": "Opposing-motor cut on this arm.",
        "properties": {
            "element_id": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "description": (
                    "Median or line element_id with a motor lane on both sides, "
                    "or null if one-way."
                ),
            },
            "one_way_travel": _nullable_string_enum(
                TRAVELS,
                description=(
                    "incoming or outgoing when element_id is null (one-way). "
                    "Null when a cut is set."
                ),
            ),
            "reasoning": _reasoning_field(),
        },
        "required": ["element_id", "one_way_travel", "reasoning"],
        "additionalProperties": False,
    }


def cross_section_item_schema(*, allow_lr, require_travel=True):
    return {
        "anyOf": [
            boundary_item_schema(),
            painted_line_separator_item_schema(),
            unmarked_line_separator_item_schema(),
            edge_separator_item_schema(),
            lane_item_schema(allow_lr=allow_lr, require_travel=require_travel),
            median_item_schema(),
            buffer_item_schema(),
            bicycle_item_schema(),
            parking_item_schema(),
        ],
        "description": "One mouth element. Order: boundary, area, separator, ..., area, boundary.",
    }


def initial_result_schema(*, allow_lr):
    return {
        "type": "object",
        "description": "Visual identification: mouth cross-section and cues. No travel.",
        "properties": {
            "cross_section": {
                "type": "array",
                "items": cross_section_item_schema(allow_lr=allow_lr, require_travel=False),
            },
        },
        "required": ["cross_section"],
        "additionalProperties": False,
    }


def arrow_update_schema(*, allow_lr):
    codes = movement_codes(allow_lr=allow_lr)
    return {
        "type": "object",
        "description": "Image-grounded rewrite of arrow.movements on one existing arrow lane.",
        "properties": {
            "index": {
                "type": "integer",
                "description": "cross_section index of that lane.",
            },
            "movements": {
                "type": "array",
                "items": {"type": "string", "enum": codes},
                "description": movement_code_description(allow_lr=allow_lr),
            },
            "reasoning": _reasoning_field(),
        },
        "required": ["index", "movements", "reasoning"],
        "additionalProperties": False,
    }


def organization_result_schema(*, allow_lr):
    return {
        "type": "object",
        "description": (
            "Organization: opposing_separator, plus optional arrow movement updates. "
            "Do not add, remove, or retype elements."
        ),
        "properties": {
            "opposing_separator": opposing_separator_schema(),
            "arrow_updates": {
                "type": "array",
                "items": arrow_update_schema(allow_lr=allow_lr),
                "description": (
                    "Rewrite arrow.movements only when the crop's painted heads differ from "
                    "the reported code. Empty if none. Do not change kind or orientation."
                ),
            },
        },
        "required": ["opposing_separator", "arrow_updates"],
        "additionalProperties": False,
    }


def zoom_box_fields(*, crop_name="initial arm crop"):
    where = (
        f"Box in the {crop_name}, [0, 1]. "
        "For possible_missing_split / possible_extra_split: box the contested reported "
        "area(s) plus the hypothesized cut, including both adjacent motor strips — "
        "not a single paint stroke. "
        "For other kinds: tight box on the contested element only; do not include "
        "neighboring lanes. Code expands for context."
    )
    return {
        "x0": {"type": "number", "description": f"Top-left x. {where}"},
        "y0": {"type": "number", "description": f"Top-left y. {where}"},
        "x1": {"type": "number", "description": f"Bottom-right x. {where}"},
        "y1": {"type": "number", "description": f"Bottom-right y. {where}"},
    }


def evaluator_result_schema():
    critic = {
        "type": "object",
        "description": "One priority mismatch between a reported element and the crop.",
        "properties": {
            "kind": {
                "type": "string",
                "enum": list(CRITIC_KINDS),
                "description": (
                    "possible_missing_split: one reported area may need one or more splits "
                    "only if a clear painted or physical cut and two motor-width strips are visible. "
                    "possible_extra_split: adjacent reported lanes may be one strip; "
                    "box both lanes and the disputed separator. "
                    "possible_wrong_separator: a reported separator type or color looks wrong. "
                    "possible_cue_misread: an arrow, road_marking, or vehicle_heading was "
                    "misidentified (wrong kind, orientation up/down, or arrow code; "
                    "invented; or omitted). An omitted cue is added; do not drop other kinds. "
                    "possible_wrong_area_type: lane/parking/bicycle/median/buffer looks "
                    "misclassified; challenge only with clear evidence for the new type."
                ),
            },
            "replace_start": {
                "type": "integer",
                "description": "Inclusive start index in the given cross_section.",
            },
            "replace_end": {
                "type": "integer",
                "description": "Exclusive end index in the given cross_section.",
            },
            **zoom_box_fields(),
            "visual_anchor": {
                "type": "string",
                "description": (
                    "Location and visible geometry of the boxed region. "
                    "For a split critic, name the contested strip and the hypothesized cut."
                ),
            },
            "inspect_content": {
                "type": "string",
                "description": (
                    "A challenge hypothesis; the inspector determines the span independently "
                    "and may reject it. For possible_cue_misread, name the hypothesized kind "
                    "(arrow, road_marking, or vehicle_heading), orientation (up or down), "
                    "and arrow movements if the cue is an arrow."
                ),
            },
            "reason": {
                "type": "string",
                "description": "Why this correspondence failure has this priority.",
            },
        },
        "required": [
            "kind", "replace_start", "replace_end",
            "x0", "y0", "x1", "y1",
            "visual_anchor", "inspect_content", "reason",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "description": "Priority mismatches where reported elements and the crop do not correspond. Empty if they do.",
        "properties": {
            "critiques": {"type": "array", "items": critic},
        },
        "required": ["critiques"],
        "additionalProperties": False,
    }


def inspect_patch_schema(*, allow_lr):
    patch = {
        "type": "object",
        "description": (
            "Independent local reading of one span or conflict-group union. "
            "Replace it only if this crop supports a different reading."
        ),
        "properties": {
            "region_id": {"type": "string", "description": "Input region_id (R1, or G1 for a conflict group)."},
            "confirmed": {
                "type": "boolean",
                "description": (
                    "True only if this crop shows the evidence needed to support the "
                    "evaluator challenge and the replacement differs. "
                    "False if the crop cannot support the issue or the current span is kept."
                ),
            },
            "items": {
                "type": "array",
                "items": cross_section_item_schema(allow_lr=allow_lr, require_travel=False),
                "description": (
                    "Replacement span if confirmed=true; empty if confirmed=false. "
                    "Internal physical cuts are type=edge. Use boundary only if this span includes a mouth end."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Why this crop confirms or rejects the challenge."
                ),
            },
        },
        "required": ["region_id", "confirmed", "items", "reasoning"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "description": "One span patch for the provided region_id.",
        "properties": {"patch": patch},
        "required": ["patch"],
        "additionalProperties": False,
    }
