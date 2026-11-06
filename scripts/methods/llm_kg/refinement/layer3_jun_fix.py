"""Layer 3: inter-junction HARD constraints + sequential re-identification.

Activates only when a scene has more than one junction.

``connecting_road`` links: the two mouths of one driving connector must agree
(one-way vs two-way, and complementary lane counts). This layer re-identifies
inconsistent ends; scene feasibility then enforces the same rule as a HARD
min-edit constraint. Assemble refuses to emit a mismatched connector.

``median`` links are not two mouths of one carriageway, so they are not
lane-balanced against each other. Instead each junction's through carriageway
along the median axis must be one-way: those two one-way roads later merge
into one two-way divided road with the median as the center lane. Scene
feasibility enforces that one-way rule as a HARD constraint; this layer
re-identifies two-way through arms first.
"""

from scripts.methods.llm_kg.junction_level.llm_gen import derive_inter_junction_roads
from scripts.methods.llm_kg.refinement.common import (
    ConnectionParseError,
    lanes_doc_from_ctx,
    layer_result,
    log_validation_error,
    make_issue,
    save_layer_result,
)
from scripts.methods.llm_kg.refinement.layer1_tr_check import (
    arm_interpretation_summary,
    arm_io_counts,
    _has_opposing_traffic,
    _lanes_from_arm,
)
from scripts.methods.llm_kg.refinement.loop import _refine_one_arm
from scripts.methods.llm_utils import BaseLLM, dump_json

# Same-road ends should agree exactly; |delta| > threshold fails the pair.
CHECK_DELTA = 0
KIND_CONNECTING_ROAD = "connecting_road"
KIND_MEDIAN_THROUGH = "median_through"
MEDIAN_THROUGH_AXIS_TOL_DEG = 25.0
CONSTRAINT_MEDIAN_THROUGH_ONE_WAY = "median_through_one_way"

_FINDING_PRIORITY = ("opposing_traffic", "lane_balance")
_FINDING_CONSTRAINT = {
    "opposing_traffic": "inter_junction_opposing_traffic",
    "lane_balance": "inter_junction_lane_balance",
}


def _primary_constraint(findings):
    kinds = {item["kind"] for item in (findings or [])}
    for kind in _FINDING_PRIORITY:
        if kind in kinds:
            return _FINDING_CONSTRAINT[kind]
    return "inter_junction_lane_balance"


def _pair_kind(item):
    return item.get("kind") or KIND_CONNECTING_ROAD


def _pair_label(left, right):
    return "j%s.arm%s/j%s.arm%s" % (
        left["junction"], left["arm"], right["junction"], right["arm"],
    )


def pair_log_label(item):
    kind = _pair_kind(item)
    left, right = item["left"], item["right"]
    if kind == KIND_MEDIAN_THROUGH:
        return "median_through %s" % _pair_label(left, right)
    return "connecting_road %s" % _pair_label(left, right)


def _pair_message(left, right, findings):
    kinds = [
        kind for kind in _FINDING_PRIORITY
        if any(item.get("kind") == kind for item in (findings or []))
    ]
    bits = []
    for kind in kinds:
        if kind == "opposing_traffic":
            bits.append("opposing traffic inconsistent (one-way vs two-way)")
        else:
            bits.append("same-road lane counts differ")
    body = "; ".join(bits) if bits else "far-end structure disagrees"
    return "Inter-junction arms %s: %s." % (_pair_label(left, right), body)


def _lane_of(end, lane_results_by_junction):
    return (lane_results_by_junction.get(int(end["junction"])) or {}).get(int(end["arm"]))


def _collect_pair_findings(left, right, *, left_opp, right_opp, left_counts, right_counts):
    findings = []
    extra = {}
    if left_opp != right_opp:
        findings.append({
            "kind": "opposing_traffic",
            "arm_opp": left_opp,
            "peer_opp": right_opp,
        })
        extra["arm_opposing_traffic"] = {
            "j%s.arm%s" % (left["junction"], left["arm"]): left_opp,
            "j%s.arm%s" % (right["junction"], right["arm"]): right_opp,
        }

    a_out, a_in = left_counts
    b_out, b_in = right_counts
    pairs = []
    # Same roadway: this outgoing meets the far-end incoming, and vice versa.
    if abs(a_out - b_in) > CHECK_DELTA:
        pairs.append({
            "travel": "outgoing", "count": a_out,
            "peer_travel": "incoming", "peer_count": b_in,
            "abs_delta": abs(a_out - b_in), "threshold": CHECK_DELTA,
        })
    if abs(a_in - b_out) > CHECK_DELTA:
        pairs.append({
            "travel": "incoming", "count": a_in,
            "peer_travel": "outgoing", "peer_count": b_out,
            "abs_delta": abs(a_in - b_out), "threshold": CHECK_DELTA,
        })
    if pairs:
        findings.append({"kind": "lane_balance", "pairs": pairs})
        extra["lane_counts"] = {
            "j%s.arm%s" % (left["junction"], left["arm"]): {
                "outgoing": a_out, "incoming": a_in,
            },
            "j%s.arm%s" % (right["junction"], right["arm"]): {
                "outgoing": b_out, "incoming": b_in,
            },
        }
    extra["discrepancies"] = [
        kind for kind in _FINDING_PRIORITY
        if any(item["kind"] == kind for item in findings)
    ]
    extra["reported_value"] = _pair_message(left, right, findings).split(": ", 1)[-1].rstrip(".")
    return findings, extra


def _end_counts(end, lane_results_by_junction):
    return arm_io_counts(_lanes_from_arm(None, _lane_of(end, lane_results_by_junction)))


def pair_inconsistent(left, right, lane_results_by_junction):
    findings, _extra = _collect_pair_findings(
        left, right,
        left_opp=_has_opposing_traffic(_lane_of(left, lane_results_by_junction)),
        right_opp=_has_opposing_traffic(_lane_of(right, lane_results_by_junction)),
        left_counts=_end_counts(left, lane_results_by_junction),
        right_counts=_end_counts(right, lane_results_by_junction),
    )
    return bool(findings)


def _median_end_inconsistent(end, lane_results_by_junction):
    arm = _lane_of(end, lane_results_by_junction)
    return arm is None or _has_opposing_traffic(arm)


def item_inconsistent(item, lane_results_by_junction):
    """True if this Layer-3 item currently violates its constraint."""
    left, right = item["left"], item["right"]
    if _pair_kind(item) == KIND_MEDIAN_THROUGH:
        return (
            _median_end_inconsistent(left, lane_results_by_junction)
            or _median_end_inconsistent(right, lane_results_by_junction)
        )
    left_arm = _lane_of(left, lane_results_by_junction)
    right_arm = _lane_of(right, lane_results_by_junction)
    if left_arm is None or right_arm is None:
        return True
    return pair_inconsistent(left, right, lane_results_by_junction)


def _ends_to_reidentify(item, lane_results_by_junction):
    left, right = item["left"], item["right"]
    if _pair_kind(item) == KIND_MEDIAN_THROUGH:
        ends = []
        if _has_opposing_traffic(_lane_of(left, lane_results_by_junction)):
            ends.append((left, right))
        if _has_opposing_traffic(_lane_of(right, lane_results_by_junction)):
            ends.append((right, left))
        return ends
    if item_inconsistent(item, lane_results_by_junction):
        left_opp = _has_opposing_traffic(_lane_of(left, lane_results_by_junction))
        right_opp = _has_opposing_traffic(_lane_of(right, lane_results_by_junction))
        # One-way vs two-way: complete the one-way mouth. Re-identifying the
        # two-way end first copies the thinner reading and collapses the road.
        if left_opp != right_opp:
            if left_opp:
                return [(right, left)]
            return [(left, right)]
        return [(left, right), (right, left)]
    return []


def _ends_from_connector(connector):
    left = {
        "junction": int(connector["from_junction"]),
        "arm": int(connector["from_arm"]),
    }
    right = {
        "junction": int(connector["to_junction"]),
        "arm": int(connector["to_arm"]),
    }
    return left, right


def _median_through_arm_ids(arms_doc, through_heading_deg):
    """Through-road arm pair of one junction along the median axis."""
    from scripts.methods.llm_kg.junction_level.llm_gen import (
        MAX_ARM_OPPOSITE_TOL_DEG,
        _arm_heading,
        _arm_opposite_err_deg,
        _axis_mean_heading,
        _signed_heading_delta,
    )

    if through_heading_deg is None:
        return None
    through_heading_deg = float(through_heading_deg)
    arms = list((arms_doc or {}).get("arms") or [])
    by_id = {int(arm["id"]): arm for arm in arms if arm.get("id") is not None}

    def axis_err(id_a, id_b):
        try:
            axis = _axis_mean_heading(_arm_heading(by_id[id_a]), _arm_heading(by_id[id_b]))
        except (KeyError, ValueError):
            return None
        return min(
            abs(_signed_heading_delta(axis, through_heading_deg)),
            abs(_signed_heading_delta(axis, (through_heading_deg + 180.0) % 360.0)),
        )

    best = None
    for road in (arms_doc or {}).get("through_roads") or []:
        ids = [int(x) for x in (road.get("arm_ids") or []) if x is not None]
        if len(ids) != 2 or ids[0] not in by_id or ids[1] not in by_id:
            continue
        err = axis_err(ids[0], ids[1])
        if err is None or err > MEDIAN_THROUGH_AXIS_TOL_DEG:
            continue
        if best is None or err < best[0]:
            best = (err, ids[0], ids[1])
    if best is not None:
        return best[1], best[2]

    opp_best = None
    for i, arm_a in enumerate(arms):
        try:
            ha = _arm_heading(arm_a)
        except ValueError:
            continue
        for arm_b in arms[i + 1:]:
            try:
                hb = _arm_heading(arm_b)
            except ValueError:
                continue
            if _arm_opposite_err_deg(ha, hb) > MAX_ARM_OPPOSITE_TOL_DEG:
                continue
            err = axis_err(int(arm_a["id"]), int(arm_b["id"]))
            if err is None or err > MEDIAN_THROUGH_AXIS_TOL_DEG:
                continue
            if opp_best is None or err < opp_best[0]:
                opp_best = (err, int(arm_a["id"]), int(arm_b["id"]))
    if opp_best is None:
        return None
    return opp_best[1], opp_best[2]


def locate_inter_junction_pairs(junctions_doc, arms_by_junction):
    """Match junction-level links to local arms.

    ``connecting_road``: opposite mouths of one driving carriageway.
    ``median``: each junction's through-road pair along the divided-road axis;
    those arms must be one-way so the two carriageways can merge later.
    """
    from scripts.methods.llm_kg.junction_level.schemas import (
        CONNECTION_TYPE_MEDIAN,
        CONNECTION_TYPE_ROAD,
        normalize_connection_type,
    )

    connectors = derive_inter_junction_roads(
        junctions_doc, arms_by_junction, strict=False,
    )
    pairs = []
    seen_median = set()
    for connector in connectors:
        ctype = normalize_connection_type(connector.get("connection_type"))
        if ctype == CONNECTION_TYPE_ROAD:
            if connector.get("from_arm") is None or connector.get("to_arm") is None:
                continue
            left, right = _ends_from_connector(connector)
            pairs.append({
                "kind": KIND_CONNECTING_ROAD,
                "left": left,
                "right": right,
                "connector": connector,
            })
            continue
        if ctype != CONNECTION_TYPE_MEDIAN:
            continue
        through = connector.get("through_heading_deg")
        for jid in connector.get("junction_ids") or []:
            ids = _median_through_arm_ids(arms_by_junction.get(int(jid)), through)
            if not ids:
                BaseLLM.log(
                    "skip median through: j%s has no through pair on heading %s"
                    % (jid, through)
                )
                continue
            key = (int(jid), min(ids), max(ids))
            if key in seen_median:
                continue
            seen_median.add(key)
            pairs.append({
                "kind": KIND_MEDIAN_THROUGH,
                "left": {"junction": int(jid), "arm": ids[0]},
                "right": {"junction": int(jid), "arm": ids[1]},
                "connector": connector,
            })
    return pairs


def _peer_issues(constraint, message, *, this_end, peer_end, lane_results_by_junction, extra_details=None):
    extra_details = extra_details or {}
    this_lane = _lane_of(this_end, lane_results_by_junction)
    peer_lane = _lane_of(peer_end, lane_results_by_junction)
    details = {
        "connector": [this_end["junction"], peer_end["junction"]],
        "peer_arm": peer_end["arm"],
        "peer_junction": peer_end["junction"],
        "this_junction": this_end["junction"],
        "this_summary": arm_interpretation_summary(this_lane),
        "peer_summary": arm_interpretation_summary(peer_lane),
        "reported_value": message,
    }
    details.update(extra_details)
    return make_issue(
        "layer3", constraint, message,
        location={
            "arm": this_end["arm"],
            "junction": this_end["junction"],
            "field": "cross_section",
        },
        details=details,
        arm_id_list=[this_end["arm"]],
    )


def _median_one_way_message(end):
    return (
        "Median-linked through arm j%s.arm%s must be one-way "
        "(no opposing traffic); this carriageway merges with the opposite "
        "junction across the median into one two-way divided road."
        % (end["junction"], end["arm"])
    )


def check(pairs, lane_results_by_junction):
    issues = []
    for item in pairs or []:
        left, right = item["left"], item["right"]
        if _pair_kind(item) == KIND_MEDIAN_THROUGH:
            for end, peer in ((left, right), (right, left)):
                if not _has_opposing_traffic(_lane_of(end, lane_results_by_junction)):
                    continue
                message = _median_one_way_message(end)
                extra = {
                    "arm_opposing_traffic": {
                        "j%s.arm%s" % (end["junction"], end["arm"]): True,
                    },
                    "required": "one_way",
                    "discrepancies": ["opposing_traffic"],
                    "reported_value": message.rsplit(": ", 1)[-1].rstrip(".")
                    if ": " in message else message,
                }
                issues.append(_peer_issues(
                    CONSTRAINT_MEDIAN_THROUGH_ONE_WAY, message,
                    this_end=end, peer_end=peer,
                    lane_results_by_junction=lane_results_by_junction,
                    extra_details=extra,
                ))
            continue
        findings, extra = _collect_pair_findings(
            left, right,
            left_opp=_has_opposing_traffic(_lane_of(left, lane_results_by_junction)),
            right_opp=_has_opposing_traffic(_lane_of(right, lane_results_by_junction)),
            left_counts=_end_counts(left, lane_results_by_junction),
            right_counts=_end_counts(right, lane_results_by_junction),
        )
        if not findings:
            continue
        constraint = _primary_constraint(findings)
        message = _pair_message(left, right, findings)
        issues.append(_peer_issues(
            constraint, message,
            this_end=left, peer_end=right,
            lane_results_by_junction=lane_results_by_junction,
            extra_details=extra,
        ))
        issues.append(_peer_issues(
            constraint, message,
            this_end=right, peer_end=left,
            lane_results_by_junction=lane_results_by_junction,
            extra_details=extra,
        ))
    if not issues:
        return None
    return ConnectionParseError(issues, "[layer3] inter-junction hard constraint failed")


def _lane_results_by_junction(contexts):
    return {
        int(jid): dict(ctx.lane_results or {})
        for jid, ctx in (contexts or {}).items()
    }


def _issue_for_end(validation_error, end):
    """Keep only the issue that targets this junction/arm so re-identify sees the right peer."""
    kept = []
    for issue in (validation_error or {}).get("issues") or []:
        location = issue.get("location") or {}
        if location.get("arm") is None or int(location["arm"]) != int(end["arm"]):
            continue
        if location.get("junction") is None or int(location["junction"]) != int(end["junction"]):
            continue
        kept.append(issue)
    return {"issues": kept} if kept else validation_error


def _apply_round(contexts, pairs, *, validation_error, round_no, max_rounds):
    from scripts.methods.llm_kg.refinement.common import (
        refinement_round_dir,
        save_refinement_round_dir,
    )

    progressed = False
    lane_map = _lane_results_by_junction(contexts)
    for item in pairs:
        ends = _ends_to_reidentify(item, lane_map)
        if not ends:
            continue
        label = pair_log_label(item)
        for end, peer in ends:
            ctx = contexts.get(int(end["junction"]))
            if ctx is None:
                BaseLLM.log(
                    "[refine:layer3] pair %s: junction %s context missing"
                    % (label, end["junction"])
                )
                continue
            arm_id = int(end["arm"])
            if arm_id not in set(ctx.all_arm_ids):
                BaseLLM.log(
                    "[refine:layer3] pair %s: arm %s not in junction %s"
                    % (label, arm_id, end["junction"])
                )
                continue
            BaseLLM.log(
                "[refine:layer3] pair %s: re-identify j%s.arm%s (peer=j%s.arm%s)"
                % (
                    label,
                    end["junction"], arm_id, peer["junction"], peer["arm"],
                )
            )
            round_dir = None
            if ctx.refinement_root is not None:
                round_dir = save_refinement_round_dir(
                    refinement_round_dir(ctx.refinement_root, "layer3", round_no),
                    validation_error=validation_error,
                )
            _refine_one_arm(
                ctx, arm_id,
                validation_error=_issue_for_end(validation_error, end),
                layer="layer3",
                round_no=round_no, max_rounds=max_rounds, round_dir=round_dir,
            )
            progressed = True
            lane_map = _lane_results_by_junction(contexts)
            still = _ends_to_reidentify(item, lane_map)
            if not still:
                BaseLLM.log(
                    "[refine:layer3] pair %s consistent after j%s.arm%s; skip peer"
                    % (label, end["junction"], arm_id)
                )
                break
            BaseLLM.log(
                "[refine:layer3] pair %s still inconsistent after j%s.arm%s"
                % (label, end["junction"], arm_id)
            )
            refreshed = check(pairs, lane_map)
            if refreshed is not None:
                validation_error = log_validation_error(
                    refreshed, round_no=round_no, max_rounds=max_rounds, layer="layer3",
                )
    return progressed


def run(contexts, pairs, *, scene_root=None, max_refine_rounds=1):
    """Check inter-junction HARD constraints and re-identify until consistent or rounds expire."""
    BaseLLM.log("[validate:layer3] start")
    if len(contexts or {}) <= 1:
        BaseLLM.log("[validate:layer3] skipped (one junction)")
        out = layer_result(passed=True, config=None, refine_rounds=0, extra={"skipped": True})
        save_layer_result(
            scene_root, "layer3",
            passed=True, refine_rounds=0, validation_error=None,
            lanes_doc={"skipped": True, "reason": "single junction"},
        )
        return out

    if not pairs:
        BaseLLM.log("[validate:layer3] no inter-junction pairs")
        out = layer_result(passed=True, config=None, refine_rounds=0)
        save_layer_result(
            scene_root, "layer3",
            passed=True, refine_rounds=0, validation_error=None,
            lanes_doc={"pairs": [], "junctions": {
                str(jid): lanes_doc_from_ctx(ctx) for jid, ctx in contexts.items()
            }},
        )
        return out

    refine_rounds = 0
    validation_error = None
    exc = check(pairs, _lane_results_by_junction(contexts))
    while exc is not None and refine_rounds < max_refine_rounds:
        refine_rounds += 1
        validation_error = log_validation_error(
            exc, round_no=refine_rounds, max_rounds=max_refine_rounds, layer="layer3",
        )
        progressed = _apply_round(
            contexts, pairs,
            validation_error=validation_error,
            round_no=refine_rounds, max_rounds=max_refine_rounds,
        )
        if not progressed:
            break
        exc = check(pairs, _lane_results_by_junction(contexts))

    passed = exc is None
    if not passed:
        validation_error = log_validation_error(exc, layer="layer3")
        BaseLLM.log("[validate:layer3] unresolved after %s round(s)" % refine_rounds)
    else:
        validation_error = None
        BaseLLM.log(
            "[validate:layer3] PASS"
            + (" after %s round(s)" % refine_rounds if refine_rounds else "")
        )

    lanes_doc = {
        "pairs": [
            {
                "kind": _pair_kind(item),
                "left": item["left"],
                "right": item["right"],
                "connector": item["connector"],
                "inconsistent": item_inconsistent(
                    item, _lane_results_by_junction(contexts),
                ),
            }
            for item in pairs
        ],
        "junctions": {
            str(jid): lanes_doc_from_ctx(ctx) for jid, ctx in contexts.items()
        },
    }
    out = layer_result(
        passed=passed, config=None, refine_rounds=refine_rounds,
        validation_error=validation_error,
    )
    save_layer_result(
        scene_root, "layer3",
        passed=out["passed"], refine_rounds=out["refine_rounds"],
        validation_error=out.get("validation_error"),
        lanes_doc=lanes_doc,
    )
    if scene_root is not None:
        dump_json(
            scene_root / "layer_3" / "pairs.json",
            {"pairs": lanes_doc["pairs"]},
        )
    return out
