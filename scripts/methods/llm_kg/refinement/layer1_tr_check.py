"""Layer 1: through-road peer CHECK (not a hard constraint) + sequential re-identification.

Through-road check: collect every relevant peer discrepancy on a pair at once
(opposing traffic, lane counts). Always runs after lane-level. The outer refine
flow keeps L1 when it is connection-valid, or when both L1 and lane-level are
invalid (then continues to layer-2). If L1 is invalid but lane-level is valid,
it restores the lane-level reading. Each inconsistent pair is processed as:

  A_old, B_old
  1. feedback for A from B_old
  2-4. re-identify A from the extended crop, then commit A_current
  5. recompute pair consistency
  6. if still inconsistent: feedback for B from A_current, re-identify B, commit B
  7. recompute pair consistency again (outer check loop)

Identification uses the lane-level flow (system prompt, initial inspect,
resolution inspect). The user message is only a compact THIS/peer interpretation
summary. The new reading fully replaces the old one.
"""

from scripts.methods.llm_kg.refinement.common import (
    ConnectionParseError,
    lanes_doc_from_ctx,
    layer_result,
    make_issue,
    save_layer_result,
)
from scripts.methods.llm_kg.refinement.loop import run_refine_loop
from scripts.methods.llm_utils import BaseLLM

# Peer lane-count |delta| threshold (fail when abs_delta > threshold).
CHECK_DELTA = 0


def _lanes_from_arm(arm, lane_arm=None):
    if lane_arm is not None:
        return [item for item in (lane_arm.get("cross_section") or []) if item.get("type") == "lane"]
    return list(arm.get("lanes") or [])


def arm_io_counts(lanes):
    """(outgoing, incoming) from fixed travel labels."""
    outgoing = sum(1 for lane in lanes if lane.get("travel") == "outgoing")
    incoming = sum(1 for lane in lanes if lane.get("travel") == "incoming")
    return outgoing, incoming


def _has_opposing_traffic(lane_arm):
    """True if the arm reading marks an opposing_separator (two-way); False if one-way / absent."""
    if not lane_arm:
        return False
    opp = lane_arm.get("opposing_separator")
    if isinstance(opp, dict) and opp.get("element_id"):
        return True
    return any(bool(item.get("opposing_separator")) for item in (lane_arm.get("cross_section") or []))


def arm_interpretation_summary(lane_arm):
    """Compact THIS/peer interpretation for the layer-1 re-identify user message."""
    if not lane_arm:
        return (
            "- outgoing lanes: unknown\n"
            "- incoming lanes: unknown\n"
            "- opposing traffic: unknown"
        )
    lanes = [item for item in (lane_arm.get("cross_section") or []) if item.get("type") == "lane"]
    out_n, in_n = arm_io_counts(lanes)
    opp = "yes" if _has_opposing_traffic(lane_arm) else "no"
    return (
        f"- outgoing lanes: {out_n}\n"
        f"- incoming lanes: {in_n}\n"
        f"- opposing traffic: {opp}"
    )


def _counts_by_arm(config, lane_results):
    arms_by_id = {int(arm["id"]): arm for arm in (config.get("arms") or [])}
    lane_results = lane_results or {}
    counts = {}
    for arm_id, arm in arms_by_id.items():
        lanes = _lanes_from_arm(arm, lane_results.get(arm_id))
        counts[arm_id] = arm_io_counts(lanes)
    return counts


_FINDING_PRIORITY = ("opposing_traffic", "lane_balance")
_FINDING_CONSTRAINT = {
    "opposing_traffic": "through_road_opposing_traffic",
    "lane_balance": "through_road_lane_balance",
}


def _primary_constraint(findings):
    kinds = {item["kind"] for item in (findings or [])}
    for kind in _FINDING_PRIORITY:
        if kind in kinds:
            return _FINDING_CONSTRAINT[kind]
    return "through_road_lane_balance"


def _pair_message(a_id, b_id, findings):
    kinds = [kind for kind in _FINDING_PRIORITY if any(item.get("kind") == kind for item in (findings or []))]
    bits = []
    for kind in kinds:
        if kind == "opposing_traffic":
            bits.append("opposing traffic inconsistent (one-way vs two-way)")
        else:
            bits.append("peer lane counts differ")
    body = "; ".join(bits) if bits else "peer structure disagrees"
    return f"Through-road arms {a_id}/{b_id}: {body}."


def _through_peer_issues(constraint, message, *, a_id, b_id, lane_results, extra_details=None):
    """Emit one issue per arm. The refine round processes the pair sequentially (A then maybe B)."""
    issues = []
    extra_details = extra_details or {}
    for arm_id, other_id in ((a_id, b_id), (b_id, a_id)):
        details = {
            "through_road": [a_id, b_id],
            "peer_arm": other_id,
            "this_summary": arm_interpretation_summary(lane_results.get(arm_id)),
            "peer_summary": arm_interpretation_summary(lane_results.get(other_id)),
            "reported_value": message,
        }
        details.update(extra_details)
        issues.append(make_issue(
            "check", constraint, message,
            location={"arm": arm_id, "field": "cross_section"},
            details=details,
            arm_id_list=[a_id, b_id],
        ))
    return issues


def _collect_pair_findings(
    a_id, b_id, *,
    a_opp, b_opp,
    counts_by_arm,
):
    """Every relevant discrepancy on this through-road pair, from each arm's view."""
    findings = []
    extra = {}
    if a_opp != b_opp:
        findings.append({"kind": "opposing_traffic", "arm_opp": a_opp, "peer_opp": b_opp})
        extra["arm_opposing_traffic"] = {str(a_id): a_opp, str(b_id): b_opp}

    a_out, a_in = counts_by_arm[a_id]
    b_out, b_in = counts_by_arm[b_id]
    pairs = []
    if abs(a_in - b_out) > CHECK_DELTA:
        pairs.append({
            "travel": "incoming", "count": a_in,
            "peer_travel": "outgoing", "peer_count": b_out,
            "abs_delta": abs(a_in - b_out), "threshold": CHECK_DELTA,
        })
    if abs(a_out - b_in) > CHECK_DELTA:
        pairs.append({
            "travel": "outgoing", "count": a_out,
            "peer_travel": "incoming", "peer_count": b_in,
            "abs_delta": abs(a_out - b_in), "threshold": CHECK_DELTA,
        })
    if pairs:
        findings.append({"kind": "lane_balance", "pairs": pairs})
        extra["lane_counts"] = {
            str(a_id): {"outgoing": a_out, "incoming": a_in},
            str(b_id): {"outgoing": b_out, "incoming": b_in},
        }
    extra["discrepancies"] = [kind for kind in _FINDING_PRIORITY if any(item["kind"] == kind for item in findings)]
    extra["reported_value"] = _pair_message(a_id, b_id, findings).split(": ", 1)[-1].rstrip(".")
    return findings, extra


def pair_inconsistent(a_id, b_id, lane_results):
    """True if this through-road pair still fails opposing-traffic or lane-balance."""
    a_id, b_id = int(a_id), int(b_id)
    lane_results = lane_results or {}
    counts_by_arm = {
        a_id: arm_io_counts(_lanes_from_arm(None, lane_results.get(a_id))),
        b_id: arm_io_counts(_lanes_from_arm(None, lane_results.get(b_id))),
    }
    findings, _extra = _collect_pair_findings(
        a_id, b_id,
        a_opp=_has_opposing_traffic(lane_results.get(a_id)),
        b_opp=_has_opposing_traffic(lane_results.get(b_id)),
        counts_by_arm=counts_by_arm,
    )
    return bool(findings)


def through_road_pairs_from_issues(issues):
    """Unique (A, B) pairs in first-seen through_road order from check issues."""
    pairs = []
    seen = set()
    for issue in issues or []:
        details = issue.get("details") or {}
        road = details.get("through_road") or []
        if len(road) < 2:
            continue
        a_id, b_id = int(road[0]), int(road[1])
        key = (a_id, b_id)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a_id, b_id))
    return pairs


def check(config, through_roads, *, lane_results=None):
    counts_by_arm = _counts_by_arm(config, lane_results)
    lane_results = lane_results or {}
    issues = []
    for item in through_roads or []:
        arm_ids = item.get("arm_ids") or []
        if len(arm_ids) < 2:
            continue
        a_id, b_id = int(arm_ids[0]), int(arm_ids[1])
        if a_id not in counts_by_arm or b_id not in counts_by_arm:
            continue
        a_opp = _has_opposing_traffic(lane_results.get(a_id))
        b_opp = _has_opposing_traffic(lane_results.get(b_id))
        findings, extra = _collect_pair_findings(
            a_id, b_id,
            a_opp=a_opp, b_opp=b_opp,
            counts_by_arm=counts_by_arm,
        )
        if not findings:
            continue
        issues.extend(_through_peer_issues(
            _primary_constraint(findings),
            _pair_message(a_id, b_id, findings),
            a_id=a_id, b_id=b_id, lane_results=lane_results,
            extra_details=extra,
        ))
    if not issues:
        return None
    return ConnectionParseError(issues, "[check] through-road peer check failed")


def run(ctx, config, through_roads, *, max_refine_rounds=2):
    BaseLLM.log("[validate:check] start")

    def _check(cfg):
        return check(
            cfg, through_roads,
            lane_results=ctx.lane_results or {},
        )

    config, refine_rounds, validation_error, passed = run_refine_loop(
        ctx, config, layer="check", max_refine_rounds=max_refine_rounds, check=_check,
    )
    if not passed:
        BaseLLM.log(f"[validate:check] unresolved after {refine_rounds} round(s)")
        out = layer_result(passed=False, config=config, refine_rounds=refine_rounds, validation_error=validation_error)
    else:
        BaseLLM.log("[validate:check] PASS" + (f" after {refine_rounds} round(s)" if refine_rounds else ""))
        out = layer_result(passed=True, config=config, refine_rounds=refine_rounds, validation_error=None)
    save_layer_result(
        ctx.refinement_root, "check",
        passed=out["passed"], refine_rounds=out["refine_rounds"],
        validation_error=out.get("validation_error"),
        lanes_doc=lanes_doc_from_ctx(ctx),
    )
    return out
