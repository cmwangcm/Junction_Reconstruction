"""Flatten per-inspect evaluator issues and inspector edits into CSV rows.

Machine dump: ``inspect_metrics.csv`` (rewritten on eval refresh).
Human labels: ``inspect_labels.csv``, keyed by ``inspect_key``. Refresh
mirrors the current metrics keys (drop gone keys, append new ones) and
never overwrites ``inspect_success`` / ``reviewer`` / ``notes`` / ``labeled_at``.
Eval reads inspects from ``conversation.json`` and rewrites ``inspect_events.json``
to match, so a metrics refresh does not keep a stale events dump.
"""
import hashlib
import csv
import json
import re
from pathlib import Path

from scripts.methods.method_utils import (
    dump_json,
    lane_level_dir,
    list_junction_artifact_roots,
    load_json_if_exists,
    resolve_case_dir,
)

INSPECT_CSV_NAME = "inspect_metrics.csv"
INSPECT_LABELS_CSV_NAME = "inspect_labels.csv"

# Identity for joining labels. inspect_id is omitted: R1/R2 can swap across
# refreshes, while kind+issue+visual_anchor identify the same challenge.
INSPECT_KEY_FIELDS = (
    "dataset",
    "case_id",
    "model",
    "reasoning",
    "junction_id",
    "arm_id",
    "kind",
    "issue",
    "visual_anchor",
)
LABEL_HUMAN_FIELDS = ("inspect_success", "reviewer", "notes", "labeled_at")
LABEL_DISPLAY_FIELDS = (
    "dataset",
    "case_id",
    "model",
    "reasoning",
    "junction_id",
    "arm_id",
    "inspect_id",
    "kind",
    "issue",
    "confirmed",
    "edit",
    "edit_reasoning",
)

INSPECT_CSV_FIELDS = [
    "inspect_key",
    "dataset",
    "case_id",
    "model",
    "reasoning",
    "junction_id",
    "arm_id",
    "inspect_id",
    "kind",
    "issue",
    "issue_reason",
    "visual_anchor",
    "confirmed",
    "edit",
    "edit_reasoning",
]

INSPECT_LABEL_FIELDS = [
    "inspect_key",
    "inspect_success",
    "reviewer",
    "notes",
    "labeled_at",
    *LABEL_DISPLAY_FIELDS,
]


def inspect_key(row):
    """Stable id from challenge content, not CSV row order or R1/R2 labels."""
    parts = [str((row or {}).get(field) or "").strip() for field in INSPECT_KEY_FIELDS]
    return hashlib.sha1("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def _compact_json(value):
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def records_from_patch_events(events):
    """One record per applied inspector patch on an arm."""
    records = []
    seen = {}
    for event in events or []:
        crops = {
            str(crop.get("region_id") or ""): crop
            for crop in (event.get("crops") or [])
            if crop.get("region_id")
        }
        for applied in event.get("applied") or []:
            region_id = str(applied.get("region_id") or "")
            crop = crops.get(region_id) or {}
            round_idx = seen.get(region_id, 0) + 1
            seen[region_id] = round_idx
            inspect_id = region_id if round_idx == 1 else "%s:%s" % (round_idx, region_id)
            if not inspect_id:
                inspect_id = str(len(records) + 1)
            members = crop.get("members") or []
            issue_reason = " | ".join(
                str(member.get("reason") or "") for member in members if member.get("reason")
            )
            records.append({
                "inspect_id": inspect_id,
                "kind": applied.get("kind") or crop.get("kind") or "",
                "issue": crop.get("inspect_content") or crop.get("note") or "",
                "issue_reason": issue_reason,
                "visual_anchor": crop.get("visual_anchor") or "",
                "confirmed": bool(applied.get("confirmed")),
                "edit": list(applied.get("after") or []),
                "edit_reasoning": applied.get("reasoning") or "",
            })
    return records


def inspect_events_from_agents(agents_by_arm):
    """Arm-id -> inspect records, for ``inspect_events.json``.

    Prefer live ``patch_events``. Frozen arms reload conversation calls without
    ``patch_events``, so fall back to parsing ``llm_calls``.
    """
    out = {}
    for arm_id in sorted(agents_by_arm or {}, key=lambda value: (len(str(value)), str(value))):
        agent = agents_by_arm[arm_id] or {}
        records = records_from_patch_events(agent.get("patch_events"))
        if not records:
            records = records_from_conversation_calls(agent.get("llm_calls"))
        if records:
            out[str(arm_id)] = records
    return out


def _message_text(message):
    content = (message or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return ""


def _parse_json_text(text):
    text = (text or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        return None


def _call_assistant_json(call):
    for message in reversed(call.get("messages") or []):
        if message.get("role") == "assistant":
            parsed = _parse_json_text(_message_text(message))
            if parsed is not None:
                return parsed
    return None


def _call_user_text(call):
    parts = []
    for message in call.get("messages") or []:
        if message.get("role") == "user":
            parts.append(_message_text(message))
    return "\n".join(parts)


def _parse_inspect_user(text):
    """Pull region_id / kind / issue / visual_anchor from an inspector user prompt."""
    region_id = ""
    kind = ""
    issue = ""
    visual_anchor = ""
    match = re.search(r"Copy region_id=([^\s.]+)", text or "")
    if match:
        region_id = match.group(1)
    match = re.search(r"Keep kind=([^\s.]+)", text or "")
    if match:
        kind = match.group(1)
    challenge = re.search(
        r"evaluator_challenge=(.*)\. visual_anchor=(.*?)(?:\.\n|\.\s*Copy region_id=|\Z)",
        text or "",
        re.S,
    )
    if challenge:
        issue = challenge.group(1).strip()
        visual_anchor = challenge.group(2).strip().rstrip(".")
    return {
        "region_id": region_id,
        "kind": kind,
        "issue": issue,
        "visual_anchor": visual_anchor,
    }


def records_from_conversation_calls(calls):
    """Recover inspect records from an arm's conversation.json call list."""
    records = []
    pending = {}
    seen = {}
    for call in calls or []:
        parsed = _call_assistant_json(call)
        if not isinstance(parsed, dict):
            continue
        if "critiques" in parsed:
            pending = {}
            for i, critic in enumerate(parsed.get("critiques") or [], start=1):
                if not isinstance(critic, dict):
                    continue
                region_id = str(critic.get("region_id") or "R%s" % i)
                pending[region_id] = critic
            continue
        raw = parsed.get("patch") if "patch" in parsed else None
        if not isinstance(raw, dict):
            continue
        user = _parse_inspect_user(_call_user_text(call))
        region_id = str(raw.get("region_id") or user.get("region_id") or "")
        critic = pending.get(region_id) or {}
        round_idx = seen.get(region_id, 0) + 1
        seen[region_id] = round_idx
        inspect_id = region_id if round_idx == 1 else "%s:%s" % (round_idx, region_id)
        if not inspect_id:
            inspect_id = str(len(records) + 1)
        confirmed = bool(raw.get("confirmed"))
        items = list(raw.get("items") or []) if confirmed else []
        records.append({
            "inspect_id": inspect_id,
            "kind": raw.get("kind") or user.get("kind") or critic.get("kind") or "",
            "issue": (
                critic.get("inspect_content") or user.get("issue") or critic.get("note") or ""
            ),
            "issue_reason": critic.get("reason") or "",
            "visual_anchor": (
                critic.get("visual_anchor") or user.get("visual_anchor") or ""
            ),
            "confirmed": confirmed,
            "edit": items,
            "edit_reasoning": raw.get("reasoning") or "",
        })
    return records


def inspect_events_from_conversation(conversation):
    """Arm-id -> inspect records parsed from ``conversation.json`` calls."""
    out = {}
    if not isinstance(conversation, dict):
        return out
    for arm_id in sorted(conversation, key=lambda value: (len(str(value)), str(value))):
        calls = conversation.get(arm_id)
        if calls is None:
            calls = conversation.get(str(arm_id))
        records = records_from_conversation_calls(calls)
        if records:
            out[str(arm_id)] = records
    return out


def records_from_lane_dir(lane_dir, *, sync_events=True):
    """Read inspects from live ``conversation.json`` and refresh ``inspect_events.json``.

    ``inspect_events.json`` used to be skipped when a rerun produced no patches,
    leaving a stale file. Conversation is rewritten every lane-level save, so it
    is the source of truth. Eval writes the dump back so the file matches.
    """
    lane_dir = Path(lane_dir)
    conversation = load_json_if_exists(lane_dir / "conversation.json")
    events_path = lane_dir / "inspect_events.json"
    if isinstance(conversation, dict) and conversation:
        events = inspect_events_from_conversation(conversation)
        if sync_events:
            dump_json(events_path, events)
    else:
        events = load_json_if_exists(events_path) or {}
        if not isinstance(events, dict):
            events = {}
    rows = []
    for arm_id in sorted(events, key=lambda value: (len(str(value)), str(value))):
        for record in events.get(str(arm_id)) or events.get(arm_id) or []:
            rows.append({"arm_id": arm_id, **record})
    return rows


def flatten_inspect_row(row):
    out = {key: row.get(key) for key in INSPECT_CSV_FIELDS}
    out["inspect_key"] = row.get("inspect_key") or inspect_key(row)
    out["confirmed"] = "" if row.get("confirmed") is None else bool(row.get("confirmed"))
    out["edit"] = _compact_json(row.get("edit"))
    return out


def collect_inspect_rows(items):
    """Flatten inspects for the same case/model runs as ``case_metrics``."""
    rows = []
    seen = set()
    for item in items or []:
        dataset = item.get("dataset")
        case_id = item.get("case_id")
        model = item.get("lane_model") or item.get("model")
        reasoning = item.get("lane_reasoning") or item.get("reasoning") or "none"
        output_dir = item.get("output_dir")
        if dataset is None or case_id is None or model is None or not output_dir:
            continue
        case_dir = resolve_case_dir(output_dir, dataset, case_id)
        for root in list_junction_artifact_roots(case_dir):
            junction_id = root.name if root.parent.name == "junctions" else ""
            lane_dir = lane_level_dir(root, model, reasoning)
            key = (str(lane_dir),)
            if key in seen:
                continue
            seen.add(key)
            for record in records_from_lane_dir(lane_dir):
                row = {
                    "dataset": dataset,
                    "case_id": case_id,
                    "model": model,
                    "reasoning": reasoning,
                    "junction_id": junction_id,
                    "arm_id": record.get("arm_id"),
                    "inspect_id": record.get("inspect_id"),
                    "kind": record.get("kind") or "",
                    "issue": record.get("issue") or "",
                    "issue_reason": record.get("issue_reason") or "",
                    "visual_anchor": record.get("visual_anchor") or "",
                    "confirmed": record.get("confirmed"),
                    "edit": record.get("edit"),
                    "edit_reasoning": record.get("edit_reasoning") or "",
                }
                row["inspect_key"] = inspect_key(row)
                rows.append(row)
    rows.sort(key=lambda row: (
        str(row.get("dataset") or ""),
        str(row.get("case_id") or ""),
        str(row.get("model") or ""),
        str(row.get("reasoning") or ""),
        str(row.get("junction_id") or ""),
        str(row.get("arm_id") or ""),
        str(row.get("inspect_id") or ""),
    ))
    return rows


def _cell(row, field):
    if not row:
        return ""
    value = row.get(field)
    return "" if value is None else str(value).strip()


def read_inspect_labels_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_inspect_labels(metric_rows, existing_label_rows):
    """Join human labels onto current inspects by ``inspect_key``.

    - Keys follow ``inspect_metrics.csv``: gone keys are dropped, new keys
      get empty human fields.
    - Existing non-empty human fields are never overwritten.
    - Display columns (issue/edit/inspect_id/...) refresh from metrics.
    """
    existing = {}
    for row in existing_label_rows or []:
        key = _cell(row, "inspect_key")
        if not key:
            continue
        if key not in existing:
            existing[key] = dict(row)
            continue
        for field in LABEL_HUMAN_FIELDS:
            if not _cell(existing[key], field) and _cell(row, field):
                existing[key][field] = row.get(field)

    out = []
    seen = set()
    for mrow in metric_rows or []:
        key = _cell(mrow, "inspect_key")
        if not key or key in seen:
            continue
        seen.add(key)
        old = existing.get(key) or {}
        merged = {field: "" for field in INSPECT_LABEL_FIELDS}
        merged["inspect_key"] = key
        for field in LABEL_DISPLAY_FIELDS:
            merged[field] = mrow.get(field, "")
        for field in LABEL_HUMAN_FIELDS:
            merged[field] = old.get(field, "") if old.get(field) is not None else ""
        out.append(merged)
    return out


def write_inspect_labels_csv(metric_rows, case_metrics_path):
    """Merge/write ``inspect_labels.csv`` beside ``case_metrics.csv``."""
    path = Path(case_metrics_path).with_name(INSPECT_LABELS_CSV_NAME)
    existing = read_inspect_labels_csv(path)
    rows = merge_inspect_labels(metric_rows, existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INSPECT_LABEL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_inspect_metrics_csv(items, case_metrics_path):
    """Write machine ``inspect_metrics.csv`` and merge ``inspect_labels.csv``."""
    path = Path(case_metrics_path).with_name(INSPECT_CSV_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [flatten_inspect_row(row) for row in collect_inspect_rows(items)]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INSPECT_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    write_inspect_labels_csv(rows, case_metrics_path)
    return path
