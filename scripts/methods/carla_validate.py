#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""在 CARLA 中加载 XODR，用 wp.next 遍历路网并校验覆盖。

校验内容：
1. CARLA 可达的 road/lane 是否覆盖 XODR 中的全部相关 road/lane
2. CARLA 的 road/lane 边是否覆盖 XODR 完整可驾驶图

XODR 可驾驶图由两部分组成：
- junction.connection：incoming <-> connecting
- connecting road 的 road.link + lane.link：connecting -> outgoing
"""

import argparse
import json
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

from scripts.methods.method_utils import resolve_case_dir, scenario_xodr_path_for_model

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CARLA_REPORT_NAME = "carla_validation_report.json"


def _import_carla():
    try:
        import carla
    except ImportError as exc:
        raise ImportError(
            "carla is required for CARLA validation; install the CARLA Python API or use a traffic env"
        ) from exc
    return carla


# ---------------------------------------------------------------------------
# XODR I/O
# ---------------------------------------------------------------------------

def read_xodr_text(xodr_path):
    p = Path(xodr_path)
    if not p.exists():
        raise FileNotFoundError("XODR not found: %s" % xodr_path)
    return p.read_text(encoding="utf-8", errors="ignore")


def _resolve_xodr_path(xodr_dir):
    d = Path(xodr_dir).expanduser().resolve()
    if not d.exists():
        raise FileNotFoundError("XODR dir not found: %s" % d)
    if not d.is_dir():
        raise NotADirectoryError("XODR dir is not a directory: %s" % d)
    return d / "scenario.xodr"


# ---------------------------------------------------------------------------
# XODR 解析：完整可驾驶图
# ---------------------------------------------------------------------------

def _parse_link_item(elem):
    if elem is None:
        return None
    eid = elem.attrib.get("elementId")
    return {
        "type": elem.attrib.get("elementType"),
        "id": int(eid) if eid is not None else None,
        "contactPoint": elem.attrib.get("contactPoint"),
    }


def _parse_all_roads(root):
    """roads_info[road_id] -> junction / pred / succ / lane_sections。"""
    roads_info = {}
    for road in root.findall("road"):
        rid_str = road.attrib.get("id")
        if rid_str is None:
            continue
        rid = int(rid_str)
        link = road.find("link")
        pred = succ = None
        if link is not None:
            pred = _parse_link_item(link.find("predecessor"))
            succ = _parse_link_item(link.find("successor"))
        roads_info[rid] = {
            "junction": int(road.attrib.get("junction", "-1")),
            "pred": pred,
            "succ": succ,
            "lane_sections": road.findall("./lanes/laneSection"),
        }
    return roads_info


def _road_side_at_junction(root):
    """side_map[(road_id, junction_id)] = 'start' | 'end'。"""
    side_map = {}
    for road in root.findall("road"):
        rid_str = road.attrib.get("id")
        if rid_str is None:
            continue
        rid = int(rid_str)
        link = road.find("link")
        if link is None:
            continue
        pred = link.find("predecessor")
        succ = link.find("successor")
        if pred is not None and pred.attrib.get("elementType") == "junction":
            jid = pred.attrib.get("elementId")
            if jid is not None:
                side_map[(rid, int(jid))] = "start"
        if succ is not None and succ.attrib.get("elementType") == "junction":
            jid = succ.attrib.get("elementId")
            if jid is not None:
                side_map[(rid, int(jid))] = "end"
    return side_map


def _lane_dir_at_road_side(road_side, lane_id):
    """端点处该 lane 是驶入(in)还是驶出(out)。负 lane 沿参考线，正 lane 逆参考线。"""
    if lane_id == 0:
        return "none"
    along_ref = lane_id < 0
    if road_side == "start":
        return "out" if along_ref else "in"
    if road_side == "end":
        return "in" if along_ref else "out"
    raise ValueError("Invalid road_side: %s" % road_side)


def _iter_driving_lanes(lane_section):
    for side_tag in ("left", "right"):
        side = lane_section.find(side_tag)
        if side is None:
            continue
        for lane in side.findall("lane"):
            if lane.attrib.get("type") == "driving":
                yield lane


def _lane_link_id(lane, tag):
    llink = lane.find("link")
    if llink is None:
        return None
    item = llink.find(tag)
    if item is None:
        return None
    raw = item.attrib.get("id")
    if raw is None:
        return None
    lid = int(raw)
    return None if lid == 0 else lid


def _augment_connector_road_links(roads_info, road_edges, lane_edges):
    """用 connecting road 的 road.link + lane.link 补全跨 road 边。"""
    for rid, info in roads_info.items():
        if info["junction"] < 0:
            continue
        lane_sections = info["lane_sections"]
        if not lane_sections:
            continue

        pred = info["pred"]
        if pred is not None and pred["type"] == "road" and pred["id"] is not None:
            pred_road = pred["id"]
            for lane in _iter_driving_lanes(lane_sections[0]):
                lid_str = lane.attrib.get("id")
                if lid_str is None:
                    continue
                lid = int(lid_str)
                if lid == 0:
                    continue
                pred_lane = _lane_link_id(lane, "predecessor")
                if pred_lane is None:
                    continue
                road_edges[(pred_road, rid)] += 1
                lane_edges[(pred_road, pred_lane, rid, lid)] += 1

        succ = info["succ"]
        if succ is not None and succ["type"] == "road" and succ["id"] is not None:
            succ_road = succ["id"]
            for lane in _iter_driving_lanes(lane_sections[-1]):
                lid_str = lane.attrib.get("id")
                if lid_str is None:
                    continue
                lid = int(lid_str)
                if lid == 0:
                    continue
                succ_lane = _lane_link_id(lane, "successor")
                if succ_lane is None:
                    continue
                road_edges[(rid, succ_road)] += 1
                lane_edges[(rid, lid, succ_road, succ_lane)] += 1


def parse_xodr_drivable_edges(xodr_text):
    """解析完整可驾驶有向边：(road_edges, lane_edges)。"""
    root = ET.fromstring(xodr_text)
    roads_info = _parse_all_roads(root)
    road_side_at_junction = _road_side_at_junction(root)
    road_edges = defaultdict(int)
    lane_edges = defaultdict(int)

    for junction in root.findall("junction"):
        jid_str = junction.attrib.get("id")
        if jid_str is None:
            continue
        jid = int(jid_str)
        for conn in junction.findall("connection"):
            incoming = conn.attrib.get("incomingRoad")
            connecting = conn.attrib.get("connectingRoad")
            cp = conn.attrib.get("contactPoint")
            if incoming is None or connecting is None or cp not in ("start", "end"):
                continue
            in_road = int(incoming)
            con_road = int(connecting)
            in_side = road_side_at_junction.get((in_road, jid))
            if in_side is None:
                continue
            road_dirs_found = set()
            for lane_link in conn.findall("laneLink"):
                lane_from = lane_link.attrib.get("from")
                lane_to = lane_link.attrib.get("to")
                if lane_from is None or lane_to is None:
                    continue
                lf = int(lane_from)
                lt = int(lane_to)
                if lf == 0 or lt == 0:
                    continue
                in_dir = _lane_dir_at_road_side(in_side, lf)
                con_dir = _lane_dir_at_road_side(cp, lt)
                if in_dir == "in" and con_dir == "out":
                    lane_edges[(in_road, lf, con_road, lt)] += 1
                    road_dirs_found.add((in_road, con_road))
                elif in_dir == "out" and con_dir == "in":
                    lane_edges[(con_road, lt, in_road, lf)] += 1
                    road_dirs_found.add((con_road, in_road))
            for edge in road_dirs_found:
                road_edges[edge] += 1

    _augment_connector_road_links(roads_info, road_edges, lane_edges)
    return road_edges, lane_edges


# ---------------------------------------------------------------------------
# CARLA 路网遍历
# ---------------------------------------------------------------------------

def _waypoint_key(wp, s_quant=0.2):
    return (wp.road_id, wp.section_id, wp.lane_id, int(round(wp.s / s_quant)))


def _seed_waypoints(carla_map, seed_dist=2.0, driving_only=True):
    carla = _import_carla()
    seeds = list(carla_map.generate_waypoints(seed_dist))
    if driving_only:
        seeds = [w for w in seeds if w.lane_type == carla.LaneType.Driving]
    return seeds


def traverse_with_next(
    carla_map,
    step=0.5,
    s_quant=0.2,
    max_nodes=400000,
    seed_dist=2.0,
    seed_driving_only=True,
):
    """BFS 遍历驾驶道，返回统计、样例、覆盖集合与边计数。"""
    carla = _import_carla()
    seeds = _seed_waypoints(carla_map, seed_dist=seed_dist, driving_only=seed_driving_only)
    if not seeds:
        raise RuntimeError("No seed waypoints. Try larger seed_dist or --seed-all-lanes.")

    q = deque(seeds)
    visited = set()
    stats = defaultdict(int)
    lanes_covered = set()
    road_edge_counts = defaultdict(int)
    lane_edge_counts = defaultdict(int)
    examples_deadends = []
    examples_branch = []

    while q and len(visited) < max_nodes:
        wp = q.popleft()
        if wp is None:
            continue
        key = _waypoint_key(wp, s_quant=s_quant)
        if key in visited:
            continue
        visited.add(key)
        if wp.lane_type != carla.LaneType.Driving:
            stats["non_driving_skipped"] += 1
            continue

        lanes_covered.add((wp.road_id, wp.lane_id))
        nxt = wp.next(step)
        if not nxt:
            stats["dead_end_next_empty"] += 1
            if len(examples_deadends) < 20:
                loc = wp.transform.location
                examples_deadends.append(
                    "dead_end road=%s lane=%s s=%.2f junc=%s x=%.2f y=%.2f"
                    % (wp.road_id, wp.lane_id, wp.s, wp.is_junction, loc.x, loc.y)
                )
            continue

        seen = set()
        unique_nxt = []
        for w in nxt:
            if w is None:
                continue
            nk = (w.road_id, w.lane_id, round(w.s, 2))
            if nk in seen:
                continue
            seen.add(nk)
            unique_nxt.append(w)

        if len(unique_nxt) > 1:
            stats["branch_points"] += 1
            if len(examples_branch) < 20:
                cand = ", ".join(
                    "(r%s,l%s,s%.2f)" % (w.road_id, w.lane_id, w.s) for w in unique_nxt[:6]
                )
                examples_branch.append(
                    "branch road=%s lane=%s s=%.2f -> %s: %s"
                    % (wp.road_id, wp.lane_id, wp.s, len(unique_nxt), cand)
                )

        for w in unique_nxt:
            stats["edges_traversed"] += 1
            if wp.road_id == w.road_id and wp.lane_id == w.lane_id:
                q.append(w)
                continue
            road_edge_counts[(wp.road_id, w.road_id)] += 1
            lane_edge_counts[(wp.road_id, wp.lane_id, w.road_id, w.lane_id)] += 1
            q.append(w)

    stats["visited_nodes"] = len(visited)
    stats["driving_lanes_covered"] = len(lanes_covered)
    stats["seed_count"] = len(seeds)
    stats["queue_remaining"] = len(q)
    stats["hit_max_nodes"] = int(len(visited) >= max_nodes)
    stats["seed_dist"] = seed_dist
    stats["seed_driving_only"] = int(seed_driving_only)
    return (
        stats,
        examples_deadends,
        examples_branch,
        lanes_covered,
        road_edge_counts,
        lane_edge_counts,
    )


# ---------------------------------------------------------------------------
# 校验与报告
# ---------------------------------------------------------------------------

def _xodr_roads_and_lanes(xodr_road_edges, xodr_lane_edges):
    xodr_roads = set()
    xodr_lanes = set()
    for fr, tr in xodr_road_edges:
        xodr_roads.add(fr)
        xodr_roads.add(tr)
    for fr, fl, tr, tl in xodr_lane_edges:
        xodr_lanes.add((fr, fl))
        xodr_lanes.add((tr, tl))
        xodr_roads.add(fr)
        xodr_roads.add(tr)
    return xodr_roads, xodr_lanes


def run_validation(
    lanes_covered,
    road_edge_counts,
    lane_edge_counts,
    xodr_road_edges,
    xodr_lane_edges,
):
    """要求 XODR 的 roads/lanes/edges 被 CARLA 覆盖；extra_* 仅作诊断。"""
    carla_roads = {r for (r, _) in lanes_covered}
    carla_road_edges = set(road_edge_counts.keys())
    carla_lane_edges = set(lane_edge_counts.keys())
    xodr_road_edges_set = set(xodr_road_edges.keys())
    xodr_lane_edges_set = set(xodr_lane_edges.keys())
    xodr_roads, xodr_lanes = _xodr_roads_and_lanes(xodr_road_edges_set, xodr_lane_edges_set)

    missing_roads = sorted(xodr_roads - carla_roads)
    missing_lanes = sorted(xodr_lanes - lanes_covered)
    missing_road_edges = sorted(xodr_road_edges_set - carla_road_edges)
    missing_lane_edges = sorted(xodr_lane_edges_set - carla_lane_edges)
    extra_roads = sorted(carla_roads - xodr_roads)
    extra_lanes = sorted(lanes_covered - xodr_lanes)
    extra_road_edges = sorted(carla_road_edges - xodr_road_edges_set)
    extra_lane_edges = sorted(carla_lane_edges - xodr_lane_edges_set)
    passed = not (missing_roads or missing_lanes or missing_road_edges or missing_lane_edges)
    return {
        "passed": passed,
        "missing_roads": missing_roads,
        "missing_lanes": missing_lanes,
        "missing_road_edges": missing_road_edges,
        "missing_lane_edges": missing_lane_edges,
        "extra_roads": extra_roads,
        "extra_lanes": extra_lanes,
        "extra_road_edges": extra_road_edges,
        "extra_lane_edges": extra_lane_edges,
    }


def _print_list_field(name, values, top_extra):
    if not values:
        print("  %s: []" % name)
    elif len(values) <= top_extra:
        print("  %s: %s" % (name, values))
    else:
        print("  %s: 共 %s 项" % (name, len(values)))


def print_report(
    stats,
    deadends,
    branches,
    lanes_covered,
    road_edge_counts,
    lane_edge_counts,
    xodr_road_edges,
    xodr_lane_edges,
    report,
    top_lane_edges=80,
    top_extra=80,
):
    print("\n----- 统计 -----")
    for k in sorted(stats.keys()):
        print("  %s: %s" % (k, stats[k]))

    print("\n----- dead_end 示例 (最多 20) -----")
    for s in (deadends or [])[:20]:
        print(" ", s)
    if not deadends:
        print("  (无)")

    print("\n----- branch 示例 (最多 20) -----")
    for s in (branches or [])[:20]:
        print(" ", s)
    if not branches:
        print("  (无)")

    print("\n----- 覆盖 (road_id, lane_id): %s -----" % len(lanes_covered))
    if len(lanes_covered) <= 200:
        print(sorted(lanes_covered))
    else:
        print("(数量过多不打印)")

    print("\n----- CARLA road 边 (next) -----")
    for (fr, tr), cnt in sorted(road_edge_counts.items()):
        print("  r%s -> r%s: %s" % (fr, tr, cnt))
    if not road_edge_counts:
        print("  (无)")

    print("\n----- CARLA lane 边 (前 %s 条) -----" % top_lane_edges)
    ranked = sorted(lane_edge_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for (fr, fl, tr, tl), cnt in ranked[:top_lane_edges]:
        print("  r%s/l%s -> r%s/l%s: %s" % (fr, fl, tr, tl, cnt))
    if not lane_edge_counts:
        print("  (无)")
    elif len(lane_edge_counts) > top_lane_edges:
        print("  ... 共 %s 条" % len(lane_edge_counts))

    print("\n----- XODR 完整可驾驶 road 边（有方向）-----")
    for (fr, tr), cnt in sorted(xodr_road_edges.items()):
        print("  r%s -> r%s: %s" % (fr, tr, cnt))
    if not xodr_road_edges:
        print("  (无)")

    print("\n----- XODR 完整可驾驶 lane 边（有方向）-----")
    for (fr, fl, tr, tl), cnt in sorted(xodr_lane_edges.items()):
        print("  r%s/l%s -> r%s/l%s: %s" % (fr, fl, tr, tl, cnt))
    if not xodr_lane_edges:
        print("  (无)")

    print("\n----- 覆盖校验: CARLA vs XODR -----")
    print("  missing_roads: %s" % (report["missing_roads"] or []))
    print("  missing_lanes: %s" % (report["missing_lanes"] or []))
    print("  missing_road_edges: %s" % (report["missing_road_edges"] or []))
    print("  missing_lane_edges: %s" % (report["missing_lane_edges"] or []))

    print("\n----- CARLA 额外项（不影响覆盖是否通过）-----")
    _print_list_field("extra_roads", report["extra_roads"], top_extra)
    _print_list_field("extra_lanes", report["extra_lanes"], top_extra)
    _print_list_field("extra_road_edges", report["extra_road_edges"], top_extra)
    _print_list_field("extra_lane_edges", report["extra_lane_edges"], top_extra)


# ---------------------------------------------------------------------------
# 校验后演示车辆
# ---------------------------------------------------------------------------

def _spawn_too_close(wp, picked, min_dist):
    for other in picked:
        dx = wp.transform.location.x - other.transform.location.x
        dy = wp.transform.location.y - other.transform.location.y
        if dx * dx + dy * dy < min_dist * min_dist:
            return True
    return False


def _pick_spawn_waypoints(carla_map, n, seed_dist=5.0):
    """在入口负车道上选取互相错开的 spawn 点（避开远端与路口口门）。"""
    carla = _import_carla()
    candidates = []
    for wp in carla_map.generate_waypoints(seed_dist):
        if wp.lane_type != carla.LaneType.Driving:
            continue
        if wp.is_junction or wp.lane_id >= 0 or wp.s < 8.0:
            continue
        try:
            until = wp.next_until_lane_end(2.0)
        except Exception:
            until = []
        if until and (until[-1].s - wp.s) < 15.0:
            continue
        candidates.append(wp)

    random.shuffle(candidates)
    picked = []
    used_roads = set()
    picked_keys = set()

    for min_dist, prefer_new_road in ((12.0, True), (8.0, False)):
        for wp in candidates:
            if len(picked) >= n:
                break
            if prefer_new_road and wp.road_id in used_roads and len(used_roads) < n:
                continue
            key = (wp.road_id, wp.lane_id, round(wp.s, 1))
            if key in picked_keys or _spawn_too_close(wp, picked, min_dist):
                continue
            picked.append(wp)
            picked_keys.add(key)
            used_roads.add(wp.road_id)
        if len(picked) >= n:
            break
    return picked


def spawn_demo_vehicles(client, world, carla_map, n=3, duration=20.0, tm_port=8000):
    """在入口道投放 autopilot 车辆，运行一段时间后销毁。"""
    if n <= 0:
        return []

    vehicle_bps = [
        bp for bp in world.get_blueprint_library().filter("vehicle.*")
        if int(bp.get_attribute("number_of_wheels")) == 4
    ]
    if not vehicle_bps:
        print("[Spawn] 未找到可用车辆 blueprint，跳过")
        return []

    spawn_wps = _pick_spawn_waypoints(carla_map, n)
    if not spawn_wps:
        print("[Spawn] 未找到合适 spawn 点，跳过")
        return []

    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(False)
    tm_port_used = tm.get_port()

    actors = []
    print("\n[Spawn] 尝试在入口道投放 %s 辆车，运行 %.0fs ..." % (len(spawn_wps), duration))
    for i, wp in enumerate(spawn_wps):
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            colors = bp.get_attribute("color").recommended_values
            if colors:
                bp.set_attribute("color", random.choice(colors))
        tf = wp.transform
        tf.location.z += 0.3
        vehicle = world.try_spawn_actor(bp, tf)
        if vehicle is None:
            print("  [%s] spawn 失败 road=%s lane=%s s=%.1f" % (i, wp.road_id, wp.lane_id, wp.s))
            continue
        vehicle.set_autopilot(True, tm_port_used)
        tm.ignore_lights_percentage(vehicle, 100.0)
        tm.vehicle_percentage_speed_difference(vehicle, random.uniform(-10.0, 10.0))
        actors.append(vehicle)
        print("  [%s] id=%s road=%s lane=%s s=%.1f" % (i, vehicle.id, wp.road_id, wp.lane_id, wp.s))

    if not actors:
        print("[Spawn] 全部 spawn 失败")
        return []

    print("[Spawn] 已投放 %s 辆，等待 %.0fs ..." % (len(actors), duration))
    try:
        time.sleep(duration)
    finally:
        print("[Spawn] 清理车辆 ...")
        for v in actors:
            if v.is_alive:
                v.destroy()
    return actors


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

HOST = "localhost"
PORT = 2000
TIMEOUT = 10.0
STEP = 0.5
S_QUANT = 0.2
MAX_NODES = 400000
SEED_DIST = 2.0
SEED_DRIVING_ONLY = True
TM_PORT = 8000


def validate(xodr_dir, spawn_vehicles=0, spawn_duration=30.0, verbose=True):
    carla = _import_carla()
    client = carla.Client(HOST, PORT)
    client.set_timeout(TIMEOUT)

    xodr_path = _resolve_xodr_path(xodr_dir)
    xodr_text = read_xodr_text(str(xodr_path))
    xodr_road_edges, xodr_lane_edges = parse_xodr_drivable_edges(xodr_text)

    world = client.generate_opendrive_world(xodr_text)
    carla_map = world.get_map()

    if verbose:
        print("[Map] %s" % carla_map.name)
        print("[XODR] %s" % xodr_path)
        print(
            "[Config] step=%s s_quant=%s max_nodes=%s seed_dist=%s seed_driving_only=%s"
            % (STEP, S_QUANT, MAX_NODES, SEED_DIST, SEED_DRIVING_ONLY)
        )

    stats, deadends, branches, lanes_covered, road_edge_counts, lane_edge_counts = traverse_with_next(
        carla_map,
        step=STEP,
        s_quant=S_QUANT,
        max_nodes=MAX_NODES,
        seed_dist=SEED_DIST,
        seed_driving_only=SEED_DRIVING_ONLY,
    )
    report = run_validation(lanes_covered, road_edge_counts, lane_edge_counts, xodr_road_edges, xodr_lane_edges)
    if verbose:
        print_report(stats, deadends, branches, lanes_covered, road_edge_counts, lane_edge_counts, xodr_road_edges, xodr_lane_edges, report)

    if report["passed"]:
        print("\n[Result] PASSED: CARLA 路网完整覆盖 XODR 的 road/lane 及可行驶连接")
    else:
        print("\n[Result] FAILED: CARLA 缺少 XODR 声明的部分 road/lane 或连接")

    report["xodr_path"] = str(xodr_path)
    report["xodr_dir"] = str(Path(xodr_dir).resolve())
    # Persist before optional spawn demo so a native crash during cleanup cannot lose the result.
    _write_carla_report(xodr_dir, report)

    if spawn_vehicles > 0:
        spawn_demo_vehicles(client, world, carla_map, n=spawn_vehicles, duration=spawn_duration, tm_port=TM_PORT)

    return report


def _write_carla_report(xodr_dir, report):
    xodr_dir = Path(xodr_dir)
    xodr_dir.mkdir(parents=True, exist_ok=True)
    report_path = xodr_dir / CARLA_REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path


def run_carla_validate(dataset="nhtsa_2023", case_id="28620", output_dir="results", model="openai/gpt-5.6-sol", reasoning="none", spawn_vehicles=0, spawn_duration=30.0, verbose=True):
    """Load ``scenario.xodr`` (case-root or ``junctions/{id}/`` xodr_result) and validate in CARLA."""
    run_dir = resolve_case_dir(output_dir, dataset, case_id)
    run_name = "%s/%s" % (dataset, case_id)
    xodr_path = scenario_xodr_path_for_model(run_dir, model, reasoning)
    xodr_dir = xodr_path.parent.resolve()
    if verbose:
        print("[INFO] case=%s model=%s reasoning=%s" % (run_name, model, reasoning), flush=True)
        print("[INFO] xodr_dir: %s" % xodr_dir, flush=True)
        print("[INFO] xodr: %s" % xodr_path, flush=True)
    report = validate(xodr_dir, spawn_vehicles=spawn_vehicles, spawn_duration=spawn_duration, verbose=verbose)
    report_path = _write_carla_report(xodr_dir, report)
    if not report["passed"]:
        err_path = xodr_dir / "carla_validation_error.json"
        err_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        if verbose:
            print("[INFO] validation error saved: %s" % err_path, flush=True)
    elif verbose:
        print("[INFO] validation report saved: %s" % report_path, flush=True)
    return report


def run_carla_validate_subprocess(dataset="nhtsa_2023", case_id="28620", output_dir="results", model="openai/gpt-5.6-sol", reasoning="none", spawn_vehicles=0, spawn_duration=30.0, verbose=True):
    """Run CARLA validation in a child process so native crashes cannot kill the batch parent."""
    run_dir = resolve_case_dir(output_dir, dataset, case_id)
    xodr_path = scenario_xodr_path_for_model(run_dir, model, reasoning)
    xodr_dir = xodr_path.parent.resolve()
    report_path = xodr_dir / CARLA_REPORT_NAME
    if report_path.is_file():
        report_path.unlink()

    cmd = [
        sys.executable,
        "-m",
        "scripts.methods.carla_validate",
        "--dataset",
        str(dataset),
        "--case-id",
        str(case_id),
        "--output-dir",
        str(output_dir),
        "--model",
        str(model),
        "--reasoning",
        str(reasoning),
        "--spawn-vehicles",
        str(int(spawn_vehicles)),
        "--spawn-duration",
        str(float(spawn_duration)),
    ]
    cmd.append("--verbose" if verbose else "--no-verbose")

    if verbose:
        print("[INFO] carla validate subprocess: %s" % " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))

    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["subprocess_returncode"] = proc.returncode
        report["crashed"] = False
        return report

    report = {
        "passed": False,
        "crashed": True,
        "subprocess_returncode": proc.returncode,
        "error": "carla validate subprocess exited without writing %s (returncode=%s)"
        % (CARLA_REPORT_NAME, proc.returncode),
        "xodr_dir": str(xodr_dir),
    }
    _write_carla_report(xodr_dir, report)
    err_path = xodr_dir / "carla_validation_error.json"
    err_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if verbose:
        print("[INFO] carla validate subprocess crashed; error saved: %s" % err_path, flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="在 CARLA 中加载 refinement/{model}_{reasoning}/xodr_result/scenario.xodr 并校验路网覆盖与连接")
    ap.add_argument("--dataset", default="nhtsa_2023", help="Dataset folder under dataset/, e.g. nhtsa_2023")
    ap.add_argument("--case-id", default="28661", help="Case folder name under the dataset")
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--model", default="openai/gpt-5.6-sol")
    ap.add_argument("--reasoning", default="none", help="Reasoning effort used in the refinement result folder name")
    ap.add_argument("--spawn-vehicles", type=int, default=0, help="校验后投放车辆数（0 关闭）")
    ap.add_argument("--spawn-duration", type=float, default=30.0, help="车辆运行时长（秒）")
    ap.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    report = run_carla_validate(
        dataset=args.dataset,
        case_id=args.case_id,
        output_dir=args.output_dir,
        model=args.model,
        reasoning=args.reasoning,
        spawn_vehicles=args.spawn_vehicles,
        spawn_duration=args.spawn_duration,
        verbose=args.verbose,
    )
    raise SystemExit(0 if report["passed"] else 2)
