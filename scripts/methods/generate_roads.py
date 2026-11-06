import math
import argparse
import shutil
from pathlib import Path
from typing import Any, Optional, Union

from scenariogeneration import xodr

from scripts.methods.road_utils import (
    DEFAULT_ESMINI_PATH,
    OUTPUT_FILENAME,
    SpiralCleanupScenario,
    load_connections,
    run_esmini_for_scenario_generator,
    MultiLaneCommonJunctionCreator,
)
from scripts.methods.method_utils import connections_path_for_model, resolve_case_dir
from scripts.methods.connection_parser import normalize_driving_side, odr_left_right_lanes
from scripts.methods.llm_kg.junction_level.schemas import (
    CONNECTION_TYPE_MEDIAN,
    DEFAULT_MEDIAN_LANE_TYPE,
    NON_DRIVING_LANE_TYPES,
    normalize_connection_type,
    normalize_lane_type,
)


JUNCTION_ROAD_START_ID = 1000
JUNCTION_ROAD_ID_STEP = 100
JUNCTION_ID_BASE = 100


class Scenario(SpiralCleanupScenario):
    def __init__(
        self,
        connections_path: Union[str, Path] = "connections.json",
        *,
        lane_width: float = 3.5,
        driving_side=None,
    ):
        super().__init__()
        self.cfg = load_connections(connections_path)
        self._remap_junction_ids()
        self.parameters = {}
        self.lane_width = lane_width
        self.driving_side = normalize_driving_side(
            driving_side if driving_side is not None else self.cfg.get("driving_side")
        )

    def _traffic_rule(self):
        """left driving_side is right-hand traffic; otherwise left-hand traffic."""
        if self.driving_side == "left":
            return xodr.TrafficRule.RHT
        return xodr.TrafficRule.LHT

    def _remap_junction_ids(self) -> None:
        """Map junction ids into a range that cannot collide with road ids or 0."""
        junctions = self.cfg.get("junctions", [])
        if not junctions:
            return
        id_map = {
            int(j["id"]): JUNCTION_ID_BASE + int(j["id"]) for j in junctions
        }
        if all(id_map[int(j["id"])] == int(j["id"]) for j in junctions):
            return
        for j in junctions:
            j["id"] = id_map[int(j["id"])]
        for c in self.cfg.get("connectors", []):
            c["junctions"] = [id_map[int(x)] for x in c["junctions"]]

    @staticmethod
    def _ordered_junctions(connector: dict) -> tuple[int, int]:
        a, b = map(int, connector["junctions"])
        return (a, b) if a > b else (b, a)

    @staticmethod
    def _unit_vector(heading_deg: float) -> tuple[float, float, float]:
        h = math.radians(float(heading_deg))
        return h, math.cos(h), math.sin(h)

    @staticmethod
    def _connection_key(conn: dict) -> tuple[int, int]:
        return int(conn["from_arm"]), int(conn["to_arm"])

    @staticmethod
    def _nonnegative(value: float, name: str) -> float:
        value = float(value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _connector_connection_type(connector: dict) -> str:
        return normalize_connection_type(
            connector.get("connection_type") or connector.get("type")
        )

    @classmethod
    def _is_non_driving_connector(cls, connector: dict) -> bool:
        if cls._connector_connection_type(connector) == CONNECTION_TYPE_MEDIAN:
            return True
        lane_type = str(connector.get("lane_type") or "").strip().lower()
        return lane_type in NON_DRIVING_LANE_TYPES

    @classmethod
    def _connector_lane_type(cls, connector: dict) -> str:
        return (
            normalize_lane_type(
                connector.get("lane_type"),
                connection_type=cls._connector_connection_type(connector),
            )
            or str(connector.get("lane_type") or "").strip().lower()
            or DEFAULT_MEDIAN_LANE_TYPE
        )

    def _lane_id(self, meta: dict, jid: int, lane_index: int, role: str) -> int:
        """Map logical from/to lane index to OpenDRIVE signed lane id.

        Must match ``odr_left_right_lanes``: at the junction (successor) end,
        left driving_side puts incoming on right (-), outgoing on left (+);
        right driving_side flips those sides.

        Incoming ``from_lane_id`` 1 is the leftmost lane of that direction.
        Under right driving_side that lane is the outermost one, so its
        OpenDRIVE ``|id|`` is ``N`` rather than ``1``. Outgoing ``to_lane_id``
        is already innermost-first and is not reversed. Counts are omitted for
        dual-carriageway sections, whose links are already ``|id|`` pairs.
        """
        idx = abs(int(lane_index))
        at_successor = (
            True if meta["type"] == "road" else jid == meta["to_junction"]
        )
        # left: from → -, to → + at successor; right: opposite.
        positive = (role == "to") == at_successor
        if self.driving_side == "right":
            positive = not positive
        if role == "from" and self.driving_side == "right":
            count_key = "from_n_successor" if at_successor else "from_n_predecessor"
            n = int(meta.get(count_key) or 0)
            if n >= idx >= 1:
                idx = n - idx + 1
        signed = idx if positive else -idx
        median_n = int(meta.get("median_n") or 0)
        median_side = meta.get("median_side")
        if median_n and median_side == "right" and signed < 0:
            signed -= median_n
        elif median_n and median_side == "left" and signed > 0:
            signed += median_n
        return signed

    @staticmethod
    def _is_contiguous_abs_ids(lane_ids):
        abs_ids = [abs(int(x)) for x in lane_ids]
        return abs_ids == list(range(abs_ids[0], abs_ids[0] + len(abs_ids)))

    @classmethod
    def _connection_lane_bundles(cls, lane_pairs):
        """Keep contiguous 1:1 parallel links as one multilane bundle; split merges/diverges."""
        if not lane_pairs:
            return []
        from_ids = [fr for fr, _ in lane_pairs]
        to_ids = [to for _, to in lane_pairs]
        if (
            len(from_ids) == len(set(from_ids))
            and len(to_ids) == len(set(to_ids))
            and cls._is_contiguous_abs_ids(from_ids)
            and cls._is_contiguous_abs_ids(to_ids)
        ):
            return [lane_pairs]
        return [[pair] for pair in lane_pairs]

    def _solve_junction_positions(self) -> dict[int, tuple[float, float]]:
        junctions = self.cfg["junctions"]
        first = junctions[0]
        positions = {
            int(first["id"]): (
                float(first.get("x", 0.0)),
                float(first.get("y", 0.0)),
            )
        }
        # Single-junction files keep the original anchor-only placement.
        if len(junctions) == 1:
            return positions

        radii = {
            int(j["id"]): float(j["radius"])
            for j in junctions
        }

        driving = [
            c for c in self.cfg.get("connectors", [])
            if (not self._is_non_driving_connector(c)) or c.get("dual_carriageway")
        ]

        changed = True
        while changed:
            changed = False
            for c in driving:
                from_jid, to_jid = self._ordered_junctions(c)
                _, dx, dy = self._unit_vector(c["heading_deg"])
                if c.get("center_gap") is not None:
                    gap = float(c["center_gap"])
                else:
                    gap = radii[from_jid] + float(c["length"]) + radii[to_jid]

                if to_jid in positions and from_jid not in positions:
                    tx, ty = positions[to_jid]
                    positions[from_jid] = (tx - gap * dx, ty - gap * dy)
                    changed = True
                elif from_jid in positions and to_jid not in positions:
                    fx, fy = positions[from_jid]
                    positions[to_jid] = (fx + gap * dx, fy + gap * dy)
                    changed = True

        for j in junctions:
            jid = int(j["id"])
            if jid in positions:
                continue
            if "x" in j and "y" in j:
                positions[jid] = (float(j["x"]), float(j["y"]))

        return positions

    def _signed_heading_delta(self, from_deg, to_deg):
        return (float(to_deg) - float(from_deg) + 180.0) % 360.0 - 180.0

    def _axis_mean_heading(self, heading_a, heading_b):
        ha = float(heading_a) % 360.0
        hb = float(heading_b) % 360.0
        if abs(self._signed_heading_delta(ha, hb)) > 90.0:
            hb = (hb + 180.0) % 360.0
        return (ha + 0.5 * self._signed_heading_delta(ha, hb)) % 360.0

    def _infer_through_heading(self, jid, road_to_junction):
        headings = [
            float(r["heading_deg"])
            for r in self.cfg.get("roads", [])
            if road_to_junction.get(int(r["id"])) == int(jid)
        ]
        best = None
        for i, ha in enumerate(headings):
            for hb in headings[i + 1:]:
                err = abs(abs(self._signed_heading_delta(ha, hb)) - 180.0)
                if err > 20.0:
                    continue
                if best is None or err < best[0]:
                    best = (err, ha)
        return None if best is None else best[1]

    def _connector_through_heading(self, connector, road_to_junction):
        if connector.get("through_heading_deg") is not None:
            return float(connector["through_heading_deg"])
        a, b = map(int, connector["junctions"])
        ha = self._infer_through_heading(a, road_to_junction)
        hb = self._infer_through_heading(b, road_to_junction)
        if ha is not None and hb is not None:
            return self._axis_mean_heading(ha, hb)
        if ha is not None:
            return ha
        if hb is not None:
            return hb
        return float(connector["heading_deg"])

    def _through_roads(self, jid, road_to_junction):
        roads = [
            r for r in self.cfg.get("roads", [])
            if road_to_junction.get(int(r["id"])) == int(jid)
        ]
        best = None
        for i, ra in enumerate(roads):
            ha = float(ra["heading_deg"])
            for rb in roads[i + 1:]:
                hb = float(rb["heading_deg"])
                err = abs(abs(self._signed_heading_delta(ha, hb)) - 180.0)
                if err > 20.0:
                    continue
                if best is None or err < best[0]:
                    best = (err, ra, rb)
        if best is not None:
            return [best[1], best[2]]
        return roads[:1]

    def _through_half_width(self, jid, road_to_junction):
        plan = self._dual_plan_for_jid(jid)
        if plan is not None:
            n = (
                int(plan.get("plan_left_n") or 0)
                + int(plan.get("median_n") or 1)
                + int(plan.get("plan_right_n") or 0)
            )
            if n:
                return 0.5 * n * self.lane_width
        half = 0.0
        for r in self._through_roads(jid, road_to_junction):
            n = int(r.get("incoming_lanes") or 0) + int(r.get("outgoing_lanes") or 0)
            half = max(half, 0.5 * n * self.lane_width)
        return half or self.lane_width

    def _carriageway_lane_count(self, jid, road_to_junction):
        n = 0
        for r in self._through_roads(jid, road_to_junction):
            n = max(
                n,
                int(r.get("incoming_lanes") or 0) + int(r.get("outgoing_lanes") or 0),
            )
        return n or 1

    def _stem_outward_heading(self, jid, through, road_to_junction):
        for r in self.cfg.get("roads", []):
            if road_to_junction.get(int(r["id"])) != int(jid):
                continue
            heading = float(r["heading_deg"])
            if abs(self._signed_heading_delta(heading, through)) <= 25.0:
                continue
            if abs(self._signed_heading_delta(heading, through + 180.0)) <= 25.0:
                continue
            return heading
        for connector in self.cfg.get("connectors", []):
            if self._is_non_driving_connector(connector):
                continue
            if int(jid) not in map(int, connector["junctions"]):
                continue
            heading = float(connector["heading_deg"])
            from_jid, to_jid = self._ordered_junctions(connector)
            if int(jid) == to_jid:
                heading = (heading + 180.0) % 360.0
            return heading
        return None

    def _road_facing_twin(self, jid, my_xy, twin_xy, road_to_junction):
        target = math.degrees(math.atan2(twin_xy[1] - my_xy[1], twin_xy[0] - my_xy[0])) % 360.0
        best = None
        for r in self._through_roads(jid, road_to_junction):
            err = abs(self._signed_heading_delta(float(r["heading_deg"]), target))
            if best is None or err < best[0]:
                best = (err, r)
        return None if best is None else best[1]

    def _median_left_right_junctions(self, connector, through, road_to_junction, image_xy):
        a, b = map(int, connector["junctions"])
        _, ldx, ldy = self._unit_vector(through + 90.0)

        def side_sign(jid, other):
            stem = self._stem_outward_heading(jid, through, road_to_junction)
            if stem is not None:
                _, sx, sy = self._unit_vector(stem)
                return 1.0 if sx * ldx + sy * ldy >= 0.0 else -1.0
            vx = image_xy[jid][0] - image_xy[other][0]
            vy = image_xy[jid][1] - image_xy[other][1]
            return 1.0 if vx * ldx + vy * ldy >= 0.0 else -1.0

        sa, sb = side_sign(a, b), side_sign(b, a)
        if sa >= 0.0 and sb < 0.0:
            return a, b
        if sb >= 0.0 and sa < 0.0:
            return b, a
        return (a, b) if sa >= sb else (b, a)

    def _dual_section_counts(self, plan, s_along_through):
        left_n = int(plan["plan_left_n"])
        right_n = int(plan["plan_right_n"])
        median_n = int(plan.get("median_n") or 1)
        if s_along_through:
            return left_n, median_n + right_n, "right"
        return median_n + right_n, left_n, "left"

    def _s_along_through(self, heading_deg, through_deg):
        return abs(self._signed_heading_delta(heading_deg, through_deg)) <= 90.0

    def _set_center_adjacent_lane_type(self, road, side, lane_type_name) -> None:
        lane_type = getattr(xodr.LaneType, str(lane_type_name))
        for section in road.lanes.lanesections:
            lanes = section.leftlanes if side == "left" else section.rightlanes
            if lanes:
                lanes[0].lane_type = lane_type

    def _median_odr_id(self, meta):
        """OpenDRIVE lane id of the center-adjacent median on a dual section."""
        if not int((meta or {}).get("median_n") or 0):
            return None
        side = meta.get("median_side")
        if side == "left":
            return 1
        if side == "right":
            return -1
        return None

    @staticmethod
    def _road_link_id(link):
        if link is None:
            return None
        try:
            return int(link.element_id)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _lane_type_name(lane) -> str:
        value = getattr(lane, "lane_type", None)
        name = getattr(value, "name", None)
        return str(name or value or "").strip().lower()

    def _is_median_lane(self, lane) -> bool:
        return self._lane_type_name(lane) in ("restricted", "median")

    def _connecting_road_abuts_median(self, road, road_meta) -> bool:
        """True when this connecting road's innermost driving lane sits next to a median."""
        ends = (
            ("predecessor", self._road_link_id(getattr(road, "predecessor", None))),
            ("successor", self._road_link_id(getattr(road, "successor", None))),
        )
        for role, other_id in ends:
            median_id = self._median_odr_id(road_meta.get(other_id))
            if median_id is None:
                continue
            for section in road.lanes.lanesections:
                for lane in list(section.leftlanes) + list(section.rightlanes):
                    if self._is_median_lane(lane):
                        continue
                    linked = lane.get_linked_lane_id(role)
                    if linked is None:
                        continue
                    if abs(int(linked) - int(median_id)) == 1:
                        return True
        return False

    def _attach_median_lane(self, road) -> None:
        """Add a center-adjacent median lane on the empty side of a one-way through."""
        lane_type = getattr(xodr.LaneType, DEFAULT_MEDIAN_LANE_TYPE)
        for section in road.lanes.lanesections:
            if any(self._is_median_lane(lane) for lane in list(section.leftlanes) + list(section.rightlanes)):
                continue
            median = xodr.Lane(lane_type=lane_type, a=self.lane_width)
            median.add_roadmark(xodr.RoadMark(xodr.RoadMarkType.solid, 0.2))
            if section.rightlanes and not section.leftlanes:
                section.add_left_lane(median)
            elif section.leftlanes and not section.rightlanes:
                section.add_right_lane(median)
            elif not section.leftlanes and not section.rightlanes:
                section.add_left_lane(median)

    def _add_dual_median_through(self, junction, jid, road_meta, used_roads) -> None:
        """Continue the median as a non-driving lane on the through connecting road.

        Dual approach roads already carry a center-adjacent median. The through
        connecting road between those dual sections should include the same
        strip as a typed lane, not as a separate junction connection / laneLink.
        """
        plan = self._dual_plan_for_jid(jid)
        if plan is None:
            return
        cid = int(plan["id"])
        if cid not in used_roads:
            return
        if not int((road_meta.get(cid) or {}).get("median_n") or 0):
            return
        attached = set()
        for road in junction.junction_roads:
            pred_id = self._road_link_id(getattr(road, "predecessor", None))
            succ_id = self._road_link_id(getattr(road, "successor", None))
            if cid not in (pred_id, succ_id):
                continue
            other = succ_id if pred_id == cid else pred_id
            if not int((road_meta.get(other) or {}).get("median_n") or 0):
                continue
            pair = frozenset((pred_id, succ_id))
            if pair in attached:
                continue
            if not self._connecting_road_abuts_median(road, road_meta):
                continue
            self._attach_median_lane(road)
            attached.add(pair)

    def _rewrite_arm_id(self, junction_cfg, old_id, new_id) -> None:
        old_id, new_id = int(old_id), int(new_id)
        for conn in junction_cfg.get("connections") or []:
            if int(conn["from_arm"]) == old_id:
                conn["from_arm"] = new_id
            if int(conn["to_arm"]) == old_id:
                conn["to_arm"] = new_id

    def _promote_median_to_dual(self, road_to_junction) -> None:
        """Merge opposite one-way carriageways into one two-way road with a median lane.

        Refinement requires each median-linked junction's through carriageway to
        be one-way. Those two one-way roads become the left and right sides of
        this dual section; the median occupies the center-adjacent lane.
        """
        self._suppressed_road_ids = getattr(self, "_suppressed_road_ids", set())
        image_xy = {
            int(j["id"]): (float(j.get("x", 0.0)), float(j.get("y", 0.0)))
            for j in self.cfg["junctions"]
        }
        radii = {int(j["id"]): float(j["radius"]) for j in self.cfg["junctions"]}
        junctions = {int(j["id"]): j for j in self.cfg["junctions"]}
        for connector in self.cfg.get("connectors", []):
            if not self._is_non_driving_connector(connector):
                continue
            from_jid, to_jid = self._ordered_junctions(connector)
            through = self._connector_through_heading(connector, road_to_junction)
            left_jid, right_jid = self._median_left_right_junctions(
                connector, through, road_to_junction, image_xy,
            )
            fx, fy = image_xy[from_jid]
            tx, ty = image_xy[to_jid]
            _, tdx, tdy = self._unit_vector(through)
            stagger = (tx - fx) * tdx + (ty - fy) * tdy
            heading = through if stagger >= 0.0 else (through + 180.0) % 360.0
            facing_from = self._road_facing_twin(
                from_jid, image_xy[from_jid], image_xy[to_jid], road_to_junction,
            )
            facing_to = self._road_facing_twin(
                to_jid, image_xy[to_jid], image_xy[from_jid], road_to_junction,
            )
            cid = int(connector["id"])
            if facing_from is not None:
                self._suppressed_road_ids.add(int(facing_from["id"]))
                self._rewrite_arm_id(junctions[from_jid], facing_from["id"], cid)
            if facing_to is not None:
                self._suppressed_road_ids.add(int(facing_to["id"]))
                self._rewrite_arm_id(junctions[to_jid], facing_to["id"], cid)
            connector["dual_carriageway"] = True
            connector["heading_deg"] = heading
            connector["center_gap"] = abs(stagger)
            connector["length"] = max(1.0, abs(stagger) - float(radii[from_jid]) - float(radii[to_jid]))
            connector["through_heading_deg"] = through
            connector["plan_left_jid"] = left_jid
            connector["plan_right_jid"] = right_jid
            connector["plan_left_n"] = self._carriageway_lane_count(left_jid, road_to_junction)
            connector["plan_right_n"] = self._carriageway_lane_count(right_jid, road_to_junction)
            connector["median_n"] = 1
            s_along = self._s_along_through(heading, through)
            left_n, right_n, _ = self._dual_section_counts(connector, s_along)
            # Successor end of the connector is to_jid; store counts for create_road fallback.
            if s_along:
                connector["outgoing_lanes"] = left_n
                connector["incoming_lanes"] = right_n
            else:
                connector["outgoing_lanes"] = left_n
                connector["incoming_lanes"] = right_n
            self._rewrite_dual_through_connections(
                connector, road_to_junction, junctions,
            )

    def _dual_drive_lanes(self, plan) -> int:
        return max(
            1,
            min(
                int(plan.get("plan_left_n") or 1),
                int(plan.get("plan_right_n") or 1),
            ),
        )

    def _dual_outer_road_id(self, jid, connector, road_to_junction):
        """The remaining through arm at ``jid`` after the facing stub is suppressed."""
        through = float(connector["through_heading_deg"])
        best = None
        for road in self.cfg.get("roads", []):
            rid = int(road["id"])
            if rid in getattr(self, "_suppressed_road_ids", set()):
                continue
            if road_to_junction.get(rid) != int(jid):
                continue
            heading = float(road["heading_deg"])
            delta = min(
                abs(self._signed_heading_delta(heading, through)),
                abs(self._signed_heading_delta(heading, through + 180.0)),
            )
            if delta > 25.0:
                continue
            if best is None or delta < best[0]:
                best = (delta, rid)
        return None if best is None else best[1]

    def _upsert_lane_connection(self, junction_cfg, from_arm, to_arm, n_lanes) -> None:
        links = [
            {"from_lane_id": i, "to_lane_id": i}
            for i in range(1, int(n_lanes) + 1)
        ]
        for conn in junction_cfg.get("connections") or []:
            if int(conn["from_arm"]) == int(from_arm) and int(conn["to_arm"]) == int(to_arm):
                conn["lane_links"] = links
                return
        junction_cfg.setdefault("connections", []).append({
            "from_arm": int(from_arm),
            "to_arm": int(to_arm),
            "lane_links": links,
        })

    def _rewrite_dual_through_connections(self, connector, road_to_junction, junctions) -> None:
        """Both dual directions must continue through each T.

        Median merge turns two one-way throughs into one two-way section, but
        the original connections only keep the old one-way through. Without the
        reverse links, half the dual ends at the junction disk.
        """
        cid = int(connector["id"])
        n_lanes = self._dual_drive_lanes(connector)
        for jid in map(int, connector["junctions"]):
            outer_id = self._dual_outer_road_id(jid, connector, road_to_junction)
            if outer_id is None:
                continue
            self._upsert_lane_connection(junctions[jid], cid, outer_id, n_lanes)
            self._upsert_lane_connection(junctions[jid], outer_id, cid, n_lanes)

    def _dual_plan_for_jid(self, jid):
        for connector in self.cfg.get("connectors", []):
            if connector.get("dual_carriageway") and int(jid) in map(int, connector["junctions"]):
                return connector
        return None

    def _snap_median_through_headings(self, road_to_junction) -> None:
        """Align through arms of a median pair onto one divided-road axis."""
        for connector in self.cfg.get("connectors", []):
            if not self._is_non_driving_connector(connector):
                continue
            through = self._connector_through_heading(connector, road_to_junction)
            opposite = (through + 180.0) % 360.0
            jids = {int(x) for x in connector["junctions"]}
            for road in self.cfg.get("roads", []):
                if road_to_junction.get(int(road["id"])) not in jids:
                    continue
                heading = float(road["heading_deg"])
                d0 = abs(self._signed_heading_delta(heading, through))
                d180 = abs(self._signed_heading_delta(heading, opposite))
                if min(d0, d180) > 25.0:
                    continue
                road["heading_deg"] = through if d0 <= d180 else opposite

    def _effective_radii(self, radii, road_to_junction) -> dict[int, float]:
        """Size median-paired T disks to cover the dual, without overlapping."""
        out = {jid: float(radius) for jid, radius in radii.items()}
        for connector in self.cfg.get("connectors", []):
            if not self._is_non_driving_connector(connector):
                continue
            gap = float(connector.get("center_gap") or connector.get("length") or 0.0)
            max_r = max(10.0, 0.5 * gap - 1.0) if gap > 2.0 else None
            for jid in map(int, connector["junctions"]):
                half = self._through_half_width(jid, road_to_junction)
                radius = min(out[jid], max(10.0, 2.5 * half))
                if max_r is not None:
                    radius = min(radius, max_r)
                out[jid] = radius
        return out

    def _build_road_to_junction(self) -> dict[int, int]:
        junctions = self.cfg["junctions"]
        road_ids = {int(r["id"]) for r in self.cfg["roads"]}

        if len(junctions) == 1:
            jid = int(junctions[0]["id"])
            return {rid: jid for rid in road_ids}

        mapping = {}
        for j in junctions:
            jid = int(j["id"])
            for conn in j["connections"]:
                for key in ("from_arm", "to_arm"):
                    rid = int(conn[key])
                    if rid in road_ids:
                        mapping.setdefault(rid, jid)
        return mapping

    def _build_connection_biases(
        self,
        junction_cfg: dict,
    ) -> dict[tuple[int, int], tuple[float, float]]:
        connections = junction_cfg["connections"]

        max_split_length = 0.0
        max_merge_length = 0.0

        for conn in connections:
            if not conn.get("channelized", False):
                continue

            max_split_length = max(
                max_split_length,
                self._nonnegative(
                    conn.get("split_length", 0.0),
                    "split_length",
                ),
            )
            max_merge_length = max(
                max_merge_length,
                self._nonnegative(
                    conn.get("merge_length", 0.0),
                    "merge_length",
                ),
            )

        biases = {}

        for conn in connections:
            key = self._connection_key(conn)

            if conn.get("channelized", False):
                own_split = self._nonnegative(
                    conn.get("split_length", 0.0),
                    "split_length",
                )
                own_merge = self._nonnegative(
                    conn.get("merge_length", 0.0),
                    "merge_length",
                )

                split = max_split_length - own_split
                merge = max_merge_length - own_merge

            else:
                split = max_split_length
                merge = max_merge_length

                if "split_length" in conn:
                    split = self._nonnegative(
                        conn["split_length"],
                        "split_length",
                    )

                if "merge_length" in conn:
                    merge = self._nonnegative(
                        conn["merge_length"],
                        "merge_length",
                    )

            biases[key] = (split, merge)

        return biases

    def _add_external_roads(
        self,
        roads: dict,
        road_meta: dict,
        road_contact: dict,
        junction_radius: dict[int, float],
        junction_xy: dict[int, tuple[float, float]],
        road_to_junction: dict[int, int],
    ) -> None:
        extra_ids = getattr(self, "_suppressed_road_ids", set())
        for r in self.cfg["roads"]:
            rid = int(r["id"])
            if rid in extra_ids:
                continue
            jid = road_to_junction.get(rid)
            if jid is None:
                continue
            jx, jy = junction_xy[jid]
            length = float(r["length"])

            arm_heading, dx, dy = self._unit_vector(r["heading_deg"])
            road_heading = arm_heading + math.pi
            contact_x = jx + junction_radius[jid] * dx
            contact_y = jy + junction_radius[jid] * dy

            plan = self._dual_plan_for_jid(jid)
            if plan is not None:
                through = float(plan["through_heading_deg"])
                arm_h = float(r["heading_deg"])
                if (
                    abs(self._signed_heading_delta(arm_h, through)) > 25.0
                    and abs(self._signed_heading_delta(arm_h, through + 180.0)) > 25.0
                ):
                    plan = None
            median_side = None
            median_n = 0
            if plan is not None:
                through = float(plan["through_heading_deg"])
                s_along = self._s_along_through(math.degrees(road_heading) % 360.0, through)
                left_n, right_n, median_side = self._dual_section_counts(plan, s_along)
                median_n = int(plan.get("median_n") or 1)
            else:
                left_n, right_n = odr_left_right_lanes(
                    r.get("outgoing_lanes", 0), r.get("incoming_lanes", 0), self.driving_side,
                )
            road = xodr.create_road(
                xodr.Line(length),
                id=rid,
                left_lanes=left_n,
                right_lanes=right_n,
                lane_width=self.lane_width,
            )
            if median_side:
                self._set_center_adjacent_lane_type(road, median_side, DEFAULT_MEDIAN_LANE_TYPE)
            road.planview.set_start_point(
                contact_x + length * dx,
                contact_y + length * dy,
                road_heading,
            )

            roads[rid] = road
            meta = {"type": "road", "junction": jid}
            if plan is not None:
                meta["median_side"] = median_side
                meta["median_n"] = median_n
                meta["plan_left_jid"] = plan["plan_left_jid"]
                meta["plan_right_jid"] = plan["plan_right_jid"]
            else:
                meta["from_n_successor"] = int(r.get("incoming_lanes") or 0)
            road_meta[rid] = meta
            road_contact[rid] = {
                jid: (contact_x, contact_y, road_heading, "successor")
            }

    def _add_connector_roads(
        self,
        roads: dict,
        road_meta: dict,
        road_contact: dict,
        junction_radius: dict[int, float],
        junction_xy: dict[int, tuple[float, float]],
    ) -> None:
        for c in self.cfg.get("connectors", []):
            rid = int(c["id"])
            from_jid, to_jid = self._ordered_junctions(c)
            if self._is_non_driving_connector(c) and not c.get("dual_carriageway"):
                continue

            heading, dx, dy = self._unit_vector(c["heading_deg"])
            fx, fy = junction_xy[from_jid]
            tx, ty = junction_xy[to_jid]
            start_x = fx + junction_radius[from_jid] * dx
            start_y = fy + junction_radius[from_jid] * dy
            end_x = tx - junction_radius[to_jid] * dx
            end_y = ty - junction_radius[to_jid] * dy
            chord_x = end_x - start_x
            chord_y = end_y - start_y
            length = math.hypot(chord_x, chord_y)
            if length >= 1.0:
                heading = math.atan2(chord_y, chord_x)
            else:
                length = max(1.0, float(c["length"]))
            median_side = None
            median_n = 0
            if c.get("dual_carriageway"):
                through = float(c["through_heading_deg"])
                s_along = self._s_along_through(float(c["heading_deg"]), through)
                left_n, right_n, median_side = self._dual_section_counts(c, s_along)
                median_n = int(c.get("median_n") or 1)
            else:
                left_n, right_n = odr_left_right_lanes(
                    c.get("outgoing_lanes", c.get("backward_lanes", 0)),
                    c.get("incoming_lanes", c.get("forward_lanes", 0)),
                    self.driving_side,
                )
            road = xodr.create_road(
                xodr.Line(length),
                id=rid,
                left_lanes=left_n,
                right_lanes=right_n,
                lane_width=self.lane_width,
            )
            if median_side:
                self._set_center_adjacent_lane_type(road, median_side, DEFAULT_MEDIAN_LANE_TYPE)
            road.planview.set_start_point(start_x, start_y, heading)

            roads[rid] = road
            meta = {
                "type": "connector",
                "from_junction": from_jid,
                "to_junction": to_jid,
            }
            if c.get("dual_carriageway"):
                meta["median_side"] = median_side
                meta["median_n"] = median_n
                meta["plan_left_jid"] = c["plan_left_jid"]
                meta["plan_right_jid"] = c["plan_right_jid"]
            else:
                meta["from_n_successor"] = int(
                    c.get("incoming_lanes", c.get("forward_lanes", 0)) or 0
                )
                meta["from_n_predecessor"] = int(
                    c.get("outgoing_lanes", c.get("backward_lanes", 0)) or 0
                )
            road_meta[rid] = meta
            road_contact[rid] = {
                from_jid: (start_x, start_y, heading + math.pi, "predecessor"),
                to_jid: (end_x, end_y, heading, "successor"),
            }

    def _add_junctions(
        self,
        odr,
        roads: dict,
        road_meta: dict,
        road_contact: dict,
    ) -> list:
        created = []
        for j in self.cfg["junctions"]:
            jid = int(j["id"])
            junction = MultiLaneCommonJunctionCreator(
                id=jid,
                name=f"junction_{jid}",
                startnum=JUNCTION_ROAD_START_ID + jid * JUNCTION_ROAD_ID_STEP,
            )

            used_roads = {
                int(conn[key])
                for conn in j["connections"]
                for key in ("from_arm", "to_arm")
            }

            for rid in sorted(used_roads):
                if rid not in road_contact or jid not in road_contact[rid]:
                    continue
                x, y, heading, side = road_contact[rid][jid]
                junction.add_incoming_road_cartesian_geometry(
                    roads[rid],
                    x=x,
                    y=y,
                    heading=heading,
                    road_connection=side,
                )

            biases = self._build_connection_biases(j)

            for conn in j["connections"]:
                fr_id, to_id = self._connection_key(conn)
                if fr_id not in road_meta or to_id not in road_meta:
                    continue
                if (road_meta.get(fr_id) or {}).get("lane_type") in NON_DRIVING_LANE_TYPES:
                    continue
                if (road_meta.get(to_id) or {}).get("lane_type") in NON_DRIVING_LANE_TYPES:
                    continue

                lane_pairs = [
                    (
                        self._lane_id(road_meta[fr_id], jid, int(link["from_lane_id"]), "from"),
                        self._lane_id(road_meta[to_id], jid, int(link["to_lane_id"]), "to"),
                    )
                    for link in conn["lane_links"]
                ]
                lane_pairs.sort(key=lambda pair: (abs(pair[0]), abs(pair[1])))
                split, merge = biases[(fr_id, to_id)]

                for bundle in self._connection_lane_bundles(lane_pairs):
                    junction.add_multilane_connection(
                        road_one_id=fr_id,
                        road_two_id=to_id,
                        lane_one_ids=[fr for fr, _ in bundle],
                        lane_two_ids=[to for _, to in bundle],
                        straight_prefix_length=split,
                        straight_suffix_length=merge,
                    )

            odr.add_junction_creator(junction)
            created.append((junction, jid, used_roads))
        return created

    def road(self, **kwargs):
        junctions = self.cfg["junctions"]
        junction_radius = {
            int(j["id"]): float(j["radius"])
            for j in junctions
        }
        road_to_junction = self._build_road_to_junction()
        self._snap_median_through_headings(road_to_junction)
        self._promote_median_to_dual(road_to_junction)
        junction_xy = self._solve_junction_positions()
        road_to_junction = self._build_road_to_junction()
        junction_radius = self._effective_radii(junction_radius, road_to_junction)

        roads = {}
        road_meta = {}
        road_contact = {}

        self._add_external_roads(
            roads,
            road_meta,
            road_contact,
            junction_radius,
            junction_xy,
            road_to_junction,
        )
        self._add_connector_roads(
            roads,
            road_meta,
            road_contact,
            junction_radius,
            junction_xy,
        )

        odr = xodr.OpenDrive("scenario")

        for road in roads.values():
            odr.add_road(road)

        created = self._add_junctions(odr, roads, road_meta, road_contact)

        rule = self._traffic_rule()
        for road in odr.roads.values():
            road.rule = rule

        odr.adjust_roads_and_lanes()
        # After lane linking: attach median as a typed lane, never a junction laneLink.
        for junction, jid, used_roads in created:
            self._add_dual_median_through(junction, jid, road_meta, used_roads)
        return odr


def generate_xodr_from_connections(
    connections_path: Union[str, Path],
    *,
    output_dir: Optional[Union[str, Path]] = None,
    lane_width: float = 3.5,
    prettyprint_output: bool = False,
    run_esmini: bool = False,
    esmini_path: Optional[Union[str, Path]] = None,
    verbose: bool = True,
    **esmini_kwargs: Any,
) -> Path:
    """Build OpenDRIVE XML from a connections JSON via :class:`Scenario`.

    Args:
        run_esmini: If True, launch esmini to preview the generated road network.
        esmini_path: Directory containing the esmini executable. Defaults to
            :data:`~scripts.methods.road_utils.DEFAULT_ESMINI_PATH` when
            ``run_esmini`` is True.
        **esmini_kwargs: Extra keyword arguments forwarded to
            :func:`~scripts.methods.road_utils.run_esmini_for_scenario_generator`.
    """
    connections_path = Path(connections_path).resolve()
    output_dir = Path(output_dir or connections_path.parent).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    xodr_path = (output_dir / OUTPUT_FILENAME).resolve()

    scenario = Scenario(connections_path, lane_width=lane_width)
    if verbose:
        print("[INFO] Generating OpenDRIVE from connections...", flush=True)
    _, road_files = scenario.generate(
        generation_folder=str(output_dir),
        prettyprint=prettyprint_output,
    )

    if road_files:
        produced = Path(road_files[0]).resolve()
        if produced != xodr_path:
            shutil.copyfile(produced, xodr_path)
    if not xodr_path.exists():
        raise FileNotFoundError("OpenDRIVE was not written to: %s" % xodr_path)

    if verbose:
        print("[INFO] XODR written to: %s" % xodr_path, flush=True)

    if run_esmini:
        resolved_esmini = Path(esmini_path or DEFAULT_ESMINI_PATH).expanduser()
        if not resolved_esmini.is_dir():
            raise FileNotFoundError(
                "esmini directory not found: %s (set esmini_path=...)" % resolved_esmini
            )
        if verbose:
            print("[INFO] Launching esmini preview: %s" % resolved_esmini, flush=True)
        run_esmini_for_scenario_generator(
            scenario,
            str(resolved_esmini),
            **esmini_kwargs,
        )

    return xodr_path


def run_generate_roads(dataset="nhtsa_2023", case_id="28620", output_dir="results", model="openai/gpt-5.6-sol", reasoning="none", run_esmini=False, esmini_path=None, lane_width=3.5, verbose=True, connections_path=None):
    """Read ``connections.json`` and write ``scenario.xodr`` in the same directory.

    Default location is case-root ``refinement/{model}_{reasoning}/xodr_result/``,
    or ``junctions/{id}/refinement/.../xodr_result/`` for a single-junction layout.
    Pass ``connections_path`` to write next to that file.
    """
    run_dir = resolve_case_dir(output_dir, dataset, case_id)
    run_name = "%s/%s" % (dataset, case_id)
    if connections_path is not None:
        connections_path = Path(connections_path)
        xodr_dir = connections_path.parent.resolve()
    else:
        connections_path = connections_path_for_model(run_dir, model, reasoning)
        xodr_dir = connections_path.parent.resolve()
    if verbose:
        print("[INFO] case=%s model=%s reasoning=%s" % (run_name, model, reasoning), flush=True)
    xodr_path = generate_xodr_from_connections(
        connections_path,
        output_dir=xodr_dir,
        lane_width=lane_width,
        run_esmini=run_esmini,
        esmini_path=esmini_path,
        verbose=verbose,
    )
    return {"xodr_path": xodr_path, "connections_path": connections_path, "xodr_dir": xodr_dir, "run_dir": run_dir}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate OpenDRIVE from refinement/{model}_{reasoning}/xodr_result/connections.json.")
    parser.add_argument("--dataset", default="nhtsa_2023", help="Dataset folder under dataset/, e.g. nhtsa_2023")
    parser.add_argument("--case-id", default="28661", help="Case folder name under the dataset")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--model", default="openai/gpt-5.6-sol")
    parser.add_argument("--reasoning", default="none", help="Reasoning effort used in the refinement result folder name")
    parser.add_argument(
        "--run-esmini",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Launch esmini preview after XODR generation.",
    )
    parser.add_argument(
        "--esmini-path",
        default=DEFAULT_ESMINI_PATH,
        help="Directory containing the esmini executable.",
    )
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    run_generate_roads(
        dataset=args.dataset,
        case_id=args.case_id,
        output_dir=args.output_dir,
        model=args.model,
        reasoning=args.reasoning,
        run_esmini=args.run_esmini,
        esmini_path=args.esmini_path,
        verbose=args.verbose,
    )
