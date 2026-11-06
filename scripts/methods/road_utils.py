"""OpenDRIVE 道路生成共用工具：配置、几何、螺旋线后处理、esmini 临时目录、带清理的 Scenario 输出。"""

from __future__ import annotations

import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional, Union

from scenariogeneration import ScenarioGenerator

CURV_EPS = 1e-6
OUTPUT_FILENAME = "scenario.xodr"
DEFAULT_ESMINI_PATH = "esmini"
ESMINI_LEGACY_STAGING_DIR = "generated"


def load_connections(path: Union[str, Path] = "connections.json") -> dict[str, Any]:
    """Load a connections JSON file as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def postprocess_spirals(xodr_path: Union[str, Path]) -> None:
    """In-place: degenerate spirals -> line or arc; near-zero curvatures -> literal 0.0."""
    tree = ET.parse(xodr_path)
    root = tree.getroot()

    for geom in root.iter("geometry"):
        spiral = geom.find("spiral")
        if spiral is None:
            continue

        cs = float(spiral.get("curvStart", "0"))
        ce = float(spiral.get("curvEnd", "0"))

        cs_zero = abs(cs) < CURV_EPS
        ce_zero = abs(ce) < CURV_EPS

        if cs_zero and ce_zero:
            geom.remove(spiral)
            ET.SubElement(geom, "line")
        elif (
            not cs_zero
            and not ce_zero
            and abs(cs - ce) < max(CURV_EPS, CURV_EPS * max(abs(cs), abs(ce)))
        ):
            geom.remove(spiral)
            ET.SubElement(geom, "arc", {"curvature": str((cs + ce) / 2.0)})
        else:
            if cs_zero:
                spiral.set("curvStart", "0.0")
            if ce_zero:
                spiral.set("curvEnd", "0.0")

    tree.write(xodr_path, xml_declaration=True, encoding="utf-8")


def cleanup_esmini_default_staging_dir(base_dir: Optional[Union[str, Path]] = None) -> None:
    """Remove ``<base>/generated`` if present (esmini default staging path)."""
    root = Path.cwd() if base_dir is None else Path(base_dir)
    staging = root / ESMINI_LEGACY_STAGING_DIR
    if staging.is_dir():
        shutil.rmtree(staging, ignore_errors=True)


def run_esmini_for_scenario_generator(
    generator: ScenarioGenerator,
    esminipath: str,
    **esmini_kwargs: Any,
) -> None:
    """Run esmini with a temp ``generation_path``; drop leftover ``generated/`` in cwd."""
    from scenariogeneration import esmini

    esmini_kwargs.pop("generation_path", None)
    try:
        with tempfile.TemporaryDirectory(prefix="esmini_staging_") as tmp_dir:
            esmini(
                generator,
                esminipath,
                generation_path=tmp_dir,
                **esmini_kwargs,
            )
    finally:
        cleanup_esmini_default_staging_dir()


class SpiralCleanupScenario(ScenarioGenerator):
    """Generate into a temp dir, postprocess spirals, then copy results to generation_folder."""

    basename: str = "scenario"

    def generate(
        self,
        generation_folder=None,
        override_parameters=None,
        write_relative_road_path=True,
        name_separator=None,
        prettyprint=True,
    ):
        if generation_folder is None:
            generation_folder = "."

        generation_folder = Path(generation_folder)
        generation_folder.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="scenariogen_") as tmp_dir:
            scenario_files, road_files = super().generate(
                generation_folder=tmp_dir,
                override_parameters=override_parameters,
                write_relative_road_path=write_relative_road_path,
                name_separator=name_separator,
                prettyprint=prettyprint,
            )

            tmp_xodr_dir = Path(tmp_dir) / "xodr"
            produced = (
                sorted(tmp_xodr_dir.glob("*.xodr"))
                if tmp_xodr_dir.is_dir()
                else []
            )

            final_road_files: list[str] = []

            for i, src in enumerate(produced):
                postprocess_spirals(src)

                if i == 0:
                    dest = generation_folder / OUTPUT_FILENAME
                else:
                    dest = generation_folder / f"{Path(OUTPUT_FILENAME).stem}_{i}{Path(OUTPUT_FILENAME).suffix}"

                shutil.copyfile(src, dest)
                final_road_files.append(str(dest))

        return scenario_files, final_road_files


from typing import Sequence

import numpy as np
import pyclothoids as pcloth

from scenariogeneration.xodr import (
    CommonJunctionCreator,
    ContactPoint,
    ElementType,
    LaneDef,
    Line,
    Spiral,
    Connection,
    MixingDrivingDirection,
    UndefinedRoadNetwork,
    create_road,
)

STD_START_CLOTH = 1 / 1000000000

# 硬上限：prefix + suffix 必须严格小于弦长的此比例（_validate_straight_parts_length 据此报错）。
MAX_STRAIGHT_TOTAL_FRACTION = 0.75
# 最终 clamp 目标：把 prefix + suffix 收进弦长的此比例以内（略低于 0.75，留数值安全余量）。
# 上游（generate_roads 的 bias 方案）传来的任何长度都会被收进这个范围 -> 永不触发上面的硬上限。
STRAIGHT_CLAMP_FRACTION = 0.7
# 自动取值（prefix/suffix 传 None）时，用掉上面 clamp 预算的多大比例。
# 取小让中间曲线用满大部分弦长 -> 平滑、大半径转弯；接近 1 会退化成急弯。
STRAIGHT_RATIO_OF_MAX = 0.3


class MultiLaneCommonJunctionCreator(CommonJunctionCreator):
    def add_multilane_connection(
        self,
        road_one_id: int,
        road_two_id: int,
        lane_one_ids: Sequence[int],
        lane_two_ids: Sequence[int],
        straight_prefix_length: Optional[float] = 0.0,
        straight_suffix_length: Optional[float] = 0.0,
    ) -> None:
        lane_one_ids = list(lane_one_ids)
        lane_two_ids = list(lane_two_ids)

        self._validate_continuous_lane_ids(lane_one_ids, "lane_one_ids")
        self._validate_continuous_lane_ids(lane_two_ids, "lane_two_ids")

        if len(lane_one_ids) != len(lane_two_ids):
            raise ValueError(
                "lane_one_ids and lane_two_ids must have the same length"
            )

        idx1 = self._get_list_index(road_one_id)
        idx2 = self._get_list_index(road_two_id)

        lane_one_id = lane_one_ids[0]
        lane_two_id = lane_two_ids[0]

        if (
            self._get_connection_type(idx2) == self._get_connection_type(idx1)
            and np.sign(lane_one_id) == np.sign(lane_two_id)
        ) or (
            self._get_connection_type(idx2) != self._get_connection_type(idx1)
            and np.sign(lane_one_id) != np.sign(lane_two_id)
        ):
            raise MixingDrivingDirection(
                "driving direction not consistent when trying to make "
                "connection between roads:"
                + str(road_one_id)
                + " and "
                + str(road_two_id)
            )

        start_offset = self._get_offset_before_lane(idx1, lane_one_id)
        end_offset = self._get_offset_before_lane(idx2, lane_two_id)

        if self._get_connection_type(idx1) == "successor":
            angle_offset_start = np.sign(lane_one_id) * np.pi / 2
        else:
            angle_offset_start = -np.sign(lane_one_id) * np.pi / 2

        if self._get_connection_type(idx2) == "successor":
            angle_offset_end = -np.sign(lane_two_id) * np.pi / 2
        else:
            angle_offset_end = np.sign(lane_two_id) * np.pi / 2

        if self._circular_junction:
            an1 = self._angles[idx2] - self._angles[idx1] - np.pi

            if an1 > np.pi:
                an1 = -(2 * np.pi - an1)

            start_x = -self._radie[idx1]
            start_y = start_offset * np.sin(angle_offset_start)
            start_h = 0

            end_x = self._radie[idx2] * np.cos(an1) + end_offset * np.cos(
                an1 + angle_offset_end
            )
            end_y = self._radie[idx2] * np.sin(an1) + end_offset * np.sin(
                an1 + angle_offset_end
            )
            end_h = an1

        elif self._generic_junction:
            start_x = self._x[idx1] + start_offset * np.cos(
                self._h[idx1] + angle_offset_start
            )
            start_y = self._y[idx1] + start_offset * np.sin(
                self._h[idx1] + angle_offset_start
            )
            start_h = self._h[idx1]

            end_x = self._x[idx2] + end_offset * np.cos(
                self._h[idx2] - angle_offset_end
            )
            end_y = self._y[idx2] + end_offset * np.sin(
                self._h[idx2] - angle_offset_end
            )
            end_h = self._h[idx2] - np.pi

        else:
            raise UndefinedRoadNetwork("junction geometry is not defined")

        # 把 None 的 prefix/suffix 解析成“可用余量的 ratio”：需要 start/end 坐标，
        # 所以在几何端点算出来之后、生成连接道几何之前做这一步。
        straight_prefix_length, straight_suffix_length = self._resolve_straight_lengths(
            start_x=start_x,
            start_y=start_y,
            start_h=start_h,
            end_x=end_x,
            end_y=end_y,
            end_h=end_h,
            prefix_length=straight_prefix_length,
            suffix_length=straight_suffix_length,
        )

        roadgeoms = self._create_shifted_connection_geometry(
            start_x=start_x,
            start_y=start_y,
            start_h=start_h,
            end_x=end_x,
            end_y=end_y,
            end_h=end_h,
            straight_prefix_length=straight_prefix_length,
            straight_suffix_length=straight_suffix_length,
        )

        # Match create_road's sequential +=. Python 3.12+ sum() uses compensated
        # summation and can be 1 ULP shorter, which makes scenariogeneration pad
        # a leftover laneSection of length ~1e-14 that crashes CARLA.
        road_length = 0
        for geom in roadgeoms:
            road_length += geom.length
        lane_count = len(lane_one_ids)

        start_widths = [
            self._get_lane_width(lane_id, idx1) for lane_id in lane_one_ids
        ]
        end_widths = [
            self._get_lane_width(lane_id, idx2) for lane_id in lane_two_ids
        ]

        lane_def = LaneDef(
            0,
            road_length,
            lane_count,
            lane_count,
            None,
            start_widths,
            end_widths,
        )

        if self._get_connection_type(idx1) == "successor":
            if lane_one_id < 0:
                num_left_lanes = 0
                num_right_lanes = lane_def
                connecting_lane_ids = [-(i + 1) for i in range(lane_count)]
            else:
                num_left_lanes = lane_def
                num_right_lanes = 0
                connecting_lane_ids = [i + 1 for i in range(lane_count)]
        else:
            if lane_one_id < 0:
                num_left_lanes = lane_def
                num_right_lanes = 0
                connecting_lane_ids = [i + 1 for i in range(lane_count)]
            else:
                num_left_lanes = 0
                num_right_lanes = lane_def
                connecting_lane_ids = [-(i + 1) for i in range(lane_count)]

        tmp_junc_road = create_road(
            roadgeoms,
            self.startnum,
            left_lanes=num_left_lanes,
            right_lanes=num_right_lanes,
            lane_width=start_widths[0],
            road_type=self.id,
        )

        if self._height is not None:
            tmp_junc_road.add_elevation(0, self._height, 0, 0, 0)
            tmp_junc_road.add_superelevation(0, 0, 0, 0, 0)

        pred_lane_offset = np.sign(lane_one_id) * (abs(lane_one_id) - 1)

        if self._get_connection_type(idx1) == "predecessor":
            pred_lane_offset = -pred_lane_offset

        succ_lane_offset = np.sign(lane_two_id) * (abs(lane_two_id) - 1)

        if self._get_connection_type(idx2) == "predecessor":
            succ_lane_offset = -succ_lane_offset

        tmp_junc_road.add_predecessor(
            ElementType.road,
            road_one_id,
            contact_point=self._get_contact_point_connecting_road(
                road_one_id
            ),
            lane_offset=pred_lane_offset,
        )

        tmp_junc_road.add_successor(
            ElementType.road,
            road_two_id,
            contact_point=self._get_contact_point_connecting_road(
                road_two_id
            ),
            lane_offset=succ_lane_offset,
        )

        self._set_offset_for_incoming_road(
            idx1,
            tmp_junc_road.id,
            -pred_lane_offset,
        )

        self._set_offset_for_incoming_road(
            idx2,
            tmp_junc_road.id,
            -succ_lane_offset,
        )

        self.junction_roads.append(tmp_junc_road)

        connection = Connection(
            road_one_id,
            tmp_junc_road.id,
            ContactPoint.start,
        )

        for source_lane_id, connecting_lane_id in zip(
            lane_one_ids,
            connecting_lane_ids,
        ):
            connection.add_lanelink(source_lane_id, connecting_lane_id)

        self.junction.add_connection(connection)

        self.startnum += 1

    @classmethod
    def _resolve_straight_lengths(
        cls,
        start_x: float,
        start_y: float,
        start_h: float,
        end_x: float,
        end_y: float,
        end_h: float,
        prefix_length: Optional[float],
        suffix_length: Optional[float],
    ) -> tuple[float, float]:
        """解析并 clamp 直线 prefix/suffix 长度。

        None = 自动：按弦长取 STRAIGHT_RATIO_OF_MAX 份额的 clamp 预算；显式数值原样采用。
        最后无论来源，都把 prefix + suffix clamp 到 STRAIGHT_CLAMP_FRACTION * chord 以内
        （超出则两侧按比例缩小）。这样上游 generate_roads 的 bias 方案传进来的任何长度都不会
        超出几何可行范围，也不会触发 _validate_straight_parts_length 的硬上限报错。
        """
        chord_length = float(np.hypot(end_x - start_x, end_y - start_y))
        budget = STRAIGHT_CLAMP_FRACTION * chord_length

        if prefix_length is None or suffix_length is None:
            auto_total = STRAIGHT_RATIO_OF_MAX * budget
            fixed = (0.0 if prefix_length is None else float(prefix_length)) + (
                0.0 if suffix_length is None else float(suffix_length)
            )
            auto_count = (prefix_length is None) + (suffix_length is None)
            share = max(0.0, auto_total - fixed) / auto_count if auto_count else 0.0
            prefix = share if prefix_length is None else float(prefix_length)
            suffix = share if suffix_length is None else float(suffix_length)
        else:
            prefix = float(prefix_length)
            suffix = float(suffix_length)

        prefix = max(0.0, prefix)
        suffix = max(0.0, suffix)
        total = prefix + suffix
        if total > budget and total > 0.0:
            scale = budget / total
            prefix *= scale
            suffix *= scale
        return prefix, suffix

    def _create_shifted_connection_geometry(
        self,
        start_x: float,
        start_y: float,
        start_h: float,
        end_x: float,
        end_y: float,
        end_h: float,
        straight_prefix_length: float = 0.0,
        straight_suffix_length: float = 0.0,
    ) -> list:
        if straight_prefix_length < 0 or straight_suffix_length < 0:
            raise ValueError(
                "straight_prefix_length and straight_suffix_length "
                "must be non-negative"
            )

        if straight_prefix_length == 0 and straight_suffix_length == 0:
            return self._solve_g2_as_spirals(
                start_x=start_x,
                start_y=start_y,
                start_h=start_h,
                end_x=end_x,
                end_y=end_y,
                end_h=end_h,
            )

        return self._solve_g2_with_straight_prefix_and_suffix(
            start_x=start_x,
            start_y=start_y,
            start_h=start_h,
            end_x=end_x,
            end_y=end_y,
            end_h=end_h,
            prefix_length=straight_prefix_length,
            suffix_length=straight_suffix_length,
        )

    @staticmethod
    def _solve_g2_as_spirals(
        start_x: float,
        start_y: float,
        start_h: float,
        end_x: float,
        end_y: float,
        end_h: float,
    ) -> list:
        clothoids = pcloth.SolveG2(
            start_x,
            start_y,
            start_h,
            STD_START_CLOTH,
            end_x,
            end_y,
            end_h,
            STD_START_CLOTH,
        )

        return [
            Spiral(x.KappaStart, x.KappaEnd, length=x.length)
            for x in clothoids
        ]

    @classmethod
    def _solve_g2_with_straight_prefix_and_suffix(
        cls,
        start_x: float,
        start_y: float,
        start_h: float,
        end_x: float,
        end_y: float,
        end_h: float,
        prefix_length: float,
        suffix_length: float,
    ) -> list:
        cls._validate_straight_parts_length(
            start_x=start_x,
            start_y=start_y,
            start_h=start_h,
            end_x=end_x,
            end_y=end_y,
            end_h=end_h,
            prefix_length=prefix_length,
            suffix_length=suffix_length,
        )

        curve_start_x = start_x + prefix_length * np.cos(start_h)
        curve_start_y = start_y + prefix_length * np.sin(start_h)
        curve_start_h = start_h

        curve_end_x = end_x - suffix_length * np.cos(end_h)
        curve_end_y = end_y - suffix_length * np.sin(end_h)
        curve_end_h = end_h

        roadgeoms = []

        if prefix_length > 0:
            roadgeoms.append(Line(prefix_length))

        roadgeoms.extend(
            cls._solve_g2_as_spirals(
                start_x=curve_start_x,
                start_y=curve_start_y,
                start_h=curve_start_h,
                end_x=curve_end_x,
                end_y=curve_end_y,
                end_h=curve_end_h,
            )
        )

        if suffix_length > 0:
            roadgeoms.append(Line(suffix_length))

        return roadgeoms

    @staticmethod
    def _validate_straight_parts_length(
        start_x: float,
        start_y: float,
        start_h: float,
        end_x: float,
        end_y: float,
        end_h: float,
        prefix_length: float,
        suffix_length: float,
    ) -> None:
        chord_length = np.hypot(end_x - start_x, end_y - start_y)

        if chord_length <= 1e-6:
            raise ValueError("connection start and end points are too close")

        if prefix_length + suffix_length >= MAX_STRAIGHT_TOTAL_FRACTION * chord_length:
            raise ValueError(
                "straight prefix/suffix lengths are too large for this connection: "
                f"{prefix_length=}, {suffix_length=}, {chord_length=}"
            )

        curve_start_x = start_x + prefix_length * np.cos(start_h)
        curve_start_y = start_y + prefix_length * np.sin(start_h)

        curve_end_x = end_x - suffix_length * np.cos(end_h)
        curve_end_y = end_y - suffix_length * np.sin(end_h)

        curve_chord_length = np.hypot(
            curve_end_x - curve_start_x,
            curve_end_y - curve_start_y,
        )

        if curve_chord_length <= 1e-6:
            raise ValueError(
                "remaining curve part is too short after applying "
                "straight_prefix_length and straight_suffix_length"
            )

    @staticmethod
    def _validate_continuous_lane_ids(
        lane_ids: list[int],
        name: str,
    ) -> None:
        if not lane_ids:
            raise ValueError(f"{name} cannot be empty")

        if any(lane_id == 0 for lane_id in lane_ids):
            raise ValueError(f"{name} cannot contain lane id 0")

        signs = {np.sign(lane_id) for lane_id in lane_ids}

        if len(signs) != 1:
            raise ValueError(
                f"{name} must be on the same side of the road: {lane_ids}"
            )

        abs_ids = [abs(lane_id) for lane_id in lane_ids]
        expected = list(range(abs_ids[0], abs_ids[0] + len(abs_ids)))

        if abs_ids != expected:
            raise ValueError(
                f"{name} must be continuous and ordered from inner to outer, "
                f"got {lane_ids}, expected absolute lane ids {expected}"
            )

    def _get_offset_before_lane(self, road_idx: int, lane_id: int) -> float:
        lane_section = self.incoming_roads[road_idx].lanes.lanesections[
            self._get_connecting_lane_section(road_idx)
        ]

        offset = 0.0
        lane_count_before = abs(lane_id) - 1

        if lane_id < 0:
            lanes = lane_section.rightlanes
        else:
            lanes = lane_section.leftlanes

        if lane_count_before >= len(lanes):
            raise ValueError(
                f"lane id {lane_id} is out of range for road "
                f"{self.incoming_roads[road_idx].id}"
            )

        for lane_iter in range(lane_count_before):
            offset += lanes[lane_iter].get_width(0)

        return offset

    def _get_lane_width(self, lane_id: int, road_idx: int) -> float:
        lane_section = self.incoming_roads[road_idx].lanes.lanesections[
            self._get_connecting_lane_section(road_idx)
        ]

        lane_index = abs(lane_id) - 1

        if lane_id < 0:
            lanes = lane_section.rightlanes
        else:
            lanes = lane_section.leftlanes

        if lane_index >= len(lanes):
            raise ValueError(
                f"lane id {lane_id} is out of range for road "
                f"{self.incoming_roads[road_idx].id}"
            )

        return lanes[lane_index].get_width(0)