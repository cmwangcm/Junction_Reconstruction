#!/usr/bin/env python3
"""Convert ``dataset/nhtsa_raw/<year>/`` into ``dataset/nhtsa_<year>/<case_id>/``.

Each case becomes:

    <case_id>/
      <scene-diagram-stem>.pdf
      <scene-diagram-stem>.png
      meta.json                 # {"driving_side": "left"} unless already present

Scene Diagram PDFs come from ``cases/<id>/Docs/Scene Diagram/``.
Existing ``arms.json`` / ``lanes.json`` / ``meta.json`` are left untouched
(those are GT labels, not produced from the raw dump).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_reader import pdf_to_png

DATASET_ROOT = REPO_ROOT / "dataset"
NHTSA_RAW_ROOT = DATASET_ROOT / "nhtsa_raw"
DEFAULT_DRIVING_SIDE = "left"


def out_dataset_root(year, out_dataset=None):
    if out_dataset:
        name = str(out_dataset).strip("/\\")
        if name.startswith("dataset/"):
            name = name[len("dataset/") :]
        return DATASET_ROOT / name
    return DATASET_ROOT / ("nhtsa_%s" % year)


def iter_dirs(root):
    if not root.is_dir():
        return
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if path.is_dir() and not path.name.startswith("."):
            yield path


def scene_diagram_pdfs(case_dir):
    scene_dir = case_dir / "Docs" / "Scene Diagram"
    if not scene_dir.is_dir():
        return []
    seen = set()
    pdfs = []
    for pattern in ("*.pdf", "*.PDF"):
        for path in sorted(scene_dir.glob(pattern)):
            key = path.name.lower()
            if key in seen:
                continue
            seen.add(key)
            pdfs.append(path)
    return pdfs


def write_meta(out_case, driving_side, overwrite_meta):
    meta_path = out_case / "meta.json"
    if meta_path.is_file() and not overwrite_meta:
        return False
    meta_path.write_text(
        json.dumps({"driving_side": driving_side}, indent=2) + "\n",
        encoding="utf-8",
    )
    return True


def export_pdf(pdf, out_case, dpi, page_no, skip_existing):
    out_pdf = out_case / pdf.name
    out_png = out_case / (pdf.stem + ".png")
    if skip_existing and out_png.is_file() and out_pdf.is_file():
        return False
    out_case.mkdir(parents=True, exist_ok=True)
    if not out_pdf.is_file() or not skip_existing:
        shutil.copy2(pdf, out_pdf)
    if not out_png.is_file() or not skip_existing:
        pdf_to_png(pdf, out_png, dpi=dpi, page_no=page_no)
    return True


def convert_year(
    year,
    *,
    out_dataset=None,
    dpi=300,
    page_no=0,
    skip_existing=True,
    driving_side=DEFAULT_DRIVING_SIDE,
    overwrite_meta=False,
):
    cases_root = NHTSA_RAW_ROOT / str(year) / "cases"
    year_out = out_dataset_root(year, out_dataset)
    stats = {
        "converted": 0,
        "skipped_pdf": 0,
        "meta_written": 0,
        "no_pdf": 0,
        "cases": 0,
    }

    for case_dir in iter_dirs(cases_root):
        pdfs = scene_diagram_pdfs(case_dir)
        if not pdfs:
            stats["no_pdf"] += 1
            continue
        stats["cases"] += 1
        out_case = year_out / case_dir.name
        for pdf in pdfs:
            if export_pdf(pdf, out_case, dpi, page_no, skip_existing):
                stats["converted"] += 1
            else:
                stats["skipped_pdf"] += 1
        if write_meta(out_case, driving_side, overwrite_meta):
            stats["meta_written"] += 1

    return year_out, stats


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "将 dataset/nhtsa_raw/<year> 转成 dataset/nhtsa_<year>/<case_id>/ "
            "（Scene Diagram PDF + PNG + meta.json）。"
        )
    )
    parser.add_argument("--year", default="2022", help="nhtsa_raw 下的年份目录，如 2022")
    parser.add_argument(
        "--out-dataset",
        default=None,
        help="输出数据集名，默认 nhtsa_<year>，例如 nhtsa_2022",
    )
    parser.add_argument("--dpi", type=int, default=300, help="渲染 DPI（默认 300）")
    parser.add_argument("--page", type=int, default=0, help="PDF 页索引（默认 0）")
    parser.add_argument(
        "--driving-side",
        default=DEFAULT_DRIVING_SIDE,
        choices=("left", "right"),
        help="写入 meta.json 的 driving_side（与现有 nhtsa_2022 一致，默认 left）",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="目标 PDF+PNG 已存在则跳过渲染（默认开启）",
    )
    parser.add_argument(
        "--overwrite-meta",
        action="store_true",
        help="覆盖已有 meta.json；默认保留 GT 目录里已有的 meta",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cases_root = NHTSA_RAW_ROOT / args.year / "cases"
    if not cases_root.is_dir():
        print("错误: 未找到 cases 目录: %s" % cases_root, file=sys.stderr)
        return 1

    year_out, stats = convert_year(
        args.year,
        out_dataset=args.out_dataset,
        dpi=args.dpi,
        page_no=args.page,
        skip_existing=args.skip_existing,
        driving_side=args.driving_side,
        overwrite_meta=args.overwrite_meta,
    )
    print(
        "完成: cases=%d  新转 PDF/PNG=%d  跳过已有=%d  写入 meta=%d  无 Scene Diagram PDF=%d"
        % (
            stats["cases"],
            stats["converted"],
            stats["skipped_pdf"],
            stats["meta_written"],
            stats["no_pdf"],
        )
    )
    print("输出根目录: %s" % year_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
