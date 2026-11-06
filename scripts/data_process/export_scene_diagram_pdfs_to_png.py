#!/usr/bin/env python3
"""
从 dataset/nhtsa_raw/<year>/cases/<case_id>/Docs/Scene Diagram/ 读取 PDF，
转为增强对比度 PNG，并连同 PDF 写入 dataset/nhtsa_<year>/<case_id>/。
同一 case 多份 PDF 时，各自输出为 <pdf 主文件名>.png。
"""

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.data_reader import pdf_to_png

DATASET_ROOT = REPO_ROOT / "dataset"
NHTSA_RAW_ROOT = DATASET_ROOT / "nhtsa_raw"


def out_year_root(year):
    return DATASET_ROOT / ("nhtsa_%s" % year)


def iter_dirs(root):
    if not root.is_dir():
        return
    for path in sorted(root.iterdir(), key=lambda p: p.name):
        if path.is_dir():
            yield path


def list_pdfs(folder):
    return sorted(folder.glob("*.pdf")) + sorted(folder.glob("*.PDF"))


def scene_diagram_pdfs(case_dir):
    scene_dir = case_dir / "Docs" / "Scene Diagram"
    if not scene_dir.is_dir():
        return []
    return list_pdfs(scene_dir)


def export_pdf(pdf, out_case, dpi, page_no, skip_existing):
    """Copy PDF and write PNG under out_case. Returns True if converted, False if skipped."""
    out_pdf = out_case / pdf.name
    out_png = out_case / (pdf.stem + ".png")
    if skip_existing and out_png.is_file():
        return False
    out_case.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, out_pdf)
    pdf_to_png(pdf, out_png, dpi=dpi, page_no=page_no)
    return True


def export_year(year, dpi, page_no, skip_existing):
    """Returns (converted, skipped_existing, cases_without_pdf)."""
    cases_root = NHTSA_RAW_ROOT / year / "cases"
    year_out = out_year_root(year)
    converted = 0
    skipped = 0
    no_pdf = 0

    for case_dir in iter_dirs(cases_root):
        pdfs = scene_diagram_pdfs(case_dir)
        if not pdfs:
            no_pdf += 1
            continue
        out_case = year_out / case_dir.name
        for pdf in pdfs:
            if export_pdf(pdf, out_case, dpi, page_no, skip_existing):
                converted += 1
            else:
                skipped += 1

    return converted, skipped, no_pdf


def parse_args():
    parser = argparse.ArgumentParser(
        description="将 nhtsa_raw 的 Scene Diagram PDF 转为 PNG，写入 dataset/nhtsa_<year>/<case_id>/。"
    )
    parser.add_argument("--year", default="2022", help="年份目录名，如 2022")
    parser.add_argument("--dpi", type=int, default=300, help="渲染 DPI（默认 300）")
    parser.add_argument("--page", type=int, default=0, help="PDF 页索引（默认 0）")
    parser.add_argument("--skip-existing", default=True, help="目标 PNG 已存在则跳过")
    return parser.parse_args()


def main():
    args = parse_args()
    cases_root = NHTSA_RAW_ROOT / args.year / "cases"
    if not cases_root.is_dir():
        print("错误: 未找到 cases 目录: %s" % cases_root, file=sys.stderr)
        return 1

    converted, skipped, no_pdf = export_year(
        args.year, args.dpi, args.page, args.skip_existing
    )
    print(
        "完成: 转换 %d 个 PDF，跳过已有 %d 个，无 Scene Diagram 或无 PDF 的 case 数 %d。"
        % (converted, skipped, no_pdf)
    )
    print("输出根目录: %s" % out_year_root(args.year))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
