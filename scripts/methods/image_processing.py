"""Preprocess junction images for arm-level (and shared lane-level) inference.

Writes under ``{output_dir}/{dataset}/{case_id}/images/``:
  - ``*_cropped.png``
  - ``*_cropped_square.png`` (optional ``*_maxedge{N}.png``)
  - ``*_compressed.jpg|png``  (LLM input)
  - ``meta.json``             (paths + size metadata)
"""

import argparse
import json
from pathlib import Path

from scripts.methods.image_utils import (
    crop_white_margins,
    load_rgb_image,
    pad_to_square,
    resize_long_edge,
    save_image_under_limit,
)
from scripts.methods.llm_utils import dump_json
from scripts.methods.method_utils import prepare_output_dir, resolve_case_inputs, resolve_run_dir


def images_dir_for_case(case_dir):
    return Path(case_dir) / "images"


def meta_path_for_case(case_dir):
    return images_dir_for_case(case_dir) / "meta.json"


def _media_type_for_path(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _pick_unique_image(images_dir, patterns):
    for pattern in patterns:
        image_paths = sorted(Path(images_dir).glob(pattern))
        if len(image_paths) == 1:
            return image_paths[0].resolve()
        if len(image_paths) > 1:
            raise ValueError(f"multiple images matching {pattern!r} found in: {images_dir}")
    return None


def _resolve_from_meta(meta, *, compressed):
    key = "compressed" if compressed else "square"
    value = meta.get(key)
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return path.resolve()
    raise FileNotFoundError(f"meta.json points to missing {key} image: {path}")


def resolve_processed_image(case_dir, compressed=True):
    """Return a processed image path under ``case_dir/images``.

    compressed=True  -> ``*_compressed.*``
    compressed=False -> ``*_square.png``
    """
    case_dir = Path(case_dir)
    meta_path = meta_path_for_case(case_dir)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        picked = _resolve_from_meta(meta, compressed=compressed)
        if picked is not None:
            return picked
    images_dir = images_dir_for_case(case_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"processed images dir not found: {images_dir}")
    if compressed:
        patterns = ("*_compressed.jpg", "*_compressed.jpeg", "*_compressed.png")
        label = "compressed"
    else:
        patterns = ("*_square.png",)
        label = "square"
    picked = _pick_unique_image(images_dir, patterns)
    if picked is not None:
        return picked
    raise FileNotFoundError(f"processed {label} image not found in: {images_dir}")


def load_processed_image(case_dir, compressed=True):
    """Load the stored processed image.

    compressed selects compressed vs ``*_square.png``. Returns dict with keys:
    path, image, raw_bytes, media_type, meta (or None), compressed.
    """
    case_dir = Path(case_dir)
    path = resolve_processed_image(case_dir, compressed=compressed)
    meta = None
    meta_path = meta_path_for_case(case_dir)
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_bytes = path.read_bytes()
    if compressed and meta and meta.get("media_type"):
        media_type = meta["media_type"]
    else:
        media_type = _media_type_for_path(path)
    image = load_rgb_image(path)
    return {
        "path": path,
        "image": image,
        "raw_bytes": raw_bytes,
        "media_type": media_type,
        "meta": meta,
        "compressed": compressed,
    }


def process_case_image(dataset="nhtsa_2023", case_id="28661", output_dir="results", max_edge=None, images_dir=None):
    """Crop / pad / optionally resize / compress the case image into ``images/``.

    ``images_dir`` overrides the default ``{case_dir}/images`` (used when the
    processed scene lives under ``junctions/{id}/images``).
    """
    image_path, run_name = resolve_case_inputs(dataset=dataset, case_id=case_id)
    case_dir = resolve_run_dir(output_dir, run_name)
    target_dir = Path(images_dir) if images_dir is not None else images_dir_for_case(case_dir)
    images_dir = prepare_output_dir(target_dir, clear=images_dir is not None)

    cropped_path = crop_white_margins(image_path, images_dir)
    square_path = pad_to_square(cropped_path, images_dir)

    work_path = square_path
    image = load_rgb_image(work_path)
    resized_path = None
    if max_edge is not None:
        image = resize_long_edge(image, max_edge)
        resized_path = images_dir / f"{work_path.stem}_maxedge{max_edge}.png"
        image.save(resized_path, format="PNG", optimize=True)
        work_path = resized_path

    compressed_path, raw_bytes, media_type = save_image_under_limit(image, images_dir, work_path.stem)

    image = load_rgb_image(compressed_path)
    meta = {
        "name": run_name,
        "source": str(image_path.resolve()),
        "cropped": str(Path(cropped_path).resolve()),
        "square": str(Path(square_path).resolve()),
        "resized": str(Path(resized_path).resolve()) if resized_path is not None else None,
        "compressed": str(Path(compressed_path).resolve()),
        "max_edge": max_edge,
        "width": image.width,
        "height": image.height,
        "bytes": len(raw_bytes),
        "media_type": media_type,
    }
    dump_json(images_dir / "meta.json", meta)
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess a junction image for LLM-KG inference.")
    parser.add_argument("--dataset", default="openaerialmap")
    parser.add_argument("--case-id", default="2002")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--max-edge", type=int, default=None)
    args = parser.parse_args()

    process_case_image(
        dataset=args.dataset,
        case_id=args.case_id,
        output_dir=args.output_dir,
        max_edge=args.max_edge,
    )
