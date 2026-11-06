"""Image loading, resizing, and bbox helpers for vision LLM pipelines."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Union

from PIL import Image

BBox = tuple[float, float, float, float]
PxBox = tuple[int, int, int, int]
PathLike = Union[str, Path]


def crop_white_margins(image_path, output_dir, threshold=250, padding=20):
    """Crop near-white outer margins and save the resulting PNG."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    image = load_rgb_image(image_path)
    channel_mask = image.point(lambda value: 255 if value < threshold else 0)
    content_box = channel_mask.convert("L").getbbox()
    if content_box is None:
        content_box = (0, 0, image.width, image.height)
    left = max(0, content_box[0] - padding)
    top = max(0, content_box[1] - padding)
    right = min(image.width, content_box[2] + padding)
    bottom = min(image.height, content_box[3] + padding)
    crop_box = (left, top, right, bottom)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_cropped.png"
    image.crop(crop_box).save(output_path, format="PNG", optimize=True)
    return output_path


def pad_to_square(image_path, output_dir, fill=(255, 255, 255)):
    """Center-pad an image onto a square canvas and save as *_square.png."""
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    image = load_rgb_image(image_path)
    side = max(image.width, image.height)
    if image.width == image.height:
        canvas = image
    else:
        canvas = Image.new("RGB", (side, side), fill)
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_square.png"
    canvas.save(output_path, format="PNG", optimize=True)
    return output_path


def load_rgb_image(path: PathLike) -> Image.Image:
    return Image.open(path).convert("RGB")


def resize_long_edge(img: Image.Image, edge: int) -> Image.Image:
    ratio = edge / max(img.width, img.height)
    if abs(ratio - 1.0) < 1e-3:
        return img
    return img.resize(
        (max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
        Image.LANCZOS,
    )


DEFAULT_MAX_IMAGE_BYTES = 1024 * 1024


def _save_image_bytes(img, fmt, **save_kwargs):
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def encode_image_under_limit(img, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    """Encode image so raw bytes stay under max_bytes.

    Order: keep PNG if small enough; else grayscale + JPEG quality ladder;
    if still too large, repeatedly downscale and retry JPEG.
    Returns (raw_bytes, media_type).
    """
    raw = _save_image_bytes(img.convert("RGB"), "PNG", optimize=True)
    if len(raw) <= max_bytes:
        return raw, "image/png"

    work = img.convert("L")
    qualities = (85, 75, 65, 55, 45, 35, 25, 15)
    while True:
        for quality in qualities:
            raw = _save_image_bytes(work, "JPEG", quality=quality, optimize=True)
            if len(raw) <= max_bytes:
                return raw, "image/jpeg"
        if min(work.width, work.height) <= 64:
            raw = _save_image_bytes(work, "JPEG", quality=10, optimize=True)
            return raw, "image/jpeg"
        new_w = max(1, int(work.width * 0.85))
        new_h = max(1, int(work.height * 0.85))
        work = work.resize((new_w, new_h), Image.LANCZOS)


def compress_image_under_limit(img, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    """If image encoding exceeds max_bytes, compress it (grayscale/JPEG/resize) under the limit.

    Returns a PIL Image decoded from the compressed bytes.
    """
    raw, _media_type = encode_image_under_limit(img, max_bytes=max_bytes)
    out = Image.open(io.BytesIO(raw))
    out.load()
    return out


def save_image_under_limit(img, output_dir, stem, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    """Compress under max_bytes and save to output_dir as ``{stem}_compressed.{ext}``.

    Returns (output_path, raw_bytes, media_type).
    """
    raw, media_type = encode_image_under_limit(img, max_bytes=max_bytes)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if media_type == "image/jpeg" else "png"
    output_path = output_dir / f"{stem}_compressed.{ext}"
    output_path.write_bytes(raw)
    return output_path, raw, media_type


def encode_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.standard_b64encode(buf.getvalue()).decode()


def encode_image_b64_under_limit(img, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    raw, media_type = encode_image_under_limit(img, max_bytes=max_bytes)
    return base64.standard_b64encode(raw).decode(), media_type


def image_content_block(img: Image.Image) -> dict:
    data, media_type = encode_image_b64_under_limit(img)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }


def image_data_url_under_limit(img, max_bytes=DEFAULT_MAX_IMAGE_BYTES):
    raw, media_type = encode_image_under_limit(img, max_bytes=max_bytes)
    payload = base64.standard_b64encode(raw).decode("ascii")
    return "data:%s;base64,%s" % (media_type, payload)


def normalize_bbox(x0, y0, x1, y1) -> BBox:
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    return (
        max(0.0, x0),
        max(0.0, y0),
        min(1.0, x1),
        min(1.0, y1),
    )


def bbox_to_px(bbox: BBox, width: int, height: int) -> PxBox:
    return (
        int(bbox[0] * width),
        int(bbox[1] * height),
        int(bbox[2] * width),
        int(bbox[3] * height),
    )
