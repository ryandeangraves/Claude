"""Helpers for images produced by the hosted ``image_generation`` tool."""
import base64
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai.types.responses.response_output_item import ImageGenerationCall


@dataclass
class GeneratedImage:
    """One image produced by the agent."""

    data: bytes  # decoded image bytes
    output_format: str  # png | webp | jpeg
    revised_prompt: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    path: Optional[Path] = None  # set once saved to disk


def extract_images(result) -> List[GeneratedImage]:
    """Pull every completed image out of a ``RunResult``."""
    images: List[GeneratedImage] = []
    for item in result.new_items:
        raw = getattr(item, "raw_item", None)
        if isinstance(raw, ImageGenerationCall) and raw.result:
            images.append(
                GeneratedImage(
                    data=base64.b64decode(raw.result),
                    output_format=raw.output_format or "png",
                    revised_prompt=raw.revised_prompt,
                    size=raw.size,
                    quality=raw.quality,
                )
            )
    return images


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit] or "image"


def save_images(images: List[GeneratedImage], out_dir: Path, stem: str = "image") -> List[Path]:
    """Write images to ``out_dir`` and record each path on the image."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    paths: List[Path] = []
    for i, img in enumerate(images, start=1):
        path = out_dir / f"{ts}-{_slug(stem)}-{i}.{img.output_format}"
        path.write_bytes(img.data)
        img.path = path
        paths.append(path)
    return paths
