"""Image-generation agent.

Uses the OpenAI Agents SDK's hosted ``ImageGenerationTool`` (backed by the
``gpt-image-*`` models via the Responses API).  Generated images are returned
as base64 and can be written to disk with :func:`save_images`.
"""
import base64
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from agents import Agent, ImageGenerationTool, RunConfig, Runner, set_default_openai_key
from openai.types.responses.response_output_item import ImageGenerationCall

from src.agent.config import AgentConfig, load_config

IMAGE_AGENT_NAME = "Image Generator"

IMAGE_INSTRUCTIONS = """\
You are an image-generation assistant.

When the user describes an image, call the image_generation tool with a
clear, detailed prompt that captures subject, style, composition, lighting
and any text that must appear. Generate exactly one image per request unless
the user explicitly asks for more. After generating, reply with one short
sentence describing what you produced. Do not generate images of real,
identifiable people.
"""


@dataclass
class GeneratedImage:
    """One image produced by the agent."""

    data: bytes  # decoded image bytes
    output_format: str  # png | webp | jpeg
    revised_prompt: Optional[str] = None
    size: Optional[str] = None
    quality: Optional[str] = None
    path: Optional[Path] = None  # set once saved to disk


@dataclass
class ImageRunResult:
    message: str
    images: List[GeneratedImage] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


def build_image_agent(
    config: AgentConfig,
    *,
    image_model: Optional[str] = None,
    size: str = "1024x1024",
    quality: str = "auto",
    output_format: str = "png",
    background: str = "auto",
) -> Agent:
    """Create the image-generation agent (no network calls).

    ``image_model`` defaults to ``config.image_model`` (env ``OPENAI_IMAGE_MODEL``).
    """
    tool = ImageGenerationTool(
        tool_config={
            "type": "image_generation",
            "model": image_model or config.image_model,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "background": background,
        }
    )
    return Agent(
        name=IMAGE_AGENT_NAME,
        instructions=IMAGE_INSTRUCTIONS,
        model=config.model,
        tools=[tool],
    )


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


def generate_image(
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    out_dir: Optional[Path] = None,
    **agent_kwargs,
) -> ImageRunResult:
    """Generate image(s) for ``prompt``; optionally save them under ``out_dir``."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    config = config or load_config()
    set_default_openai_key(config.api_key, use_for_tracing=config.tracing_enabled)

    result = Runner.run_sync(
        build_image_agent(config, **agent_kwargs),
        prompt,
        max_turns=config.max_turns,
        run_config=RunConfig(
            workflow_name="second-brain-image",
            tracing_disabled=not config.tracing_enabled,
        ),
    )

    images = extract_images(result)
    if out_dir is not None:
        save_images(images, out_dir, stem=prompt)

    usage = result.context_wrapper.usage
    return ImageRunResult(
        message=str(result.final_output),
        images=images,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
    )
