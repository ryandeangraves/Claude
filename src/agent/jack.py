"""Jack: the specialist for image-related tasks and large audits.

Frank (and any other agent) issues Jack a job when it involves generating
or auditing images, or auditing something large.  Image generation uses
``config.image_model`` (default ``gpt-image-2.5-sunburst``); audits run on
the Auditor model (``config.auditor_model``, default ``gpt-6-astra``).
"""
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from agents import Agent, ImageGenerationTool, RunConfig, RunContextWrapper, Runner, function_tool, set_default_openai_key

from src.agent.auditor import AuditReport, audit_image_async, audit_text_async
from src.agent.config import AgentConfig, load_config
from src.agent.image import GeneratedImage, extract_images, save_images

JACK_NAME = "Jack"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_AUDIT_IMAGE_BYTES = 20 * 1024 * 1024

JACK_INSTRUCTIONS = """\
You are Jack, the specialist for image work and large audits. Other agents
(usually Frank) issue you jobs; you do them and report back. If asked who
you are, say so.

Your jobs:
- generate images (image_generation tool),
- audit an image file (audit_image_file tool) against the request it was
  meant to satisfy,
- audit a large piece of text (audit_text tool) - a long document, a batch
  of items, or anything needing a thorough review.

When generating an image, write a clear, detailed prompt covering subject,
style, composition, lighting and any text that must appear, and produce
exactly one image unless asked for more. Never generate images of real,
identifiable people. For an audit, run the audit tool and report the
verdict, score and every failed item verbatim. Be concise; if a job is
outside image work or auditing, say so and do not attempt it.
"""


@dataclass
class JackContext:
    """Per-run state shared with Jack's tools."""

    config: AgentConfig
    out_dir: Optional[Path] = None
    audits: List[AuditReport] = field(default_factory=list)


@dataclass
class JackResult:
    """Everything Jack produced for one job."""

    output: str
    images: List[GeneratedImage] = field(default_factory=list)
    audits: List[AuditReport] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ---------------------------------------------------------------------------
# Audit tools (run the Auditor model)
# ---------------------------------------------------------------------------


@function_tool
async def audit_text(ctx: RunContextWrapper[JackContext], text: str, request: str) -> str:
    """Audit a piece of text against the request it was meant to satisfy.

    Args:
        text: The text to audit (an answer, a draft, a message).
        request: What the text was supposed to accomplish.
    """
    report = await audit_text_async(text, request, ctx.context.config)
    ctx.context.audits.append(report)
    return json.dumps(report.model_dump())


@function_tool
async def audit_image_file(ctx: RunContextWrapper[JackContext], path: str, request: str) -> str:
    """Audit an image file on disk against the request it was meant to satisfy.

    Args:
        path: Path to a png, jpg, jpeg, webp or gif file.
        request: What the image was supposed to show.
    """
    p = Path(path).expanduser()
    if p.suffix.lower() not in IMAGE_EXTENSIONS:
        return f"error: {path} is not an image file (expected one of {sorted(IMAGE_EXTENSIONS)})."
    if not p.is_file():
        return f"error: {path} does not exist."
    if p.stat().st_size > MAX_AUDIT_IMAGE_BYTES:
        return f"error: {path} is larger than {MAX_AUDIT_IMAGE_BYTES // (1024 * 1024)} MB."
    report = await audit_image_async(p, request, ctx.context.config)
    ctx.context.audits.append(report)
    return json.dumps(report.model_dump())


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def build_jack(
    config: AgentConfig,
    *,
    image_model: Optional[str] = None,
    size: str = "auto",
    quality: str = "auto",
    output_format: str = "png",
    background: str = "auto",
) -> Agent[JackContext]:
    """Create Jack (no network calls).

    ``image_model`` defaults to ``config.image_model`` (env ``OPENAI_IMAGE_MODEL``).
    """
    image_tool = ImageGenerationTool(
        tool_config={
            "type": "image_generation",
            "model": image_model or config.image_model,
            "size": size,
            "quality": quality,
            "output_format": output_format,
            "background": background,
        }
    )
    return Agent[JackContext](
        name=JACK_NAME,
        instructions=JACK_INSTRUCTIONS,
        model=config.model,
        tools=[image_tool, audit_text, audit_image_file],
    )


async def run_jack_async(
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    out_dir: Optional[Path] = None,
    **agent_kwargs,
) -> JackResult:
    """Give Jack a job. Any generated images are saved under ``out_dir``."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    config = config or load_config()
    set_default_openai_key(config.api_key, use_for_tracing=config.tracing_enabled)
    context = JackContext(config=config, out_dir=out_dir)

    result = await Runner.run(
        build_jack(config, **agent_kwargs),
        prompt,
        context=context,
        max_turns=config.max_turns,
        run_config=RunConfig(workflow_name="jack", tracing_disabled=not config.tracing_enabled),
    )

    images = extract_images(result)
    if out_dir is not None:
        save_images(images, out_dir, stem=prompt)

    usage = result.context_wrapper.usage
    return JackResult(
        output=str(result.final_output),
        images=images,
        audits=context.audits,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
    )


def run_jack(prompt: str, config: Optional[AgentConfig] = None, **kwargs) -> JackResult:
    """Synchronous wrapper around :func:`run_jack_async`."""
    return asyncio.run(run_jack_async(prompt, config, **kwargs))
