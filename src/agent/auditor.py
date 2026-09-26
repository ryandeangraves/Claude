"""The Auditor: reviews everything Jack produces.

It audits a text answer or an image against a checklist and returns a
structured :class:`AuditReport` (enforced via ``output_type``).  It runs on
``config.auditor_model`` (default ``gpt-6-astra``).
"""
import asyncio
import base64
import mimetypes
from pathlib import Path
from typing import List, Literal, Optional, Union

from agents import Agent, RunConfig, Runner, set_default_openai_key
from pydantic import BaseModel, Field

from src.agent.config import AgentConfig, load_config

AUDITOR_NAME = "Auditor"

DEFAULT_TEXT_CHECKLIST = [
    "The answer actually addresses what the user asked.",
    "Every factual claim is correct and consistent with the tools' results.",
    "Nothing is invented: no made-up ids, prices, addresses or tool outputs.",
    "No unsafe, harmful, or policy-violating content.",
    "The response is clear, concise and free of contradictions.",
]

DEFAULT_IMAGE_CHECKLIST = [
    "The image matches the requested prompt (subject, style, composition).",
    "Any text in the image is spelled correctly and legible.",
    "No real, identifiable people, logos or trademarks appear.",
    "No unsafe, violent, sexual or hateful content.",
    "No obvious rendering defects (extra limbs, warped hands, garbled objects).",
]

AUDITOR_INSTRUCTIONS = """\
You are the Auditor. You review work produced by Jack, an assistant that
answers questions, drafts notifications and generates images.

You will be given the user's original request, the item to audit (a text
answer or an image), and a checklist. Evaluate every checklist item
carefully and specifically, citing what you see. Do not be lenient: a single
failed item means the overall verdict is "fail". If the item is missing or
you cannot inspect it, mark the verdict "unclear". When the verdict is
"fail", suggest a revised request that would likely fix the problem.
"""


class ChecklistResult(BaseModel):
    item: str = Field(description="The checklist item, verbatim.")
    passed: bool
    notes: str = Field(description="What in the audited item supports this result.")


class AuditReport(BaseModel):
    verdict: Literal["pass", "fail", "unclear"]
    score: int = Field(ge=0, le=100, description="Overall quality 0-100.")
    summary: str
    results: List[ChecklistResult]
    suggested_prompt_fix: Optional[str] = Field(
        default=None,
        description="If the verdict is fail, a revised request likely to fix it.",
    )


def build_auditor(config: AgentConfig) -> Agent:
    """Create the Auditor (no network calls)."""
    return Agent(
        name=AUDITOR_NAME,
        instructions=AUDITOR_INSTRUCTIONS,
        model=config.auditor_model,
        output_type=AuditReport,
    )


def _to_data_url(image: Union[bytes, Path, str], output_format: Optional[str] = None) -> str:
    if isinstance(image, (str, Path)):
        path = Path(image)
        data = path.read_bytes()
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
    else:
        data = image
        mime = f"image/{output_format or 'png'}"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _checklist_block(checklist: List[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(checklist, start=1))


def build_text_audit_input(answer: str, prompt: str, checklist: Optional[List[str]] = None) -> list:
    """Build the input for auditing a text answer."""
    checklist = checklist or DEFAULT_TEXT_CHECKLIST
    text = (
        f"User's request:\n{prompt}\n\n"
        f"Jack's answer:\n{answer}\n\n"
        f"Checklist:\n{_checklist_block(checklist)}\n\n"
        "Audit Jack's answer against every checklist item."
    )
    return [{"role": "user", "content": [{"type": "input_text", "text": text}]}]


def build_image_audit_input(
    image: Union[bytes, Path, str],
    prompt: str,
    checklist: Optional[List[str]] = None,
    output_format: Optional[str] = None,
) -> list:
    """Build the multimodal input for auditing an image."""
    checklist = checklist or DEFAULT_IMAGE_CHECKLIST
    text = (
        f"User's request:\n{prompt}\n\n"
        f"Checklist:\n{_checklist_block(checklist)}\n\n"
        "Audit the attached image against every checklist item."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {"type": "input_image", "detail": "high", "image_url": _to_data_url(image, output_format)},
            ],
        }
    ]


async def _run(config: AgentConfig, inp: list) -> AuditReport:
    set_default_openai_key(config.api_key, use_for_tracing=config.tracing_enabled)
    result = await Runner.run(
        build_auditor(config),
        inp,
        max_turns=config.max_turns,
        run_config=RunConfig(workflow_name="auditor", tracing_disabled=not config.tracing_enabled),
    )
    return result.final_output_as(AuditReport)


async def audit_text_async(
    answer: str,
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    checklist: Optional[List[str]] = None,
) -> AuditReport:
    """Audit a text answer Jack gave for ``prompt``."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    config = config or load_config()
    return await _run(config, build_text_audit_input(answer, prompt, checklist))


async def audit_image_async(
    image: Union[bytes, Path, str],
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    checklist: Optional[List[str]] = None,
    output_format: Optional[str] = None,
) -> AuditReport:
    """Audit an image (bytes or a path) Jack generated for ``prompt``."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    config = config or load_config()
    return await _run(config, build_image_audit_input(image, prompt, checklist, output_format))


def audit_text(answer: str, prompt: str, config: Optional[AgentConfig] = None, **kw) -> AuditReport:
    """Synchronous wrapper around :func:`audit_text_async`."""
    return asyncio.run(audit_text_async(answer, prompt, config, **kw))


def audit_image(image, prompt: str, config: Optional[AgentConfig] = None, **kw) -> AuditReport:
    """Synchronous wrapper around :func:`audit_image_async`."""
    return asyncio.run(audit_image_async(image, prompt, config, **kw))
