"""Auditor agent.

Reviews an image (generated or supplied) against a checklist and returns a
structured :class:`AuditReport`.  The model is asked for a strict JSON schema
via ``output_type`` so the verdict is machine-readable.
"""
import base64
import mimetypes
from pathlib import Path
from typing import List, Literal, Optional, Union

from agents import Agent, RunConfig, Runner, set_default_openai_key
from pydantic import BaseModel, Field

from src.agent.config import AgentConfig, load_config

AUDITOR_NAME = "Image Auditor"

DEFAULT_CHECKLIST = [
    "The image matches the requested prompt (subject, style, composition).",
    "Any text in the image is spelled correctly and legible.",
    "No real, identifiable people, logos or trademarks appear.",
    "No unsafe, violent, sexual or hateful content.",
    "No obvious rendering defects (extra limbs, warped hands, garbled objects).",
]

AUDITOR_INSTRUCTIONS = """\
You are a meticulous image auditor.

You will be given an image and the prompt it was generated from, plus a
checklist. Inspect the image carefully and evaluate every checklist item.
Be specific: cite what you see. Do not be lenient — a single failed item
means the overall verdict is "fail". If you cannot see the image, say so and
mark the verdict "unclear".
"""


class ChecklistResult(BaseModel):
    item: str = Field(description="The checklist item, verbatim.")
    passed: bool
    notes: str = Field(description="What in the image supports this result.")


class AuditReport(BaseModel):
    verdict: Literal["pass", "fail", "unclear"]
    score: int = Field(ge=0, le=100, description="Overall quality 0-100.")
    summary: str
    results: List[ChecklistResult]
    suggested_prompt_fix: Optional[str] = Field(
        default=None,
        description="If the verdict is fail, a revised prompt likely to fix it.",
    )


def build_auditor_agent(config: AgentConfig) -> Agent:
    """Create the auditor agent (no network calls)."""
    return Agent(
        name=AUDITOR_NAME,
        instructions=AUDITOR_INSTRUCTIONS,
        model=config.model,
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


def build_audit_input(
    image: Union[bytes, Path, str],
    prompt: str,
    checklist: Optional[List[str]] = None,
    output_format: Optional[str] = None,
) -> list:
    """Build the multimodal input list for the auditor."""
    checklist = checklist or DEFAULT_CHECKLIST
    items = "\n".join(f"{i}. {c}" for i, c in enumerate(checklist, start=1))
    text = (
        f"Original prompt:\n{prompt}\n\n"
        f"Checklist:\n{items}\n\n"
        "Audit the attached image against every checklist item."
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": text},
                {
                    "type": "input_image",
                    "detail": "high",
                    "image_url": _to_data_url(image, output_format),
                },
            ],
        }
    ]


def audit_image(
    image: Union[bytes, Path, str],
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    checklist: Optional[List[str]] = None,
    output_format: Optional[str] = None,
) -> AuditReport:
    """Audit ``image`` (bytes or a path) against ``prompt`` and a checklist."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    config = config or load_config()
    set_default_openai_key(config.api_key, use_for_tracing=config.tracing_enabled)

    result = Runner.run_sync(
        build_auditor_agent(config),
        build_audit_input(image, prompt, checklist, output_format),
        max_turns=config.max_turns,
        run_config=RunConfig(
            workflow_name="second-brain-audit",
            tracing_disabled=not config.tracing_enabled,
        ),
    )
    return result.final_output_as(AuditReport)
