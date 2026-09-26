"""Generate an image, then have the auditor review it.

Optionally retries generation using the auditor's suggested prompt fix.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.agent.auditor import AuditReport, audit_image
from src.agent.config import AgentConfig, load_config
from src.agent.image import GeneratedImage, generate_image


@dataclass
class Attempt:
    prompt: str
    image: Optional[GeneratedImage]
    report: Optional[AuditReport]


@dataclass
class PipelineResult:
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def final(self) -> Optional[Attempt]:
        return self.attempts[-1] if self.attempts else None

    @property
    def passed(self) -> bool:
        return bool(self.final and self.final.report and self.final.report.verdict == "pass")


def generate_and_audit(
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    out_dir: Optional[Path] = None,
    max_attempts: int = 1,
    checklist: Optional[List[str]] = None,
) -> PipelineResult:
    """Generate, audit, and (if ``max_attempts`` > 1) retry on a failed audit."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    config = config or load_config()
    result = PipelineResult()
    current_prompt = prompt

    for _ in range(max_attempts):
        gen = generate_image(current_prompt, config, out_dir=out_dir)
        if not gen.images:
            result.attempts.append(Attempt(current_prompt, None, None))
            break

        image = gen.images[0]
        report = audit_image(
            image.data,
            current_prompt,
            config,
            checklist=checklist,
            output_format=image.output_format,
        )
        result.attempts.append(Attempt(current_prompt, image, report))

        if report.verdict == "pass" or not report.suggested_prompt_fix:
            break
        current_prompt = report.suggested_prompt_fix

    return result
