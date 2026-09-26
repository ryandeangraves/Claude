"""Run Jack, then have the Auditor review everything he produced.

The text answer is always audited. Every generated image is audited too.
If anything fails and the Auditor suggests a fix, the request is retried
with that fix, up to ``max_attempts`` times.
"""
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from src.agent.auditor import AuditReport, audit_image_async, audit_text_async
from src.agent.config import AgentConfig, load_config
from src.agent.jack import JackResult, run_jack_async


@dataclass
class Attempt:
    prompt: str
    result: JackResult
    text_report: AuditReport
    image_reports: List[AuditReport] = field(default_factory=list)

    @property
    def reports(self) -> List[AuditReport]:
        return [self.text_report, *self.image_reports]

    @property
    def passed(self) -> bool:
        return all(r.verdict == "pass" for r in self.reports)

    @property
    def suggested_fix(self) -> Optional[str]:
        for r in self.reports:
            if r.verdict != "pass" and r.suggested_prompt_fix:
                return r.suggested_prompt_fix
        return None


@dataclass
class PipelineResult:
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def final(self) -> Optional[Attempt]:
        return self.attempts[-1] if self.attempts else None

    @property
    def passed(self) -> bool:
        return bool(self.final and self.final.passed)


async def run_audited_async(
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    out_dir: Optional[Path] = None,
    max_attempts: int = 1,
    text_checklist: Optional[List[str]] = None,
    image_checklist: Optional[List[str]] = None,
    **jack_kwargs,
) -> PipelineResult:
    """Run Jack and audit his text and images; retry on failure if allowed."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    config = config or load_config()
    result = PipelineResult()
    current_prompt = prompt

    for _ in range(max_attempts):
        jack = await run_jack_async(current_prompt, config, out_dir=out_dir, **jack_kwargs)
        text_report = await audit_text_async(jack.output, current_prompt, config, checklist=text_checklist)
        image_reports = [
            await audit_image_async(
                img.data, current_prompt, config, checklist=image_checklist, output_format=img.output_format
            )
            for img in jack.images
        ]
        attempt = Attempt(current_prompt, jack, text_report, image_reports)
        result.attempts.append(attempt)

        if attempt.passed or not attempt.suggested_fix:
            break
        current_prompt = attempt.suggested_fix

    return result


def run_audited(prompt: str, config: Optional[AgentConfig] = None, **kwargs) -> PipelineResult:
    """Synchronous wrapper around :func:`run_audited_async`."""
    return asyncio.run(run_audited_async(prompt, config, **kwargs))
