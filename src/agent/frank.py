"""Frank: the top-level agent the user talks to.

Frank holds the conversation and issues jobs to Jack for anything concrete:
validating contact details, drafting notifications, generating images, and
auditing text or images ("Hey Frank, audit this").  With a session name,
Frank remembers earlier turns, so "audit this" can refer to something Jack
produced previously.

If ``auto_audit`` is on, every job's output is also audited automatically
and retried with the Auditor's suggested fix.
"""
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

from agents import Agent, RunConfig, RunContextWrapper, Runner, SQLiteSession, function_tool, set_default_openai_key

from src.agent.config import AgentConfig, load_config
from src.agent.jack import JackResult, run_jack_async
from src.agent.pipeline import PipelineResult, run_audited_async

FRANK_NAME = "Frank"
DEFAULT_SESSION_DB = Path("generated") / "frank-sessions.db"

FRANK_INSTRUCTIONS = """\
You are Frank, the Second Brain assistant. If asked who you are, say so.

You talk with the user and decide what needs doing. You do not do concrete
work yourself: you issue jobs to Jack with the issue_job tool. Jobs include
validating an email address or phone number, drafting an order confirmation
or shipping message, generating an image, and auditing a piece of text or an
image file. Write each job as a clear, self-contained request; when the
user says "audit this", work out from the conversation what "this" is
(quote the text, or give the saved image path) and what it was meant to
satisfy, and put both in the job.

Report Jack's results faithfully, including audit verdicts and failed items.
Answer general questions directly. Be concise.
"""


@dataclass
class FrankContext:
    """Per-run state shared with Frank's tools."""

    config: AgentConfig
    out_dir: Optional[Path] = None
    auto_audit: bool = False
    max_attempts: int = 1
    jobs: List[Union[JackResult, PipelineResult]] = field(default_factory=list)


@dataclass
class FrankResult:
    output: str
    jobs: List[Union[JackResult, PipelineResult]]
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def _describe_report(label: str, rep) -> List[str]:
    failed = [c.item for c in rep.results if not c.passed]
    lines = [f"- audit of {label}: {rep.verdict} (score {rep.score}). {rep.summary}"]
    if failed:
        lines.append("  failed items: " + "; ".join(failed))
    return lines


def _summarise_job(job: JackResult) -> str:
    lines = [f"Jack's report:\n{job.output}"]
    for img in job.images:
        lines.append(f"Generated image saved at: {img.path}" if img.path else "Generated an image (not saved).")
    for i, rep in enumerate(job.audits, start=1):
        lines.extend(_describe_report(f"item {i}", rep))
    return "\n".join(lines)


def _summarise_pipeline(pipeline: PipelineResult) -> str:
    final = pipeline.final
    if final is None:
        return "Jack produced nothing."
    lines = [_summarise_job(final.result)]
    lines.append(f"Automatic audit: {'PASS' if final.passed else 'FAIL'} after {len(pipeline.attempts)} attempt(s).")
    lines.extend(_describe_report("text", final.text_report))
    for i, rep in enumerate(final.image_reports, start=1):
        lines.extend(_describe_report(f"image {i}", rep))
    return "\n".join(lines)


@function_tool
async def issue_job(ctx: RunContextWrapper[FrankContext], request: str) -> str:
    """Issue a job to Jack and get back his report.

    Args:
        request: A clear, self-contained description of the job. For an
            audit, include what to audit (the text, or the image file path)
            and what it was meant to satisfy.
    """
    fc = ctx.context
    if fc.auto_audit:
        pipeline = await run_audited_async(request, fc.config, out_dir=fc.out_dir, max_attempts=fc.max_attempts)
        fc.jobs.append(pipeline)
        return _summarise_pipeline(pipeline)
    job = await run_jack_async(request, fc.config, out_dir=fc.out_dir)
    fc.jobs.append(job)
    return _summarise_job(job)


def build_frank(config: AgentConfig) -> Agent[FrankContext]:
    """Create Frank (no network calls)."""
    return Agent[FrankContext](
        name=FRANK_NAME,
        instructions=FRANK_INSTRUCTIONS,
        model=config.model,
        tools=[issue_job],
    )


async def run_frank_async(
    prompt: str,
    config: Optional[AgentConfig] = None,
    *,
    out_dir: Optional[Path] = None,
    auto_audit: bool = False,
    max_attempts: int = 1,
    session: Optional[str] = None,
    session_db: Optional[Path] = None,
) -> FrankResult:
    """Talk to Frank. ``session`` names a persistent conversation to continue."""
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    config = config or load_config()
    set_default_openai_key(config.api_key, use_for_tracing=config.tracing_enabled)
    context = FrankContext(config=config, out_dir=out_dir, auto_audit=auto_audit, max_attempts=max_attempts)

    memory = None
    if session:
        db = Path(session_db or DEFAULT_SESSION_DB)
        db.parent.mkdir(parents=True, exist_ok=True)
        memory = SQLiteSession(session, db)

    result = await Runner.run(
        build_frank(config),
        prompt,
        context=context,
        session=memory,
        max_turns=config.max_turns,
        run_config=RunConfig(workflow_name="frank", tracing_disabled=not config.tracing_enabled),
    )

    usage = result.context_wrapper.usage
    return FrankResult(
        output=str(result.final_output),
        jobs=context.jobs,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
    )


def run_frank(prompt: str, config: Optional[AgentConfig] = None, **kwargs) -> FrankResult:
    """Synchronous wrapper around :func:`run_frank_async`."""
    return asyncio.run(run_frank_async(prompt, config, **kwargs))
