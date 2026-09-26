"""Build and run the OpenAI agent."""
from dataclasses import dataclass
from typing import Optional

from agents import Agent, RunConfig, Runner, set_default_openai_key

from src.agent.config import AgentConfig, load_config
from src.agent.tools import ALL_TOOLS

AGENT_NAME = "Second Brain Assistant"

INSTRUCTIONS = """\
You are the assistant for the Second Brain application.

You can validate email addresses and phone numbers and draft (never send)
order-confirmation emails and shipping SMS messages using the tools provided.
Use a tool whenever the answer depends on it rather than guessing.
Be concise. If a request is outside what your tools can do, say so plainly.
"""


@dataclass
class AgentRunResult:
    """A small, serialisable summary of an agent run."""

    output: str
    input_tokens: int
    output_tokens: int
    requests: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def build_agent(config: AgentConfig) -> Agent:
    """Create the agent (does not make any network calls)."""
    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        model=config.model,
        tools=list(ALL_TOOLS),
    )


def _run_config(config: AgentConfig) -> RunConfig:
    return RunConfig(
        workflow_name="second-brain-agent",
        tracing_disabled=not config.tracing_enabled,
    )


def run_agent(prompt: str, config: Optional[AgentConfig] = None) -> AgentRunResult:
    """Run the agent on ``prompt`` and return its final answer plus token usage.

    ``config`` defaults to :func:`load_config`, which reads OPENAI_API_KEY from
    the environment.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")

    config = config or load_config()
    set_default_openai_key(config.api_key, use_for_tracing=config.tracing_enabled)

    result = Runner.run_sync(
        build_agent(config),
        prompt,
        max_turns=config.max_turns,
        run_config=_run_config(config),
    )

    usage = result.context_wrapper.usage
    return AgentRunResult(
        output=str(result.final_output),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        requests=usage.requests,
    )
