"""Shared fakes for the agent tests (no network)."""
import base64
from dataclasses import dataclass
from types import SimpleNamespace

from openai.types.responses.response_output_item import ImageGenerationCall

from src.agent.auditor import DEFAULT_IMAGE_CHECKLIST, AuditReport
from src.agent.config import AgentConfig

FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"
PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def config(**overrides) -> AgentConfig:
    base = dict(api_key=FAKE_KEY, model="gpt-test", auditor_model="gpt-6-astra",
                max_turns=3, tracing_enabled=False)
    base.update(overrides)
    return AgentConfig(**base)


@dataclass
class FakeUsage:
    input_tokens: int = 5
    output_tokens: int = 3
    requests: int = 1


class FakeRunResult:
    def __init__(self, final_output, new_items=()):
        self.final_output = final_output
        self.new_items = list(new_items)
        self.context_wrapper = SimpleNamespace(usage=FakeUsage())

    def final_output_as(self, cls):
        return self.final_output


def image_call(result=PNG_B64, fmt="png"):
    return ImageGenerationCall(
        id="ig_1", type="image_generation_call", status="completed", result=result,
        output_format=fmt, revised_prompt="a red fox, watercolor", size="1024x1024", quality="high",
    )


def report(verdict="pass", fix=None, checklist=DEFAULT_IMAGE_CHECKLIST) -> AuditReport:
    return AuditReport(
        verdict=verdict, score=90 if verdict == "pass" else 40, summary=f"summary:{verdict}",
        results=[{"item": c, "passed": verdict == "pass", "notes": "seen"} for c in checklist],
        suggested_prompt_fix=fix,
    )


def stub_runner(monkeypatch, module, factory):
    """Replace ``Runner.run`` in ``module`` with an async stub. Returns (calls, keys)."""
    calls, keys = [], []

    async def fake_run(agent, inp, **kwargs):
        calls.append((agent, inp, kwargs))
        return factory(agent, inp)

    monkeypatch.setattr(f"{module}.Runner.run", fake_run)
    monkeypatch.setattr(f"{module}.set_default_openai_key",
                        lambda key, use_for_tracing=True: keys.append((key, use_for_tracing)))
    return calls, keys
