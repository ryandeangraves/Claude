"""Unit tests for the OpenAI agent integration.

These tests never hit the network: the SDK's Runner is stubbed out.
"""
import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext

from src.agent import (
    AgentRunResult,
    ConfigError,
    build_agent,
    load_config,
    run_agent,
)
from src.agent.agent import AGENT_NAME
from src.agent.config import DEFAULT_MAX_TURNS, DEFAULT_MODEL, AgentConfig
from src.agent.tools import ALL_TOOLS

FAKE_KEY = "sk-test-0123456789abcdefghijklmnop"


def _config(**overrides) -> AgentConfig:
    base = dict(api_key=FAKE_KEY, model="gpt-test", max_turns=3, tracing_enabled=False)
    base.update(overrides)
    return AgentConfig(**base)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_defaults(self):
        cfg = load_config({"OPENAI_API_KEY": FAKE_KEY})
        assert cfg.api_key == FAKE_KEY
        assert cfg.model == DEFAULT_MODEL
        assert cfg.max_turns == DEFAULT_MAX_TURNS
        assert cfg.tracing_enabled is True

    def test_overrides(self):
        cfg = load_config(
            {
                "OPENAI_API_KEY": FAKE_KEY,
                "OPENAI_AGENT_MODEL": "gpt-4.1",
                "OPENAI_AGENT_MAX_TURNS": "5",
                "OPENAI_AGENT_TRACING": "false",
            }
        )
        assert cfg.model == "gpt-4.1"
        assert cfg.max_turns == 5
        assert cfg.tracing_enabled is False

    def test_missing_key(self):
        with pytest.raises(ConfigError, match="OPENAI_API_KEY is not set"):
            load_config({})

    def test_blank_key(self):
        with pytest.raises(ConfigError, match="OPENAI_API_KEY is not set"):
            load_config({"OPENAI_API_KEY": "   "})

    def test_malformed_key(self):
        with pytest.raises(ConfigError, match="does not look like an OpenAI key"):
            load_config({"OPENAI_API_KEY": "not-a-key"})

    @pytest.mark.parametrize("value", ["abc", "0", "-2"])
    def test_bad_max_turns(self, value):
        with pytest.raises(ConfigError, match="OPENAI_AGENT_MAX_TURNS"):
            load_config({"OPENAI_API_KEY": FAKE_KEY, "OPENAI_AGENT_MAX_TURNS": value})

    def test_repr_hides_key(self):
        cfg = load_config({"OPENAI_API_KEY": FAKE_KEY})
        assert FAKE_KEY not in repr(cfg)
        assert "***" in repr(cfg)


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------


class TestBuildAgent:
    def test_agent_shape(self):
        agent = build_agent(_config())
        assert agent.name == AGENT_NAME
        assert agent.model == "gpt-test"
        assert "Second Brain" in agent.instructions
        assert [t.name for t in agent.tools] == [t.name for t in ALL_TOOLS]

    def test_tool_names(self):
        names = {t.name for t in ALL_TOOLS}
        assert names == {
            "check_email_address",
            "check_phone_number",
            "draft_order_confirmation",
            "draft_shipping_sms",
        }


# ---------------------------------------------------------------------------
# Tools (invoked the way the SDK invokes them: JSON args in, string out)
# ---------------------------------------------------------------------------


def _invoke(tool, **kwargs) -> str:
    args = json.dumps(kwargs)
    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call_test",
        tool_arguments=args,
    )
    return asyncio.run(tool.on_invoke_tool(ctx, args))


def _tool(name):
    return next(t for t in ALL_TOOLS if t.name == name)


class TestTools:
    def test_check_email_valid(self):
        assert _invoke(_tool("check_email_address"), email="a@example.com") == "valid"

    def test_check_email_invalid(self):
        assert _invoke(_tool("check_email_address"), email="nope") == "invalid"

    def test_check_phone_valid(self):
        assert _invoke(_tool("check_phone_number"), phone="+14155552671") == "valid"

    def test_check_phone_invalid(self):
        assert _invoke(_tool("check_phone_number"), phone="415-555-2671") == "invalid"

    def test_draft_order_confirmation(self):
        out = _invoke(
            _tool("draft_order_confirmation"),
            user_id="u1",
            email="a@example.com",
            order_id="ord_9",
            order_total=12.5,
        )
        data = json.loads(out) if isinstance(out, str) else out
        assert data["channel"] == "email"
        assert data["recipient"] == "a@example.com"
        assert data["subject"] == "Order Confirmation #ord_9"
        assert "$12.50" in data["body"]

    def test_draft_shipping_sms(self):
        out = _invoke(
            _tool("draft_shipping_sms"),
            user_id="u1",
            phone="+14155552671",
            order_id="ord_9",
            tracking_number="TRACK123",
        )
        data = json.loads(out) if isinstance(out, str) else out
        assert data["channel"] == "sms"
        assert data["recipient"] == "+14155552671"
        assert "TRACK123" in data["body"]


# ---------------------------------------------------------------------------
# run_agent (Runner stubbed)
# ---------------------------------------------------------------------------


@dataclass
class _FakeUsage:
    input_tokens: int = 11
    output_tokens: int = 7
    requests: int = 1


class _FakeRunResult:
    final_output = "hello from the fake model"
    context_wrapper = SimpleNamespace(usage=_FakeUsage())


class TestRunAgent:
    def test_returns_output_and_usage(self, monkeypatch):
        captured = {}

        def fake_run_sync(agent, prompt, **kwargs):
            captured["agent"] = agent
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return _FakeRunResult()

        keys = []
        monkeypatch.setattr("src.agent.agent.Runner.run_sync", fake_run_sync)
        monkeypatch.setattr(
            "src.agent.agent.set_default_openai_key",
            lambda key, use_for_tracing=True: keys.append((key, use_for_tracing)),
        )

        result = run_agent("hi", _config())

        assert isinstance(result, AgentRunResult)
        assert result.output == "hello from the fake model"
        assert (result.input_tokens, result.output_tokens, result.requests) == (11, 7, 1)
        assert result.total_tokens == 18
        assert captured["prompt"] == "hi"
        assert captured["agent"].name == AGENT_NAME
        assert captured["kwargs"]["max_turns"] == 3
        assert captured["kwargs"]["run_config"].tracing_disabled is True
        assert keys == [(FAKE_KEY, False)]

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValueError, match="prompt must not be empty"):
            run_agent("   ", _config())

    def test_uses_env_config_when_none_given(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        monkeypatch.setattr("src.agent.agent.Runner.run_sync", lambda *a, **k: _FakeRunResult())
        monkeypatch.setattr("src.agent.agent.set_default_openai_key", lambda *a, **k: None)
        assert run_agent("hi").output == "hello from the fake model"

    def test_missing_key_surfaces_config_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError):
            run_agent("hi")
