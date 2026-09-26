"""Unit tests for config, Jack's tools and Jack himself.

These tests never hit the network: the SDK's Runner is stubbed out.
"""
import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from agents.tool_context import ToolContext
from openai.types.responses.response_output_item import ImageGenerationCall

from src.agent import ConfigError, JackContext, JackResult, build_jack, load_config, run_jack
from src.agent.config import (
    DEFAULT_AUDITOR_MODEL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MAX_TURNS,
    DEFAULT_MODEL,
    AgentConfig,
)
from src.agent.jack import JACK_NAME
from src.agent.tools import ALL_TOOLS
from tests.unit.agent_helpers import FAKE_KEY, PNG_BYTES, FakeRunResult, config, image_call, stub_runner


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_defaults(self):
        cfg = load_config({"OPENAI_API_KEY": FAKE_KEY})
        assert cfg.api_key == FAKE_KEY
        assert cfg.model == DEFAULT_MODEL
        assert cfg.auditor_model == DEFAULT_AUDITOR_MODEL == "gpt-6-astra"
        assert cfg.image_model == DEFAULT_IMAGE_MODEL == "gpt-image-2.5-sunburst"
        assert cfg.max_turns == DEFAULT_MAX_TURNS
        assert cfg.tracing_enabled is True

    def test_overrides(self):
        cfg = load_config(
            {
                "OPENAI_API_KEY": FAKE_KEY,
                "OPENAI_AGENT_MODEL": "gpt-4.1",
                "OPENAI_AUDITOR_MODEL": "gpt-6-sol",
                "OPENAI_IMAGE_MODEL": "gpt-image-2",
                "OPENAI_AGENT_MAX_TURNS": "5",
                "OPENAI_AGENT_TRACING": "false",
            }
        )
        assert cfg.model == "gpt-4.1"
        assert cfg.auditor_model == "gpt-6-sol"
        assert cfg.image_model == "gpt-image-2"
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
# Jack's construction
# ---------------------------------------------------------------------------


class TestBuildJack:
    def test_shape(self):
        agent = build_jack(config(), size="1536x1024", quality="high")
        assert agent.name == JACK_NAME == "Jack"
        assert "Jack" in agent.instructions
        assert agent.model == "gpt-test"
        names = [t.name for t in agent.tools]
        assert names == [t.name for t in ALL_TOOLS] + ["image_generation", "audit_text", "audit_image_file"]
        image_tool = agent.tools[len(ALL_TOOLS)]
        assert image_tool.tool_config["model"] == "gpt-image-2.5-sunburst"
        assert image_tool.tool_config["size"] == "1536x1024"
        assert image_tool.tool_config["quality"] == "high"

    def test_image_model_override(self):
        cfg = config(image_model="gpt-image-2")
        idx = len(ALL_TOOLS)
        assert build_jack(cfg).tools[idx].tool_config["model"] == "gpt-image-2"
        assert build_jack(cfg, image_model="gpt-image-1.5").tools[idx].tool_config["model"] == "gpt-image-1.5"

    def test_tool_names(self):
        assert {t.name for t in ALL_TOOLS} == {
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
    ctx = ToolContext(context=None, tool_name=tool.name, tool_call_id="call_test", tool_arguments=args)
    return asyncio.run(tool.on_invoke_tool(ctx, args))


def _tool(name):
    return next(t for t in ALL_TOOLS if t.name == name)


def _as_dict(out):
    return json.loads(out) if isinstance(out, str) else out


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
        data = _as_dict(_invoke(
            _tool("draft_order_confirmation"),
            user_id="u1", email="a@example.com", order_id="ord_9", order_total=12.5,
        ))
        assert data["channel"] == "email"
        assert data["recipient"] == "a@example.com"
        assert data["subject"] == "Order Confirmation #ord_9"
        assert "$12.50" in data["body"]

    def test_draft_shipping_sms(self):
        data = _as_dict(_invoke(
            _tool("draft_shipping_sms"),
            user_id="u1", phone="+14155552671", order_id="ord_9", tracking_number="TRACK123",
        ))
        assert data["channel"] == "sms"
        assert data["recipient"] == "+14155552671"
        assert "TRACK123" in data["body"]


# ---------------------------------------------------------------------------
# run_jack (Runner stubbed)
# ---------------------------------------------------------------------------


class TestRunJack:
    def test_text_only(self, monkeypatch):
        calls, keys = stub_runner(monkeypatch, "src.agent.jack", lambda a, i: FakeRunResult("hello"))
        r = run_jack("hi", config())
        assert isinstance(r, JackResult)
        assert r.output == "hello" and r.images == [] and r.audits == []
        assert (r.input_tokens, r.output_tokens, r.requests) == (5, 3, 1)
        assert r.total_tokens == 8
        agent, inp, kwargs = calls[0]
        assert agent.name == "Jack" and inp == "hi"
        assert kwargs["max_turns"] == 3
        assert kwargs["run_config"].tracing_disabled is True
        assert isinstance(kwargs["context"], JackContext)
        assert keys == [(FAKE_KEY, False)]

    def test_with_image_saved(self, monkeypatch, tmp_path):
        stub_runner(monkeypatch, "src.agent.jack",
                    lambda a, i: FakeRunResult("Here is your fox.", [SimpleNamespace(raw_item=image_call())]))
        r = run_jack("a red fox", config(), out_dir=tmp_path)
        assert len(r.images) == 1
        assert r.images[0].data == PNG_BYTES
        assert r.images[0].path is not None and r.images[0].path.exists()
        assert r.images[0].path.parent == tmp_path

    def test_empty_prompt_rejected(self):
        with pytest.raises(ValueError, match="prompt must not be empty"):
            run_jack("   ", config())

    def test_uses_env_config_when_none_given(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
        stub_runner(monkeypatch, "src.agent.jack", lambda a, i: FakeRunResult("hello"))
        assert run_jack("hi").output == "hello"

    def test_missing_key_surfaces_config_error(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ConfigError):
            run_jack("hi")


# ---------------------------------------------------------------------------
# Jack's audit tools (Auditor stubbed)
# ---------------------------------------------------------------------------


def _jack_tool(name):
    return next(t for t in build_jack(config()).tools if t.name == name)


def _invoke_with_ctx(tool, jc, **kwargs):
    args = json.dumps(kwargs)
    ctx = ToolContext(context=jc, tool_name=tool.name, tool_call_id="c1", tool_arguments=args)
    return asyncio.run(tool.on_invoke_tool(ctx, args))


class TestJackAuditTools:
    def test_audit_text_tool(self, monkeypatch):
        from tests.unit.agent_helpers import report
        seen = {}

        async def fake(text, request, cfg, **kw):
            seen.update(text=text, request=request, cfg=cfg)
            return report("fail", "tighten it")

        monkeypatch.setattr("src.agent.jack.audit_text_async", fake)
        jc = JackContext(config=config())
        out = json.loads(_invoke_with_ctx(_jack_tool("audit_text"), jc, text="draft", request="say hi"))
        assert out["verdict"] == "fail" and out["suggested_prompt_fix"] == "tighten it"
        assert seen == dict(text="draft", request="say hi", cfg=jc.config)
        assert len(jc.audits) == 1

    def test_audit_image_file_tool(self, monkeypatch, tmp_path):
        from tests.unit.agent_helpers import report
        seen = {}

        async def fake(path, request, cfg, **kw):
            seen.update(path=path, request=request)
            return report("pass")

        monkeypatch.setattr("src.agent.jack.audit_image_async", fake)
        img = tmp_path / "fox.png"
        img.write_bytes(PNG_BYTES)
        jc = JackContext(config=config())
        out = json.loads(_invoke_with_ctx(_jack_tool("audit_image_file"), jc, path=str(img), request="a fox"))
        assert out["verdict"] == "pass"
        assert seen == dict(path=img, request="a fox")
        assert len(jc.audits) == 1

    @pytest.mark.parametrize("name, expect", [("notes.txt", "not an image"), ("missing.png", "does not exist")])
    def test_audit_image_file_rejects_bad_paths(self, tmp_path, name, expect):
        (tmp_path / "notes.txt").write_text("x")
        jc = JackContext(config=config())
        out = _invoke_with_ctx(_jack_tool("audit_image_file"), jc, path=str(tmp_path / name), request="r")
        assert out.startswith("error:") and expect in out
        assert jc.audits == []
