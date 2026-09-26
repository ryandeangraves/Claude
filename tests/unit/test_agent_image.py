"""Unit tests for the image-generation, auditor and pipeline agents.

The SDK's Runner is stubbed; nothing here touches the network.
"""
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from openai.types.responses.response_output_item import ImageGenerationCall

from src.agent import (
    AuditReport,
    GeneratedImage,
    audit_image,
    build_auditor_agent,
    build_image_agent,
    generate_and_audit,
    generate_image,
)
from src.agent.auditor import AUDITOR_NAME, DEFAULT_CHECKLIST, build_audit_input
from src.agent.config import AgentConfig
from src.agent.image import IMAGE_AGENT_NAME, extract_images, save_images

PNG_BYTES = b"\x89PNG\r\n\x1a\nfake"
PNG_B64 = base64.b64encode(PNG_BYTES).decode()


def _config() -> AgentConfig:
    return AgentConfig(api_key="sk-test-0123456789abcdefghijklmnop", model="gpt-test",
                       auditor_model="gpt-6-astra", max_turns=3, tracing_enabled=False)


@dataclass
class _Usage:
    input_tokens: int = 5
    output_tokens: int = 3
    requests: int = 1


def _image_call(result=PNG_B64, fmt="png", status="completed"):
    return ImageGenerationCall(
        id="ig_1", type="image_generation_call", status=status, result=result,
        output_format=fmt, revised_prompt="a red fox, watercolor", size="1024x1024", quality="high",
    )


class _FakeRunResult:
    def __init__(self, final_output, new_items=()):
        self.final_output = final_output
        self.new_items = list(new_items)
        self.context_wrapper = SimpleNamespace(usage=_Usage())

    def final_output_as(self, cls):
        return self.final_output


def _stub_runner(monkeypatch, module, factory):
    calls = []

    def fake_run_sync(agent, inp, **kwargs):
        calls.append((agent, inp, kwargs))
        return factory(agent, inp)

    monkeypatch.setattr(f"{module}.Runner.run_sync", fake_run_sync)
    monkeypatch.setattr(f"{module}.set_default_openai_key", lambda *a, **k: None)
    return calls


# ---------------------------------------------------------------------------
# Image agent
# ---------------------------------------------------------------------------


class TestImageAgent:
    def test_build(self):
        agent = build_image_agent(_config(), size="1536x1024", quality="high")
        assert agent.name == IMAGE_AGENT_NAME
        assert agent.model == "gpt-test"
        assert len(agent.tools) == 1
        cfg = agent.tools[0].tool_config
        assert cfg["type"] == "image_generation"
        assert cfg["model"] == "gpt-image-2.5-sunburst"
        assert cfg["size"] == "1536x1024"
        assert cfg["quality"] == "high"

    def test_build_image_model_override(self):
        cfg = AgentConfig(api_key="sk-test-0123456789abcdefghijklmnop", image_model="gpt-image-2")
        assert build_image_agent(cfg).tools[0].tool_config["model"] == "gpt-image-2"
        assert build_image_agent(cfg, image_model="gpt-image-1.5").tools[0].tool_config["model"] == "gpt-image-1.5"

    def test_extract_images_skips_empty_and_non_image_items(self):
        items = [
            SimpleNamespace(raw_item=_image_call()),
            SimpleNamespace(raw_item=_image_call(result=None)),
            SimpleNamespace(raw_item="not an image"),
            SimpleNamespace(),
        ]
        imgs = extract_images(SimpleNamespace(new_items=items))
        assert len(imgs) == 1
        assert imgs[0].data == PNG_BYTES
        assert imgs[0].output_format == "png"
        assert imgs[0].revised_prompt == "a red fox, watercolor"

    def test_save_images(self, tmp_path):
        imgs = [GeneratedImage(data=PNG_BYTES, output_format="png"),
                GeneratedImage(data=b"x", output_format="webp")]
        paths = save_images(imgs, tmp_path / "out", stem="A Red Fox!!")
        assert [p.suffix for p in paths] == [".png", ".webp"]
        assert all("a-red-fox" in p.name for p in paths)
        assert paths[0].read_bytes() == PNG_BYTES
        assert imgs[0].path == paths[0]

    def test_generate_image(self, monkeypatch, tmp_path):
        calls = _stub_runner(
            monkeypatch, "src.agent.image",
            lambda agent, inp: _FakeRunResult("Here is your fox.", [SimpleNamespace(raw_item=_image_call())]),
        )
        r = generate_image("a red fox", _config(), out_dir=tmp_path)
        assert r.message == "Here is your fox."
        assert len(r.images) == 1
        assert r.images[0].path is not None and r.images[0].path.exists()
        assert (r.input_tokens, r.output_tokens, r.requests) == (5, 3, 1)
        agent, inp, kwargs = calls[0]
        assert agent.name == IMAGE_AGENT_NAME
        assert inp == "a red fox"
        assert kwargs["max_turns"] == 3

    def test_generate_image_empty_prompt(self):
        with pytest.raises(ValueError):
            generate_image("", _config())


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


def _report(verdict="pass", fix=None):
    return AuditReport(
        verdict=verdict, score=90 if verdict == "pass" else 40, summary="ok",
        results=[{"item": c, "passed": verdict == "pass", "notes": "seen"} for c in DEFAULT_CHECKLIST],
        suggested_prompt_fix=fix,
    )


class TestAuditor:
    def test_build(self):
        agent = build_auditor_agent(_config())
        assert agent.name == AUDITOR_NAME
        assert agent.model == "gpt-6-astra"
        assert agent.output_type is AuditReport

    def test_build_audit_input_from_bytes(self):
        inp = build_audit_input(PNG_BYTES, "a red fox", ["item one"], output_format="webp")
        assert len(inp) == 1 and inp[0]["role"] == "user"
        text, image = inp[0]["content"]
        assert "a red fox" in text["text"] and "1. item one" in text["text"]
        assert image["type"] == "input_image"
        assert image["image_url"].startswith("data:image/webp;base64,")
        assert image["image_url"].endswith(PNG_B64)

    def test_build_audit_input_from_path(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(PNG_BYTES)
        inp = build_audit_input(p, "a red fox")
        text, image = inp[0]["content"]
        assert image["image_url"].startswith("data:image/png;base64,")
        for c in DEFAULT_CHECKLIST:
            assert c in text["text"]

    def test_audit_image(self, monkeypatch):
        calls = _stub_runner(monkeypatch, "src.agent.auditor",
                             lambda agent, inp: _FakeRunResult(_report("fail", "add more contrast")))
        report = audit_image(PNG_BYTES, "a red fox", _config())
        assert report.verdict == "fail"
        assert report.suggested_prompt_fix == "add more contrast"
        agent, inp, _ = calls[0]
        assert agent.name == AUDITOR_NAME
        assert inp[0]["content"][1]["type"] == "input_image"

    def test_audit_report_schema_validates(self):
        with pytest.raises(Exception):
            AuditReport(verdict="maybe", score=5, summary="", results=[])
        with pytest.raises(Exception):
            AuditReport(verdict="pass", score=101, summary="", results=[])

    def test_audit_empty_prompt(self):
        with pytest.raises(ValueError):
            audit_image(PNG_BYTES, "", _config())


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def _stub(self, monkeypatch, reports):
        gens, audits = [], []
        it = iter(reports)

        def fake_generate(prompt, config, out_dir=None):
            gens.append(prompt)
            return SimpleNamespace(images=[GeneratedImage(data=PNG_BYTES, output_format="png")])

        def fake_audit(data, prompt, config, checklist=None, output_format=None):
            audits.append((prompt, output_format))
            return next(it)

        monkeypatch.setattr("src.agent.pipeline.generate_image", fake_generate)
        monkeypatch.setattr("src.agent.pipeline.audit_image", fake_audit)
        return gens, audits

    def test_pass_first_try(self, monkeypatch):
        gens, audits = self._stub(monkeypatch, [_report("pass")])
        res = generate_and_audit("a red fox", _config(), max_attempts=3)
        assert res.passed and len(res.attempts) == 1
        assert gens == ["a red fox"]
        assert audits == [("a red fox", "png")]

    def test_retries_with_suggested_fix(self, monkeypatch):
        gens, _ = self._stub(monkeypatch, [_report("fail", "a red fox, sharper"), _report("pass")])
        res = generate_and_audit("a red fox", _config(), max_attempts=3)
        assert res.passed and len(res.attempts) == 2
        assert gens == ["a red fox", "a red fox, sharper"]

    def test_stops_at_max_attempts(self, monkeypatch):
        self._stub(monkeypatch, [_report("fail", "v2"), _report("fail", "v3"), _report("fail", "v4")])
        res = generate_and_audit("a red fox", _config(), max_attempts=2)
        assert not res.passed and len(res.attempts) == 2
        assert res.final.prompt == "v2"

    def test_stops_when_no_fix_suggested(self, monkeypatch):
        self._stub(monkeypatch, [_report("fail", None)])
        res = generate_and_audit("a red fox", _config(), max_attempts=3)
        assert not res.passed and len(res.attempts) == 1

    def test_no_image_produced(self, monkeypatch):
        monkeypatch.setattr("src.agent.pipeline.generate_image",
                            lambda *a, **k: SimpleNamespace(images=[]))
        res = generate_and_audit("a red fox", _config())
        assert not res.passed
        assert res.final.image is None and res.final.report is None

    def test_bad_max_attempts(self):
        with pytest.raises(ValueError):
            generate_and_audit("x", _config(), max_attempts=0)
