"""Unit tests for the Auditor, the audited pipeline, and Frank (no network)."""
import asyncio
from types import SimpleNamespace

import pytest

from src.agent import (
    AuditReport,
    FrankContext,
    audit_image,
    audit_text,
    build_auditor,
    build_frank,
    run_audited,
    run_frank,
)
from src.agent.auditor import (
    AUDITOR_NAME,
    DEFAULT_IMAGE_CHECKLIST,
    DEFAULT_TEXT_CHECKLIST,
    build_image_audit_input,
    build_text_audit_input,
)
from src.agent.frank import FRANK_NAME, SPECIALIST_TOOLS, issue_job_to_jack
from src.agent.tools import ALL_TOOLS
from src.agent.image import GeneratedImage, extract_images, save_images
from src.agent.jack import JackResult
from tests.unit.agent_helpers import PNG_B64, PNG_BYTES, FakeRunResult, config, image_call, report, stub_runner


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


class TestImageHelpers:
    def test_extract_images_skips_empty_and_non_image_items(self):
        items = [
            SimpleNamespace(raw_item=image_call()),
            SimpleNamespace(raw_item=image_call(result=None)),
            SimpleNamespace(raw_item="not an image"),
            SimpleNamespace(),
        ]
        imgs = extract_images(SimpleNamespace(new_items=items))
        assert len(imgs) == 1
        assert imgs[0].data == PNG_BYTES and imgs[0].output_format == "png"
        assert imgs[0].revised_prompt == "a red fox, watercolor"

    def test_save_images(self, tmp_path):
        imgs = [GeneratedImage(data=PNG_BYTES, output_format="png"),
                GeneratedImage(data=b"x", output_format="webp")]
        paths = save_images(imgs, tmp_path / "out", stem="A Red Fox!!")
        assert [p.suffix for p in paths] == [".png", ".webp"]
        assert all("a-red-fox" in p.name for p in paths)
        assert paths[0].read_bytes() == PNG_BYTES
        assert imgs[0].path == paths[0]


# ---------------------------------------------------------------------------
# Auditor
# ---------------------------------------------------------------------------


class TestAuditor:
    def test_build(self):
        agent = build_auditor(config())
        assert agent.name == AUDITOR_NAME
        assert agent.model == "gpt-6-astra"
        assert agent.output_type is AuditReport
        assert "Jack" in agent.instructions

    def test_text_input(self):
        inp = build_text_audit_input("the answer", "the question", ["only item"])
        assert len(inp) == 1 and inp[0]["role"] == "user"
        (text,) = inp[0]["content"]
        assert "the question" in text["text"] and "the answer" in text["text"]
        assert "1. only item" in text["text"]

    def test_text_input_default_checklist(self):
        (text,) = build_text_audit_input("a", "q")[0]["content"]
        for c in DEFAULT_TEXT_CHECKLIST:
            assert c in text["text"]

    def test_image_input_from_bytes(self):
        inp = build_image_audit_input(PNG_BYTES, "a red fox", ["item one"], output_format="webp")
        text, image = inp[0]["content"]
        assert "a red fox" in text["text"] and "1. item one" in text["text"]
        assert image["type"] == "input_image"
        assert image["image_url"].startswith("data:image/webp;base64,")
        assert image["image_url"].endswith(PNG_B64)

    def test_image_input_from_path(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(PNG_BYTES)
        text, image = build_image_audit_input(p, "a red fox")[0]["content"]
        assert image["image_url"].startswith("data:image/png;base64,")
        for c in DEFAULT_IMAGE_CHECKLIST:
            assert c in text["text"]

    def test_audit_text(self, monkeypatch):
        calls, _ = stub_runner(monkeypatch, "src.agent.auditor", lambda a, i: FakeRunResult(report("pass")))
        rep = audit_text("answer", "question", config())
        assert rep.verdict == "pass"
        agent, inp, _ = calls[0]
        assert agent.name == AUDITOR_NAME
        assert inp[0]["content"][0]["type"] == "input_text"

    def test_audit_image(self, monkeypatch):
        calls, _ = stub_runner(monkeypatch, "src.agent.auditor",
                               lambda a, i: FakeRunResult(report("fail", "add more contrast")))
        rep = audit_image(PNG_BYTES, "a red fox", config())
        assert rep.verdict == "fail"
        assert rep.suggested_prompt_fix == "add more contrast"
        assert calls[0][1][0]["content"][1]["type"] == "input_image"

    def test_report_schema_validates(self):
        with pytest.raises(Exception):
            AuditReport(verdict="maybe", score=5, summary="", results=[])
        with pytest.raises(Exception):
            AuditReport(verdict="pass", score=101, summary="", results=[])

    def test_empty_prompt(self):
        with pytest.raises(ValueError):
            audit_text("a", "", config())
        with pytest.raises(ValueError):
            audit_image(PNG_BYTES, "", config())


# ---------------------------------------------------------------------------
# Audited pipeline (Jack + Auditor stubbed at the function level)
# ---------------------------------------------------------------------------


def _jack_result(output="ok", n_images=0):
    return JackResult(output=output,
                      images=[GeneratedImage(data=PNG_BYTES, output_format="png") for _ in range(n_images)])


def _stub_pipeline(monkeypatch, jack_results, text_reports, image_reports=()):
    prompts, text_audits, image_audits = [], [], []
    jr, tr, ir = iter(jack_results), iter(text_reports), iter(image_reports)

    async def fake_jack(prompt, config, out_dir=None, **kw):
        prompts.append(prompt)
        return next(jr)

    async def fake_text(answer, prompt, config, checklist=None):
        text_audits.append((answer, prompt))
        return next(tr)

    async def fake_image(data, prompt, config, checklist=None, output_format=None):
        image_audits.append((prompt, output_format))
        return next(ir)

    monkeypatch.setattr("src.agent.pipeline.run_jack_async", fake_jack)
    monkeypatch.setattr("src.agent.pipeline.audit_text_async", fake_text)
    monkeypatch.setattr("src.agent.pipeline.audit_image_async", fake_image)
    return prompts, text_audits, image_audits


class TestRunAudited:
    def test_text_passes_first_try(self, monkeypatch):
        prompts, text_audits, image_audits = _stub_pipeline(monkeypatch, [_jack_result("42")], [report("pass")])
        res = run_audited("what is 6*7", config(), max_attempts=3)
        assert res.passed and len(res.attempts) == 1
        assert prompts == ["what is 6*7"]
        assert text_audits == [("42", "what is 6*7")]
        assert image_audits == []

    def test_images_are_audited_too(self, monkeypatch):
        _, _, image_audits = _stub_pipeline(
            monkeypatch, [_jack_result("here", 2)], [report("pass")], [report("pass"), report("pass")]
        )
        res = run_audited("draw", config())
        assert res.passed
        assert len(res.final.image_reports) == 2
        assert image_audits == [("draw", "png"), ("draw", "png")]

    def test_one_failed_image_fails_attempt(self, monkeypatch):
        _stub_pipeline(monkeypatch, [_jack_result("here", 1)], [report("pass")], [report("fail")])
        res = run_audited("draw", config())
        assert not res.passed
        assert res.final.text_report.verdict == "pass"

    def test_retries_with_suggested_fix(self, monkeypatch):
        prompts, _, _ = _stub_pipeline(
            monkeypatch, [_jack_result(), _jack_result()], [report("fail", "be specific"), report("pass")]
        )
        res = run_audited("q", config(), max_attempts=3)
        assert res.passed and len(res.attempts) == 2
        assert prompts == ["q", "be specific"]

    def test_stops_at_max_attempts(self, monkeypatch):
        _stub_pipeline(monkeypatch, [_jack_result()] * 3,
                       [report("fail", "v2"), report("fail", "v3"), report("fail", "v4")])
        res = run_audited("q", config(), max_attempts=2)
        assert not res.passed and len(res.attempts) == 2
        assert res.final.prompt == "v2"

    def test_stops_when_no_fix_suggested(self, monkeypatch):
        _stub_pipeline(monkeypatch, [_jack_result()], [report("fail", None)])
        res = run_audited("q", config(), max_attempts=3)
        assert not res.passed and len(res.attempts) == 1

    def test_bad_max_attempts(self):
        with pytest.raises(ValueError):
            run_audited("x", config(), max_attempts=0)


# ---------------------------------------------------------------------------
# Frank
# ---------------------------------------------------------------------------


def _invoke_issue_job(fc, request):
    import json
    from agents.tool_context import ToolContext
    args = json.dumps({"request": request})
    ctx = ToolContext(context=fc, tool_name="issue_job_to_jack", tool_call_id="c1", tool_arguments=args)
    return asyncio.run(issue_job_to_jack.on_invoke_tool(ctx, args))


class TestFrank:
    def test_build(self):
        agent = build_frank(config())
        assert agent.name == FRANK_NAME == "Frank"
        assert agent.model == "gpt-test"
        names = [t.name for t in agent.tools]
        assert names == [t.name for t in ALL_TOOLS] + ["issue_job_to_jack"]  # own work + specialists
        assert [t.name for t in SPECIALIST_TOOLS] == ["issue_job_to_jack"]
        assert "Jack" in agent.instructions and "audit this" in agent.instructions
        assert "image" in agent.instructions and "large audit" in agent.instructions

    def test_issue_job_runs_jack(self, monkeypatch, tmp_path):
        seen = {}

        async def fake_jack(request, cfg, out_dir=None, **kw):
            seen.update(request=request, cfg=cfg, out_dir=out_dir)
            jr = _jack_result("drafted", 1)
            jr.images[0].path = tmp_path / "x.png"
            jr.audits = [report("fail", "fix")]
            return jr

        monkeypatch.setattr("src.agent.frank.run_jack_async", fake_jack)
        fc = FrankContext(config=config(), out_dir=tmp_path)
        out = _invoke_issue_job(fc, "draft it")

        assert seen == dict(request="draft it", cfg=fc.config, out_dir=tmp_path)
        assert len(fc.jobs) == 1 and isinstance(fc.jobs[0], JackResult)
        assert "drafted" in out and "x.png" in out
        assert "audit of item 1: fail" in out and "failed items:" in out

    def test_issue_job_auto_audit_uses_pipeline(self, monkeypatch, tmp_path):
        seen = {}

        async def fake_run_audited(request, cfg, out_dir=None, max_attempts=1):
            seen.update(request=request, max_attempts=max_attempts)
            from src.agent.pipeline import Attempt, PipelineResult
            return PipelineResult([Attempt(request, _jack_result("done"), report("pass"), [])])

        monkeypatch.setattr("src.agent.frank.run_audited_async", fake_run_audited)
        fc = FrankContext(config=config(), out_dir=tmp_path, auto_audit=True, max_attempts=2)
        out = _invoke_issue_job(fc, "do it")
        assert seen == dict(request="do it", max_attempts=2)
        assert "Automatic audit: PASS after 1 attempt(s)." in out
        assert len(fc.jobs) == 1

    def test_run_frank(self, monkeypatch):
        calls, _ = stub_runner(monkeypatch, "src.agent.frank", lambda a, i: FakeRunResult("Frank says hi"))
        r = run_frank("hello", config(), max_attempts=2)
        assert r.output == "Frank says hi" and r.jobs == [] and r.total_tokens == 8
        agent, inp, kwargs = calls[0]
        assert agent.name == "Frank" and inp == "hello"
        assert isinstance(kwargs["context"], FrankContext)
        assert kwargs["context"].max_attempts == 2
        assert kwargs["session"] is None

    def test_run_frank_with_session(self, monkeypatch, tmp_path):
        from agents import SQLiteSession
        calls, _ = stub_runner(monkeypatch, "src.agent.frank", lambda a, i: FakeRunResult("ok"))
        db = tmp_path / "s" / "sessions.db"
        run_frank("hello", config(), session="ryan", session_db=db)
        session = calls[0][2]["session"]
        assert isinstance(session, SQLiteSession)
        assert session.session_id == "ryan"
        assert db.parent.is_dir()

    def test_empty_prompt(self):
        with pytest.raises(ValueError):
            run_frank(" ", config())
