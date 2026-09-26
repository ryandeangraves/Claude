"""Command-line entry point. You talk to Frank. He does his own work and
sends image tasks and large audits to Jack.

    python -m src.agent "Hey Frank, draw a watercolor red fox"
    python -m src.agent "Hey Frank, audit generated/<file>.png against 'a watercolor red fox'"
    python -m src.agent "Hey Frank, audit this" --session ryan   # remembers earlier turns
    python -m src.agent "..." --auto-audit --attempts 2          # audit every job automatically
    python -m src.agent jack "Draw a watercolor red fox"          # give Jack a job directly
    python -m src.agent audit ./generated/fox.png --prompt "a watercolor red fox"
    python -m src.agent audit --text "some draft" --prompt "what it should say"
"""
import argparse
import json
import sys
from pathlib import Path

from src.agent.auditor import audit_image, audit_text
from src.agent.config import ConfigError, load_config
from src.agent.frank import run_frank
from src.agent.jack import JackResult, run_jack
from src.agent.pipeline import PipelineResult, run_audited


def _usage_line(r) -> str:
    return (
        f"[usage] requests={r.requests} in={r.input_tokens} "
        f"out={r.output_tokens} total={r.input_tokens + r.output_tokens}"
    )


def _print_report(label: str, report) -> None:
    print(f"--- audit ({label}): {report.verdict.upper()} score={report.score}")
    print(json.dumps(report.model_dump(), indent=2))


def _print_job(job: JackResult, usage: bool) -> None:
    print(job.output)
    for img in job.images:
        print(f"saved {img.path}")
    for i, rep in enumerate(job.audits, start=1):
        _print_report(f"item {i}", rep)
    if usage:
        print(_usage_line(job), file=sys.stderr)


def _print_pipeline(res: PipelineResult, usage: bool) -> None:
    for i, a in enumerate(res.attempts, start=1):
        print(f"=== attempt {i}: {a.prompt}")
        _print_job(a.result, usage)
        _print_report("text", a.text_report)
        for j, rep in enumerate(a.image_reports, start=1):
            _print_report(f"image {j}", rep)


def _job_ok(job) -> bool:
    if isinstance(job, PipelineResult):
        return job.passed
    return all(a.verdict == "pass" for a in job.audits)


def _cmd_frank(args, config) -> int:
    r = run_frank(
        args.prompt, config, out_dir=Path(args.out), auto_audit=args.auto_audit,
        max_attempts=args.attempts, session=args.session,
    )
    print(r.output)
    for n, job in enumerate(r.jobs, start=1):
        if args.verbose:
            print(f"\n##### job {n} (Jack)")
            _print_pipeline(job, args.usage) if isinstance(job, PipelineResult) else _print_job(job, args.usage)
        else:
            jr = job.final.result if isinstance(job, PipelineResult) and job.final else job
            for img in getattr(jr, "images", []):
                print(f"saved {img.path}")
    if args.usage:
        print(_usage_line(r), file=sys.stderr)
    return 0 if all(_job_ok(j) for j in r.jobs) else 1


def _cmd_jack(args, config) -> int:
    kw = dict(size=args.size, quality=args.quality)
    if args.auto_audit:
        res = run_audited(args.prompt, config, out_dir=Path(args.out), max_attempts=args.attempts, **kw)
        _print_pipeline(res, args.usage)
        return 0 if res.passed else 1
    job = run_jack(args.prompt, config, out_dir=Path(args.out), **kw)
    _print_job(job, args.usage)
    return 0 if _job_ok(job) else 1


def _cmd_audit(args, config) -> int:
    if args.text is not None:
        report = audit_text(args.text, args.prompt, config)
    else:
        report = audit_image(Path(args.image), args.prompt, config)
    print(json.dumps(report.model_dump(), indent=2))
    return 0 if report.verdict == "pass" else 1


def _add_common(p):
    p.add_argument("--auto-audit", action="store_true", help="Audit every job's output automatically.")
    p.add_argument("--attempts", type=int, default=1, help="Retries when an automatic audit fails.")
    p.add_argument("--out", default="generated", help="Where generated images are saved.")
    p.add_argument("--usage", action="store_true", help="Print token usage to stderr.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.agent", description="Talk to Frank.")
    sub = parser.add_subparsers(dest="command")

    fr = sub.add_parser("frank", help="Talk to Frank (default).")
    fr.add_argument("prompt")
    fr.add_argument("--session", help="Name of a conversation to remember across runs.")
    fr.add_argument("--verbose", action="store_true", help="Show each job's full output.")
    _add_common(fr)
    fr.set_defaults(func=_cmd_frank)

    jk = sub.add_parser("jack", help="Give Jack an image or large-audit job directly.")
    jk.add_argument("prompt")
    jk.add_argument("--size", default="auto")
    jk.add_argument("--quality", default="auto", choices=["auto", "low", "medium", "high", "xhigh", "max"])
    _add_common(jk)
    jk.set_defaults(func=_cmd_jack)

    aud = sub.add_parser("audit", help="Run the Auditor model on an image file or a piece of text.")
    aud.add_argument("image", nargs="?", help="Path to an image file.")
    aud.add_argument("--text", help="Text to audit instead of an image.")
    aud.add_argument("--prompt", required=True, help="The original request the item should satisfy.")
    aud.set_defaults(func=_cmd_audit)

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in {"frank", "jack", "audit", "-h", "--help"}:
        argv = ["frank", *argv]  # `python -m src.agent "question"` talks to Frank

    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "audit" and (args.image is None) == (args.text is None):
        parser.error("audit needs exactly one of: an image path, or --text")

    try:
        config = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return args.func(args, config)


if __name__ == "__main__":
    sys.exit(main())
