"""Command-line entry point.

    python -m src.agent chat "Is +14155552671 a valid phone number?"
    python -m src.agent image "a watercolor fox" --out ./generated
    python -m src.agent audit ./generated/fox.png --prompt "a watercolor fox"
    python -m src.agent pipeline "a watercolor fox" --out ./generated --attempts 2
"""
import argparse
import json
import sys
from pathlib import Path

from src.agent.agent import run_agent
from src.agent.auditor import audit_image
from src.agent.config import ConfigError, load_config
from src.agent.image import generate_image
from src.agent.pipeline import generate_and_audit


def _usage_line(r) -> str:
    return (
        f"[usage] requests={r.requests} in={r.input_tokens} "
        f"out={r.output_tokens} total={r.input_tokens + r.output_tokens}"
    )


def _cmd_chat(args, config) -> int:
    r = run_agent(args.prompt, config)
    print(r.output)
    if args.usage:
        print(_usage_line(r), file=sys.stderr)
    return 0


def _cmd_image(args, config) -> int:
    r = generate_image(args.prompt, config, out_dir=Path(args.out), size=args.size, quality=args.quality)
    print(r.message)
    for img in r.images:
        print(f"saved {img.path}")
    if args.usage:
        print(_usage_line(r), file=sys.stderr)
    return 0 if r.images else 1


def _cmd_audit(args, config) -> int:
    report = audit_image(Path(args.image), args.prompt, config)
    print(json.dumps(report.model_dump(), indent=2))
    return 0 if report.verdict == "pass" else 1


def _cmd_pipeline(args, config) -> int:
    res = generate_and_audit(args.prompt, config, out_dir=Path(args.out), max_attempts=args.attempts)
    for i, a in enumerate(res.attempts, start=1):
        print(f"--- attempt {i}: {a.prompt}")
        if a.image and a.image.path:
            print(f"saved {a.image.path}")
        if a.report:
            print(json.dumps(a.report.model_dump(), indent=2))
        else:
            print("no image produced")
    return 0 if res.passed else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="Ask the general assistant.")
    p.add_argument("prompt")
    p.add_argument("--usage", action="store_true")
    p.set_defaults(func=_cmd_chat)

    p = sub.add_parser("image", help="Generate an image.")
    p.add_argument("prompt")
    p.add_argument("--out", default="generated")
    p.add_argument("--size", default="1024x1024")
    p.add_argument("--quality", default="auto", choices=["auto", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--usage", action="store_true")
    p.set_defaults(func=_cmd_image)

    p = sub.add_parser("audit", help="Audit an existing image file.")
    p.add_argument("image")
    p.add_argument("--prompt", required=True, help="The prompt the image should match.")
    p.set_defaults(func=_cmd_audit)

    p = sub.add_parser("pipeline", help="Generate, then audit (retry on fail).")
    p.add_argument("prompt")
    p.add_argument("--out", default="generated")
    p.add_argument("--attempts", type=int, default=1)
    p.set_defaults(func=_cmd_pipeline)

    args = parser.parse_args(argv)
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return args.func(args, config)


if __name__ == "__main__":
    sys.exit(main())
