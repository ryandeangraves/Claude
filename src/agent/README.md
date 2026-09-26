# OpenAI Agents

Agents built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python):

| Agent | Entry point | What it does |
|---|---|---|
| Assistant | `run_agent` / `chat` | Validates contact details and drafts order notifications using `src/notifications` as tools |
| Image Generator | `generate_image` / `image` | Generates images with the hosted `image_generation` tool (`gpt-image-1`) and saves them to disk |
| Image Auditor | `audit_image` / `audit` | Inspects an image against a checklist and returns a structured pass/fail report |
| Pipeline | `generate_and_audit` / `pipeline` | Generates, audits, and retries with the auditor's suggested prompt fix |

## Setup

```bash
pip install -r requirements.txt        # or: pip install -e ".[dev]"
cp .env.example .env                   # then paste your key into .env
export OPENAI_API_KEY=sk-...           # or export it directly
```

The key is read only from the environment. It is never logged, and
`AgentConfig`'s `repr` masks it.

## Run

```bash
python -m src.agent chat "Is +14155552671 a valid phone number?" --usage
python -m src.agent image "a watercolor red fox" --out ./generated --size 1024x1024 --quality high
python -m src.agent audit ./generated/<file>.png --prompt "a watercolor red fox"
python -m src.agent pipeline "a watercolor red fox" --out ./generated --attempts 2
```

`--usage` prints request and token counts to stderr so you can watch spend.
`audit` and `pipeline` exit 0 on a passing verdict and 1 otherwise, so they
can gate a script. Generated files land in `generated/` (git-ignored).

## Configure

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI key |
| `OPENAI_AGENT_MODEL` | `gpt-4.1-mini` | Text model for all agents (the auditor needs a vision-capable model, which this is) |
| `OPENAI_AGENT_MAX_TURNS` | `10` | Cap on model/tool round-trips per run |
| `OPENAI_AGENT_TRACING` | `1` | Set `0` to disable OpenAI tracing uploads |

The image model, size, quality, format and background are arguments to
`build_image_agent` / `generate_image`.

## Use from code

```python
from src.agent import generate_and_audit

res = generate_and_audit("a watercolor red fox", out_dir="generated", max_attempts=2)
print(res.passed, res.final.image.path, res.final.report.summary)
```

The audit report is a pydantic model:

```python
AuditReport(
    verdict="pass" | "fail" | "unclear",
    score=0..100,
    summary=str,
    results=[ChecklistResult(item, passed, notes), ...],
    suggested_prompt_fix=str | None,
)
```

Pass your own `checklist=[...]` to `audit_image` or `generate_and_audit` to
change what is audited.

## Tests

```bash
python -m pytest tests/unit/test_agent.py tests/unit/test_agent_image.py
```

Tests stub the SDK runner and never call the API.
