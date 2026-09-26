# Frank and Jack

Two OpenAI agents built on the [OpenAI Agents SDK](https://github.com/openai/openai-agents-python).

| Agent | Role | Model |
|---|---|---|
| **Frank** | The one you talk to. Holds the conversation and issues jobs to Jack. Never does the work himself. | `OPENAI_AGENT_MODEL` (default `gpt-4.1-mini`) |
| **Jack** | Does the jobs: validates emails and phone numbers, drafts order notifications, generates images, audits text and images. | `OPENAI_AGENT_MODEL`; images via `OPENAI_IMAGE_MODEL` (default `gpt-image-2.5-sunburst`) |
| Auditor model | What Jack's audit tools run. Returns a structured pass/fail report. | `OPENAI_AUDITOR_MODEL` (default `gpt-6-astra`) |

So "Hey Frank, audit this" becomes a job Frank issues to Jack, and Jack runs
the Auditor model on it. "Hey Frank, draw me a fox" becomes an image job.

## Setup

```bash
pip install -r requirements.txt        # or: pip install -e ".[dev]"
cp .env.example .env                   # then paste your key into .env
export OPENAI_API_KEY=sk-...           # or export it directly
```

The key is read only from the environment. It is never logged, and
`AgentConfig`'s `repr` masks it.

## Talk to Frank

```bash
python -m src.agent "Hey Frank, draw a watercolor red fox"
python -m src.agent "Hey Frank, audit generated/<file>.png against 'a watercolor red fox'"
python -m src.agent "Hey Frank, draft a confirmation email for order 42 to a@example.com, total 19.99"
python -m src.agent "Hey Frank, audit that draft" --session ryan
```

`--session NAME` gives Frank memory across runs, stored in
`generated/frank-sessions.db`, so "audit this" can point at something Jack
made in an earlier turn. Without a session, put what to audit in the message.

Other flags: `--verbose` prints each job's full output and audit reports,
`--usage` prints token counts to stderr, `--out DIR` changes where images are
saved (default `generated/`, git-ignored).

### Automatic auditing

```bash
python -m src.agent "Hey Frank, draw a watercolor red fox" --auto-audit --attempts 2
```

With `--auto-audit`, every job's text and every generated image is audited
automatically, and a failing job is retried with the Auditor's suggested fix
up to `--attempts` times. The exit code is 0 only if every job passed.

## Jack and the Auditor directly

```bash
python -m src.agent jack "Draw a watercolor red fox" --quality xhigh
python -m src.agent audit ./generated/<file>.png --prompt "a watercolor red fox"
python -m src.agent audit --text "Jack's draft" --prompt "what it should say"
```

`--quality` accepts `auto`, `low`, `medium`, `high`, `xhigh`, `max`; the last
two only exist on the gpt-image-2.5 models. Sunburst is the highest-fidelity
variant; set `OPENAI_IMAGE_MODEL=gpt-image-2.5-flare` for the faster sibling.

## Configure

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI key |
| `OPENAI_AGENT_MODEL` | `gpt-4.1-mini` | Frank and Jack |
| `OPENAI_AUDITOR_MODEL` | `gpt-6-astra` | Vision model behind audits |
| `OPENAI_IMAGE_MODEL` | `gpt-image-2.5-sunburst` | Model behind the `image_generation` tool |
| `OPENAI_AGENT_MAX_TURNS` | `10` | Cap on model/tool round-trips per run |
| `OPENAI_AGENT_TRACING` | `1` | Set `0` to disable OpenAI tracing uploads |

## Use from code

```python
from src.agent import run_frank, run_jack, audit_image, run_audited

r = run_frank("Hey Frank, draw a watercolor red fox", out_dir="generated", session="ryan")
print(r.output)                      # Frank's reply
for job in r.jobs:                   # what Jack did
    print(job.output, [i.path for i in job.images], [a.verdict for a in job.audits])

job = run_jack("Audit generated/fox.png against 'a watercolor red fox'")
report = audit_image("generated/fox.png", "a watercolor red fox")   # the Auditor model directly
res = run_audited("Draw a watercolor red fox", max_attempts=2)       # generate, audit, retry
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

Pass your own `checklist=[...]` to `audit_text` / `audit_image`, or
`text_checklist=` / `image_checklist=` to `run_audited`, to change what is
audited. Jack's `audit_image_file` tool only accepts png, jpg, jpeg, webp or
gif files up to 20 MB.

## Tests

```bash
python -m pytest tests/unit/test_agent.py tests/unit/test_auditor.py
```

Tests stub the SDK runner and never call the API.
