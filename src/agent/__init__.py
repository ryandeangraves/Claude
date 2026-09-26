"""OpenAI agents for the Second Brain app.

* Frank (:func:`run_frank`) is the top-level assistant the user talks to. He
  issues jobs to Jack ("Hey Frank, audit this" / "draw me a fox").
* Jack (:func:`run_jack`) does the jobs: validates contact details, drafts
  notifications, generates images, and audits text or images.
* The Auditor model (:func:`audit_text`, :func:`audit_image`) is what Jack's
  audit tools run; it returns a structured :class:`AuditReport`.
* :func:`run_audited` runs Jack, audits everything he produced, and retries
  with the Auditor's suggested fix. Frank uses it when ``auto_audit`` is on.
"""
from src.agent.auditor import AuditReport, audit_image, audit_text, build_auditor
from src.agent.config import AgentConfig, ConfigError, load_config
from src.agent.frank import FrankContext, FrankResult, build_frank, run_frank
from src.agent.image import GeneratedImage
from src.agent.jack import JackContext, JackResult, build_jack, run_jack
from src.agent.pipeline import Attempt, PipelineResult, run_audited

__all__ = [
    "AgentConfig",
    "Attempt",
    "AuditReport",
    "ConfigError",
    "FrankContext",
    "FrankResult",
    "GeneratedImage",
    "JackContext",
    "JackResult",
    "PipelineResult",
    "audit_image",
    "audit_text",
    "build_auditor",
    "build_frank",
    "build_jack",
    "load_config",
    "run_audited",
    "run_frank",
    "run_jack",
]
