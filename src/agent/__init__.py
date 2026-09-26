"""OpenAI Agent integration.

Three agents are available:

* :func:`run_agent`          - general assistant with notification tools
* :func:`generate_image`     - image generation (hosted image_generation tool)
* :func:`audit_image`        - structured audit of an image against a checklist
* :func:`generate_and_audit` - generate, then audit, with optional retry
"""
from src.agent.agent import AgentRunResult, build_agent, run_agent
from src.agent.auditor import AuditReport, audit_image, build_auditor_agent
from src.agent.config import AgentConfig, ConfigError, load_config
from src.agent.image import GeneratedImage, ImageRunResult, build_image_agent, generate_image
from src.agent.pipeline import PipelineResult, generate_and_audit

__all__ = [
    "AgentConfig",
    "AgentRunResult",
    "AuditReport",
    "ConfigError",
    "GeneratedImage",
    "ImageRunResult",
    "PipelineResult",
    "audit_image",
    "build_agent",
    "build_auditor_agent",
    "build_image_agent",
    "generate_and_audit",
    "generate_image",
    "load_config",
    "run_agent",
]
