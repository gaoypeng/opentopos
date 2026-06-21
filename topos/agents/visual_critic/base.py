"""Critic protocol, rubric loader, and factory.

A ``Critic`` evaluates inputs (images + metadata) against a ``Rubric`` and
returns a ``CriticResult``. Implementations are dispatched by name from the
rubric's ``judge_backend`` field (the YAML field name is kept for plan.json /
rubric stability — every shipped rubric.yaml still has ``judge_backend:``).

New critic backends drop into this directory; register a branch in
``make_critic()`` below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml


@dataclass
class Criterion:
    id: str
    prompt: str
    weight: float = 1.0
    # Optional hard gate: if set, this criterion scoring below ``min_floor``
    # blocks the whole object from passing regardless of the weighted total
    # (so a weak-but-compensated dimension — e.g. geometry_detail or identity —
    # can't be averaged away). None = no floor (weighted total only).
    min_floor: float | None = None


@dataclass
class Rubric:
    id: str
    judge_backend: str          # YAML field kept for stability (plan.json / yaml)
    pass_threshold: float
    criteria: list[Criterion]
    description: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "Rubric":
        criteria = [Criterion(**c) for c in (d.get("criteria") or [])]
        return cls(
            id=d["id"],
            judge_backend=d.get("judge_backend", "claude_vision"),
            pass_threshold=float(d.get("pass_threshold", 0.7)),
            criteria=criteria,
            description=d.get("description"),
            extras={k: v for k, v in d.items()
                    if k not in {"id", "judge_backend", "pass_threshold", "criteria", "description"}},
        )

    @classmethod
    def from_yaml_file(cls, path: Path) -> "Rubric":
        with path.open("r", encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f))


@dataclass
class CriticInputs:
    images: list[Path]                                       # the rendered output to grade
    reference_images: list[Path] = field(default_factory=list)  # optional user-provided target(s) to compare against
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CriticResult:
    passed: bool
    overall_score: float
    per_criterion: dict[str, dict[str, Any]]
    suggested_fixes: list[str]
    raw_response: str | None = None
    cost_usd: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "per_criterion": self.per_criterion,
            "suggested_fixes": self.suggested_fixes,
            "cost_usd": self.cost_usd,
            "usage": self.usage,
        }


@runtime_checkable
class Critic(Protocol):
    def evaluate(self, inputs: CriticInputs, rubric: Rubric) -> CriticResult: ...


def make_critic(rubric: Rubric, *, config: dict | None = None) -> Critic:
    """Factory: pick the implementation matching ``rubric.judge_backend``.

    Built-in backends:
      - ``claude_vision`` — claude CLI agent, sees images AND src/ source code
      - ``codex_cli``     — codex CLI agent, sees images AND src/ source code
      - ``gemini_cli``    — gemini CLI agent, sees images AND src/ source code
      - ``openai_vision`` — OpenAI Chat Completions API (HTTP), images only
      - ``gemini_vision`` — Gemini generateContent API (HTTP), images only

    The three CLI critics share a common ``CliVisionCritic`` implementation
    (in ``cli_critic.py``); they differ only in which agent CLI gets driven.
    API critics are single-shot HTTP — faster and cheaper but image-only.

    **Global override.** If ``visual_critic.default`` is set in user config
    (e.g. ``topos config set visual_critic.default gemini_vision``), every
    rubric is critiqued by that backend regardless of its own
    ``judge_backend:`` field. Useful for swapping the whole pipeline to a
    cheaper model without editing per-rubric YAMLs. The rubric's own field
    is treated as a fallback; the override always wins when set.
    """
    backend_name = rubric.judge_backend

    # Optional global override (one-knob swap for the entire pipeline).
    from ... import config as _cfg
    try:
        _effective = _cfg.load_effective_config()
    except Exception:  # noqa: BLE001
        _effective = {}
    _override = (
        ((_effective.get("visual_critic") or {}).get("default"))
        or _effective.get("visual_critic_default")
    )
    if _override:
        backend_name = str(_override)

    if backend_name == "claude_vision":
        from .claude_vision import ClaudeVisionCritic
        return ClaudeVisionCritic.from_config(config)
    if backend_name == "gemini_vision":
        from .gemini_vision import GeminiVisionCritic
        return GeminiVisionCritic.from_config(config)
    if backend_name == "openai_vision":
        from .openai_vision import OpenAIVisionCritic
        return OpenAIVisionCritic.from_config(config)
    if backend_name == "codex_cli":
        from .cli_critic import make_codex_cli_critic
        return make_codex_cli_critic(config)
    if backend_name == "gemini_cli":
        from .cli_critic import make_gemini_cli_critic
        return make_gemini_cli_critic(config)
    raise ValueError(f"unknown judge_backend: {backend_name!r}")


def load_rubric(name_or_path: str) -> Rubric:
    """Load a rubric by short name (from ``topos/rubrics/<name>.yaml``) or by
    absolute path.
    """
    p = Path(name_or_path)
    if p.is_file():
        return Rubric.from_yaml_file(p)
    from importlib import resources
    res = resources.files("topos").joinpath("rubrics", f"{name_or_path}.yaml")
    with res.open("r", encoding="utf-8") as f:
        return Rubric.from_dict(yaml.safe_load(f))
