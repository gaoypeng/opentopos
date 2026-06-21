"""Shared utilities for vision-critic backends.

Merges three previously-separate helper files (``_score.py`` / ``_schema.py``
/ ``_http.py``) and absorbs the prompt-rendering shape that used to live as
three near-identical ``_build_prompt`` copies in ``cli_critic`` /
``openai_vision`` / ``gemini_vision``.

What lives here:
  - ``OUTPUT_SCHEMA``           — JSON schema every critic asks the model to emit
  - ``materialise_score()``     — shared (passed, overall, per_criterion, fixes) extraction
  - ``build_critic_prompt()``   — single ``render(...)`` shape for the shared Jinja2 template
  - ``post_json_with_retries()`` — HTTP transport with 429/5xx/network retry
                                   for the API critics (openai / gemini)
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ...backends._rate_limit import is_retryable_http_status, sleep_for_backoff
from .base import Rubric


# ---------- output schema ----------

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "per_criterion": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "feedback": {"type": "string"},
                },
                "required": ["score", "feedback"],
            },
        },
        "overall_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "passed": {"type": "boolean"},
        "suggested_fixes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["per_criterion", "overall_score", "passed", "suggested_fixes"],
}


# ---------- score extraction ----------

def materialise_score(
    parsed: dict, rubric: Rubric,
) -> tuple[bool, float, dict, list[str]]:
    """Return (passed, overall_score, per_criterion, suggested_fixes).

    ``passed`` requires BOTH the model's own ``passed`` claim AND the
    threshold check — belt-and-suspenders against models that score below
    threshold but still claim pass. If the model omits ``passed``, we
    default it to the threshold check.
    """
    per_criterion = parsed.get("per_criterion") or {}
    overall = float(parsed.get("overall_score", 0.0))
    passed_claim = bool(parsed.get("passed", overall >= rubric.pass_threshold))
    passed = overall >= rubric.pass_threshold and passed_claim
    fixes = list(parsed.get("suggested_fixes") or [])

    # Per-criterion floors: a criterion may declare ``min_floor``; if the model
    # scored it below that floor the object cannot pass regardless of the
    # weighted total. This stops a box-stack from passing on framing while a
    # load-bearing dimension (geometry_detail / identity) is failing — the
    # exact compensation that let a flat, blocky object clear threshold before.
    floor_violations: list[str] = []
    for crit in rubric.criteria:
        floor = getattr(crit, "min_floor", None)
        if floor is None:
            continue
        cscore = (per_criterion.get(crit.id) or {}).get("score")
        if cscore is not None and float(cscore) < float(floor):
            passed = False
            floor_violations.append(
                f"{crit.id} scored {float(cscore):.2f} < required floor {float(floor):.2f}"
            )
    if floor_violations:
        # Surface the gate first so the fix-loop targets the blocking dimension.
        fixes = ["BLOCKING (below required floor): " + "; ".join(floor_violations)] + fixes

    return passed, overall, per_criterion, fixes


# ---------- prompt rendering ----------

def build_critic_prompt(
    rubric: Rubric,
    *,
    image_names: list[str] | None = None,
    role_hint: str | None = None,
    reference_image_names: list[str] | None = None,
    n_reference: int = 0,
) -> str:
    """Render the shared vision-judge Jinja2 prompt.

    ``image_names`` populates the template's image-list section: pass actual
    relative paths for backends where the model uses a Read tool to load
    images from disk (cli_critic); leave empty/None for API backends where
    images embed in the message body as base64.

    ``role_hint`` is forwarded as extra context to the model.

    Reference comparison: when the user provided a reference target, pass
    ``reference_image_names`` (CLI critics Read them) and/or ``n_reference``
    (API critics embed the reference images FIRST, so the count tells the model
    which leading images are the target). The template then asks the model to
    grade STRUCTURAL fidelity against the reference, not photographic match.
    """
    from ...prompts import render as render_prompt
    return render_prompt(
        "system/vision_judge_base.md.j2",
        rubric_id=rubric.id,
        pass_threshold=rubric.pass_threshold,
        image_names=image_names or [],
        criteria=rubric.criteria,
        output_schema_json=json.dumps(OUTPUT_SCHEMA, indent=2),
        role_hint=role_hint,
        reference_image_names=reference_image_names or [],
        n_reference=n_reference,
    )


# ---------- HTTP transport ----------

def post_json_with_retries(
    *,
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: int,
    max_retries: int,
    retry_base_wait_s: float,
    label: str,
) -> bytes:
    """POST ``body`` to ``url``; return response bytes.

    Retries on HTTP 429/5xx and transient network errors (``URLError``,
    ``TimeoutError``) with exponential backoff. ``label`` appears in log
    prefixes and the final ``RuntimeError`` message so failures can be
    traced back to the caller (e.g. ``openai_vision`` vs ``gemini_vision``).

    Total attempts = ``max_retries + 1``. Raises ``RuntimeError`` on the
    last attempt (whether HTTP error, network error, or — defensively —
    exhaustion with no recorded outcome).
    """
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            if attempt < max_retries and is_retryable_http_status(e.code):
                wait = sleep_for_backoff(attempt, retry_base_wait_s)
                print(
                    f"[{label}] HTTP {e.code} (attempt {attempt + 1}/"
                    f"{max_retries + 1}); slept {wait:.0f}s before retry."
                )
                continue
            raise RuntimeError(f"[{label}] HTTP {e.code}: {err_body[:500]}")
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < max_retries:
                wait = sleep_for_backoff(attempt, retry_base_wait_s)
                print(
                    f"[{label}] network error '{e}' (attempt {attempt + 1}/"
                    f"{max_retries + 1}); slept {wait:.0f}s before retry."
                )
                continue
            raise RuntimeError(f"[{label}] network error: {e}")
    # Unreachable: every iteration either returns, continues, or raises.
    raise RuntimeError(f"[{label}] exhausted retries")
