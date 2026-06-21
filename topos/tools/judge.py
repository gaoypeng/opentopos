"""``judge`` tool: load a rubric, materialise the right ``Critic``, and
evaluate against rendered images. Exposed to agents and to ToolTasks.

(Tool name stays ``judge`` for plan.json / config stability; internal types
are the renamed ``Critic`` / ``CriticInputs`` from ``topos.agents.visual_critic``.)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agents.visual_critic.base import CriticInputs, load_rubric, make_critic
from ._paths import resolve_under_workspace
from .registry import tool


INPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "workspace": {"type": "string"},
        "rubric": {"type": "string", "description": "Rubric short name (e.g. 'articulated_object_v1') or absolute path to .yaml"},
        "image_pattern": {"type": "string", "description": "Glob pattern relative to workspace (e.g. 'artifacts/*.png')"},
        "images": {"type": "array", "items": {"type": "string"}, "description": "Explicit list, takes precedence over image_pattern"},
        "metadata": {"type": "object"},
        "compare_to_reference": {"type": "boolean", "description": "If true and the workspace has reference image(s) under prompts/references/all_*, pass them to the critic as a comparison target (assembly-level judge only)."},
        "timeout_s": {"type": "integer", "description": "Critic wall-clock budget. If omitted, auto-scaled by image count (floor 300s) so an 8-view assembly judge gets ~600s while a 1-2 view part judge stays 300s."},
    },
    "required": ["workspace", "rubric"],
}

OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "passed": {"type": "boolean"},
        "overall_score": {"type": "number"},
        "per_criterion": {"type": "object"},
        "suggested_fixes": {"type": "array", "items": {"type": "string"}},
        "images_evaluated": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["passed", "overall_score", "per_criterion", "suggested_fixes"],
}


def _gather_reference_images(ws: Path) -> list[Path]:
    """Collect user-provided reference target images for an assembly judge:
    the shared ``prompts/references/all_*.{png,jpg,jpeg,webp}`` (what `topos make
    -i` drops in) plus any top-level ``design.json`` ``reference_images``."""
    refs: list[Path] = []
    refs_dir = ws / "prompts" / "references"
    if refs_dir.is_dir():
        for ext in ("png", "jpg", "jpeg", "webp"):
            refs.extend(sorted(refs_dir.glob(f"all_*.{ext}")))
    design_path = ws / "src" / "design.json"
    if design_path.is_file():
        try:
            design = json.loads(design_path.read_text(encoding="utf-8"))
            for rel in design.get("reference_images") or []:
                p = (ws / rel)
                if p.is_file() and p not in refs:
                    refs.append(p)
        except (ValueError, OSError):
            pass
    return refs


@tool(
    "judge",
    description=(
        "Evaluate rendered images against a Topos rubric and return scores plus "
        "concrete suggested fixes. With compare_to_reference, also compares the "
        "render against the user's reference image(s)."
    ),
    input_schema=INPUT_SCHEMA,
    output_schema=OUTPUT_SCHEMA,
    side_effects=False,
)
def judge(
    *,
    workspace: str,
    rubric: str,
    image_pattern: str | None = None,
    images: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    compare_to_reference: bool = False,
    timeout_s: int | None = None,
) -> dict[str, Any]:
    ws = Path(workspace).resolve()
    if not ws.is_dir():
        raise NotADirectoryError(f"workspace must be a directory: {ws}")

    img_paths: list[Path] = []
    if images:
        img_paths = [resolve_under_workspace(ws, p, label=f"images[{i}]") for i, p in enumerate(images)]
    elif image_pattern:
        img_paths = sorted(ws.glob(image_pattern))

    if not img_paths:
        # Fail loud (CLAUDE.md rule #12): zero matches is a misconfiguration,
        # not a low judge score. Letting it return passed=False would burn
        # fix-loop iters on a non-existent visual.
        raise FileNotFoundError(
            f"judge: no images matched under {ws} "
            f"(pattern={image_pattern!r}, images={images!r}). "
            "Check the workspace and the image_pattern in plan.json."
        )

    rubric_obj = load_rubric(rubric)
    # Auto-scale the critic's wall-clock budget by image count (floor 300s) when
    # the caller didn't pin one: an 8-view assembly judge running a CLI critic
    # (claude_vision) needs more than the flat 300s — a turbofan assembly judge
    # timed out at 300.2s and hard-failed a complete deliverable. A 1-2 view
    # per-part judge stays at the 300s floor.
    if timeout_s is None:
        timeout_s = max(300, 120 + 60 * len(img_paths))
    critic = make_critic(rubric_obj, config={"timeout_s": int(timeout_s)})
    # Inject workspace_path so CLI critics (claude_vision, codex_cli, gemini_cli)
    # can run INSIDE the project workspace and Read src/ files for grounding.
    # API-only critics (openai_vision, gemini_vision) ignore this field.
    md = dict(metadata or {})
    md.setdefault("workspace_path", str(ws))

    # Inject the user's request as role_hint when the caller hasn't set one.
    # Without this the critic sees rendered images plus a generic rubric and
    # has no way to tell whether the produced thing matches the request —
    # leading to absurdities like grading a turbofan engine as a "grinder or
    # shaker" because the silhouette is plausible and the rubric is identity-
    # agnostic. `make` passes the prompt inline via metadata.prompt; a hand-
    # built workspace may instead carry a prompts/intent.md — accept either.
    # Per-part judges already set a part-specific role_hint upstream, so this
    # only kicks in for assembly-level judges that lack one.
    if "role_hint" not in md:
        request = md.get("prompt")
        if not request:
            intent_path = ws / "prompts" / "intent.md"
            if intent_path.is_file():
                request = intent_path.read_text(encoding="utf-8").strip()
        if request:
            md["role_hint"] = (
                "The output renders should match this user request:\n\n"
                f"{request}\n\n"
                "Grade the renders by how well the produced object matches "
                "this request. Be critical — do NOT assume the produced object "
                "already matches because the prompt describes the target."
            )
    md.pop("prompt", None)  # control field consumed here, not a critic input

    reference_images = _gather_reference_images(ws) if compare_to_reference else []

    result = critic.evaluate(
        CriticInputs(images=img_paths, reference_images=reference_images, metadata=md),
        rubric_obj,
    )
    return {
        **result.to_dict(),
        "images_evaluated": [str(p.relative_to(ws)) if p.is_relative_to(ws) else str(p) for p in img_paths],
    }
