"""Runtime subgraph expansion strategies (ADR-0008).

When a ``SubgraphTask``'s deps finish, the runner reads the design doc the
parent agent wrote (path in ``SubgraphTask.expand_from``) and calls the
strategy registered under ``SubgraphTask.expansion_kind`` here. The strategy
returns a flat list of ``AgentTask`` / ``ToolTask`` instances which the
runner splices into the live DAG. All emitted ids are namespaced as
``<subgraph.id>__<local_id>`` so they sort under the subgraph in trajectory
listings and don't collide with sibling subgraphs.

Mirrors the API shape of ``fix_loop.build_fix_tasks``: a free function that
takes the run state (subgraph + workspace + design doc) and returns a list
of new tasks. No runner self state; strategies stay testable in isolation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from .plan_schema import _resolve_goal_template
from .tasks import AgentTask, SubgraphTask, Task, ToolTask


# --- skill sets for per-part agents (design/build/joints live in
#     plan_generator._SKILL_BY_TASK; the part-agent set is defined here) ------

_PART_SKILLS = [
    "topos_part_geometry",
    "topos_bpy_docs",
    "topos_blender_pitfalls",
]
# NB: no texture skill here — part agents write geometry only. Texture is owned
# by the design agent (authors texture.prompt in design.json) + the build
# harness (generates + UV-binds the PNG). See topos_texture_creator (design).
_HARDWARE_KEYWORDS = {"handle", "pull", "knob", "grip", "hinge", "latch", "catch", "lock"}
# Drivetrain / running-gear / insertion mechanisms — parts that read as machine
# only when built as a fixed multi-primitive assembly (crankset = axle + arms +
# chainring + pedals, etc.). Matched as substrings, so "crank" catches
# "crankset", "seat_post"/"seatpost" both hit, etc.
_MECHANICAL_KEYWORDS = {
    "crank", "chainset", "pedal", "chainring", "sprocket", "cog", "cassette",
    "gear", "wheel", "hub", "spoke", "axle", "spindle", "pulley", "derailleur",
    "seat_post", "seatpost", "steerer", "stem",
}


def _camel_to_snake(name: str) -> str:
    """``DrawerTop`` → ``drawer_top``; ``Frame`` → ``frame``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _skills_for_part(part_name: str) -> list[str]:
    skills = list(_PART_SKILLS)
    lower = part_name.lower()
    if any(kw in lower for kw in _HARDWARE_KEYWORDS):
        skills.append("topos_furniture_hardware")
    if any(kw in lower for kw in _MECHANICAL_KEYWORDS):
        skills.append("topos_mechanical_details")
    return skills


def _part_timeout_s(part_name: str) -> int:
    """Wall-clock budget for a part agent. Mechanical/drivetrain parts (crankset,
    wheel, derailleur, …) are multi-primitive assemblies that legitimately take
    longer than the furniture-calibrated flat 900s — bike part agents timed out
    at 900s, failing the whole run. Give those more headroom. This only RAISES
    the ceiling; the idle-watchdog still kills genuinely-hung agents, so a higher
    wall-clock limit only helps the slow-but-progressing ones."""
    lower = part_name.lower()
    if any(kw in lower for kw in _MECHANICAL_KEYWORDS):
        return 1500
    return 900


def _find_reference_images(workspace_root: Path, part_name: str, design_part: dict) -> list[str]:
    """Collect reference images for a part from two sources:

    1. ``design.json`` part spec: ``reference_images: ["path/to/img.jpg", ...]``
    2. ``prompts/references/<lower_name>_*.{png,jpg,jpeg,webp}`` auto-discovery
    3. ``prompts/references/all_*.{png,jpg,jpeg,webp}`` shared across all parts

    Returns workspace-relative paths.
    """
    images: list[str] = []
    for img in design_part.get("reference_images") or []:
        if (workspace_root / img).is_file():
            images.append(img)

    refs_dir = workspace_root / "prompts" / "references"
    if refs_dir.is_dir():
        snake = _camel_to_snake(part_name)
        for ext in ("png", "jpg", "jpeg", "webp"):
            for p in refs_dir.glob(f"{snake}_*.{ext}"):
                rel = str(p.relative_to(workspace_root))
                if rel not in images:
                    images.append(rel)
            for p in refs_dir.glob(f"all_*.{ext}"):
                rel = str(p.relative_to(workspace_root))
                if rel not in images:
                    images.append(rel)
    return images


def _find_extras_file(workspace_root: Path, part_name: str) -> str | None:
    """Locate ``prompts/extras_<...>.md`` for a part. Tries, in order:

    1. Exact match (``LeftArm`` → ``extras_left_arm.md``)
    2. Drop trailing segments (``DrawerTop`` → ``extras_drawer.md``)
    3. Drop leading segments (``LeftArm`` → ``extras_arm.md``)

    Returns the relpath as a string for goal_params, not the file contents.
    The Jinja resolver in plan_schema reads it.
    """
    prompts_dir = workspace_root / "prompts"
    if not prompts_dir.is_dir():
        return None
    snake = _camel_to_snake(part_name)
    parts = snake.split("_")
    # 1+2: exact, then drop trailing segments
    for i in range(len(parts), 0, -1):
        prefix = "_".join(parts[:i])
        candidate = prompts_dir / f"extras_{prefix}.md"
        if candidate.is_file():
            return f"./prompts/extras_{prefix}.md"
    # 3: drop leading segments — covers Left/Right-style prefixed siblings
    # (LeftArm, RightArm both map to extras_arm.md). Skip i=len(parts)
    # since that's already covered above.
    for i in range(1, len(parts)):
        suffix = "_".join(parts[i:])
        candidate = prompts_dir / f"extras_{suffix}.md"
        if candidate.is_file():
            return f"./prompts/extras_{suffix}.md"
    return None


# --- expander registry -----------------------------------------------------

_EXPANDERS: dict[str, Callable[..., list[Task]]] = {}


def register_expander(kind: str):
    def deco(fn: Callable[..., list[Task]]):
        _EXPANDERS[kind] = fn
        return fn
    return deco


def get_expander(kind: str) -> Callable[..., list[Task]]:
    if kind not in _EXPANDERS:
        raise ValueError(
            f"unknown expansion_kind: {kind!r}; registered: {sorted(_EXPANDERS)}"
        )
    return _EXPANDERS[kind]


def build_children(
    subgraph: SubgraphTask,
    *,
    workspace_root: Path,
    design_doc: dict | None = None,
) -> list[Task]:
    """Dispatch to the strategy registered under ``subgraph.expansion_kind``.
    Returns a list of fully-resolved ``AgentTask``/``ToolTask`` ready to be
    spliced into the runner's task list.

    ``design_doc``: pre-loaded dict (caller reads + parses ``expand_from``).
    Passed in (rather than loaded here) so tests can drive expansion with
    synthetic design docs without touching the filesystem.
    """
    if design_doc is None:
        path = workspace_root / subgraph.expand_from
        try:
            design_doc = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise RuntimeError(
                f"subgraph {subgraph.id}: expand_from not found: {path}"
            ) from None
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"subgraph {subgraph.id}: expand_from is not valid JSON: {path}: {e}"
            ) from None
    fn = get_expander(subgraph.expansion_kind)
    return fn(subgraph, workspace_root=workspace_root, design_doc=design_doc)


# --- articulated_parts -----------------------------------------------------

@register_expander("articulated_parts")
def _expand_articulated_parts(
    subgraph: SubgraphTask,
    *,
    workspace_root: Path,
    design_doc: dict,
    backend: str = "claude",
) -> list[Task]:
    """Read ``design.json["parts"]`` (or ``["children"]``) and emit per-part
    agent + texture + judge_part triplets plus a single verify_parts and
    render_parts batch. All ids are namespaced under ``subgraph.id``.

    Resulting topology (deps shown with ⇐):

        ⇐ subgraph deps (e.g., 01_agent_design)
            ├─ pp_agent_<lower>          (one per part)        — writes parts/<lower>.py
            │   └─ pp_tool_texture_<lower>                     — writes textures/<lower>.png
            │
            └─ pp_tool_verify_parts (batch, deps = all agent_<lower>)
                └─ pp_tool_render_parts
                    └─ pp_tool_judge_part_<lower>   (one per part)

    Each child task's id is ``<subgraph.id>__<local_id>``.
    """
    parts = design_doc.get("parts") or design_doc.get("children") or []
    if not parts:
        raise ValueError(
            f"articulated_parts expander: design doc at "
            f"{subgraph.expand_from!r} has no 'parts' or 'children' list"
        )

    sg = subgraph.id
    tasks: list[Task] = []
    part_names: list[str] = []
    part_agent_ids: list[str] = []
    judge_part_ids: list[str] = []

    # Pre-collect names so verify/render args can be a flat list
    for p in parts:
        name = p["name"]
        part_names.append(name)

    # 1. Per-part agent + texture
    for i, p in enumerate(parts, start=1):
        name = p["name"]
        lower = _camel_to_snake(name)
        agent_id = f"{sg}__{i:02d}_agent_part_{lower}"
        texture_id = f"{sg}__{i:02d}_tool_texture_{lower}"
        part_agent_ids.append(agent_id)

        extras_rel = _find_extras_file(workspace_root, name)
        goal_params = {"part_name": name, "lower_name": lower}
        if extras_rel:
            goal_params["extras_file"] = extras_rel
        else:
            goal_params["extras"] = ""

        # Resolve the goal_template now (expansion happens post-load)
        goal = _resolve_goal_template(
            "topos:articulated/part_geom.md.j2",
            goal_params,
            plan_dir=workspace_root,
            task_id=agent_id,
        )

        part_images = _find_reference_images(workspace_root, name, p)
        tasks.append(AgentTask(
            id=agent_id,
            goal=goal,
            backend=backend,
            deps=list(subgraph.deps),       # part-agents wait for whatever the subgraph waited for
            allowed_tools=[
                "Read", "Edit", "Write", "Glob", "Bash",
                "WebSearch", "WebFetch",
            ],
            skills=_skills_for_part(name),
            images=part_images,
            timeout_s=_part_timeout_s(name),
        ))

        tasks.append(ToolTask(
            id=texture_id,
            tool="generate_texture_image",
            args={"part_name": name, "timeout_s": 180},
            deps=[agent_id],
        ))

    # 2. Batch verify + render
    verify_id = f"{sg}__zz_tool_verify_parts"
    render_id = f"{sg}__zz_tool_render_parts"
    tasks.append(ToolTask(
        id=verify_id,
        tool="verify_parts",
        args={
            "parts_dir_relpath": "src/parts",
            "parts": part_names,
            "output_relpath": "scratch/verify_parts_result.json",
            "timeout_s": 180,
        },
        deps=list(part_agent_ids),
    ))
    tasks.append(ToolTask(
        id=render_id,
        tool="render_part",
        args={
            "parts_dir_relpath": "src/parts",
            "parts": part_names,
            "output_subdir": "artifacts/parts_render",
            "n_views": 4,
            "resolution": 384,
            "engine": "eevee",
            "coloring": "as_authored",
            "timeout_s": 240,
        },
        deps=[verify_id],
    ))

    # 3. Per-part judge (depends on the batch render)
    for i, p in enumerate(parts, start=1):
        name = p["name"]
        lower = _camel_to_snake(name)
        jid = f"{sg}__{i:02d}_tool_judge_part_{lower}"
        judge_part_ids.append(jid)
        tasks.append(ToolTask(
            id=jid,
            tool="judge",
            args={
                "rubric": "part_shape_v1",
                "image_pattern": f"artifacts/parts_render/{name}/view_*.png",
                "metadata": {
                    "part_name": name,
                    "role_hint": (
                        f"This is the '{name}' part — an articulated component. "
                        f"Score the per-part rubric based on whether the visible "
                        f"mesh reads as a well-formed {name.lower()} for its role "
                        f"in this object."
                    ),
                },
            },
            deps=[render_id],
        ))

    return tasks


# --- public helpers --------------------------------------------------------

def expansion_kinds() -> list[str]:
    """Sorted list of registered expansion strategy keys."""
    return sorted(_EXPANDERS)
