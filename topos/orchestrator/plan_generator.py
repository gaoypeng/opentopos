"""Generate the fixed articulated plan.json for a fresh workspace.

Used by ``topos make``. The articulated plan is a template (design → subgraph →
build → joints → render/export/judge); parts are discovered at runtime by the
``articulated_parts`` expander from ``src/design.json`` (ADR-0008), so the plan
needs nothing more than the project name.
"""

from __future__ import annotations


_SKILL_BY_TASK = {
    # Design agent: pure design.json author. Doesn't write Python so no bpy_docs;
    # gets texture skill so it knows the optional `texture` field exists.
    "design": ["topos_design_articulated", "topos_texture_creator"],
    # Build agent: stitches parts together, also a Python author. Gets
    # geometry_contracts (fill-ratio / inter-part collision / cavity-fit) so
    # it can emit the corresponding validation blocks. mesh_islands is a
    # related-but-experimental skill (see its STATUS section) that is not
    # wired here yet — its heuristic for picking a "main mass" baseline
    # produces false positives on parts whose main body is itself composed
    # of many small joined sub-meshes (the dominant case in practice).
    # Re-enable once a robust main-mass detector lands.
    "build": ["topos_bpy_docs", "topos_geometry_contracts"],
    # Joints agent: writes a YAML / Python joint description — small surface
    # area, can still benefit from bpy_docs (mathutils etc.).
    "joints": ["topos_joints_creator", "topos_bpy_docs"],
}


def generate_plan_articulated(project: str, prompt: str) -> dict:
    """Build a plan.json dict for an articulated project named ``project``.

    ``prompt`` is the user's request, inlined directly into the design agent's
    goal (no separate file).

    Post-ADR-0008: the plan emits a single ``SubgraphTask`` (kind=``subgraph``)
    in place of the per-part fan-out. At runtime, after ``01_agent_design``
    writes ``src/design.json``, the runner reads it and dynamically spawns
    one part-agent + texture + judge_part triplet per design.json part —
    plus a single verify_parts + render_parts batch — via the
    ``articulated_parts`` strategy in ``topos/orchestrator/expand.py``.

    The build / joints / asm-tool tasks reference the subgraph id in their
    deps; the subgraph completes when all its dynamic children resolve.
    """
    tasks: list[dict] = []

    # 01_agent_design — reads the inlined prompt, writes src/design.json.
    # WebSearch + WebFetch let the designer consult reference imagery /
    # canonical descriptions for the asset class (e.g., G1 Optimus Prime
    # color codes, Victorian wardrobe proportions) before fixing the parts
    # list. The agent decides whether to invoke; furniture flows typically
    # don't, but humanoid / mecha flows benefit substantially.
    tasks.append({
        "id": "01_agent_design",
        "kind": "agent",
        "backend": "claude",
        "goal_template": "topos:articulated/designer.md.j2",
        "goal_params": {"prompt": prompt},
        "skills": _SKILL_BY_TASK["design"],
        "allowed_tools": [
            "Read", "Edit", "Write", "Glob",
            "WebSearch", "WebFetch",
        ],
        # Soft deadline. A complex multi-part design (or a model doing a long
        # reasoning turn) can run past 5 min; the partial-message watchdog keeps
        # it alive as long as it's actually streaming, but give real headroom.
        "timeout_s": 600,
        # design.json gates the whole run (the subgraph expands from it). A
        # no-op design turn that "succeeds" without writing it used to crash
        # expansion; declaring it here makes the runner retry-once then fail loud.
        "expected_outputs": ["src/design.json"],
    })

    # 02_subgraph_parts — at runtime expands into per-part agent / texture /
    # judge_part triplets + 1 verify_parts + 1 render_parts batch, driven by
    # design.json's parts list. See topos/orchestrator/expand.py.
    SUBGRAPH_ID = "02_subgraph_parts"
    tasks.append({
        "id": SUBGRAPH_ID,
        "kind": "subgraph",
        "expand_from": "src/design.json",
        "expansion_kind": "articulated_parts",
        "deps": ["01_agent_design"],
    })

    # 03_agent_build — composes all parts written by the subgraph's children.
    # Depends on the subgraph itself; the runner blocks build until every
    # child resolves and reports success.
    tasks.append({
        "id": "03_agent_build",
        "kind": "agent",
        "backend": "claude",
        "deps": [SUBGRAPH_ID],
        "goal_file": "topos:articulated/builder.md",
        "skills": _SKILL_BY_TASK["build"],
        "allowed_tools": ["Read", "Edit", "Write", "Glob", "Bash"],
        # build composes every part and is the heaviest agent step; 5 min was
        # too tight (observed gemini builds idle-killed at the soft deadline).
        "timeout_s": 600,
        "expected_outputs": ["src/build.py"],
    })

    # 04_agent_joints — joints.yaml is derived from design.json; only needs
    # the design agent's output, runs in parallel with the parts subgraph.
    tasks.append({
        "id": "04_agent_joints",
        "kind": "agent",
        "backend": "claude",
        "deps": ["01_agent_design"],
        "goal_file": "topos:articulated/joints_writer.md",
        "skills": _SKILL_BY_TASK["joints"],
        "allowed_tools": ["Read", "Edit", "Write", "Glob", "Bash"],
        "timeout_s": 180,
        "expected_outputs": ["src/joints.yaml"],
    })

    # Assembly tool tasks
    tasks.extend([
        {
            "id": "05_tool_render_multiview",
            "kind": "tool",
            "tool": "render_multiview",
            "args": {
                "script_relpath": "src/build.py",
                "output_subdir": "artifacts/object_render",
                "n_views": 8,
                "resolution": 512,
                "engine": "eevee",
                "coloring": "as_authored",
                "timeout_s": 360,
            },
            "deps": ["03_agent_build"],
        },
        {
            "id": "06_tool_export_glb",
            "kind": "tool",
            "tool": "export_glb",
            "args": {
                "script_relpath": "src/build.py",
                "output_relpath": "artifacts/object.glb",
                "timeout_s": 300,
            },
            "deps": ["03_agent_build"],
        },
        {
            "id": "07_tool_export_urdf",
            "kind": "tool",
            "tool": "export_urdf",
            "args": {
                "script_relpath": "src/build.py",
                "joints_relpath": "src/joints.yaml",
                "output_urdf_relpath": "artifacts/object.urdf",
                "parts_subdir": "artifacts/parts",
                "timeout_s": 300,
            },
            "deps": ["03_agent_build", "04_agent_joints"],
        },
        {
            "id": "08_tool_judge",
            "kind": "tool",
            "tool": "judge",
            "args": {
                "rubric": "articulated_object_v2",
                "image_pattern": "artifacts/object_render/view_*.png",
                # the prompt is the ground truth for "what should this be" —
                # the judge turns it into a role_hint so scoring is grounded
                "metadata": {"prompt": prompt},
                # if the user gave reference image(s), compare the assembled
                # render against them (catches structure/orientation/proportion
                # mismatches a text rubric alone misses).
                "compare_to_reference": True,
            },
            "deps": ["05_tool_render_multiview"],
        },
    ])

    return {
        "project": project,
        # max_global_iters = iter 0 + (N-1) fix iters. Set to 2 → at most ONE
        # fix iter. Combined with the regression early-stop (runner halts if a
        # fix drops the assembly score) and the assembly-pass stop, this keeps
        # the fix-loop from burning 40-50k-token part rewrites chasing a
        # whole-object that isn't converging.
        "iter_policy": {"max_global_iters": 2, "stop_on": "judge_pass"},
        "tasks": tasks,
    }
