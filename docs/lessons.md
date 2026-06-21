# Lessons

Append-only running log of gotchas and insights. Each entry: date, one-line summary, link to commit/PR if applicable.

<!-- Template:
## YYYY-MM-DD — short title
Context: ...
Lesson: ...
Reference: <commit-sha or path>
-->

## 2026-05-10 — Stage 0 smoke wired in one session
Context: built L0–L5 from empty repo to passing hello-blender end-to-end.
Lesson: ClaudeVisionJudge implemented over `ClaudeCLIBackend` (subprocess) rather than Anthropic SDK was the right call — works under subscription auth, removes need for ANTHROPIC_API_KEY for judge path. JSON extraction from the CLI's `--output-format=json` envelope needed a two-step parse (envelope, then assistant text inside `result` field) plus a regex fallback for stray markdown fencing.
Reference: `topos/agents/visual_critic/claude_vision.py:_extract_json`

## 2026-05-10 — `use_empty=True` leaves `scene.world = None`
Context: Stage-0 smoke prompt told the agent to set `scene.world.color`. With `bpy.ops.wm.read_factory_settings(use_empty=True)` the factory world data block is purged, so `scene.world` is None and the assignment AttributeErrors. Either don't touch world, or run with `use_empty=False` and then delete unwanted defaults. The smoke prompt now uses `BLENDER_WORKBENCH` (which has its own built-in studio lighting and ignores world entirely) and explicitly forbids touching `scene.world`.
Reference: `examples/smoke_hello_blender/plan.json` (A1.goal)

## 2026-05-10 — LLM-based judges have intrinsic variance; tests verify wiring, not pass/fail
Context: smoke ran twice — first score 0.92 (pass), second got a lighting criterion that flipped overall below threshold (fail). The framework wired correctly both times; only the judge's perception varied.
Lesson: Stage-0 integration test now asserts only that the judge produced a well-formed score with `overall_score > 0`. Hard `judge.passed` assertions belong to Stage 1+ where the auto-fix loop can run multiple iterations to ride out variance. Generalising: any test that depends on an LLM's final answer should either (a) assert structural well-formedness, (b) allow N retries within the test, or (c) include both pass and not-pass scenarios in coverage.
Reference: `tests/integration/test_smoke.py`

## 2026-05-10 — claude CLI flag shape pinned for headless agents
Context: building `ClaudeCLIBackend.run`.
Lesson: For headless coding agents in an isolated workspace, the working flag combo is `-p <prompt> --output-format json --no-session-persistence --permission-mode bypassPermissions --add-dir <workspace> --allowed-tools "<comma list>" --append-system-prompt <…>`. `bypassPermissions` is required if the agent needs Bash; restrict capability via `--allowed-tools` instead of via permission prompts. `--strict-mcp-config` plus `--mcp-config <file>` is the way to scope MCP servers per task.
Reference: `topos/backends/claude_cli.py`

## 2026-05-11 — Three-layer prompt hierarchy + skills/ folder + jinja templating
Context: agent prompts had grown into giant escaped strings in plan.json (16+ KB). Reorg into three layers, mirroring ForgeCAD's pattern but with topos_ prefix on skills.

Architecture:
  topos/prompts/system/       — framework-level system prompts (system prompt, fix-loop, judge prompt). Loaded by runner.py / claude_vision.py from files instead of hardcoded strings.
  topos/prompts/<domain>/      — domain-generic templates (Jinja2 .md.j2). One per task role: designer / part_geom / builder / joints_writer. Referenced from plan.json via `goal_template: topos:articulated/<name>.md.j2` + `goal_params: {part_name, extras_file}`.
  examples/<slug>/prompts/     — example-specific intent + per-part extras. Filled into the domain templates via goal_params.
  topos/skills/<skill_name>/   — higher-level capability bundles (each a SKILL.md folder). Declared in plan.json `skills: [...]` per task; runner injects content into the agent prompt with `<skill name="..."> ... </skill>` markers. P0 set: topos_part_geometry, topos_joints_creator, topos_design_articulated.

Lesson: keeping the THREE layers cleanly separated (framework / domain / example) made plan.json drop from 16.6 KB → 4 KB and exposed all the prompts as plain markdown files for human review and incremental edit. The skills/ layer is for *learned* knowledge (how to avoid the transform_apply trap, the cube_add(size=1) half-extent gotcha, etc.) that belongs to no single task — it should be loaded as background context whenever a part agent runs, not duplicated in every prompt.
Reference: `topos/prompts/`, `topos/skills/`, `topos/orchestrator/plan_schema.py:_resolve_goal_template`, `topos/orchestrator/runner.py:_build_agent_prompt`

## 2026-05-11 — EEVEE area-light energies tuned for ~30cm objects; agent materials respected
Context: switched cabinet render to EEVEE_NEXT for material/bevel visibility. Initial run had 1200W area lights blowing every surface to white under AgX tone mapping (AgX is Blender 5.0.1's default view transform — good roll-off but still clips at extreme luminance). Also my `_ensure_pbr_materials` was destructively replacing the agent's already-correct PBR materials.
Lesson: dropped key/fill/rim energies to ~5W/1.75W/2.5W (linear in object extent, 5W per 0.3m). Changed `_ensure_pbr_materials` → `_force_base_color`: respect an existing first material's Principled BSDF, only override its Base Color from `obj.color`. Only mint a new material if obj has none. Score jumped from 0.47 (white-washed renders) to 0.75 (clean PBR with visible D-handle/bevel/inset/wall-thickness) on the same agent code.
Reference: `topos/tools/render/wrapper.py:_add_three_point_lights`, `:_force_base_color`

## 2026-05-11 — Prompts moved out of plan.json into topos/prompts/ via topos:scheme
Context: agent goals were giant escaped strings in plan.json — hard to read, edit, or review. Extracted each agent task's goal into `topos/prompts/<example>/<task_id>.md`. plan.json shrank from 16.6 KB → 3.6 KB and each prompt is now a normal markdown file. `load_plan` resolves `goal_file: topos:<rel>` via `importlib.resources.files('topos').joinpath('prompts', rel)` — survives `topos init` copying plan.json to any workspace because resolution doesn't depend on filesystem relativity.
Lesson: pin shared prompts to package resources, not relative paths from the plan file. Add `prompts/**/*.md` to `[tool.setuptools.package-data]`. Empty `__init__.py` in prompt dirs makes `importlib.resources` lookups robust across editable / wheel installs.
Reference: `topos/orchestrator/plan_schema.py:_resolve_goal_file`, `pyproject.toml`, `tests/test_plan_schema_goal_file.py`

## 2026-05-10 — bbox-contract multi-part assembly: design.json + per-part agents + build.py validator
Context: rebuilt articulated_drawer_cabinet from "1 agent writes 3 cube_adds" to "1 design-agent writes design.json contract + N part-agents implement parts/<name>.py independently against the contract + 1 build-agent writes build.py that asserts each part's world bbox matches design within 5mm". Multi-file structure with `src/parts/__init__.py`-free namespace packaging works through Blender's bundled Python 3.11.
Lesson: contract-based assembly works — parts get implemented in parallel with no inter-talk, the build-time bbox assertion catches contract violations. First run scored 0.74 (vs 0.66 previously); drawer is now a real 5-panel open-top box, frame is a real hollow cabinet. Handle agent's algorithm bug (used `scale=(extent/2,...)` on `cube_add(size=1)` which gives half-size bbox) was caught by the validator with `err_extents=61.6mm` but did not block downstream — by design (print WARN, don't raise so judge still gets to evaluate). Future improvement: fix-loop should ALSO trigger on bbox WARN, not just judge fail.
Reference: `examples/articulated_drawer_cabinet/plan.json`, `outputs/cab_fix_loop_demo/src/build.py`

## 2026-05-10 — Multi-file Blender scripts via sys.path insertion in wrapper
Context: `from parts.frame import build_frame` doesn't work in a `blender --background --python build.py` run by default because the script's parent dir isn't on `sys.path`. Solution: render_wrapper.py and export_wrapper.py insert `Path(script).parent.resolve()` at the front of `sys.path` before calling `runpy.run_path`. Python 3 namespace packages mean no `__init__.py` is needed inside `parts/`. Verified end-to-end with the cabinet pipeline.
Reference: `topos/tools/render/wrapper.py:_run_agent_script`

## 2026-05-10 — Per-part Blender export must bake transforms, not zero `matrix_world`
Context: per-part OBJ/GLB export for URDF was producing meshes that didn't match the whole-scene GLB. Drawer.obj was a 2m unit cube; Frame.obj had vertices at z=39. Root cause: `obj.matrix_world = Matrix.Identity(4)` zeros location AND scale, so the mesh exports as raw mesh data (unit cube at origin) — losing the object's scale entirely. The Frame z=39 came from `bpy.ops.object.join()` with an active object that had tiny z-scale (0.0075); the inverse-transform stretched the joined panels' local-frame vertices by 133×, with the node scale supposed to bring them back. Whole-scene GLB rendered correctly because GLTF viewers compose mesh × node transform; URDF viewers don't compose, so the OBJ was rendered "raw" at wrong size.
Lesson: bake transforms via `bpy.ops.object.transform_apply(...)` before per-part export. For per-part files, apply rotation+scale into mesh data (preserve location for the manifest). For whole-scene "what-you-see-is-what's-in-the-file" GLB, apply location+rotation+scale on every mesh. Switched per-part export from .obj to .glb at the same time (user request).
Reference: `topos/tools/export/wrapper.py:_bake_world_transform_into_mesh`, `:_export_one_glb_local`

## 2026-05-10 — Self-describing task IDs: `<order>_<role>_<desc>` + `_iter<N>` trajectory dirs
Context: `A1/T1/J1/FIX1` were opaque; users couldn't tell from trajectory dirs what each task did or which iteration it belonged to. Switched plan.json IDs to `01_agent_geom`, `06_tool_judge`, etc., fix tasks to a constant `99_agent_fix`. Trajectory dirs now always suffixed `_iter<N>` (including iter 0) so same-task-across-iterations dirs sort adjacently when ls'd alphabetically.
Reference: `examples/articulated_drawer_cabinet/plan.json`, `topos/orchestrator/runner.py:_trajectory_dir_for`

## 2026-05-10 — Fix loop kicks LAM-style auto-iteration into Topos
Context: implemented `iter_policy.max_global_iters > 1` in the orchestrator. On a deliberately-failing chair (palette=as_authored, iter0 score 0.58), the runner generated a FIX1 agent prompt embedding the judge's per-criterion feedback + suggested_fixes, ran it, and re-ran all ToolTasks. Iter1 passed (0.66). The fix agent didn't just tweak parameters — it re-architected the cabinet from 3 cubes into a 5-panel hollow body, an explicit response to the "drawer interpenetrates" feedback. Insight: detailed judge feedback steers significant re-design, not just parameter tuning. Failure mode observed: agent over-corrected and made the drawer fully closed (hidden) at rest — pattern-level guidance (recipes) is the next lever.
Reference: `topos/orchestrator/runner.py:_build_fix_task`, trajectories/FIX1.iter1/

## 2026-05-10 — Separate modeling from rendering (infinigen-pattern)
Context: chair example v1 had the agent owning camera/render/lights inside `src/build.py`. Two consequences: judge saw only one viewpoint (could miss hidden defects), and the agent had to relearn camera setup every task. Borrowed infinigen's two-stage pattern (geometry script + framework renderer) and split the concerns. Agent script dropped from ~50 to ~30 lines; judge now evaluates 8 octant views; same score (0.98) at ~28s/run.
Reference: `topos/tools/render/wrapper.py`, `topos/tools/render.py`, ADR 0005

## 2026-05-10 — claude CLI envelope already carries usage + total_cost_usd
Context: looking for a way to track per-task token spend. The `--output-format=json` envelope from `claude -p` includes `total_cost_usd`, full `usage` (input/output/cache_creation/cache_read), `duration_ms`, `num_turns`, and `modelUsage` broken down by model. No separate billing API call needed; the data is already in every transcript.json we save. For the chair A1 run: $0.119 with 79% cache hit rate.
Reference: `trajectories/<task_id>/transcript.json` envelope keys


## 2026-05-11 — geometry_contracts skill saves a whole fix-loop iter, but doesn't cover everything
Context: three cabinet runs compared (cab_fix_loop_demo, cab_p123_verify, cab_diag_v1). The first two needed iter 1 + 99_agent_fix to pass; cab_diag_v1 (with the new topos_geometry_contracts skill) passed in iter 0 at score 0.693, $0.25 cheaper, ~150s faster. The build agent autonomously pasted all three contract blocks AND included the optional slide-axis suppression — strong evidence skills v2 (soft-hint + agent agency) works.

Two lessons the run also surfaced:
1. `fit_quality` is stuck at 0.50 across all three runs. Contracts say "Drawer fits cavity +6mm clearance OK on X/Z" but the judge still sees a perimeter gap on the closed-drawer view — the contract checks `cavity vs drawer.world_extents` but misses that the drawer FRONT FACE should overlap the cavity OPENING (a Z mismatch the slide-axis-Y suppression masks). Future contract v2: add a "front-face coverage" check for prismatic-joint children.
2. `part_identification` regressed from 0.85 to 0.60 in cab_diag_v1 because the design agent picked too-similar wood tones for Frame/Drawer/Handle. The design prompt has no "ensure parts are color-distinct enough for the judge to read in workbench shading" hint. Worth adding.
Reference: outputs/cab_diag_v1/trajectories/10_tool_judge_iter0/score.json + comparison across three preserved runs in outputs/

## 2026-05-11 — template+instances pattern validated end-to-end (stool_4leg_v1)
First full pipeline run with the new template+instances pattern (commits f90b683, 8b14ebd, 85a766d). NL prompt "four-legged wooden stool" produced:
- Spec agent: `parts=['Seat', 'Leg', 'Stretcher']` with `## TEMPLATE PARTS` section in intent.md listing 4 leg translations. Old behavior would have been `['Seat', 'Leg_0..3', 'Stretcher_0..1']` = 6 parts.
- Design agent: emitted `instances` field with 4 translations on Leg AND independently used the same pattern on Stretcher (2 instances) even though spec only flagged Leg. Emergent generalization of the instances pattern.
- Leg part agent: $0.74 / 163s for ONE canonical `build_leg()` with square mortise block + tapered turned shaft + ring grooves. Note in the file: "build.py applies each instance's translation after copying." Cost vs hypothetical 4-separate-leg enumeration: ~$3.0 → $0.74, **75% cost reduction** on the repeated cluster.
- Build.py: imports leg builder once, .copy() + apply translation per instance. Bbox validation + inter-part collision both correctly handle instance siblings.
- Final judge: PASS 0.795 (geometry_detail 0.80: "Legs are clearly turned/bobbin-style with multiple beads and grooves"). Reference vocabulary from refs/furniture.md (Shaker / mid-century turned leg) made it into the geometry.
- Leg's per-part judge passed on iter0; seat + stretcher needed fix-loop. **Per-part judge quality on the template+instances path was actually HIGHER than on the conventional single-instance path** — small sample, but suggests focused single-template agents produce more reliable output than multi-detail single-instance ones.

Known gap surfaced by this run: joints.yaml writer agent (and URDF export) is not yet instances-aware. 13_tool_export_urdf failed with: `link 'leg' references bpy object 'Leg' that the geometry script did not produce. produced: ['Leg_0', 'Leg_1', 'Leg_2', 'Leg_3', ...]`. Joints agent still emits singular `leg` link; build.py produces `Leg_0..3`. Phase 2 follow-up: update articulated/joints_writer.md to fan out per-instance links OR collapse fixed-joint instance siblings into a single mesh before URDF emit.

Reference: outputs/stool_4leg_v1/{run_report.json, src/design.json, src/parts/leg.py, src/build.py}


## 2026-06-21 — geometry_detail is RENDER-visibility-bound, not authoring-bound (the safe render fix A/B-regressed)

Context: an autonomous self-improvement loop ran the harness with Opus 4.8 across 12 object categories (bike, cabinet, lamp, plant, truck, guitar, chair, drill, fan, propeller, robot, jet). 11/12 passed the v2 judge; `geometry_detail` was the systematically weakest criterion (avg ~0.69, the limiting criterion for 6/10 objects).

What it is NOT: a part-authoring problem. Verified two ways. (1) A part-prompt "detail pass" directive (push ≥2 concrete features) was A/B-tested on truck/propeller/fan and was a **proven no-op** (truck 0.55→0.55, propeller 0.58→0.55, fan 0.60→0.62). (2) The part code already authors rich relief — propeller `blade.py` is a 136-line NACA airfoil; truck `cab.py` booleans 6 window recesses + bevels + door seams — yet still scores ~0.55-0.58. Part count has zero/inverse correlation with detail (robot has the MOST parts, 15, and the LOWEST score, 0.55).

Root cause: the judge-facing EEVEE render (`tools/blender_render/wrapper.py`) doesn't make authored relief legible — no ambient occlusion, no `view_settings.view_transform` set (so default AgX desaturates + flattens contrast), deliberately-soft 3-point lighting, and `_force_base_color` flattens each part to one Base Color (navy windows in a navy cab). Authored surface relief casts no legible shadow and washes out; only silhouette-changing detail (spokes, apertures, fluting) survives. Strongest single quantitative predictor of geometry_detail is **intra-object color/material diversity** (3-color objects 0.55-0.62 vs 5-9 color objects 0.72-0.80) — which independently reconfirms the 2026-05-11 note that the design step needs a "make parts color-distinct enough for the judge to read" hint.

BUT the obvious render fix did NOT pan out: a gated `_legibility_pass` (view_transform=Standard + eevee.use_gtao) was A/B'd by re-rendering+re-judging losers AND high-scorers. It did not lift losers above noise (truck +0.03, propeller +0.00) and it REGRESSED organic high-scorers (plant geometry_detail 0.78→0.72, overall 0.851→0.838) — AgX was actually tone-mapping broad-leaf/organic objects well. Reverted. Lesson: don't flip the global view transform blind; the real lever is per-feature material/color contrast (don't let `_force_base_color` homogenize multi-material parts; bias design.json toward color-distinct sub-features), an aesthetic change worth an A/B + human review, not an autonomous flip. HONEST CEILING: detail genuinely exists in the meshes; realistic render-side gain is moderate (~+0.08-0.15 on render-bound losers); current high scorers are near the procedural ceiling.

Reference: workflow wux06kh1c (4-trace investigation); A/B re-render+re-judge across truck/propeller/bike/plant; outputs/{truck,propeller,fan,robot}_verify/.


## 2026-06-21 — what actually moved quality + tokens this loop (8 commits)

Context: same autonomous loop. The high-leverage, verified wins (all on master, 6001064..47e3b83):
- **Judge honesty** — `doctor` told users ClaudeVisionCritic needs ANTHROPIC_API_KEY; it doesn't (it drives the claude CLI on subscription). And `runner._snapshot` recorded `all_judges[0]` (a lenient per-part judge) as the run verdict instead of the assembly judge, so run_report history mislabeled fails as passes (bike logged 0.81 while its assembly judge scored 0.605 FAIL). Fixed (6001064, 39d51f2).
- **Token/context (biggest surprise)** — every claude-CLI agent loaded all 31 builtin tool schemas + the user's ambient Gmail/Drive/Calendar MCP tools, ~7-9K tokens re-sent EVERY turn, because `--allowed-tools` only gates permission (doesn't strip schemas) and `--strict-mcp-config` only fired when mcp_servers was non-empty (always []). Passing `--tools` + always `--strict-mcp-config` dropped it to 2 tools, mcp [] (97c2770). Separately the "MANDATORY: read each skill" banner forced simple part agents to read all 4 hinted skills (~9.5K tokens re-sent every turn); softening to genuine opt-in cut per-part base context ~47% (36K→19K cache_creation) with NO detail regression — A/B'd on a cabinet (47e3b83).
- **Generality** — v2 identity rubric + per-criterion floors (fa14cb1), object_class-aware advisory collision WARNs so lattice/mechanism objects stop chasing bogus AABB overlaps (95e09a2), and adaptive part-agent + image-scaled judge timeouts (aa0a72f) took the always-failing bike from FAIL to PASS 0.858 and lifted the harness to 11/12 categories.
- **Judge fairness** — no_obvious_errors penalised correct intentional textures (a real engine-spinner spiral misread as "a photographic animal"); scoped to true render/geometry defects only (659c2e6).

Negative results worth remembering (don't rebuild these): a bladed-assembly geometry skill is unneeded (fan/propeller pass fine); build agents don't self-render (builder.md already forbids it); part-prompt detail directives don't move geometry_detail.

Reference: git log 6001064..47e3b83; memory/harness-improvement-loop.md
