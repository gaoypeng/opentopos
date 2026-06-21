"""DAG runner with iteration / auto-fix loop support.

Single iteration:
  Topo-sort the plan, run each task once, write per-task trajectory, halt
  downstream on dep failure.

Multi-iteration (``iter_policy.max_global_iters > 1``):
  After the first pass, if the stop condition is not met, the runner
  constructs a synthetic ``FIX<N>`` AgentTask whose goal is built from
  the failing judge's ``suggested_fixes`` and per-criterion feedback,
  runs it (it reads and edits files in src/), then re-runs every
  original ToolTask. The original AgentTasks are NOT re-run — they
  already wrote initial drafts.

Trajectory directories on iter N>0 are suffixed ``.iter<N>`` so per-iteration
history is preserved without overwrite.
"""

from __future__ import annotations

import concurrent.futures
import json
import re
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ..backends.base import AgentBackend
from ..tools import registry as tool_registry
from ..workspace import Workspace
from . import expand, fix_loop
from .plan_schema import Plan, topo_sort
from .tasks import AgentTask, SubgraphTask, Task, ToolTask


# Result types live in results.py — re-export for back-compat so callers
# (tests, external consumers) doing ``from topos.orchestrator.runner import
# TaskResult`` keep working without churn.
from .results import IterationSnapshot, RunReport, TaskResult  # noqa: F401,E402


def _default_agent_system_prompt() -> str:
    """The framework-level system prompt appended to every AgentTask. Sourced
    from ``topos/prompts/system/coding_agent_base.md`` so it's editable as a
    plain file alongside the example/domain prompts."""
    from ..prompts import load_text
    return load_text("system/coding_agent_base.md")


# Subdirs under the workspace that DON'T count as "agent work product" for
# the file-presence override below. Skills are framework-injected cache,
# artifacts/ is built by tools (not agents), trajectories/ is bookkeeping.
_AGENT_OUTPUT_EXCLUDE_PREFIXES = (
    ".topos_skills/",
    "artifacts/",
    "trajectories/",
    ".trajectory/",
    "scratch/",
    "prompts/",
)


def _real_work_products(files_modified: list[Path], workspace_root: Path) -> list[Path]:
    """Filter ``files_modified`` down to non-trivial files under
    ``workspace_root/src/`` that count as real agent output.

    Used by ``_run_agent``'s file-presence override: when the CLI envelope
    flags failure but the agent actually wrote source files, the override
    trusts the disk over the envelope. The exclude list keeps cache/
    bookkeeping/artifact files from triggering a false override.
    """
    out: list[Path] = []
    for f in files_modified:
        if not f.is_file():
            continue
        try:
            rel = f.relative_to(workspace_root).as_posix()
        except ValueError:
            continue
        if not rel.startswith("src/"):
            continue
        if any(rel.startswith(p) for p in _AGENT_OUTPUT_EXCLUDE_PREFIXES):
            continue
        if f.stat().st_size == 0:
            continue
        out.append(f)
    return out


def _missing_expected_outputs(expected: list[str], workspace_root: Path) -> list[str]:
    """Return the declared workspace-relative outputs that are absent or empty.

    Backs ``_run_agent``'s no-op guard: a CLI can report success having written
    nothing. Empty ``expected`` ⇒ ``[]`` (no check). A path that exists but is a
    zero-byte file counts as missing — a truncated/empty write is not a result.
    """
    missing: list[str] = []
    for rel in expected:
        p = workspace_root / rel
        try:
            if (not p.exists()) or (p.is_file() and p.stat().st_size == 0):
                missing.append(rel)
        except OSError:
            missing.append(rel)
    return missing


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def _parse_skill_frontmatter(skill_text: str) -> tuple[str, str | None]:
    """Pull description + when_to_use from a SKILL.md's YAML frontmatter.

    Returns (description, when_to_use_or_None). YAML parsing is intentionally
    simple — we only care about two top-level string fields.
    """
    m = _FRONTMATTER_RE.match(skill_text)
    if not m:
        return "(no frontmatter)", None
    fm = m.group(1)
    description = "(no description)"
    when_to_use: str | None = None
    for line in fm.splitlines():
        if line.startswith("description:"):
            description = line.split(":", 1)[1].strip()
        elif line.startswith("when_to_use:"):
            when_to_use = line.split(":", 1)[1].strip()
    return description, when_to_use


class Runner:
    def __init__(
        self,
        *,
        workspace: Workspace,
        plan: Plan,
        backends: dict[str, AgentBackend],
        resume: bool = False,
        max_parallel: int | None = None,
        event_sink: Callable[[dict], None] | None = None,
    ):
        self.ws = workspace
        self.plan = plan
        self.backends = backends
        self.resume = resume
        self._cost_accumulator = 0.0
        # Optional event sink — see topos/plugins/supabase_event_sink.py.
        # When set, the runner emits structured lifecycle events (run_started,
        # iter_started, task_started/completed/failed/skipped, run_finished).
        # Best-effort: a sink exception is logged but never propagated, so
        # live-viz infrastructure can never break a real run.
        self._event_sink = event_sink
        # Parallel dispatch capacity. Default comes from
        # ``orchestrator.max_parallel_tasks`` config (defaults to 4 — see
        # config_defaults.yaml). ``max_parallel=1`` recovers strict
        # sequential behaviour, useful when debugging an issue that
        # parallelism could hide. Agent tasks are I/O-bound (claude CLI
        # subprocess) so a thread pool is sufficient — no GIL pressure.
        if max_parallel is None:
            from .. import config as cfg
            effective = cfg.load_effective_config()
            max_parallel = (effective.get("orchestrator") or {}).get("max_parallel_tasks", 4)
        self.max_parallel = max(1, int(max_parallel))
        # Subgraph expansion bookkeeping (ADR-0008).
        # Maps subgraph_id → (set of child task ids, the SubgraphTask itself).
        # Holding the task instance directly lets _maybe_complete_subgraphs
        # read expansion_kind without depending on task_by_id, which in
        # iter > 0 may not contain the SubgraphTask (the combined task list
        # filters by AgentTask / ToolTask).
        # While the subgraph is in flight (children dispatching/running) its
        # own TaskResult is NOT yet in `results`, so downstream tasks that
        # depend on the subgraph_id stay blocked. Once every child has a
        # result, the subgraph itself gets a TaskResult with
        # success = all(child.success). Persisted across iters so fix-loop
        # re-runs of children correctly flip the subgraph's success state.
        self._subgraph_children: dict[str, tuple[set[str], "SubgraphTask"]] = {}
        # Task instances created by runtime SubgraphTask expansion, retained
        # across iters so the fix-loop's iter-N>0 combined task list can
        # re-dispatch them. Without this, dynamic children (per-part agents,
        # texture/judge/verify/render tools created by expand.py) live only
        # inside iter 0's _execute_tasks scope and disappear afterward —
        # leaving a runtime-failed verify_parts un-rerunable even after the
        # corresponding fix task lands. Observed on optimus_prime_v4.
        self._dynamic_tasks: list[Task] = []
        # Per-zz-tool snapshot of the FULL part list (set on first narrowing).
        # The batch verify_parts / render_parts ToolTasks are the same objects
        # across iters; the incremental-narrowing pass in _execute_tasks always
        # recomputes each iter's subset from this full snapshot, never from a
        # previously-narrowed list (else iter N would lose parts re-fixed in
        # iter N that weren't re-fixed in iter N-1).
        self._zz_full_parts: dict[str, list[str]] = {}
        tool_registry._ensure_default_tools_imported()

    # ---- event sink ----

    def _emit(self, event: dict) -> None:
        """Best-effort lifecycle emit. No-op when no sink is configured;
        exceptions are logged and swallowed so a misbehaving sink can never
        crash a real run.

        Uses ``getattr`` for the sink lookup so test fixtures that build
        Runner via ``__new__`` (skipping ``__init__``) don't AttributeError
        — same pattern as ``_subgraph_children`` in ``_execute_tasks``."""
        sink = getattr(self, "_event_sink", None)
        if sink is None:
            return
        event.setdefault("ts", time.time())
        try:
            sink(event)
        except Exception as exc:  # noqa: BLE001
            print(f"[runner] event_sink raised, dropping event: {exc!r}", flush=True)

    # ---- public ----

    def _load_prior_results(self) -> dict[str, TaskResult]:
        """If `--resume` is set and run_report.json exists from a previous run,
        load each task's prior result. Successful tasks will be reused as-is;
        failed/missing ones will be re-executed. This is opt-in and conservative:
        only tasks that recorded ``success=True`` are skipped on re-run.

        Fail loud (CLAUDE.md rule #12): a corrupted run_report.json is a real
        problem the user must see — silently re-running from scratch would
        re-bill them for completed work without explanation.
        """
        report_path = self.ws.root / "run_report.json"
        if not (self.resume and report_path.is_file()):
            return {}
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"--resume: failed to parse {report_path}: {e}. "
                "The prior run_report.json is corrupted. Delete it to start "
                "fresh, or fix manually."
            ) from e
        except OSError as e:
            raise RuntimeError(
                f"--resume: cannot read {report_path}: {e}"
            ) from e
        prior: dict[str, TaskResult] = {}
        for tid, r in (data.get("results") or {}).items():
            if not r.get("success"):
                continue
            prior[tid] = TaskResult(
                id=tid,
                kind=r.get("kind", "agent"),
                success=True,
                duration_s=float(r.get("duration_s") or 0.0),
                output=r.get("output") or {},
                cost_usd=float(r.get("cost_usd") or 0.0),
                usage=r.get("usage") or {},
                iteration=int(r.get("iteration") or 0),
                note=r.get("note"),
            )
        return prior

    def run(self) -> RunReport:
        original_tasks = topo_sort(self.plan.materialised())
        results: dict[str, TaskResult] = self._load_prior_results()
        if results:
            print(f"[runner] resume: reusing {len(results)} successful task result(s) from prior run_report.json")
            self._cost_accumulator = sum(r.cost_usd for r in results.values())
        history: list[IterationSnapshot] = []
        start = time.monotonic()

        max_iter = self.plan.iter_policy.max_global_iters
        stop_on = self.plan.iter_policy.stop_on

        self._emit({
            "type":    "run_started",
            "project": self.plan.project,
            "task_ids": [t.id for t in original_tasks],
        })

        # ---- iteration 0: run original plan ----
        # In resume mode, _execute_tasks skips tasks already in `results` with
        # success=True; the rest run normally.
        self._emit({"type": "iter_started", "iter": 0})
        iter_start = time.monotonic()
        iter_cost_start = self._cost_accumulator
        self._execute_tasks(original_tasks, results, iteration=0)
        history.append(self._snapshot(results, 0, time.monotonic() - iter_start,
                                       self._cost_accumulator - iter_cost_start))

        # ---- fix loop ----
        # On iter N>0 we run one fix-agent per failing judge (so a per-part
        # judge failure only re-runs that part's agent, not the whole pipeline)
        # plus all tool tasks so the renders/judges re-evaluate.
        # Early-stop policies (in priority order):
        #   1. All judges passed → stop_condition_met
        #   2. No failing judges → build_fix_tasks returns []
        #   3. iter_policy.min_improvement: no judge moved >= delta vs prior
        #      iter → likely stuck on sampling noise; stop wasting cost.
        # max_global_iters is the hard ceiling regardless.
        min_improvement = self.plan.iter_policy.min_improvement
        prev_scores = fix_loop.judge_scores_snapshot(results)
        # Track the whole-object assembly score across iters for the regression
        # early-stop (a fix iter that drops it made the deliverable worse).
        prev_asm = fix_loop.assembly_judge_score(results)
        iteration = 0
        while iteration < max_iter - 1:  # max_global_iters=1 → no fix iters
            if fix_loop.stop_condition_met(results, stop_on):
                break
            # Two parallel fix-task generators:
            #   - judge-driven: per-criterion vision feedback ("looks wrong")
            #   - runtime-driven: per-part build/import errors ("code didn't run")
            # Both are collected for the same iteration; the runtime ones are
            # listed FIRST so the iter trajectory makes the dependency order clear
            # (a broken part can't be visually critiqued — runtime fix has to
            # land before render/judge can produce anything).
            # Include dynamic children from SubgraphTask expansion so the
            # fix-loop can target their runtime failures (the regex matches
            # namespaced ids like ``02_subgraph_parts__03_agent_part_pelvis``).
            dispatchable_tasks = list(original_tasks) + list(self._dynamic_tasks)
            runtime_fix_tasks = fix_loop.build_runtime_fix_tasks(
                results, iteration + 1, original_tasks=dispatchable_tasks,
            )
            judge_fix_tasks = fix_loop.build_fix_tasks(
                results, iteration + 1, original_tasks=dispatchable_tasks,
            )
            fix_tasks = [*runtime_fix_tasks, *judge_fix_tasks]
            if not fix_tasks:
                break
            iteration += 1
            # Iter N>0 task set:
            #  - fix_tasks: targeted fix agents (per-part + assembly)
            #  - tool tasks: always re-run (renders + judges re-evaluate),
            #    INCLUDING dynamically-expanded subgraph children
            #    (verify_parts, render_parts, per-part judges) — they live
            #    in self._dynamic_tasks across iters
            #  - non-fix agent tasks: include them too. If they succeeded in
            #    iter 0, _execute_tasks's carry-forward logic skips them (no
            #    cost). If they were SKIPPED in iter 0 because their upstream
            #    failed, a later iter that fixes the upstream now lets them
            #    actually run.
            tool_tasks = [
                t for t in dispatchable_tasks if isinstance(t, ToolTask)
            ]
            non_fix_agents = [
                t for t in dispatchable_tasks
                if isinstance(t, AgentTask) and not t.id.startswith("99_agent_fix")
            ]
            # Dedup by id, keeping the FIRST occurrence. After the refactor that
            # makes fix tasks reuse the original part agent's id, a part with a
            # fix this iter appears twice in this list: once in `fix_tasks`
            # (with is_fix_rerun=True) and once in `non_fix_agents` (the original
            # plan.json entry). The fix version is listed first, so dedup keeps
            # it — that's the one that should run.
            combined = [*fix_tasks, *non_fix_agents, *tool_tasks]
            seen_ids: set[str] = set()
            deduped: list[Task] = []
            for t in combined:
                if t.id in seen_ids:
                    continue
                seen_ids.add(t.id)
                deduped.append(t)
            self._emit({"type": "iter_started", "iter": iteration})
            iter_start = time.monotonic()
            iter_cost_start = self._cost_accumulator
            self._execute_tasks(
                deduped,
                results, iteration=iteration,
            )
            history.append(self._snapshot(results, iteration,
                                          time.monotonic() - iter_start,
                                          self._cost_accumulator - iter_cost_start))
            # Regression early-stop (highest priority): if this fix iter made the
            # whole-object ASSEMBLY judge WORSE, halt immediately — don't keep
            # spending iters (and tokens) degrading the deliverable further. The
            # assembly judge is the verdict; a per-part judge improving while the
            # assembly regresses is net-negative (observed on the bicycle:
            # assembly 0.52 → 0.475 while a part judge nudged up, yet the loop
            # kept going). ``iter_improved`` below wouldn't catch this because it
            # counts ANY judge moving up.
            cur_asm = fix_loop.assembly_judge_score(results)
            if prev_asm is not None and cur_asm is not None and cur_asm < prev_asm - 1e-9:
                print(
                    f"[runner] regression early-stop after iter {iteration}: assembly "
                    f"judge dropped {prev_asm:.3f} -> {cur_asm:.3f}. Halting fix-loop "
                    f"(the fix made the whole object worse)."
                )
                break
            prev_asm = cur_asm
            # Cost-saturation early-stop: did anything move enough to justify
            # another iter? Compare each judge's score to the prior iter's.
            cur_scores = fix_loop.judge_scores_snapshot(results)
            if min_improvement > 0 and not fix_loop.iter_improved(prev_scores, cur_scores, min_improvement):
                shared = sorted(set(prev_scores) & set(cur_scores))
                deltas = ", ".join(
                    f"{k.split('_tool_judge')[-1].lstrip('_') or 'assembly'}={cur_scores[k]-prev_scores[k]:+.2f}"
                    for k in shared
                )
                print(
                    f"[runner] saturation early-stop after iter {iteration}: no judge "
                    f"improved >= {min_improvement} (deltas: {deltas}). "
                    f"Halting fix-loop to avoid noise-fueled iteration."
                )
                break
            prev_scores = cur_scores

        duration_s = time.monotonic() - start
        # The headline verdict is the whole-object ASSEMBLY judge — the
        # deliverable. A failing per-part shape critic on a minor part must not
        # report the run as failed when the assembled object passed (and the
        # fix-loop now stops on assembly-pass too, see build_fix_tasks). Fall
        # back to the all-judges signal only if no assembly judge ran (e.g. a
        # domain with per-part judges only).
        final_judge = fix_loop.assembly_judge_passed(results)
        if final_judge is None:
            final_judge = fix_loop.latest_judge_passed(results)
        run_ok = all(r.success for r in results.values()) and (final_judge is not False)

        report = RunReport(
            project=self.plan.project,
            success=run_ok,
            results=results,
            duration_s=duration_s,
            iteration_count=iteration + 1,
            history=history,
            total_cost_usd_all_iters=self._cost_accumulator,
            final_judge_passed=final_judge,
        )
        (self.ws.root / "run_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        self._emit({
            "type":               "run_finished",
            "success":            run_ok,
            "total_cost_usd":     self._cost_accumulator,
            "final_judge_passed": final_judge,
            "duration_s":         duration_s,
        })
        return report

    # ---- task execution ----

    def _narrow_batch_parts(
        self, tasks: list[Task], refixed_parts: set[str],
    ) -> list[tuple[str, list[str]]]:
        """On a fix iter, narrow the batch ``verify_parts`` / ``render_parts``
        ToolTasks' ``args["parts"]`` to just the re-fixed parts and return
        ``[(tool_id, narrowed_names), ...]`` for the ones actually narrowed.

        Why this is sound — the load-bearing invariant: every part NOT being
        re-touched this iter has a byte-identical ``src/parts/<name>.py``, so its
        prior per-part render PNGs and verify result are still valid; only the
        re-touched parts can have changed geometry / build status. On a 1-part
        fix to a 16/21-part object this cuts the Blender re-run from O(all parts)
        to O(re-touched).

        CRITICAL GUARD — the assembly fix agent breaks the invariant. An
        assembly/whole-object fix (an ``is_fix_rerun`` AgentTask whose id does
        NOT match the per-part pattern — i.e. ``03_agent_build`` / ``99_agent_fix``)
        is licensed by its prompt to edit ANY ``src/parts/<name>.py``, yet it
        never appears in ``refixed_parts`` (which is built only from
        ``_PART_AGENT_RE``-matching per-part fixes). When such a fix is present
        this iter we therefore CANNOT safely narrow — any part may have changed —
        so we restore the full list and bail. Narrowing only fires on iters whose
        only fixes are per-part agents, where the invariant genuinely holds.

        We narrow/restore from each tool's ORIGINAL full parts list (snapshotted
        once in ``self._zz_full_parts``), never from a previously-narrowed list,
        because these batch ToolTasks are the same objects across iters —
        narrowing a prior-narrowed list would drop parts re-fixed this iter that
        weren't re-fixed last iter. ``refixed_parts`` holds snake_case names (from
        ``_PART_AGENT_RE``); ``args["parts"]`` holds the PascalCase design names,
        so we join via ``_camel_to_snake`` — the same transform that built the
        part-agent ids, hence lossless.

        Empty intersection ⇒ nothing this tool covers was re-touched; restore the
        full list (the deterministic-skip path then carries the tool forward
        without re-running it at all).
        """
        from .expand import _camel_to_snake
        from .fix_loop import _PART_AGENT_RE

        # __new__-constructed Runners (test fixtures) may skip __init__; match the
        # lazy-init posture of _subgraph_children / _dynamic_tasks.
        if not hasattr(self, "_zz_full_parts"):
            self._zz_full_parts = {}

        # An assembly/whole-object fix can edit any part file → invariant void.
        assembly_fix_present = any(
            isinstance(t, AgentTask) and getattr(t, "is_fix_rerun", False)
            and not _PART_AGENT_RE.match(t.id)
            for t in tasks
        )

        narrowed_log: list[tuple[str, list[str]]] = []
        for t in tasks:
            if not (isinstance(t, ToolTask) and (
                t.id.endswith("__zz_tool_verify_parts")
                or t.id.endswith("__zz_tool_render_parts")
            )):
                continue
            full = self._zz_full_parts.setdefault(t.id, list(t.args.get("parts", [])))
            narrowed = (
                [] if assembly_fix_present
                else [p for p in full if _camel_to_snake(p) in refixed_parts]
            )
            if narrowed:
                t.args["parts"] = narrowed
                narrowed_log.append((t.id, narrowed))
            else:
                t.args["parts"] = list(full)
        return narrowed_log

    def _execute_tasks(
        self,
        tasks: list[Task],
        results: dict[str, TaskResult],
        iteration: int,
    ) -> None:
        """Run ``tasks`` with parallel dispatch (up to ``self.max_parallel``
        concurrent tasks), respecting deps. A task becomes eligible as
        soon as every entry in its ``deps`` is recorded in ``results``;
        an upstream failure immediately marks the dependent task failed
        without running it.

        Resume mode: skip any task already in ``results`` with
        success=True (loaded from prior run_report.json).

        ``max_parallel=1`` recovers strict sequential behaviour."""
        task_by_id: dict[str, Task] = {t.id: t for t in tasks}

        # Sticky-pass set: per-part judges that already passed in a prior iter
        # are "frozen" — we skip both (a) re-judging them (~$0.27 wasted) and
        # (b) the corresponding per-part fix agent. This eliminates the
        # judge-sampling-noise false-fail issue (e.g. cab_a4_perpart Handle
        # iter2 hallucinated "no handle visible", flipping 0.66 → 0.36).
        # Assembly judge is NOT frozen — its input changes when any fix runs.
        from .fix_loop import (
            frozen_parts as _frozen_parts,
            _PART_JUDGE_RE,
            _PART_AGENT_RE,
        )
        # Parts that have a fix re-run in THIS iter's task list. Their prior
        # per-part judge result is stale (graded the iter0 geometry, not the
        # iter1 geometry), so we must NOT sticky-pass them — the judge has
        # to re-evaluate the freshly-fixed code.
        refixed_parts: set[str] = set()
        for t in tasks:
            if isinstance(t, AgentTask) and getattr(t, "is_fix_rerun", False):
                m = _PART_AGENT_RE.match(t.id)
                if m:
                    refixed_parts.add(m.group("name"))
        frozen: set[str] = (_frozen_parts(results) - refixed_parts) if iteration > 0 else set()

        # Incremental verify/render narrowing (fix iters): re-run the batch
        # verify_parts / render_parts tools over just the re-fixed parts when it's
        # safe to (see _narrow_batch_parts — it self-guards against the
        # assembly-fix case and restores full lists otherwise, so call it every
        # fix iter, not only when refixed_parts is non-empty, to undo any prior
        # iter's narrowing on these persistent task objects).
        if iteration > 0:
            for tid, names in self._narrow_batch_parts(tasks, refixed_parts):
                print(f"  [narrow] {tid}: {len(names)}/"
                      f"{len(self._zz_full_parts[tid])} parts ({', '.join(names)})")

        # Resume + carry-forward pre-filter:
        #  - At iter 0: skip tasks that have prior success (resume mode).
        #  - At iter > 0: skip non-fix AgentTask that succeeded earlier in
        #    this run (no point re-paying for design / parts / joints once
        #    they've succeeded; the fix agents handle code fixes).
        #  - At iter > 0: also skip deterministic ToolTasks whose upstream
        #    didn't actually re-run this iter (their inputs are byte-identical,
        #    so their outputs are too — e.g. export_urdf, verify_parts).
        #    Stochastic tools (judge, generate_texture_image) always re-run.
        pending: set[str] = set()
        stale_failed: set[str] = set()

        def _carry_forward(prior: TaskResult, current_iter: int, reason: str) -> None:
            """Update a carried-forward result so the final RunReport doesn't
            lie about which iter the task belongs to. Stamps the current iter
            and records the originating iter in `note` for postmortem.

            On repeated carry-forward (same task surviving N>1 iters), we leave
            ``note`` alone after the first stamp so it doesn't chain into
            ``"sticky-pass from iter 2; sticky-pass from iter 1; ..."``. The
            true origin iter stays recorded on the first call."""
            if prior.note and "from iter" in prior.note:
                # Already stamped; just bump iteration to the current wave.
                prior.iteration = current_iter
                return
            origin = prior.iteration
            prior.iteration = current_iter
            tag = f"{reason} from iter {origin}"
            prior.note = tag if not prior.note else f"{tag}; {prior.note}"

        for task in tasks:
            # Fix-loop re-runs: must execute fresh regardless of any prior
            # success under the same task ID. Prior result was the original
            # iter0 work; the fix prompt for THIS iter asks the agent to
            # rewrite the part code to address judge feedback. The new
            # execution overwrites results[task.id] — downstream tasks see
            # the fix output (because their deps point at this task ID).
            if isinstance(task, AgentTask) and getattr(task, "is_fix_rerun", False):
                pending.add(task.id)
                # CRITICAL: clear the carry-forwarded prior result so the
                # dispatcher's dep check (which is "dep_id in results") returns
                # False for downstream tasks until the fix actually writes a
                # fresh result. Without this, a consumer like verify_parts
                # whose deps include this part id sees the stale iter0 success
                # entry, marks the dep satisfied, and dispatches IN PARALLEL
                # with the fix agent — racing on the part .py file. The fix
                # writes new code; the consumer either reads the old or a
                # half-written version. (Observed 2026-05-12 on the office
                # chair run: base_star iter2 fix and verify_parts ran in
                # parallel; verify happened to pass on the stale code; later
                # render_multiview loaded the freshly-written broken version
                # and crashed. Confirmed root cause via trajectory timestamps.)
                if task.id in results:
                    del results[task.id]
                # A re-running subgraph CHILD must also invalidate its parent
                # subgraph's stale completion. Consumers (build → render →
                # judge) gate on the subgraph id, not the child id, so without
                # this the iter-0 subgraph success entry stays in ``results``
                # and they dispatch on pre-fix geometry while the child is
                # still regenerating — the fix lands too late to reach the
                # final GLB/renders, and the judge grades stale images.
                # ``_maybe_complete_subgraphs`` re-sets the subgraph to success
                # once ALL children (including this re-run) finish again.
                # (Observed on turquoise_road_bicycle 2026-05-30: iter-2 build/
                # render/export finished before the slow stem/saddle fixes.)
                for _sg_id, (_child_ids, _sg) in getattr(self, "_subgraph_children", {}).items():
                    if task.id in _child_ids and _sg_id in results:
                        del results[_sg_id]
                continue
            prior = results.get(task.id)
            if prior is not None and prior.success:
                if iteration == 0:
                    print(f"[runner] resume: skip {task.id} (prior success, ${prior.cost_usd:.3f})")
                    # iter == 0 path: prior.iteration is already 0; nothing to fix.
                    continue
                # iter > 0: carry forward successful non-fix AgentTasks (else
                # we'd re-pay $ to re-write code that already works).
                if isinstance(task, AgentTask) and not task.id.startswith("99_agent_fix"):
                    origin = prior.iteration
                    _carry_forward(prior, iteration, "carry-forward")
                    print(f"[runner] carry-forward: skip {task.id} (already succeeded iter {origin})")
                    continue
                # iter > 0: deterministic ToolTasks whose deps didn't re-run
                # produce byte-identical output — skip. `pending` already has
                # every upstream that WILL execute this iter (we process tasks
                # in [fix_tasks, non_fix_agents, tool_tasks] order, and within
                # tool_tasks the plan is topo-ordered by id convention).
                if isinstance(task, ToolTask):
                    try:
                        spec = tool_registry.get(task.tool)
                    except KeyError:
                        spec = None
                    if spec is not None and spec.deterministic and not any(
                        d in pending for d in task.deps
                    ):
                        origin = prior.iteration
                        _carry_forward(prior, iteration, "deterministic-skip")
                        print(
                            f"[runner] deterministic-skip: skip {task.id} "
                            f"(inputs unchanged since iter {origin})"
                        )
                        continue
            # iter > 0: if there's a stale FAILED result, mark it for clearing
            # so downstream tasks that check `not results[d].success` don't
            # short-circuit and cascade-skip. Without this, a task that's
            # ABOUT to be re-attempted in this iter (e.g. build whose
            # upstream is now passing) gets cascade-skipped because the
            # dispatcher's dep check sees the stale failed result before
            # the retry has run.
            if iteration > 0 and prior is not None and not prior.success:
                stale_failed.add(task.id)
            if iteration > 0 and frozen:
                pj = _PART_JUDGE_RE.match(task.id)
                if pj and pj.group("name") in frozen and prior is not None:
                    # Prior pass result stays in `results[task.id]`; downstream
                    # treats it as success without re-evaluating. Stamp the
                    # current iter so the final report reflects the right wave.
                    origin = prior.iteration
                    _carry_forward(prior, iteration, "sticky-pass")
                    print(f"[runner] sticky-pass: skip {task.id} ({pj.group('name')} already passed iter {origin})")
                    continue
            pending.add(task.id)
        # Clear stale failed results for tasks we're about to re-attempt this
        # iter. Their downstream dep checks now see "not in results" and wait
        # for the retry to populate a fresh result.
        for tid in stale_failed:
            del results[tid]
            print(f"[runner] clear-stale: removed prior failed result for {tid}")
        if not pending:
            return

        def _execute_one(task: Task) -> TaskResult:
            """Run one task on the worker thread."""
            if isinstance(task, AgentTask):
                return self._run_agent(task, iteration=iteration)
            if isinstance(task, ToolTask):
                return self._run_tool(task, iteration=iteration)
            raise TypeError(f"unknown task type: {type(task)}")

        # Lazy-init so test fixtures that build Runner via ``__new__`` (skipping
        # __init__) don't AttributeError. Production runs go through __init__
        # which sets these attributes; tests get equivalent empties.
        if not hasattr(self, "_subgraph_children"):
            self._subgraph_children = {}
        if not hasattr(self, "_dynamic_tasks"):
            self._dynamic_tasks = []

        def _maybe_complete_subgraphs() -> None:
            """After any state change (child finishes, fix-loop re-attempts a
            child), refresh each subgraph's result from current children
            status. Runs idempotently — only mutates ``results`` when the
            computed success state actually differs from what's already
            recorded, so completed-and-unchanged subgraphs are cheap.

            We deliberately do NOT delete from ``_subgraph_children`` after
            the first computation: a fix iter can flip a child from failed
            to success, and the subgraph must re-evaluate (otherwise it
            stays stuck at the iter-0 status and downstream stays blocked
            forever — observed on optimus_prime_v4 where 11/11 children
            ended up success=True but the subgraph kept iter-0's False).
            """
            for sg_id, (child_ids, sg_task) in self._subgraph_children.items():
                if not all(cid in results for cid in child_ids):
                    continue
                child_results = [results[cid] for cid in child_ids]
                success = all(r.success for r in child_results)
                prev = results.get(sg_id)
                # No-op if status unchanged — avoid churning iteration stamps
                # and the snapshot file on every wait() return.
                if prev is not None and prev.success == success:
                    continue
                results[sg_id] = TaskResult(
                    id=sg_id,
                    kind="subgraph",
                    success=success,
                    duration_s=sum(r.duration_s for r in child_results),
                    output={
                        "children": sorted(child_ids),
                        "expansion_kind": sg_task.expansion_kind,
                        "failed_children": [r.id for r in child_results if not r.success],
                    },
                    cost_usd=sum(r.cost_usd for r in child_results),
                    iteration=iteration,
                    note=None if success else "one or more children failed",
                )

        def _expand_subgraph_inline(sg: SubgraphTask) -> None:
            """Run the registered expander, splice children into task_by_id +
            pending. Runs on the main thread; expansion is fast deterministic
            Python so we don't need to dispatch it to the pool."""
            children = expand.build_children(sg, workspace_root=self.ws.root)
            child_ids: set[str] = set()
            for child in children:
                if child.id in task_by_id:
                    # An id collision means a re-expansion landed the same
                    # children twice — keep the existing entry (carry-forward
                    # / dedup is the runner's general posture).
                    continue
                task_by_id[child.id] = child
                pending.add(child.id)
                child_ids.add(child.id)
                # Persist for cross-iter dispatch — see __init__ comment on
                # self._dynamic_tasks. Without this, runtime-failed children
                # (e.g. verify_parts) can't be re-attempted in fix iters.
                self._dynamic_tasks.append(child)
            self._subgraph_children[sg.id] = (child_ids, sg)
            self._write_expanded_plan_snapshot(task_by_id)
            print(
                f"[runner] expand: {sg.id} → {len(child_ids)} children "
                f"(kind={sg.expansion_kind!r})"
            )

        running: dict[concurrent.futures.Future, str] = {}
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            while pending or running:
                # Dispatch every currently-ready task up to capacity.
                # Sort for deterministic dispatch order across runs.
                for tid in sorted(pending):
                    if len(running) >= self.max_parallel:
                        break
                    t = task_by_id[tid]
                    if not all(d in results for d in t.deps):
                        continue
                    failed = [d for d in t.deps if not results[d].success]
                    if failed:
                        # Upstream failed; record this task as skipped
                        # without dispatching to the pool.
                        results[tid] = TaskResult(
                            id=tid, kind=t.kind, success=False, duration_s=0.0,
                            note=f"skipped: upstream failed: {failed}",
                            iteration=iteration,
                        )
                        self._emit({
                            "type": "task_skipped", "task_id": tid,
                            "kind": t.kind, "iter": iteration,
                            "reason": f"upstream failed: {failed}",
                        })
                        pending.discard(tid)
                        continue
                    # SubgraphTask: expand inline (main thread). Children get
                    # added to pending; the subgraph itself stays out of
                    # `results` until its children all resolve, which keeps
                    # downstream tasks correctly blocked on the subgraph id.
                    if isinstance(t, SubgraphTask):
                        _expand_subgraph_inline(t)
                        pending.discard(tid)
                        continue
                    self._emit({
                        "type":    "task_started",
                        "task_id": tid,
                        "kind":    t.kind,
                        "iter":    iteration,
                        "backend": getattr(t, "backend", None),
                        "deps":    list(t.deps),
                    })
                    fut = pool.submit(_execute_one, t)
                    running[fut] = tid
                    pending.discard(tid)

                if not running:
                    # Possible reasons we have pending tasks but no workers:
                    #   (a) subgraph just expanded; children are in pending
                    #       but their deps (the parent agent) are satisfied
                    #       and they'll dispatch next outer-while iter.
                    #   (b) a worker-free cascade just happened: e.g., a
                    #       verify failure synchronously skipped render +
                    #       judge_parts inside the for-loop above. Now ALL
                    #       of a subgraph's children are in `results` but
                    #       _maybe_complete_subgraphs hasn't fired yet
                    #       (it's only called after wait()). If we raise
                    #       deadlock here without first checking subgraph
                    #       completion, we never get to mark the subgraph
                    #       as done — strand all subgraph-downstream tasks.
                    #       Observed on optimus_prime_v4: pelvis.py used
                    #       Blender-4 'FAST' solver enum, verify_parts
                    #       failed, render/judge cascade-skipped, deadlock.
                    _maybe_complete_subgraphs()
                    if pending:
                        ready_now = any(
                            all(d in results for d in task_by_id[tid].deps)
                            for tid in pending
                        )
                        if ready_now:
                            continue
                        raise RuntimeError(
                            f"runner deadlock: {len(pending)} task(s) pending "
                            f"with unsatisfiable deps: {sorted(pending)}"
                        )
                    break

                # Block until at least one in-flight task completes.
                done, _not_done = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                for fut in done:
                    tid = running.pop(fut)
                    t = task_by_id[tid]
                    try:
                        result = fut.result()
                    except Exception as exc:  # noqa: BLE001
                        # Surface as a failed TaskResult so downstream tasks
                        # see "upstream failed" rather than crashing the run.
                        result = TaskResult(
                            id=tid, kind=t.kind, success=False, duration_s=0.0,
                            note=f"runner exception: {type(exc).__name__}: {exc}",
                            iteration=iteration,
                        )
                    results[tid] = result
                    self._cost_accumulator += result.cost_usd
                    self._emit({
                        "type":       "task_completed" if result.success else "task_failed",
                        "task_id":    tid,
                        "kind":       result.kind,
                        "iter":       iteration,
                        "duration_s": result.duration_s,
                        "cost_usd":   result.cost_usd,
                        "success":    result.success,
                        "note":       result.note,
                    })
                # After every batch of completions, check whether any in-flight
                # subgraph has all of its children resolved. If so, synthesize
                # the subgraph's own TaskResult so its downstream deps unblock.
                _maybe_complete_subgraphs()

    def _write_expanded_plan_snapshot(self, task_by_id: dict[str, Task]) -> None:
        """Persist the post-expansion DAG to ``plan.expanded.json`` so
        ``--resume`` and human inspection both see the dynamic shape, not
        just the static plan.json. The input ``plan.json`` is never mutated.
        """
        snapshot = {
            "project": self.plan.project,
            "tasks": [self._task_to_dict(t) for t in task_by_id.values()],
        }
        path = self.ws.root / "plan.expanded.json"
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    @staticmethod
    def _task_to_dict(t: Task) -> dict:
        if isinstance(t, AgentTask):
            return {
                "id": t.id, "kind": "agent", "backend": t.backend,
                "deps": list(t.deps),
                "allowed_tools": list(t.allowed_tools),
                "skills": list(t.skills),
                "timeout_s": t.timeout_s,
                "is_fix_rerun": t.is_fix_rerun,
                # `goal` may be a long rendered prompt; omit from snapshot to
                # keep plan.expanded.json scannable. The trajectory has it.
            }
        if isinstance(t, ToolTask):
            return {
                "id": t.id, "kind": "tool", "tool": t.tool,
                "args": dict(t.args), "deps": list(t.deps),
            }
        if isinstance(t, SubgraphTask):
            return {
                "id": t.id, "kind": "subgraph",
                "expand_from": t.expand_from,
                "expansion_kind": t.expansion_kind,
                "deps": list(t.deps), "timeout_s": t.timeout_s,
            }
        raise TypeError(f"cannot serialize task of type {type(t).__name__}")

    # ---- agent ----

    def _trajectory_dir_for(self, task_id: str, iteration: int) -> Path:
        # Always suffix `_iter<N>` so trajectory dirs are self-describing
        # and sort so that each task's runs across iterations are adjacent.
        return self.ws.trajectory_dir(f"{task_id}_iter{iteration}")

    def _run_agent(self, task: AgentTask, *, iteration: int) -> TaskResult:
        backend = self.backends.get(task.backend)
        if backend is None:
            return TaskResult(
                id=task.id, kind="agent", success=False, duration_s=0.0,
                note=f"no backend registered for {task.backend!r}",
                iteration=iteration,
            )
        trajectory = self._trajectory_dir_for(task.id, iteration)
        prompt = self._build_agent_prompt(task)
        (trajectory / "prompt.txt").write_text(prompt, encoding="utf-8")

        sys_append = _default_agent_system_prompt()
        if task.system_prompt_append:
            sys_append = sys_append + "\n\nAdditional rules for this task:\n" + task.system_prompt_append

        # No-op guard: a CLI can report success while doing nothing (gemini-cli
        # intermittently ends a turn with no tool calls). When the task declares
        # expected_outputs, we retry once if they're absent after a "successful"
        # run — the flake rarely repeats. duration_s spans all attempts (real
        # wall cost). max_noop_retries=0 when nothing is declared, so unguarded
        # tasks behave exactly as before.
        start = time.monotonic()
        max_noop_retries = 1 if task.expected_outputs else 0
        attempt = 0
        while True:
            result = backend.run(
                prompt=prompt,
                workspace=self.ws.root,
                allowed_tools=task.allowed_tools,
                mcp_servers=[],
                timeout_s=task.timeout_s,
                trajectory_dir=trajectory,
                system_prompt_append=sys_append,
            )

            # Work-product override: when the CLI envelope flags failure but
            # the agent actually wrote real source files, trust the disk over
            # the envelope. Observed 2026-05-13 on cab_gemini_pro_palace5_v2:
            # gemini-3.x preview occasionally emits a totally empty final-turn
            # response after all real work is done (4 KB part .py already
            # written via write_file tool). CLI labels that "Invalid stream",
            # topos's classify_exit picks up the (recovered) 429 in stderr as
            # well, returns exit_reason="quota", marks success=False, and the
            # one failed agent cascade-skips every downstream task — losing
            # ALL artifacts even though the work was complete on disk.
            #
            # The override is safe: src/ files are validated downstream by
            # verify_parts (Blender import test) and per-part judges, so a
            # stub/partial file still fails the run at the right gate. We
            # just stop letting the CLI's "I'm done" signal be the sole
            # arbiter of agent success.
            final_success = result.success
            override_note: str | None = None
            if not result.success:
                real_outputs = _real_work_products(result.files_modified, self.ws.root)
                if real_outputs:
                    final_success = True
                    override_note = (
                        f"file-presence override: {len(real_outputs)} real src/ "
                        f"file(s) written despite exit_reason={result.exit_reason}; "
                        f"trusting disk artifacts over CLI envelope. Downstream "
                        f"verify_parts/judges will validate work product."
                    )
                    print(f"[runner] {task.id}: {override_note}")

            missing = _missing_expected_outputs(task.expected_outputs, self.ws.root)
            if final_success and missing and attempt < max_noop_retries:
                attempt += 1
                print(
                    f"[runner] {task.id}: reported success but expected output(s) "
                    f"{missing} absent (likely a no-op turn) — retrying "
                    f"({attempt}/{max_noop_retries})."
                )
                continue
            break
        duration_s = time.monotonic() - start

        # Final verdict: declared outputs missing ⇒ failure, regardless of what
        # the envelope said. Turns a silent no-op (which used to crash at
        # subgraph expansion) into a loud, attributable task failure.
        noop_note: str | None = None
        if final_success and missing:
            final_success = False
            noop_note = (
                f"expected output(s) missing after {attempt + 1} attempt(s): "
                f"{missing} — agent reported success but wrote nothing (no-op turn)."
            )
            print(f"[runner] {task.id}: {noop_note}")

        (trajectory / "result.json").write_text(json.dumps({
            "success": final_success,
            "raw_envelope_success": result.success,    # ← keep the raw signal for postmortem
            "exit_reason": result.exit_reason,
            "override_note": override_note,
            "noop_note": noop_note,
            "noop_retries": attempt,
            "files_modified": [
                str(p.relative_to(self.ws.root)) if p.is_relative_to(self.ws.root) else str(p)
                for p in result.files_modified
            ],
            "duration_s": duration_s,
            "cost_usd": result.cost_usd,
            "usage": result.usage,
            "model_usage": result.model_usage,
        }, indent=2), encoding="utf-8")
        if final_success:
            note = None
        elif noop_note is not None:
            note = noop_note
        elif override_note is None:
            note = f"agent exit_reason={result.exit_reason}"
        else:
            note = f"agent exit_reason={result.exit_reason} (override considered but skipped)"
        return TaskResult(
            id=task.id, kind="agent", success=final_success, duration_s=duration_s,
            output={"files_modified": [str(p) for p in result.files_modified],
                    "exit_reason": result.exit_reason,
                    "override_note": override_note},
            note=note,
            cost_usd=result.cost_usd,
            usage=result.usage,
            model_usage=result.model_usage,
            iteration=iteration,
        )

    def _build_agent_prompt(self, task: AgentTask) -> str:
        parts = [
            f"# Task {task.id}",
            "",
            task.goal,
        ]
        if task.skills:
            self._materialize_skills_in_workspace(task.skills)
            parts.append("")
            parts.append("---")
            parts.append("")
            parts.append(
                "# Skills available for this task\n\n"
                "**MANDATORY: Read each matching skill BEFORE writing any code.**\n\n"
                "The skills below contain accumulated lessons from real run "
                "failures — transform_apply traps, scale arithmetic, bbox "
                "contract patterns, texture binding, joint origin math. "
                "Agents that skip skills consistently produce geometry that "
                "scores below 0.4 on the detail criterion. For every skill "
                "whose `when to use` matches your task, **use the Read tool "
                "on the listed path** and apply its patterns. Skip ONLY "
                "skills whose `when to use` clearly does not apply."
            )
            from ..skills import load_skill_md
            for skill_name in task.skills:
                try:
                    skill_text = load_skill_md(skill_name)
                except FileNotFoundError:
                    parts.append(f"\n- **{skill_name}** — (skill not found; skip)")
                    continue
                description, when_to_use = _parse_skill_frontmatter(skill_text)
                rel_path = f".topos_skills/{skill_name}.md"
                parts.append("")
                parts.append(f"- **{skill_name}**")
                parts.append(f"    description: {description}")
                if when_to_use:
                    parts.append(f"    when to use: {when_to_use}")
                parts.append(f"    full content: **Read `{rel_path}` BEFORE writing code**")
        if task.images:
            existing = [
                img for img in task.images
                if (self.ws.root / img).is_file()
            ]
            if existing:
                parts.append("")
                parts.append("---")
                parts.append("")
                parts.append(
                    "# Reference images\n\n"
                    "The following reference images are available in the workspace. "
                    "**Read each image using the Read tool** to see the visual reference "
                    "before writing code. Use these images to guide your geometry, "
                    "proportions, details, and style decisions."
                )
                for img in existing:
                    parts.append(f"- `{img}`")
        if task.deps:
            parts.append("")
            parts.append(f"Upstream tasks already complete: {', '.join(task.deps)}")
        return "\n".join(parts)

    def _materialize_skills_in_workspace(self, skill_names: list[str]) -> None:
        """Copy each requested skill's SKILL.md into ``workspace/.topos_skills/``
        so the agent can Read it on-demand. The agent has Read tool access to
        anything under the workspace via ``--add-dir``."""
        from ..skills import load_skill_md
        skill_dir = self.ws.root / ".topos_skills"
        skill_dir.mkdir(parents=True, exist_ok=True)
        for name in skill_names:
            try:
                content = load_skill_md(name)
            except FileNotFoundError:
                continue
            (skill_dir / f"{name}.md").write_text(content, encoding="utf-8")

    # ---- tool ----

    def _run_tool(
        self,
        task: ToolTask,
        *,
        iteration: int,
    ) -> TaskResult:
        trajectory = self._trajectory_dir_for(task.id, iteration)
        try:
            spec = tool_registry.get(task.tool)
        except KeyError as e:
            return TaskResult(
                id=task.id, kind="tool", success=False, duration_s=0.0,
                note=str(e), iteration=iteration,
            )
        args = dict(task.args)
        if "workspace" in spec.input_schema.get("properties", {}) and "workspace" not in args:
            args["workspace"] = str(self.ws.root)
        # Plumb task identity + its trajectory dir into metadata so downstream
        # critics (cli_critic) can stage per-call image dirs and write per-call
        # transcripts under the runner-allocated trajectory path. Without this,
        # parallel judge calls collide on shared workspace paths (_critic_images/,
        # .trajectory/transcript.json) — see ADR-0008 + the Optimus Prime
        # postmortem from 2026-05-18. Only injected when the tool's schema
        # accepts `metadata`, so non-critic tools are unaffected.
        if "metadata" in spec.input_schema.get("properties", {}):
            md = dict(args.get("metadata") or {})
            md.setdefault("_task_id", task.id)
            md.setdefault("_trajectory_dir", str(trajectory))
            args["metadata"] = md

        from .atif import write_tool_trajectory, _now_iso
        started_at = _now_iso()
        start = time.monotonic()
        try:
            output = spec.func(**args)
            ok = bool(output.get("success", True)) if isinstance(output, dict) else True
            err: str | None = None
        except Exception as exc:  # noqa: BLE001
            output = {"error": repr(exc)}
            ok = False
            err = repr(exc)
        duration_s = time.monotonic() - start
        finished_at = _now_iso()

        try:
            (trajectory / "output.json").write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
            if task.tool == "judge" and isinstance(output, dict) and "passed" in output:
                (trajectory / "score.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
        except OSError as io_err:
            import sys as _sys
            print(f"[runner] trajectory write failed for {task.id}: {io_err}", file=_sys.stderr)

        # ATIF-v1.7 trajectory: structured 2-step record of this tool execution
        # (one tool_call + one observation). Readable by Harbor's trajectory
        # viewer and any other ATIF-compliant tooling. Output and exception
        # branches both feed in — failed tool calls also get a trajectory so
        # diagnostics aren't lost.
        try:
            write_tool_trajectory(
                trajectory, task_id=task.id, iteration=iteration,
                tool_name=task.tool,
                arguments=args,
                output=output if isinstance(output, dict) else {"value": output},
                duration_s=duration_s, success=ok,
                started_at=started_at, finished_at=finished_at,
            )
        except Exception as atif_err:  # noqa: BLE001
            # ATIF emission is observational — a bug here must not fail the
            # actual task. Surface in stderr but keep the run going.
            import sys as _sys
            print(f"[runner] ATIF write failed for {task.id}: {atif_err}", file=_sys.stderr)

        out_dict = output if isinstance(output, dict) else {"value": output}
        cost_usd = float(out_dict.get("cost_usd") or 0.0)
        usage = out_dict.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}

        # Slim what we hand to TaskResult.output. The full ``output`` dict has
        # already been written to ``trajectory/output.json`` and to ATIF
        # trajectory.json above, so those four keys would be pure duplicates
        # of either the trajectory archive (stdout/stderr) or this very
        # TaskResult's top-level fields (cost_usd/usage).
        slim_output = {
            k: v for k, v in out_dict.items()
            if k not in ("stdout", "stderr", "cost_usd", "usage")
        }

        return TaskResult(
            id=task.id, kind="tool", success=ok, duration_s=duration_s,
            output=slim_output, note=err,
            cost_usd=cost_usd, usage=usage,
            iteration=iteration,
        )

    # ---- fix-loop helpers ----
    # The decision logic lives in ``topos/orchestrator/fix_loop.py`` (free
    # functions, no self state). Runner.run() and Runner._snapshot call
    # ``fix_loop.foo(results, ...)`` directly.

    def _snapshot(
        self,
        results: dict[str, TaskResult],
        iteration: int,
        duration_s: float,
        cost_usd: float,
    ) -> IterationSnapshot:
        # Snapshot reports the AUTHORITATIVE assembly (whole-object) judge, the
        # single judge that grades the deliverable. NOT all_judges[0]: in a
        # subgraph plan the per-part judges are inserted before the assembly
        # judge, so all_judges[0] is a lenient per-part judge — recording its
        # score made run_report history report a part score as the run verdict
        # (masking fails as passes). Fall back to the first judge only when the
        # assembly judge hasn't run (e.g. build failed before it). The run-level
        # pass/fail decision uses _latest_judge_passed (all judges); this is the
        # reported-numbers fix only.
        judge = fix_loop.assembly_judge_result(results)
        if judge is None:
            all_judges = fix_loop.all_judge_results(results)
            judge = all_judges[0] if all_judges else None
        passed = (judge.output.get("passed") if judge else None)
        score = (judge.output.get("overall_score") if judge else None)
        return IterationSnapshot(
            iteration=iteration,
            success=all(r.success for r in results.values()),
            judge_passed=passed,
            judge_score=score,
            duration_s=duration_s,
            cost_usd=cost_usd,
        )
