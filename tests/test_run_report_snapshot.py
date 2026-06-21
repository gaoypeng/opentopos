"""The run-history snapshot must record the AUTHORITATIVE assembly (whole-object)
judge, not ``all_judges[0]``.

Why this matters: in a subgraph plan the per-part judges are inserted *before*
the assembly judge, so ``all_judges[0]`` is a lenient per-part judge. Recording
its score as the run verdict made ``run_report.json`` history report (e.g.) a
0.85 part score for a run whose whole-object judge actually scored 0.55 FAIL —
masking failures as passes and corrupting every cross-run analysis. This pins
the fix: the snapshot follows ``fix_loop.assembly_judge_result()``.
"""

from collections import OrderedDict

from topos.orchestrator.results import TaskResult
from topos.orchestrator.runner import Runner


def _judge(task_id: str, passed: bool, score: float) -> TaskResult:
    return TaskResult(
        id=task_id, kind="tool", success=True, duration_s=1.0,
        output={"passed": passed, "overall_score": score},
    )


def test_snapshot_uses_assembly_judge_not_first_part_judge():
    # Insertion order: a lenient PART judge first, then the real assembly judge.
    results = OrderedDict()
    results["06_tool_judge_part_frame"] = _judge("06_tool_judge_part_frame", True, 0.85)
    results["08_tool_judge"] = _judge("08_tool_judge", False, 0.55)

    runner = object.__new__(Runner)  # _snapshot uses no other instance state
    snap = runner._snapshot(results, iteration=0, duration_s=1.0, cost_usd=0.0)

    assert snap.judge_score == 0.55, "history must report the assembly judge, not the part judge"
    assert snap.judge_passed is False


def test_snapshot_falls_back_to_first_judge_when_no_assembly_judge():
    # Build failed before the assembly judge ran: still record what we have.
    results = OrderedDict()
    results["06_tool_judge_part_frame"] = _judge("06_tool_judge_part_frame", True, 0.85)

    runner = object.__new__(Runner)
    snap = runner._snapshot(results, iteration=0, duration_s=1.0, cost_usd=0.0)
    assert snap.judge_score == 0.85
