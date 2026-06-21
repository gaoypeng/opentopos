"""Per-criterion ``min_floor`` is a hard gate that blocks a pass even when the
weighted total clears the threshold.

Why it exists: the weighted average let a "box-stack" object pass on easy
criteria (framing, no-errors) while a load-bearing dimension — geometry_detail
or identity — scored badly. The floor makes those dimensions un-compensatable,
so the fix-loop is forced to address the thing that actually makes the object
low-quality. These tests pin that gate.
"""

from topos.agents.visual_critic.base import Criterion, Rubric, load_rubric
from topos.agents.visual_critic.critic_utils import materialise_score


def _parsed(overall, scores):
    return {
        "overall_score": overall,
        "passed": True,
        "per_criterion": {k: {"score": v, "feedback": "f"} for k, v in scores.items()},
        "suggested_fixes": [],
    }


def test_floor_blocks_pass_even_above_threshold():
    rubric = Rubric(
        id="t", judge_backend="x", pass_threshold=0.5,
        criteria=[
            Criterion(id="identity", prompt="p", weight=0.5, min_floor=0.45),
            Criterion(id="easy", prompt="p", weight=0.5),
        ],
    )
    # Weighted overall 0.70 clears 0.5, but identity is below its 0.45 floor.
    passed, overall, _pc, fixes = materialise_score(
        _parsed(0.70, {"identity": 0.30, "easy": 1.0}), rubric
    )
    assert passed is False
    assert any("floor" in f.lower() for f in fixes), "must surface the blocking floor in fixes"


def test_floor_satisfied_allows_pass():
    rubric = Rubric(
        id="t", judge_backend="x", pass_threshold=0.5,
        criteria=[Criterion(id="identity", prompt="p", weight=1.0, min_floor=0.45)],
    )
    passed, *_ = materialise_score(_parsed(0.70, {"identity": 0.60}), rubric)
    assert passed is True


def test_no_floor_behaves_as_before():
    rubric = Rubric(
        id="t", judge_backend="x", pass_threshold=0.5,
        criteria=[Criterion(id="a", prompt="p", weight=1.0)],
    )
    passed, *_ = materialise_score(_parsed(0.70, {"a": 0.30}), rubric)
    assert passed is True  # no floor → weighted total alone decides


def test_shipped_v2_rubric_has_floors_and_normalised_weights():
    r = load_rubric("articulated_object_v2")
    by_id = {c.id: c for c in r.criteria}
    assert by_id["identity_recognizability"].min_floor == 0.45
    assert by_id["geometry_detail"].min_floor == 0.45
    assert abs(sum(c.weight for c in r.criteria) - 1.0) < 1e-6
