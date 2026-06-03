"""Pin the cross-tool subprocess success judgment.

Bug origin: Blender's ``--background --python script.py`` exits 0 even when
``script.py`` raises a Python exception. The previous ``ok = (returncode ==
0)`` check in render_multiview, export_glb, and export_urdf trusted this lie
and reported ``success=True`` despite zero artifacts and a python traceback
in stderr. Downstream tasks (assembly judge, fix-loop) consumed the false
success and crashed seconds later with confusing ``FileNotFoundError``.

The replacement detector in ``topos.tools._success`` requires:
  1. exit_code == 0
  2. not timed out
  3. no ``Traceback (most recent call last):`` line in stderr
  4. if the tool produces artifacts, at least one was written

Each layer catches a distinct failure mode; the AND keeps detection
sound (false negatives are worse than false positives — a false-success
silently propagates broken state to the rest of the pipeline)."""

from __future__ import annotations

from topos.tools._success import judge_subprocess_success, stderr_has_python_crash


# --- stderr crash detector ------------------------------------------------


def test_stderr_with_traceback_header_detected():
    """The canonical Python crash signature must trip the detector — this is
    what Blender emits when an agent's script raises."""
    s = (
        "Blender 5.0.1 (hash a3db93c5b259 built ...)\n"
        "Traceback (most recent call last):\n"
        "  File \"/.../src/build.py\", line 82, in <module>\n"
        "AssertionError: BaseStar footprint wrong\n"
        "\nBlender quit\n"
    )
    assert stderr_has_python_crash(s) is True


def test_stderr_with_word_traceback_in_prose_not_detected():
    """Prose mention of 'traceback' (e.g. log lines from the test infra) must
    NOT fire the detector — only the canonical header at start-of-line."""
    s = "INFO: retrying request; suppressing traceback for now\n"
    assert stderr_has_python_crash(s) is False


def test_empty_or_none_stderr():
    assert stderr_has_python_crash("") is False
    assert stderr_has_python_crash(None) is False


def test_traceback_anywhere_in_buffer_detected():
    """The traceback marker can appear deep in the buffer (after lots of
    other log output) — the detector must still find it."""
    s = "INFO: line 1\n" * 200 + "Traceback (most recent call last):\n"
    assert stderr_has_python_crash(s) is True


# --- judge_subprocess_success: each failure mode -------------------------


def test_success_when_all_signals_clean():
    """exit_code 0 + not timed_out + no crash + artifacts present → True."""
    assert judge_subprocess_success(
        returncode=0, timed_out=False, stderr="Blender 5.0.1\nBlender quit\n",
        artifacts=["a.png", "b.png"], expects_artifacts=True,
    ) is True


def test_fail_on_timeout_even_when_exit_zero():
    """A timeout signal trumps a clean exit — the process was killed."""
    assert judge_subprocess_success(
        returncode=0, timed_out=True, stderr="", artifacts=["a.png"],
    ) is False


def test_fail_on_nonzero_exit():
    assert judge_subprocess_success(
        returncode=2, timed_out=False, stderr="", artifacts=["a.png"],
    ) is False


def test_fail_on_stderr_traceback_with_clean_exit():
    """The original bug: exit 0 + traceback in stderr → must report failure.
    This is the scenario observed on the chair iter1 base_star crash."""
    s = (
        "Blender 5.0.1\n"
        "Traceback (most recent call last):\n"
        "  File \"src/parts/base_star.py\", line 177, in build_base_star\n"
        "AssertionError: BaseStar footprint wrong\n"
    )
    assert judge_subprocess_success(
        returncode=0, timed_out=False, stderr=s, artifacts=[], expects_artifacts=True,
    ) is False


def test_fail_on_empty_artifacts_when_expected():
    """exit clean, no crash, but the artifact-producing tool didn't write
    anything — that's a silent failure (e.g. script returned early before
    the export step). Caller declared expects_artifacts=True; must fail."""
    assert judge_subprocess_success(
        returncode=0, timed_out=False, stderr="OK\n", artifacts=[],
        expects_artifacts=True,
    ) is False


def test_success_with_empty_artifacts_when_not_expected():
    """Some tools don't produce file artifacts (their effect is elsewhere);
    they pass expects_artifacts=False. Empty artifact list is fine there."""
    assert judge_subprocess_success(
        returncode=0, timed_out=False, stderr="OK\n", artifacts=[],
        expects_artifacts=False,
    ) is True


def test_success_signal_priority_documented_via_traceback_with_artifacts():
    """If the script crashed AFTER producing some files, the artifact-count
    check would pass but the traceback check should still fail the call.
    The detector treats traceback as definitive — partial output isn't
    "successful" execution."""
    s = "Traceback (most recent call last):\n  AssertionError: late crash\n"
    assert judge_subprocess_success(
        returncode=0, timed_out=False, stderr=s, artifacts=["a.png", "b.png"],
        expects_artifacts=True,
    ) is False


# --- end-to-end: the chair iter1 regression scenario ---------------------


def test_chair_iter1_regression_render_multiview_no_longer_false_success(tmp_path):
    """Exact reproduction of the bug observed 2026-05-12 on the office chair
    run: render_multiview reports success=True with 0 artifacts and a python
    traceback in stderr. After the fix, success must be False."""
    from unittest.mock import patch
    from topos.tools.registry import get
    from topos.tools import blender_render as _br_pkg  # noqa: F401 register

    (tmp_path / "src").mkdir()
    (tmp_path / "src/build.py").write_text("# placeholder\n")

    # Reproduce the chair iter1 stderr verbatim shape: traceback header +
    # path to build.py + AssertionError. exit_code 0 (the Blender lie).
    stderr_with_crash = (
        'Traceback (most recent call last):\n'
        '  File "/lab/yipeng/topos/topos/tools/blender_render/wrapper.py", line 218, in _run_agent_script\n'
        '    runpy.run_path(path, run_name="__main__")\n'
        '  File "/lab/yipeng/topos/outputs/office_chair_v1/src/build.py", line 82, in <module>\n'
        '    obj = builder()\n'
        '  File "/lab/yipeng/topos/outputs/office_chair_v1/src/parts/base_star.py", line 177, in build_base_star\n'
        '    assert abs(width_x - target) < 0.010\n'
        'AssertionError: BaseStar footprint wrong: x=0.1200 y=0.1200 target=0.7200\n'
    )
    fake_spawn = (
        "Blender 5.0.1\n\nBlender quit\n",   # stdout (no INFO render-write lines)
        stderr_with_crash,                   # stderr (the traceback)
        0,                                   # exit_code 0 — the Blender lie
        1.3,                                 # duration_s
        [],                                  # artifacts: ZERO (the smoking gun)
        False,                               # timed_out
    )

    spec = get("render_multiview")
    with patch(
        "topos.tools.blender_render.tool._spawn_wrapper",
        return_value=fake_spawn,
    ):
        result = spec.func(
            workspace=str(tmp_path),
            script_relpath="src/build.py",
        )

    # The regression: result must report failure, not success.
    assert result["success"] is False, (
        "render_multiview with python traceback + zero artifacts must be False; "
        "this is the chair iter1 silent-failure scenario the fix is for"
    )
    # And the diagnostic info is preserved (so fix agent can see what crashed)
    assert "AssertionError" in result["stderr"]
    assert "base_star.py" in result["stderr"]


def test_export_glb_false_success_also_caught(tmp_path):
    """Same bug surface on export_glb — Blender exit 0 + traceback + no
    .glb file. Pre-fix, the artifact existence check caught this for GLB
    (out.is_file()) but the traceback check makes the diagnosis explicit."""
    from unittest.mock import patch
    from topos.tools.registry import get
    from topos.tools import export as _export_pkg  # noqa: F401

    (tmp_path / "src").mkdir()
    (tmp_path / "src/build.py").write_text("# placeholder\n")
    # The .glb does NOT exist on disk — script crashed before reaching export.

    class _FakeProc:
        def __init__(self):
            self.returncode = 0       # The lie
            self.timed_out = False
            self.stdout = "Blender 5.0.1\n\nBlender quit\n"
            self.stderr = (
                "Traceback (most recent call last):\n"
                "  File \"src/parts/base_star.py\", line 177, in build_base_star\n"
                "AssertionError: BaseStar footprint wrong\n"
            )

    spec = get("export_glb")
    with (
        patch("topos.tools.export.glb.resolve_blender_binary", return_value="blender"),
        patch("topos.tools.export.glb.run_process", return_value=_FakeProc()),
    ):
        result = spec.func(
            workspace=str(tmp_path),
            script_relpath="src/build.py",
        )
    assert result["success"] is False
    # File legitimately wasn't created
    assert result["glb_path"] == ""
    assert result["byte_size"] == 0
