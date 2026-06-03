"""End-to-end test that the ``warnings`` field on tool outputs surfaces
``[*_WARN]`` lines as a structured list — independently of however much
stdout the subprocess produces. Uses a fake ``run_process`` so we don't
need Blender or a real subprocess.

Historical note: this used to also assert that ``stdout_tail`` (a
truncated view) did NOT contain the warnings, proving the dedicated
``warnings`` field was necessary. Tool outputs now carry FULL stdout
(no truncation), so that negative assertion no longer applies — but the
``warnings`` field is still load-bearing because agents shouldn't have
to grep 100KB+ of Blender output to find contract violations.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from topos.tools.registry import get
from topos.tools import export as _export_pkg  # noqa: F401  registers export_glb
from topos.tools import blender_render as _br_pkg  # noqa: F401  registers render*


class _FakeProc:
    """Stand-in for a finished subprocess result. ``run_process`` returns
    this; the tool builds its result dict from it."""
    def __init__(self, stdout: str = "", stderr: str = "",
                 returncode: int = 0, timed_out: bool = False):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timed_out = timed_out


def _stdout_with_warnings_and_post_spam(*warnings: str) -> str:
    """Compose a stdout buffer: warnings up front, then 8KB of trailing
    gltf INFO lines (~the real failure shape — agents would otherwise
    have to scan ~400 lines of noise to find the contract lines)."""
    head = "\n".join(warnings) + "\n"
    info_spam = "\n".join(f"INFO: Primitives created: cube.{i:03d}" for i in range(400))
    return head + info_spam


# --- export_glb -----------------------------------------------------------


def test_export_glb_surfaces_warnings_as_dedicated_field(tmp_path: Path):
    """build.py prints contract warnings, then the gltf exporter floods
    INFO lines. Agents read the structured ``warnings`` field rather than
    parsing the full stdout themselves."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/build.py").write_text("# placeholder\n")
    artifact = tmp_path / "artifacts" / "object.glb"
    artifact.parent.mkdir()
    artifact.write_bytes(b"GLB_BYTES")  # pretend export succeeded

    stdout = _stdout_with_warnings_and_post_spam(
        "[ATTACHMENT_WARN] torso_to_left_smokestack (LeftSmokestack→Torso): min gap 80.2mm",
        "[ATTACHMENT_WARN] pelvis_to_torso (Torso→Pelvis): min gap 21.9mm",
        "[COLLISION_WARN] LeftThigh <-> RightThigh: AABB overlap 5850 cm³",
    )
    fake_proc = _FakeProc(stdout=stdout)

    spec = get("export_glb")
    with (
        patch("topos.tools.export.glb.resolve_blender_binary", return_value="blender"),
        patch("topos.tools.export.glb.run_process", return_value=fake_proc),
    ):
        result = spec.func(
            workspace=str(tmp_path),
            script_relpath="src/build.py",
        )

    # The warnings field must be present + extracted from the full stdout.
    assert "warnings" in result, "export_glb result missing 'warnings' field"
    warnings = result["warnings"]
    assert len(warnings) == 3, f"expected 3 warnings, got {warnings!r}"
    assert any("torso_to_left_smokestack" in w for w in warnings)
    assert any("80.2mm" in w for w in warnings)
    assert any("COLLISION_WARN] LeftThigh" in w for w in warnings)

    # The full stdout must round-trip into the result dict — no truncation.
    assert result["stdout"] == stdout
    assert "ATTACHMENT_WARN" in result["stdout"]  # warnings present in raw stream too


# --- render_multiview -----------------------------------------------------


def test_render_multiview_surfaces_warnings(tmp_path: Path):
    """``build.py`` runs here too (the render wrapper imports it to assemble
    the scene before rendering). Same warning-extraction contract."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/build.py").write_text("# placeholder\n")

    stdout = _stdout_with_warnings_and_post_spam(
        "[HOLLOW_WARN] Frame: spec declares cavity but actual_vol = 99% of outer",
    )

    # _spawn_wrapper returns a 6-tuple
    # Shape: (stdout, stderr, exit_code, duration_s, artifacts, timed_out)
    fake_spawn_result = (stdout, "", 0, 1.5, [], False)

    spec = get("render_multiview")
    with patch(
        "topos.tools.blender_render.tool._spawn_wrapper",
        return_value=fake_spawn_result,
    ):
        result = spec.func(
            workspace=str(tmp_path),
            script_relpath="src/build.py",
        )

    assert result["warnings"] == [
        "[HOLLOW_WARN] Frame: spec declares cavity but actual_vol = 99% of outer"
    ]
    # Full stdout round-trips
    assert result["stdout"] == stdout


# --- render_multiview (warning surfacing) ---------------------------------


def test_render_multiview_also_surfaces_warnings(tmp_path: Path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/build.py").write_text("# x\n")

    stdout = _stdout_with_warnings_and_post_spam(
        "[FIT_WARN] Drawer: 12mm of unused clearance in cabinet cavity",
    )
    fake_spawn_result = (stdout, "", 0, 1.0, [], False)

    spec = get("render_multiview")
    with patch(
        "topos.tools.blender_render.tool._spawn_wrapper",
        return_value=fake_spawn_result,
    ):
        result = spec.func(
            workspace=str(tmp_path),
            script_relpath="src/build.py",
        )
    assert any("FIT_WARN] Drawer" in w for w in result["warnings"])


# --- empty-stdout fallback ------------------------------------------------


def test_no_warnings_gives_empty_list_not_missing(tmp_path: Path):
    """Schema advertises ``warnings`` as a property; when stdout has none,
    the field must be ``[]`` (not absent) so downstream code doesn't
    have to ``output.get('warnings') or []`` defensively."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src/build.py").write_text("# x\n")
    artifact = tmp_path / "artifacts" / "object.glb"
    artifact.parent.mkdir()
    artifact.write_bytes(b"GLB")
    fake_proc = _FakeProc(stdout="INFO: gltf done\n")

    spec = get("export_glb")
    with (
        patch("topos.tools.export.glb.resolve_blender_binary", return_value="blender"),
        patch("topos.tools.export.glb.run_process", return_value=fake_proc),
    ):
        result = spec.func(workspace=str(tmp_path), script_relpath="src/build.py")
    assert result["warnings"] == []
