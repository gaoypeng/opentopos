"""Environment probe: Python, claude CLI, Blender, API key, MCP, config file presence.

Used by `topos doctor`. Each check returns a `CheckResult` with status
ok/warn/fail, a one-line summary, and optional hint. Blender autodetection
walks PATH and a small set of common locations.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import config as cfg

Status = Literal["ok", "warn", "fail"]


@dataclass
class CheckResult:
    name: str
    status: Status
    summary: str
    hint: str | None = None
    detected_value: str | None = None  # e.g., the discovered binary path


# ---------- individual checks ----------

def check_python() -> CheckResult:
    v = sys.version_info
    if v >= (3, 10):
        return CheckResult("python", "ok", f"Python {v.major}.{v.minor}.{v.micro}")
    return CheckResult(
        "python", "fail",
        f"Python {v.major}.{v.minor}.{v.micro} too old (need ≥3.10)",
    )


def check_claude_cli() -> CheckResult:
    path = shutil.which("claude")
    if not path:
        return CheckResult(
            "claude_cli", "fail", "claude CLI not on PATH",
            hint="Install from https://claude.ai/code; ensure `claude` is in PATH",
        )
    try:
        out = subprocess.run(
            [path, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=10
        )
        ver = (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr) else "(unknown version)"
    except (subprocess.SubprocessError, OSError) as e:
        return CheckResult(
            "claude_cli", "warn", f"claude found at {path} but --version failed: {e}",
            detected_value=path,
        )
    return CheckResult("claude_cli", "ok", f"claude: {path} ({ver})", detected_value=path)


_BLENDER_CANDIDATE_PATTERNS = [
    "/usr/bin/blender",
    "/usr/local/bin/blender",
    "/snap/bin/blender",
    "/opt/blender/blender",
    "/Applications/Blender.app/Contents/MacOS/Blender",
]


def _expand_home_blender_candidates() -> list[Path]:
    home = Path.home()
    found: list[Path] = []
    if (home / "bin" / "blender").is_file():
        found.append(home / "bin" / "blender")
    # blender-*/blender style installs under $HOME
    for child in home.glob("blender*"):
        candidate = child / "blender"
        if candidate.is_file():
            found.append(candidate)
    return found


def _vendored_blender_candidates() -> list[Path]:
    """Project-vendored Blender — preferred when present so a self-contained
    checkout pins its own Blender version. Probed under the repo root (parent
    of the ``topos`` package). Covers the plain ``./vendor/blender/blender`` /
    ``./blender/blender`` layouts plus a versioned-extract directory
    (e.g. ``./vendor/blender-5.0-linux-x64/blender``)."""
    repo_root = Path(__file__).resolve().parents[1]
    found: list[Path] = []
    for fixed in (
        repo_root / "vendor" / "blender" / "blender",
        repo_root / "blender" / "blender",
    ):
        if fixed.is_file():
            found.append(fixed)
    for parent in (repo_root / "vendor", repo_root):
        for child in sorted(parent.glob("blender-*")):
            candidate = child / "blender"
            if candidate.is_file():
                found.append(candidate)
    return found


def discover_blender() -> Path | None:
    # Project-vendored Blender wins: a self-contained checkout should use its
    # own pinned binary even when a system Blender is also on PATH.
    for p in _vendored_blender_candidates():
        return p
    on_path = shutil.which("blender")
    if on_path:
        return Path(on_path)
    for p in _BLENDER_CANDIDATE_PATTERNS:
        if Path(p).is_file():
            return Path(p)
    for p in _expand_home_blender_candidates():
        return p
    return None


def check_blender(effective_cfg: dict) -> CheckResult:
    configured = (effective_cfg.get("blender") or {}).get("binary")
    if configured:
        # Resolve ``~`` / project-relative values (e.g. ./vendor/blender/blender)
        # the same way the runtime does, so doctor validates what will actually run.
        from .tools._blender_subprocess import resolve_blender_path
        configured = resolve_blender_path(configured)  # ./vendor/... → absolute
        if Path(configured).is_file():
            try:
                out = subprocess.run(
                    [configured, "--version"], capture_output=True, text=True, encoding="utf-8", timeout=15
                )
                ver = (out.stdout or "").strip().splitlines()[0] if out.stdout else "(unknown)"
                return CheckResult(
                    "blender", "ok", f"Blender: {configured} ({ver})",
                    detected_value=configured,
                )
            except (subprocess.SubprocessError, OSError) as e:
                return CheckResult(
                    "blender", "warn",
                    f"configured binary {configured} did not run cleanly: {e}",
                    detected_value=configured,
                )
        return CheckResult(
            "blender", "fail",
            f"configured blender.binary={configured} does not exist",
            hint=f"Run: topos config set blender.binary <path>",
            detected_value=configured,
        )
    found = discover_blender()
    if found:
        return CheckResult(
            "blender", "warn",
            f"Blender not configured; found candidate at {found}",
            hint=f"Run: topos config set blender.binary {found}",
            detected_value=str(found),
        )
    return CheckResult(
        "blender", "fail",
        "Blender not configured and no candidate found in PATH or common locations",
        hint="Install Blender 4.x, then: topos config set blender.binary <path>",
    )


def check_anthropic_key(effective_cfg: dict) -> CheckResult:
    """Whether ANTHROPIC_API_KEY availability matches the configured auth mode.

    The key is required ONLY when ``backends.claude.auth == 'api_key'``. Under
    the default ``subscription`` auth, both the coding agent AND the Claude
    vision critic (``claude_vision``) drive the claude CLI, which authenticates
    via the user's subscription login — no key needed. (Earlier this check
    falsely claimed ClaudeVisionCritic would fail without the key; it won't —
    that critic is a thin façade over the claude CLI, not the HTTP API.)
    """
    auth = ((effective_cfg.get("backends") or {}).get("claude") or {}).get("auth", "subscription")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return CheckResult(
            "anthropic_api_key", "ok",
            "ANTHROPIC_API_KEY is set (covers api_key auth and any HTTP-API tools)",
        )
    if auth == "api_key":
        return CheckResult(
            "anthropic_api_key", "fail",
            "backends.claude.auth=api_key but ANTHROPIC_API_KEY is not set — the "
            "claude coding agent and claude_vision critic will fail.",
            hint=("Set the key, or switch to subscription auth: "
                  "topos config set backends.claude.auth subscription"),
        )
    return CheckResult(
        "anthropic_api_key", "ok",
        f"ANTHROPIC_API_KEY not set — not required (backends.claude.auth={auth}). "
        "The claude_vision critic judges via the same CLI/subscription path.",
    )


def check_mcp_importable() -> CheckResult:
    if importlib.util.find_spec("mcp"):
        return CheckResult("mcp_sdk", "ok", "mcp SDK importable")
    return CheckResult(
        "mcp_sdk", "warn",
        "mcp SDK not installed (only needed once tools are wired)",
        hint="pip install 'topos[mcp]'  (or: pip install mcp)",
    )


def check_image_gen_key(effective_cfg: dict) -> CheckResult:
    """Soft check: image_gen backend (default Gemini) needs an API key, but
    nothing today's pipeline requires breaks if it's missing — the texture
    layer is opt-in. So this is a `warn` at most, never a `fail`."""
    image_gen = effective_cfg.get("image_gen") or {}
    default = image_gen.get("default", "gemini")
    if default == "stub":
        return CheckResult("image_gen", "ok", "image_gen.default=stub (no API key needed)")
    if default != "gemini":
        return CheckResult("image_gen", "ok", f"image_gen.default={default} (no check implemented)")
    gconf = image_gen.get("gemini") or {}
    if gconf.get("api_key"):
        model = gconf.get("model", "gemini-3.1-flash-image-preview")
        return CheckResult(
            "image_gen", "ok",
            f"image_gen.gemini key configured (model={model})",
        )
    return CheckResult(
        "image_gen", "warn",
        "image_gen.gemini.api_key not set (texture generation will fail until set; flat materials still work)",
        hint=(
            "Get a key at https://aistudio.google.com/app/apikey, then run: "
            "topos config set image_gen.gemini.api_key <key>"
        ),
    )


def check_user_config() -> CheckResult:
    path = cfg.user_config_path()
    if path.is_file():
        return CheckResult("user_config", "ok", f"User config: {path}", detected_value=str(path))
    return CheckResult(
        "user_config", "warn",
        f"User config not present at {path}",
        hint="Run: topos config init",
    )


def check_repo_config() -> CheckResult:
    path = cfg.repo_config_path()
    if path:
        return CheckResult("repo_config", "ok", f"Repo override: {path}", detected_value=str(path))
    return CheckResult(
        "repo_config", "ok",
        "No repo-local override (this is fine; that file is optional)",
    )


# ---------- orchestrator ----------

def run_all() -> list[CheckResult]:
    effective = cfg.load_effective_config()
    return [
        check_python(),
        check_claude_cli(),
        check_blender(effective),
        check_anthropic_key(effective),
        check_image_gen_key(effective),
        check_mcp_importable(),
        check_user_config(),
        check_repo_config(),
    ]
