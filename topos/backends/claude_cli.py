"""Claude CLI backend. Spawns ``claude -p ...`` as a subprocess and consumes
its JSON output.

Permission model: in headless execution we own the workspace, so we pass
``--permission-mode bypassPermissions``. The capability surface is still
controlled by ``--allowed-tools`` (which we always set) and by the MCP
server's own tool filter; passing an empty allow-list disables everything
including Bash/Read/Edit/Write.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .. import config as cfg
from .._fs_diff import new_or_modified, snapshot_mtimes
from ..process import run_process_with_watchdog
from .dialect import CLAUDE_STREAM
from ._rate_limit import TokenBucket, make_bucket_from_config
from ._retry import run_with_retries
from ._utils import assert_prompt_within_limit, classify_exit
from .base import AgentRunResult, AuthMode, McpServerConfig


def _envelope_is_error(envelope: dict | None) -> bool:
    """True if the claude --output-format=json envelope flags a failure
    (``is_error`` set, or ``subtype`` starts with ``error``). The CLI returns
    rc=0 on internal errors / quota exhaustion, so this is the only signal."""
    if not isinstance(envelope, dict):
        return False
    if envelope.get("is_error"):
        return True
    subtype = (envelope.get("subtype") or "").lower()
    return subtype.startswith("error")


def _parse_stream_json_final_result(stdout: str) -> dict | None:
    """Walk a stream-json buffer and return the final ``type: result``
    event — the per-turn analog of what the older ``--output-format json``
    mode wrote as a single envelope.

    The claude CLI's stream-json output ships in one of two shapes
    depending on version and pipe context:
      - JSONL: one event per line (``{...}\\n{...}\\n``)
      - JSON array: all events in a single top-level array (``[{...},{...}]``)
    Both are handled. We pick the LAST result-typed event (not the last
    item) in case trailing heartbeats land after the result envelope.

    Returns ``None`` when no result event is recoverable (e.g. the CLI
    crashed before emitting one). Caller should fall back to raw stdout.
    """
    if not stdout or not stdout.strip():
        return None
    # Shape 1: single JSON array containing all events.
    try:
        whole = json.loads(stdout)
        if isinstance(whole, list):
            events = whole
        elif isinstance(whole, dict):
            events = [whole]
        else:
            events = []
        last_result: dict | None = None
        for ev in events:
            if isinstance(ev, dict) and ev.get("type") == "result":
                last_result = ev
        if last_result is not None or events:
            return last_result
    except json.JSONDecodeError:
        pass
    # Shape 2: JSONL — one event per line.
    last_result = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "result":
            last_result = ev
    return last_result


def _stream_events(stdout: str) -> list[dict]:
    """Parse the stream-json buffer (JSONL or single JSON array) into a list of
    event dicts; malformed lines are skipped."""
    if not stdout or not stdout.strip():
        return []
    try:
        whole = json.loads(stdout)
        if isinstance(whole, list):
            return [e for e in whole if isinstance(e, dict)]
        if isinstance(whole, dict):
            return [whole]
    except json.JSONDecodeError:
        pass
    out: list[dict] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _fallback_usage(events: list[dict]) -> dict:
    """When there is no terminal ``result`` envelope (agent killed mid-turn),
    recover usage from the last ``assistant`` event so a killed run still
    reports tokens for post-mortem (cost stays 0 under subscription auth)."""
    for ev in reversed(events):
        if ev.get("type") == "assistant":
            usage = (ev.get("message") or {}).get("usage")
            if isinstance(usage, dict):
                return usage
    return {}


_SHARED_RATE_BUCKETS: dict[str, TokenBucket | None] = {}
_SHARED_RATE_LOCK = __import__("threading").Lock()


def _shared_bucket(rate_per_minute: float | None) -> TokenBucket | None:
    """One process-global bucket keyed by rate setting, so every
    ``ClaudeCLIBackend.from_config()`` invocation shares the same throttle.
    Without this, each agent task constructs its own bucket and the rate
    cap becomes per-task instead of per-process."""
    if not rate_per_minute or rate_per_minute <= 0:
        return None
    key = f"claude:{float(rate_per_minute):.4f}"
    with _SHARED_RATE_LOCK:
        b = _SHARED_RATE_BUCKETS.get(key)
        if b is None:
            b = make_bucket_from_config(rate_per_minute)
            _SHARED_RATE_BUCKETS[key] = b
        return b


@dataclass
class ClaudeCLIBackend:
    """Subscription mode uses the user's existing claude login.
    API-key mode requires ``ANTHROPIC_API_KEY`` in env at call time.

    Quota resilience (subscription rate limits + paid API quotas alike):

    - ``max_quota_retries``    — on ``exit_reason == "quota"``, sleep then
      retry up to N times (default 3 = up to 4 total attempts).
    - ``quota_retry_wait_s``   — initial backoff in seconds; doubled each
      retry (default 60 → 60/120/240, ceiling ~7 min).
    - ``rate_per_minute``      — token bucket cap on calls/min; None or 0
      disables. Set to ~6 for Pro plan to stay under ~50 msg / 5h.
    """

    name: str = "claude"
    auth_mode: AuthMode = "subscription"
    cli: str = "claude"
    model: str | None = None        # If set, passed as `--model <value>` to the CLI.
                                     # None falls back to the claude CLI's own default.
    extra_args: tuple[str, ...] = ()
    default_timeout_s: int = 600
    permission_mode: str = "bypassPermissions"
    max_quota_retries: int = 3
    quota_retry_wait_s: float = 60.0
    # A watchdog idle-kill (exit_reason=="timeout") is often a transient
    # transport/CLI stall on a single turn, not a wedged agent — retry it a
    # bounded number of times before giving up. Without this a one-off hang on
    # the root design task fails the whole run.
    max_timeout_retries: int = 1
    timeout_retry_wait_s: float = 10.0
    rate_per_minute: float | None = None
    # Watchdog: ``timeout_s`` is the SOFT deadline. Past it, kill only when
    # the agent has been *idle* (no workspace file mtime changes, no NEW
    # stream-json work events) for ``watchdog_idle_grace_s``. "Work events"
    # = ``"type":"assistant"`` / ``"type":"user"`` (tool result) / partial
    # ``"type":"stream_event"`` deltas (we run with --include-partial-messages
    # so an in-progress long turn keeps refreshing the idle timer instead of
    # looking dead). We explicitly EXCLUDE ``rate_limit_event`` and ``system``
    # so a CLI stuck emitting only quota heartbeats correctly idle-trips.
    # Reason: complex fix tasks (21-part assembly) legitimately run 15-25 min,
    # but a stuck agent stops streaming entirely. ``watchdog_hard_max_s`` is
    # an absolute ceiling regardless of activity — last-resort safety so a
    # runaway can't burn the whole 5h subscription window.
    watchdog_idle_grace_s: int = 300
    watchdog_hard_max_s: int = 3600

    @classmethod
    def from_config(cls, conf: dict | None = None) -> "ClaudeCLIBackend":
        conf = conf or {}
        effective = cfg.load_effective_config()
        claude_conf = (effective.get("backends") or {}).get("claude") or {}
        merged = {**claude_conf, **conf}
        rate_raw = merged.get("rate_per_minute")
        return cls(
            name="claude",
            auth_mode=merged.get("auth", "subscription"),
            cli=merged.get("cli", "claude"),
            model=merged.get("model") or None,
            extra_args=tuple(merged.get("extra_args") or ()),
            default_timeout_s=int(merged.get("timeout_s") or 600),
            permission_mode=merged.get("permission_mode", "bypassPermissions"),
            max_quota_retries=int(merged.get("max_quota_retries", 3)),
            quota_retry_wait_s=float(merged.get("quota_retry_wait_s", 60.0)),
            max_timeout_retries=int(merged.get("max_timeout_retries", 1)),
            timeout_retry_wait_s=float(merged.get("timeout_retry_wait_s", 10.0)),
            rate_per_minute=float(rate_raw) if rate_raw else None,
            watchdog_idle_grace_s=int(merged.get("watchdog_idle_grace_s", 300)),
            watchdog_hard_max_s=int(merged.get("watchdog_hard_max_s", 3600)),
        )

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        allowed_tools: list[str],
        mcp_servers: list[McpServerConfig],
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
        system_prompt_append: str | None = None,
        trajectory_dir: Path | None = None,
    ) -> AgentRunResult:
        """Run with quota-aware retry + optional rate-limit throttle.

        The subprocess invocation is in ``_run_once``; the shared
        ``run_with_retries`` owns the cross-attempt policy (token-bucket
        acquire, retry on quota / transient watchdog timeout, backoff)."""
        bucket = _shared_bucket(self.rate_per_minute)
        return run_with_retries(
            lambda: self._run_once(
                prompt=prompt,
                workspace=workspace,
                allowed_tools=allowed_tools,
                mcp_servers=mcp_servers,
                timeout_s=timeout_s,
                env=env,
                system_prompt_append=system_prompt_append,
                trajectory_dir=trajectory_dir,
            ),
            # quota = subscription/API cap or transient server throttle (backoff);
            # timeout = watchdog idle-kill, usually a transient single-turn stall.
            retryable={"quota": self.max_quota_retries, "timeout": self.max_timeout_retries},
            base_wait_s={"quota": self.quota_retry_wait_s, "timeout": self.timeout_retry_wait_s},
            before_each=(bucket.acquire if bucket is not None else None),
            label="claude_cli",
        )

    def _run_once(
        self,
        *,
        prompt: str,
        workspace: Path,
        allowed_tools: list[str],
        mcp_servers: list[McpServerConfig],
        timeout_s: int | None = None,
        env: dict[str, str] | None = None,
        system_prompt_append: str | None = None,
        trajectory_dir: Path | None = None,
    ) -> AgentRunResult:
        """Single subprocess call — no retry, no rate limiting."""
        workspace = workspace.resolve()
        if not workspace.is_dir():
            raise NotADirectoryError(f"workspace must exist: {workspace}")
        trajectory_dir = trajectory_dir or (workspace / ".trajectory")
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        assert_prompt_within_limit(prompt, self.name)

        cmd: list[str] = [
            self.cli,
            "-p", prompt,
            # stream-json emits one JSON event per line covering each turn
            # (system / assistant message / tool_use / tool_result /
            # rate_limit / final result). We persist the raw stream as
            # ``transcript.jsonl`` for per-turn analysis, then also extract
            # the trailing ``result`` event into ``transcript.json`` so
            # existing readers (cli_critic, spec agent) that expect the
            # single-envelope shape keep working.
            "--output-format", "stream-json",
            "--verbose",          # required by claude CLI for stream-json
            # Emit partial ``stream_event`` deltas as a turn generates, so the
            # watchdog can tell "model is actively producing a long turn" from
            # "model call hung" — the former keeps refreshing the idle timer.
            "--include-partial-messages",
            "--no-session-persistence",
            "--permission-mode", self.permission_mode,
            "--add-dir", str(workspace),
            # Topos agents are headless workers with self-contained prompts.
            # Skip user/project settings (which may contain SessionStart hooks
            # like superpowers that inject ~15K tokens of conflicting skill
            # instructions and cause agents to hang). Auth and core CLI
            # functionality are unaffected — only hooks/plugins are suppressed.
            "--setting-sources", "",
        ]

        if self.model:
            cmd.extend(["--model", self.model])

        if allowed_tools is not None:
            cmd.extend(["--allowed-tools", ",".join(allowed_tools)])
            # --allowed-tools only gates PERMISSION; the full builtin tool set's
            # schemas still load into context and are re-read every turn. --tools
            # restricts which builtin schemas load at all, so mirror allowed_tools
            # to strip the ~26 unused builtin schemas (Task/Cron/Workflow/Monitor/
            # NotebookEdit/ToolSearch/…) from every agent's cached prefix. Topos
            # agents only ever use a small builtin subset (Read/Edit/Write/Glob/
            # Bash/Web*); orchestrator capabilities run as ToolTasks, not as agent
            # tool-calls, so they never appear in allowed_tools.
            cmd.extend(["--tools", ",".join(allowed_tools)])

        if system_prompt_append:
            cmd.extend(["--append-system-prompt", system_prompt_append])

        # Always pin MCP to ONLY topos-provided servers (empty by default) — pass
        # --strict-mcp-config unconditionally so the CLI does NOT discover the
        # user's ambient ~/.claude.json MCP config, which otherwise leaks
        # Gmail/Drive/Calendar tool schemas + dead needs-auth handshakes into
        # every headless agent's context, re-read every turn. (Previously this
        # only fired when mcp_servers was non-empty, but the orchestrator always
        # passes [], so ambient MCP always leaked in.)
        mcp_config_path = trajectory_dir / "mcp_config.json"
        mcp_config_path.write_text(json.dumps(
            {"mcpServers": {s.name: s.to_claude_dict() for s in mcp_servers}},
            indent=2,
        ), encoding="utf-8")
        cmd.extend(["--mcp-config", str(mcp_config_path), "--strict-mcp-config"])

        cmd.extend(self.extra_args)

        # Auth sanity
        if self.auth_mode == "api_key" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ClaudeCLIBackend(auth_mode='api_key') requires ANTHROPIC_API_KEY in env"
            )

        before = snapshot_mtimes(workspace)
        start = time.monotonic()
        # Watchdog: ``timeout_s`` is the SOFT deadline. Past it, we only kill
        # if no NEW ``"type":"assistant"`` or ``"type":"user"`` event has landed
        # in stdout for ``idle_grace_s`` — those are stream-json's "real work"
        # events (model turn or tool result). ``rate_limit_event``/``system``
        # are deliberately excluded so a CLI emitting only quota heartbeats
        # while not making progress idle-trips correctly.
        # ``tool_pending_substrings`` suppresses idle-kill while the agent is
        # waiting on a tool: each ``"type":"tool_use"`` block (Bash, Edit,
        # MCP tool call, ...) opens a pending slot; the matching
        # ``"type":"tool_result"`` closes it. A long Bash that spawns a 10-min
        # Blender render produces no stream events between the two — without
        # this signal the watchdog would idle-trip while the tool is still
        # legitimately running. ``hard_max_s`` is still the absolute ceiling.
        # ``done_event_substring="\"type\":\"result\""`` lets the watchdog
        # log "agent reported terminal result before kill" if it kills
        # during the brief teardown window after success — a benign race
        # that would otherwise look like a wedge.
        proc = run_process_with_watchdog(
            cmd,
            cwd=workspace,
            env=env,
            soft_timeout_s=timeout_s or self.default_timeout_s,
            idle_grace_s=self.watchdog_idle_grace_s,
            hard_max_s=self.watchdog_hard_max_s,
            **CLAUDE_STREAM.watchdog_kwargs(),
        )
        duration_s = time.monotonic() - start
        files_modified = new_or_modified(workspace, before)

        # Persist the stream-json event log + extract the final result event
        # for backwards-compat with consumers that read transcript.json.
        # Normalize to JSONL for ``transcript.jsonl``, and DROP the partial
        # ``stream_event`` deltas (kept only live, for the watchdog) so
        # transcripts stay readable / small.
        transcript_jsonl_path = trajectory_dir / "transcript.jsonl"
        transcript_path = trajectory_dir / "transcript.json"
        events = _stream_events(proc.stdout)
        persisted = [e for e in events if e.get("type") != "stream_event"]
        transcript_jsonl_path.write_text(
            "".join(json.dumps(e, separators=(",", ":")) + "\n" for e in persisted),
            encoding="utf-8",
        )
        parsed = _parse_stream_json_final_result(proc.stdout)
        if parsed is not None:
            transcript_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        else:
            # No final result event recovered — likely an early crash. Save
            # the stdout verbatim so post-mortem can still see what came back.
            transcript_path.write_text(proc.stdout, encoding="utf-8")

        # stderr always goes to trajectory for postmortem
        (trajectory_dir / "stderr.log").write_text(proc.stderr, encoding="utf-8")

        exit_reason = classify_exit(
            proc.returncode, proc.timed_out,
            stderr=proc.stderr,
            stdout=proc.stdout,
            envelope_error=_envelope_is_error(parsed if isinstance(parsed, dict) else None),
            have_envelope=isinstance(parsed, dict),
        )

        # Envelope keys produced by claude --output-format=json include
        # total_cost_usd, usage, modelUsage, duration_ms, num_turns, etc.
        # Surface the spending data so the runner can aggregate.
        cost_usd = 0.0
        usage_dict: dict = {}
        model_usage: dict = {}
        if isinstance(parsed, dict):
            try:
                cost_usd = float(parsed.get("total_cost_usd") or 0.0)
            except (TypeError, ValueError):
                cost_usd = 0.0
            if isinstance(parsed.get("usage"), dict):
                usage_dict = parsed["usage"]
            if isinstance(parsed.get("modelUsage"), dict):
                model_usage = parsed["modelUsage"]
        else:
            # Killed mid-turn (no terminal result envelope): recover token usage
            # from the last assistant event so the run still reports tokens.
            usage_dict = _fallback_usage(events)

        return AgentRunResult(
            success=(exit_reason == "completed"),
            files_modified=files_modified,
            stdout=proc.stdout,
            stderr=proc.stderr,
            transcript_path=transcript_path,
            exit_reason=exit_reason,
            duration_s=duration_s,
            cost_usd=cost_usd,
            usage=usage_dict,
            model_usage=model_usage,
        )
