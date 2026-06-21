"""``check_anthropic_key`` must reflect the *configured* claude auth mode, not
make a blanket "ClaudeVisionCritic needs the key" claim.

Why this matters: the Claude vision critic (``claude_vision``) drives the
claude CLI, which authenticates via the user's subscription login — exactly
like the coding agent. So under the default ``auth: subscription`` it works
*without* ANTHROPIC_API_KEY. The key is required ONLY when
``backends.claude.auth == 'api_key'``. The old check warned unconditionally
and falsely told users the vision critic would fail; this test pins the
corrected, config-aware behavior so the false claim can't come back.
"""

from topos.doctor import check_anthropic_key


def test_subscription_auth_without_key_is_ok(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = check_anthropic_key({"backends": {"claude": {"auth": "subscription"}}})
    assert r.status == "ok"
    # The bug we are fixing: it must NOT claim the vision critic fails.
    assert "will fail" not in r.summary.lower()


def test_default_auth_is_subscription_so_no_key_is_ok(monkeypatch):
    # auth unspecified → defaults to subscription → key not required.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = check_anthropic_key({})
    assert r.status == "ok"


def test_api_key_auth_without_key_fails(monkeypatch):
    # This is the only genuinely-broken case: api_key auth with no key.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    r = check_anthropic_key({"backends": {"claude": {"auth": "api_key"}}})
    assert r.status == "fail"


def test_key_present_is_ok_regardless_of_auth(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    r = check_anthropic_key({"backends": {"claude": {"auth": "api_key"}}})
    assert r.status == "ok"
