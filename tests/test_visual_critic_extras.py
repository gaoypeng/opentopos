"""Tests for GeminiVisionCritic + CliVisionCritic (no real network)."""
from __future__ import annotations
import base64, json
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from topos.agents.visual_critic.base import (Critic, CriticInputs, CriticResult, Criterion, Rubric, make_critic)
from topos.agents.visual_critic.gemini_vision import (GeminiVisionCritic, _build_payload, _materialise as gv_materialise)
from topos.agents.visual_critic.cli_critic import (CliVisionCritic, _extract_json as cli_extract_json, _materialise as cli_materialise)


def _png() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

def _rubric() -> Rubric:
    return Rubric(id="t", judge_backend="gemini_vision", pass_threshold=0.6,
                  criteria=[Criterion(id="a", prompt="p")])

def _critique() -> dict:
    return {"overall_score": 0.7, "passed": True,
            "per_criterion": {"a": {"score": 0.7, "feedback": "good"}},
            "suggested_fixes": []}


# ---------- GeminiVisionCritic ----------

def test_gemini_vision_implements_protocol():
    assert isinstance(GeminiVisionCritic(api_key="fake"), Critic)


def test_gemini_vision_factory_dispatch(monkeypatch):
    # Isolate from any user-config visual_critic.default override (the rubric's
    # own judge_backend must drive dispatch here).
    import topos.config as _cfg
    monkeypatch.setattr(_cfg, "load_effective_config", lambda: {})
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    r = _rubric()
    c = make_critic(r)
    assert isinstance(c, GeminiVisionCritic)


def test_gemini_payload_uses_inline_data(tmp_path: Path):
    img = tmp_path / "x.png"; img.write_bytes(_png())
    body = _build_payload("p", [img])
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "p"}
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert base64.b64decode(parts[1]["inline_data"]["data"]) == _png()
    assert body["generationConfig"]["response_mime_type"] == "application/json"


def test_gemini_materialise_pulls_text_part():
    envelope = {"candidates": [{"content": {"parts": [{"text": json.dumps(_critique())}]}}],
                "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150}}
    r = _rubric()
    res = gv_materialise(envelope, r, raw="x", duration_s=1.2, model="gemini-3-pro")
    assert res.passed is True and res.overall_score == 0.7
    assert res.usage["input_tokens"] == 100
    assert res.usage["output_tokens"] == 50
    assert res.usage["total_tokens"] == 150
    assert res.usage["duration_s"] == 1.2
    assert res.usage["model"] == "gemini-3-pro"


def test_gemini_evaluate_raises_without_key(tmp_path: Path, monkeypatch):
    """No env, no config-key fallback."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from topos import config as cfg
    monkeypatch.setattr(cfg, "load_effective_config",
                         lambda: {"visual_critic": {"gemini_vision": {}}, "image_gen": {"gemini": {}}})
    img = tmp_path / "x.png"; img.write_bytes(_png())
    c = GeminiVisionCritic(api_key=None)
    with pytest.raises(RuntimeError, match="needs an API key"):
        c.evaluate(CriticInputs(images=[img]), _rubric())


def test_gemini_evaluate_falls_back_to_image_gen_key(tmp_path: Path, monkeypatch):
    """Auth chain: image_gen.gemini.api_key works for vision too."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from topos import config as cfg
    monkeypatch.setattr(cfg, "load_effective_config",
                         lambda: {"visual_critic": {"gemini_vision": {}},
                                  "image_gen": {"gemini": {"api_key": "shared-google-key"}}})
    c = GeminiVisionCritic.from_config()
    assert c.api_key == "shared-google-key"


# ---------- CliVisionCritic JSON extraction ----------

def test_cli_extract_json_from_plain_text():
    text = "Here's my critique:\n\n" + json.dumps(_critique())
    out = cli_extract_json(text)
    assert out["overall_score"] == 0.7


def test_cli_extract_json_from_fenced_json():
    text = "```json\n" + json.dumps(_critique()) + "\n```"
    out = cli_extract_json(text)
    assert out["overall_score"] == 0.7


def test_cli_extract_json_from_gemini_envelope():
    """gemini -o json wraps the model's output in an envelope; the JSON we want
    is in `result` or `content` field. The extractor walks both."""
    inner = json.dumps(_critique())
    envelope = {"result": inner, "usage": {"tokens": 50}}
    out = cli_extract_json(json.dumps(envelope))
    assert out["overall_score"] == 0.7


def test_cli_extract_json_raises_on_garbage():
    with pytest.raises(ValueError, match="could not extract"):
        cli_extract_json("just plain text with no JSON anywhere")


def test_cli_materialise_basic():
    r = _rubric()
    out = cli_materialise(_critique(), r, raw="x", cost_usd=0.05,
                           usage={"total_tokens": 100})
    assert isinstance(out, CriticResult)
    assert out.passed is True and out.cost_usd == 0.05
    assert out.usage["total_tokens"] == 100


# ---------- factory dispatch for codex_cli / gemini_cli ----------

def test_codex_cli_critic_factory(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    r = Rubric(id="t", judge_backend="codex_cli", pass_threshold=0.6,
               criteria=[Criterion(id="a", prompt="p")])
    # Neutralize any global visual_critic.default override so the test exercises
    # rubric-driven routing, not whatever the local machine's config pins.
    with patch("topos.config.load_effective_config", return_value={"visual_critic": {}}):
        c = make_critic(r)
    assert isinstance(c, CliVisionCritic)
    assert c.label == "codex"


def test_gemini_cli_critic_factory(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake")
    r = Rubric(id="t", judge_backend="gemini_cli", pass_threshold=0.6,
               criteria=[Criterion(id="a", prompt="p")])
    with patch("topos.config.load_effective_config", return_value={"visual_critic": {}}):
        c = make_critic(r)
    assert isinstance(c, CliVisionCritic)
    assert c.label == "gemini"
