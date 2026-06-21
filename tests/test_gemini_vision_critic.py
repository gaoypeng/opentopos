"""Unit tests for the GeminiVisionCritic.

Network calls are mocked — we don't hit the real Gemini API. The point is
to verify: payload construction, response parsing, JSON-extraction edge
cases, and registration in ``make_critic``."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from topos.agents.visual_critic.base import (
    Criterion,
    CriticInputs,
    Rubric,
    make_critic,
)
from topos.agents._json_extract import try_load as _try_load
from topos.agents.visual_critic.gemini_vision import (
    GeminiVisionCritic,
    _build_payload,
    _extract_text,
    _materialise,
)


def _make_rubric() -> Rubric:
    return Rubric(
        id="test_rubric",
        judge_backend="gemini_vision",
        pass_threshold=0.7,
        criteria=[
            Criterion(id="visible", prompt="Is anything visible?", weight=0.5),
            Criterion(id="clean", prompt="Is the render clean?", weight=0.5),
        ],
        description="test",
    )


def _make_test_png(path: Path) -> Path:
    """Write a 1×1 PNG so the file is a valid image-bytes blob."""
    # Minimal valid PNG (1×1 transparent)
    data = bytes.fromhex(
        "89504e470d0a1a0a0000000d4948445200000001000000010806000000"
        "1f15c4890000000d49444154789c63000100000005000100"
        "0d0a2db40000000049454e44ae426082"
    )
    path.write_bytes(data)
    return path


# --- Payload construction --------------------------------------------------


def test_build_payload_includes_text_and_image_parts(tmp_path: Path):
    img = _make_test_png(tmp_path / "view_00.png")
    payload = _build_payload("evaluate this render", [img])
    contents = payload["contents"]
    assert len(contents) == 1
    parts = contents[0]["parts"]
    # text part first
    assert parts[0] == {"text": "evaluate this render"}
    # then one inline_data part per image
    assert "inline_data" in parts[1]
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    decoded = base64.b64decode(parts[1]["inline_data"]["data"])
    assert decoded == img.read_bytes()


def test_build_payload_forces_json_response():
    payload = _build_payload("x", [])
    assert payload["generationConfig"]["response_mime_type"] == "application/json"
    assert "tools" not in payload, "default payload must not include grounding tool"


def test_build_payload_grounding_swaps_json_for_search_tool():
    """When grounding is on, the API rejects response_mime_type, so we
    drop it and rely on the prompt asking for JSON. The model self-decides
    whether to actually invoke google_search."""
    payload = _build_payload("x", [], use_google_search=True)
    assert payload["tools"] == [{"google_search": {}}]
    assert "response_mime_type" not in payload["generationConfig"], \
        "response_mime_type is mutually exclusive with google_search; must be dropped"


def test_build_payload_multiple_images(tmp_path: Path):
    imgs = [
        _make_test_png(tmp_path / "view_00.png"),
        _make_test_png(tmp_path / "view_01.jpg"),
    ]
    payload = _build_payload("two views", imgs)
    parts = payload["contents"][0]["parts"]
    # 1 text + 2 image parts
    assert len(parts) == 3
    assert parts[1]["inline_data"]["mime_type"] == "image/png"
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"


# --- Response parsing ------------------------------------------------------


def test_extract_text_pulls_first_candidate_text():
    envelope = {
        "candidates": [
            {"content": {"parts": [{"text": '{"score": 0.8}'}]}, "finishReason": "STOP"}
        ]
    }
    assert _extract_text(envelope) == '{"score": 0.8}'


def test_extract_text_raises_on_no_candidates():
    with pytest.raises(RuntimeError, match="no candidates"):
        _extract_text({"promptFeedback": "blocked"})


def test_extract_text_raises_on_finishreason_other_with_no_text():
    envelope = {
        "candidates": [
            {"content": {"parts": []}, "finishReason": "SAFETY"}
        ]
    }
    with pytest.raises(RuntimeError, match="no text part"):
        _extract_text(envelope)


def test_try_load_handles_raw_json():
    assert _try_load('{"a": 1}') == {"a": 1}


def test_try_load_handles_markdown_fenced_json():
    s = '```json\n{"a": 1}\n```'
    assert _try_load(s) == {"a": 1}


def test_try_load_returns_none_on_bad_text():
    assert _try_load("not json at all") is None


# --- Materialise ----------------------------------------------------------


def test_materialise_passes_when_overall_meets_threshold():
    rubric = _make_rubric()
    envelope = {
        "candidates": [{"content": {"parts": [{
            "text": json.dumps({
                "per_criterion": {
                    "visible": {"score": 0.9, "feedback": "yes"},
                    "clean":   {"score": 0.8, "feedback": "yes"},
                },
                "overall_score": 0.85,
                "passed": True,
                "suggested_fixes": [],
            })
        }]}}],
        "usageMetadata": {
            "promptTokenCount": 1234,
            "candidatesTokenCount": 56,
            "totalTokenCount": 1290,
        },
    }
    cr = _materialise(envelope, rubric, raw="raw text", duration_s=2.5, model="m")
    assert cr.passed is True
    assert cr.overall_score == 0.85
    assert "visible" in cr.per_criterion
    assert cr.usage["input_tokens"] == 1234
    assert cr.usage["output_tokens"] == 56
    assert cr.usage["duration_s"] == 2.5
    assert cr.usage["model"] == "m"
    assert cr.cost_usd == 0.0


def test_materialise_fails_when_below_threshold():
    rubric = _make_rubric()  # pass_threshold = 0.7
    envelope = {
        "candidates": [{"content": {"parts": [{
            "text": json.dumps({
                "per_criterion": {"visible": {"score": 0.2, "feedback": "no"}},
                "overall_score": 0.2,
                "passed": True,           # model claims pass but score < threshold
                "suggested_fixes": ["add lights"],
            })
        }]}}],
    }
    cr = _materialise(envelope, rubric, raw="x", duration_s=1.0, model="m")
    assert cr.passed is False
    assert cr.suggested_fixes == ["add lights"]


# --- Factory registration --------------------------------------------------


def test_make_critic_dispatches_to_gemini(monkeypatch):
    # Isolate from any user-config visual_critic.default override so this tests
    # the rubric→critic dispatch path, not the global override.
    import topos.config as _cfg
    monkeypatch.setattr(_cfg, "load_effective_config", lambda: {})
    rubric = _make_rubric()
    # We avoid hitting the network; from_config should succeed with no api_key
    # (the RuntimeError is raised in evaluate(), not in construction).
    critic = make_critic(rubric)
    assert isinstance(critic, GeminiVisionCritic)


def test_evaluate_raises_when_no_api_key(tmp_path: Path):
    img = _make_test_png(tmp_path / "v.png")
    critic = GeminiVisionCritic(api_key=None, model="gemini-3-pro")
    with pytest.raises(RuntimeError, match="API key"):
        critic.evaluate(CriticInputs(images=[img]), _make_rubric())


def test_evaluate_raises_on_empty_image_list():
    critic = GeminiVisionCritic(api_key="dummy")
    with pytest.raises(ValueError, match="no images supplied"):
        critic.evaluate(CriticInputs(images=[]), _make_rubric())


# --- End-to-end with mocked HTTP ------------------------------------------


def test_full_evaluate_with_mocked_http(tmp_path: Path):
    """Mock urlopen so we can exercise the full evaluate() pipeline without
    hitting Gemini. Verifies that a well-formed envelope flows through
    payload-build → HTTP → parse → materialise without crashes."""
    img = _make_test_png(tmp_path / "v.png")
    rubric = _make_rubric()

    fake_envelope = {
        "candidates": [{"content": {"parts": [{
            "text": json.dumps({
                "per_criterion": {
                    "visible": {"score": 0.9, "feedback": "ok"},
                    "clean":   {"score": 0.8, "feedback": "ok"},
                },
                "overall_score": 0.85,
                "passed": True,
                "suggested_fixes": [],
            })
        }]}}],
        "usageMetadata": {"promptTokenCount": 100, "candidatesTokenCount": 50, "totalTokenCount": 150},
    }

    class FakeResponse:
        def __init__(self, body: bytes):
            self.body = body
        def read(self):
            return self.body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen",
               return_value=FakeResponse(json.dumps(fake_envelope).encode())):
        critic = GeminiVisionCritic(api_key="fake-key", model="gemini-3-pro")
        cr = critic.evaluate(CriticInputs(images=[img]), rubric)

    assert cr.passed is True
    assert cr.overall_score == 0.85
    assert cr.per_criterion["visible"]["score"] == 0.9
    assert cr.usage["total_tokens"] == 150
