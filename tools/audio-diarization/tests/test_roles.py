import json
from types import SimpleNamespace

import pytest

from diarize_transcribe.alignment import Turn, Word
from diarize_transcribe.roles import guess_roles

anthropic = pytest.importorskip("anthropic")


def make_turns():
    return [
        Turn(0, 0, 1, [Word("Olá, bem-vindo à reunião.", 0, 1, 0)]),
        Turn(1, 1, 2, [Word("Obrigado, quero falar do orçamento.", 1, 2, 1)]),
        Turn(2, 2, 3, [Word("Hmm.", 2, 3, 2)]),
    ]


def fake_client(monkeypatch, response=None, exc=None):
    calls = {}

    class FakeMessages:
        def create(self, **kwargs):
            calls.update(kwargs)
            if exc:
                raise exc
            return response

    class FakeClient:
        def __init__(self, *a, **k):
            self.beta = SimpleNamespace(messages=FakeMessages())

    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    return calls


def response_with(payload, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=json.dumps(payload))],
    )


def test_maps_roles_by_1_based_speaker_number(monkeypatch):
    payload = {"speakers": [{"speaker_number": 1, "role": "Gestor"}, {"speaker_number": 2, "role": "Cliente"},
                            {"speaker_number": 9, "role": "ignored"}]}
    calls = fake_client(monkeypatch, response_with(payload))
    roles = guess_roles(make_turns(), context="reunião de vendas")
    assert roles == ["Gestor", "Cliente", "Speaker 3"]
    assert "reunião de vendas" in calls["messages"][0]["content"]
    assert "Speaker 2: Obrigado" in calls["messages"][0]["content"]
    assert calls["output_config"]["format"]["type"] == "json_schema"


def test_refusal_returns_none(monkeypatch):
    fake_client(monkeypatch, response_with({}, stop_reason="refusal"))
    assert guess_roles(make_turns()) is None


def test_api_error_returns_none(monkeypatch):
    fake_client(monkeypatch, exc=anthropic.APIConnectionError(request=SimpleNamespace()))
    assert guess_roles(make_turns()) is None


def test_empty_turns():
    assert guess_roles([]) is None
