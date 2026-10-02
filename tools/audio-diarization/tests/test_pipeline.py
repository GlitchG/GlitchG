"""ASR settings that work around a NeMo bug (no NeMo import needed)."""

from types import SimpleNamespace

from diarize_transcribe import pipeline


class FakeASR:
    def __init__(self):
        self.chunking = []
        self.attention = []

    def change_subsampling_conv_chunking_factor(self, factor):
        self.chunking.append(factor)

    def change_attention_model(self, *args):
        self.attention.append(args[0])

    def transcribe(self, paths, timestamps, verbose):
        word = {"word": "olá", "start": 0.1, "end": 0.4}
        return [SimpleNamespace(timestamp={"word": [word]})]


def _run(monkeypatch, duration):
    model = FakeASR()
    monkeypatch.setattr(pipeline, "load_asr", lambda name: model)
    words = pipeline.transcribe_words("x.wav", duration)
    return model, words


def test_chunking_never_set_to_minus_one_after_long_audio(monkeypatch):
    model, _ = _run(monkeypatch, duration=30 * 60)
    assert -1 not in model.chunking
    assert model.attention == ["rel_pos_local_attn", "rel_pos"]


def test_short_audio_uses_safe_chunking(monkeypatch):
    model, words = _run(monkeypatch, duration=134)
    assert model.chunking == [1]
    assert [w.text for w in words] == ["olá"]
