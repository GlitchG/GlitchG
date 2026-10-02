"""Batch queue in the web UI, with the NeMo pipeline mocked out."""

import zipfile
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
app = pytest.importorskip("app")

from diarize_transcribe.alignment import Segment, Turn, Word  # noqa: E402
from diarize_transcribe.pipeline import Result  # noqa: E402


def fake_run(path, num_speakers=None):
    if "broken" in str(path):
        raise RuntimeError("cannot decode audio")
    w = Word("olá", 0.0, 0.5, speaker=0)
    return Result(turns=[Turn(0, 0.0, 0.5, [w])], segments=[Segment(0.0, 0.5, 0)], words=[w], duration=60.0)


def test_queue_continues_after_a_failed_file(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "run", fake_run)
    files = [str(tmp_path / n) for n in ("a.m4a", "broken.m4a", "b.ogg")]

    updates = list(app.process_batch(files, None, "Eugenia", False, "txt", True))
    text, outputs, table = updates[-1]

    assert "✅" in table and "❌ RuntimeError: cannot decode audio" in table
    assert "===== a.m4a =====" in text and "===== b.ogg =====" in text
    assert "Eugenia: olá" in text
    zip_path = Path(outputs[0])
    assert zip_path.name == "transcripts.zip"
    assert sorted(zipfile.ZipFile(zip_path).namelist()) == ["a.txt", "b.txt"]
    # Progress is streamed: one update per file plus the final one.
    assert len(updates) == len(files) + 1


def test_same_name_files_do_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "run", fake_run)
    (tmp_path / "x").mkdir()
    (tmp_path / "y").mkdir()
    files = [str(tmp_path / "x" / "call.m4a"), str(tmp_path / "y" / "call.m4a")]
    *_, (_, outputs, _) = app.process_batch(files, None, "", False, "md", False)
    assert sorted(Path(p).name for p in outputs[1:]) == ["call (2).md", "call.md"]


def test_empty_queue_shows_a_message():
    with pytest.raises(gr.Error):
        next(app.process_batch([], None, "", False, "txt", True))
