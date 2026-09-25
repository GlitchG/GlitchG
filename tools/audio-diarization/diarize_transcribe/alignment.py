"""Pure-Python logic: parse diarization output, assign words to speakers, build turns.

No NeMo / torch imports here, so this module is fast to import and easy to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass
class Segment:
    """One speaker-activity interval from the diarizer."""

    start: float
    end: float
    speaker: int


@dataclass
class Word:
    text: str
    start: float
    end: float
    speaker: int | None = None


@dataclass
class Turn:
    """Consecutive words spoken by one speaker."""

    speaker: int
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words).strip()


def parse_segments(raw: Iterable[str | Sequence]) -> list[Segment]:
    """Parse NeMo Sortformer output.

    `SortformerEncLabelModel.diarize()` returns strings like ``"0.000 3.120 speaker_0"``
    (one list per file). Tuples/lists ``(start, end, speaker)`` are accepted too.
    """
    segments: list[Segment] = []
    for item in raw:
        if isinstance(item, str):
            parts = item.split()
            if len(parts) < 3:
                continue
            start, end, spk = parts[0], parts[1], parts[2]
        else:
            start, end, spk = item[0], item[1], item[2]
        spk_str = str(spk)
        spk_idx = int(spk_str.rsplit("_", 1)[-1]) if "_" in spk_str else int(spk_str)
        segments.append(Segment(float(start), float(end), spk_idx))
    segments.sort(key=lambda s: (s.start, s.end))
    return segments


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speakers(words: list[Word], segments: list[Segment]) -> list[Word]:
    """Give every word the speaker whose segments overlap it the most.

    Words that fall in a gap (no overlap at all) get the speaker of the nearest
    segment, so no text is lost. With overlapping speech the dominant speaker wins.
    """
    if not segments:
        for w in words:
            w.speaker = 0
        return words

    for w in words:
        per_speaker: dict[int, float] = {}
        for seg in segments:
            if seg.start > w.end:
                break
            ov = _overlap(w.start, w.end, seg.start, seg.end)
            if ov > 0:
                per_speaker[seg.speaker] = per_speaker.get(seg.speaker, 0.0) + ov
        if per_speaker:
            w.speaker = max(per_speaker, key=per_speaker.get)
        else:
            mid = (w.start + w.end) / 2
            nearest = min(segments, key=lambda s: min(abs(mid - s.start), abs(mid - s.end)))
            w.speaker = nearest.speaker
    return words


def build_turns(words: list[Word], max_pause: float = 2.0) -> list[Turn]:
    """Group consecutive same-speaker words into turns.

    A new turn also starts after a pause longer than ``max_pause`` seconds, which keeps
    long monologues readable.
    """
    turns: list[Turn] = []
    for w in words:
        if w.speaker is None:
            continue
        last = turns[-1] if turns else None
        if last and last.speaker == w.speaker and (w.start - last.end) <= max_pause:
            last.words.append(w)
            last.end = max(last.end, w.end)
        else:
            turns.append(Turn(speaker=w.speaker, start=w.start, end=w.end, words=[w]))
    return turns


def speaker_label(speaker: int, roles: Sequence[str] | None = None) -> str:
    """Map a speaker index to a display name.

    Sortformer numbers speakers in order of first appearance, so ``roles[0]`` is
    whoever speaks first (e.g. ``["Manager", "Client"]`` for a sales call).
    """
    if roles and speaker < len(roles) and roles[speaker].strip():
        return roles[speaker].strip()
    return f"Speaker {speaker + 1}"
