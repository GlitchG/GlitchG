"""Render speaker turns as TXT, Markdown, SRT or JSON."""

from __future__ import annotations

import json
from typing import Sequence

from .alignment import Segment, Turn, speaker_label

FORMATS = ("txt", "md", "srt", "json")


def fmt_clock(seconds: float) -> str:
    """0:01:05 style timestamp for transcripts."""
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def fmt_srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3_600_000:02d}:{ms % 3_600_000 // 60_000:02d}:{ms % 60_000 // 1000:02d},{ms % 1000:03d}"


def to_txt(turns: list[Turn], roles: Sequence[str] | None = None, timestamps: bool = True) -> str:
    lines = []
    for t in turns:
        prefix = f"[{fmt_clock(t.start)}] " if timestamps else ""
        lines.append(f"{prefix}{speaker_label(t.speaker, roles)}: {t.text}")
    return "\n\n".join(lines) + "\n"


def to_markdown(turns: list[Turn], roles: Sequence[str] | None = None, timestamps: bool = True) -> str:
    lines = []
    for t in turns:
        ts = f" _({fmt_clock(t.start)})_" if timestamps else ""
        lines.append(f"**{speaker_label(t.speaker, roles)}**{ts}: {t.text}")
    return "\n\n".join(lines) + "\n"


def to_srt(turns: list[Turn], roles: Sequence[str] | None = None) -> str:
    blocks = []
    for i, t in enumerate(turns, 1):
        blocks.append(
            f"{i}\n{fmt_srt_time(t.start)} --> {fmt_srt_time(t.end)}\n"
            f"{speaker_label(t.speaker, roles)}: {t.text}\n"
        )
    return "\n".join(blocks)


def to_json(
    turns: list[Turn],
    roles: Sequence[str] | None = None,
    segments: list[Segment] | None = None,
) -> str:
    payload = {
        "speakers": sorted({speaker_label(t.speaker, roles) for t in turns}),
        "turns": [
            {
                "speaker": speaker_label(t.speaker, roles),
                "speaker_index": t.speaker,
                "start": round(t.start, 3),
                "end": round(t.end, 3),
                "text": t.text,
            }
            for t in turns
        ],
    }
    if segments is not None:
        payload["diarization_segments"] = [
            {"start": s.start, "end": s.end, "speaker": speaker_label(s.speaker, roles)} for s in segments
        ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def render(
    fmt: str,
    turns: list[Turn],
    roles: Sequence[str] | None = None,
    timestamps: bool = True,
    segments: list[Segment] | None = None,
) -> str:
    if fmt == "txt":
        return to_txt(turns, roles, timestamps)
    if fmt == "md":
        return to_markdown(turns, roles, timestamps)
    if fmt == "srt":
        return to_srt(turns, roles)
    if fmt == "json":
        return to_json(turns, roles, segments)
    raise ValueError(f"Unknown format {fmt!r}; choose one of {FORMATS}")
