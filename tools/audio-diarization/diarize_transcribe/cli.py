"""Command line: python -m diarize_transcribe.cli meeting.mp3 --roles "Manager,Client"."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .formatters import FORMATS, render
from .pipeline import DEFAULT_ASR_MODEL, DEFAULT_DIAR_MODEL, run


def parse_roles(value: str | None) -> list[str] | None:
    return [r.strip() for r in value.split(",")] if value else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audio -> transcript with speaker roles (NVIDIA Nemotron 3 Diarization).")
    p.add_argument("audio", nargs="+", help="Audio/video file(s): wav, mp3, m4a, ogg, mp4...")
    p.add_argument(
        "--roles",
        help='Comma-separated names in order of first appearance, e.g. "Interviewer,Candidate". '
        "Unnamed speakers stay 'Speaker N'.",
    )
    p.add_argument(
        "--auto-roles",
        action="store_true",
        help="Let Claude guess roles from the conversation (needs ANTHROPIC_API_KEY). Ignored if --roles is set.",
    )
    p.add_argument("--context", help='Optional description for --auto-roles, e.g. "sales call with a hotel owner"')
    p.add_argument(
        "--speakers",
        type=int,
        help="How many people talk. Extra speaker labels are merged by voice. Defaults to the number of --roles.",
    )
    p.add_argument("--no-merge", action="store_true", help="Don't merge speaker labels that have the same voice")
    p.add_argument("-f", "--format", choices=FORMATS, default="txt", help="Output format (default: txt)")
    p.add_argument("-o", "--out-dir", type=Path, help="Write <name>.<format> here instead of printing")
    p.add_argument("--no-timestamps", action="store_true", help="Hide [h:mm:ss] in txt/md output")
    p.add_argument("--max-pause", type=float, default=2.0, help="Start a new paragraph after this pause (s)")
    p.add_argument("--diar-model", default=DEFAULT_DIAR_MODEL)
    p.add_argument("--asr-model", default=DEFAULT_ASR_MODEL)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    roles = parse_roles(args.roles)

    for audio in args.audio:
        result = run(
            audio,
            diar_model=args.diar_model,
            asr_model=args.asr_model,
            max_pause=args.max_pause,
            num_speakers=args.speakers or (len(roles) if roles else None),
            merge_similar_voices=not args.no_merge,
        )
        file_roles = roles
        if not file_roles and args.auto_roles:
            from .roles import guess_roles

            file_roles = guess_roles(result.turns, args.context)
        text = render(args.format, result.turns, file_roles, timestamps=not args.no_timestamps, segments=result.segments)
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            out = args.out_dir / f"{Path(audio).stem}.{args.format}"
            out.write_text(text, encoding="utf-8")
            print(f"Saved {out}", file=sys.stderr)
        else:
            print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
