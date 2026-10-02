"""Web UI: upload or record audio, get a transcript with speaker roles.

Run:  python app.py            (opens http://127.0.0.1:7860)
      python app.py --share    (public gradio.live link, handy on Colab)

Two tabs: "One file" (upload or record) and "Several files" (a queue: files are
transcribed one after another and you get one transcript per file plus a ZIP).
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import zipfile
from pathlib import Path

import gradio as gr

from diarize_transcribe.cli import parse_roles
from diarize_transcribe.formatters import render
from diarize_transcribe.pipeline import run
from diarize_transcribe.roles import guess_roles

log = logging.getLogger("whosaid")


def copyable_textbox(**kwargs) -> gr.Textbox:
    """Textbox with a copy button; the argument name changed in Gradio 6."""
    try:
        return gr.Textbox(buttons=["copy"], **kwargs)
    except TypeError:
        return gr.Textbox(show_copy_button=True, **kwargs)


def _file_path(f) -> str:
    """gr.File gives plain paths in Gradio 4+, objects with .name in older versions."""
    return f if isinstance(f, str) else getattr(f, "path", None) or f.name


def transcribe_one(
    audio_path: str,
    out_dir: Path,
    n_speakers: float | None,
    roles_text: str,
    auto_roles: bool,
    fmt: str,
    timestamps: bool,
) -> tuple[str, Path, str]:
    """Transcribe one file and save it into ``out_dir``. Returns (text, saved file, summary)."""
    roles = parse_roles(roles_text)
    num_speakers = int(n_speakers) if n_speakers else (len(roles) if roles else None)
    result = run(audio_path, num_speakers=num_speakers)
    if not roles and auto_roles:
        roles = guess_roles(result.turns)
    text = render(fmt, result.turns, roles, timestamps=timestamps, segments=result.segments)

    out = out_dir / f"{Path(audio_path).stem}.{fmt}"
    n = 2
    while out.exists():  # two uploads with the same name
        out = out_dir / f"{Path(audio_path).stem} ({n}).{fmt}"
        n += 1
    out.write_text(text, encoding="utf-8")

    speakers = len({t.speaker for t in result.turns})
    summary = f"{result.duration / 60:.1f} min · {speakers} speaker(s) · {len(result.turns)} turns"
    return text, out, summary


def process(
    audio_path: str | None, n_speakers: float | None, roles_text: str, auto_roles: bool, fmt: str, timestamps: bool
):
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")
    try:
        text, out, summary = transcribe_one(
            audio_path, Path(tempfile.mkdtemp()), n_speakers, roles_text, auto_roles, fmt, timestamps
        )
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001 - show the real reason instead of a bare "Error"
        log.exception("Transcription failed for %s", audio_path)
        raise gr.Error(f"{type(exc).__name__}: {exc}", duration=None) from exc
    return text, [str(out)], summary


def _status_table(names: list[str], states: list[str]) -> str:
    rows = "\n".join(f"| {i + 1} | {name} | {state} |" for i, (name, state) in enumerate(zip(names, states)))
    return f"| # | File | Status |\n|---|---|---|\n{rows}"


def process_batch(
    files, n_speakers: float | None, roles_text: str, auto_roles: bool, fmt: str, timestamps: bool
):
    """Transcribe a queue of files one by one, streaming progress to the UI.

    A file that fails is marked in the table and the queue carries on with the next one.
    """
    if not files:
        raise gr.Error("Add one or more audio files first.")
    paths = [_file_path(f) for f in files]
    names = [Path(p).name for p in paths]
    states = ["⏳ waiting"] * len(paths)
    out_dir = Path(tempfile.mkdtemp())
    outputs: list[Path] = []
    texts: list[str] = []

    for i, path in enumerate(paths):
        states[i] = "🎧 transcribing…"
        yield "\n\n".join(texts), [str(p) for p in outputs], _status_table(names, states)
        try:
            text, out, summary = transcribe_one(path, out_dir, n_speakers, roles_text, auto_roles, fmt, timestamps)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the queue
            log.exception("Transcription failed for %s", path)
            states[i] = f"❌ {type(exc).__name__}: {exc}"
            continue
        states[i] = f"✅ {summary}"
        outputs.append(out)
        texts.append(f"===== {names[i]} =====\n\n{text}")

    files_out = [str(p) for p in outputs]
    if len(outputs) > 1:
        zip_path = out_dir / "transcripts.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in outputs:
                z.write(p, arcname=p.name)
        files_out.insert(0, str(zip_path))
    yield "\n\n".join(texts), files_out, _status_table(names, states)


with gr.Blocks(title="Who Said What") as demo:
    gr.Markdown(
        "# Who Said What\n"
        "Audio → transcript with speaker roles. Diarization: **NVIDIA Nemotron 3 Diarization** "
        "(up to 8 speakers). ASR: **Parakeet TDT 0.6B v3** (25 European languages, incl. RU / PT / EN)."
    )
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("One file"):
                    audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
                    btn = gr.Button("Transcribe", variant="primary")
                with gr.Tab("Several files"):
                    files = gr.File(
                        file_count="multiple",
                        file_types=["audio", "video"],
                        label="Audio files (select or drop several at once)",
                    )
                    btn_batch = gr.Button("Transcribe all", variant="primary")
                    gr.Markdown(
                        "Files are processed one after another. The settings below apply to every file."
                    )
            n_speakers = gr.Number(
                label="How many people are speaking? (optional, fixes over-split speakers)",
                precision=0,
                minimum=1,
                maximum=8,
                value=None,
            )
            roles = gr.Textbox(
                label="Roles (optional, in order of first speaking)",
                placeholder="Manager, Client",
            )
            auto = gr.Checkbox(value=False, label="Guess roles with Claude (needs ANTHROPIC_API_KEY)")
            fmt = gr.Radio(["txt", "md", "srt", "json"], value="txt", label="Format")
            ts = gr.Checkbox(value=True, label="Show timestamps")
        with gr.Column():
            summary = gr.Markdown()
            transcript = copyable_textbox(label="Transcript", lines=24)
            file_out = gr.File(label="Download", file_count="multiple")

    inputs = [n_speakers, roles, auto, fmt, ts]
    outputs = [transcript, file_out, summary]
    # Same concurrency group: one transcription at a time, the rest wait in Gradio's queue.
    btn.click(process, [audio, *inputs], outputs, concurrency_id="model", concurrency_limit=1)
    btn_batch.click(process_batch, [files, *inputs], outputs, concurrency_id="model", concurrency_limit=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true", help="Open the page in the default browser")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo.queue().launch(share=a.share, server_port=a.port, inbrowser=a.inbrowser, show_error=True)
