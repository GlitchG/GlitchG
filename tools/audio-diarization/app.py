"""Web UI: upload or record audio, get a transcript with speaker roles.

Run:  python app.py            (opens http://127.0.0.1:7860)
      python app.py --share    (public gradio.live link, handy on Colab)
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import gradio as gr

from diarize_transcribe.cli import parse_roles
from diarize_transcribe.formatters import render
from diarize_transcribe.pipeline import run
from diarize_transcribe.roles import guess_roles


def process(audio_path: str | None, roles_text: str, auto_roles: bool, fmt: str, timestamps: bool):
    if not audio_path:
        raise gr.Error("Upload or record an audio file first.")
    result = run(audio_path)
    roles = parse_roles(roles_text)
    if not roles and auto_roles:
        roles = guess_roles(result.turns)
    text = render(fmt, result.turns, roles, timestamps=timestamps, segments=result.segments)

    out = Path(tempfile.mkdtemp()) / f"{Path(audio_path).stem}.{fmt}"
    out.write_text(text, encoding="utf-8")

    speakers = len({t.speaker for t in result.turns})
    summary = f"{result.duration / 60:.1f} min · {speakers} speaker(s) · {len(result.turns)} turns"
    return text, str(out), summary


with gr.Blocks(title="Who Said What") as demo:
    gr.Markdown(
        "# Who Said What\n"
        "Audio → transcript with speaker roles. Diarization: **NVIDIA Nemotron 3 Diarization** "
        "(up to 8 speakers). ASR: **Parakeet TDT 0.6B v3** (25 European languages, incl. RU / PT / EN)."
    )
    with gr.Row():
        with gr.Column():
            audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
            roles = gr.Textbox(
                label="Roles (optional, in order of first speaking)",
                placeholder="Manager, Client",
            )
            auto = gr.Checkbox(value=False, label="Guess roles with Claude (needs ANTHROPIC_API_KEY)")
            fmt = gr.Radio(["txt", "md", "srt", "json"], value="txt", label="Format")
            ts = gr.Checkbox(value=True, label="Show timestamps")
            btn = gr.Button("Transcribe", variant="primary")
        with gr.Column():
            summary = gr.Markdown()
            transcript = gr.Textbox(label="Transcript", lines=24, show_copy_button=True)
            file_out = gr.File(label="Download")
    btn.click(process, [audio, roles, auto, fmt, ts], [transcript, file_out, summary])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--inbrowser", action="store_true", help="Open the page in the default browser")
    a = ap.parse_args()
    demo.queue().launch(share=a.share, server_port=a.port, inbrowser=a.inbrowser)
