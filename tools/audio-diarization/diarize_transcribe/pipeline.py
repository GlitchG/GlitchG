"""Audio -> speaker-attributed transcript, using NVIDIA NeMo models.

1. Convert any audio (mp3, m4a, ogg voice notes, video...) to 16 kHz mono WAV.
2. Diarize with Nemotron 3 Diarization ("who spoke when", up to 8 speakers).
3. Transcribe with Parakeet TDT 0.6B v3 (25 European languages, word timestamps).
4. Assign each word to the speaker active at that moment and group into turns.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .alignment import Segment, Turn, Word, assign_speakers, build_turns, parse_segments

log = logging.getLogger(__name__)

DEFAULT_DIAR_MODEL = "nvidia/Nemotron-3-Diarization"
# Older 4-speaker Streaming Sortformer; same API, used if the main model can't load.
FALLBACK_DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
DEFAULT_ASR_MODEL = "nvidia/parakeet-tdt-0.6b-v3"

SAMPLE_RATE = 16_000
# Parakeet's full attention handles ~24 min; beyond that switch to local attention.
LONG_AUDIO_SEC = 20 * 60


@dataclass
class Result:
    turns: list[Turn]
    segments: list[Segment]
    words: list[Word]
    duration: float


def _device() -> str:
    """NVIDIA GPU if present, else CPU.

    WHOSAID_DEVICE overrides it, e.g. ``WHOSAID_DEVICE=mps`` to try the Apple Silicon GPU
    (not officially supported by NeMo; unsupported ops fall back to CPU).
    """
    import torch

    override = os.environ.get("WHOSAID_DEVICE")
    if override:
        if override == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_wav_16k_mono(src: str | Path, dst_dir: str | Path) -> Path:
    """Convert anything ffmpeg can read into the 16 kHz mono WAV the models expect."""
    src = Path(src)
    dst = Path(dst_dir) / f"{src.stem}_16k.wav"
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:  # portable ffmpeg from pip, so no Homebrew/apt install is needed
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            ffmpeg = None
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)],
            check=True,
        )
        return dst
    # Fallback without ffmpeg (handles wav/flac/ogg; mp3/m4a usually need ffmpeg).
    import librosa
    import soundfile as sf

    audio, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    sf.write(str(dst), audio, SAMPLE_RATE)
    return dst


def _duration(path: Path) -> float:
    import soundfile as sf

    return sf.info(str(path)).duration


@lru_cache(maxsize=2)
def load_diarizer(model_name: str = DEFAULT_DIAR_MODEL):
    from nemo.collections.asr.models import SortformerEncLabelModel

    try:
        model = SortformerEncLabelModel.from_pretrained(model_name, map_location=_device())
    except Exception as exc:  # noqa: BLE001 - surface the reason, then fall back
        if model_name == FALLBACK_DIAR_MODEL:
            raise
        log.warning("Could not load %s (%s); falling back to %s", model_name, exc, FALLBACK_DIAR_MODEL)
        model = SortformerEncLabelModel.from_pretrained(FALLBACK_DIAR_MODEL, map_location=_device())
    model.eval()
    return model


@lru_cache(maxsize=2)
def load_asr(model_name: str = DEFAULT_ASR_MODEL):
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location=_device())
    model.eval()
    return model


def diarize(wav: Path, model_name: str = DEFAULT_DIAR_MODEL) -> list[Segment]:
    model = load_diarizer(model_name)
    predicted = model.diarize(audio=[str(wav)], batch_size=1, verbose=False)
    return parse_segments(predicted[0])


def transcribe_words(wav: Path, duration: float, model_name: str = DEFAULT_ASR_MODEL) -> list[Word]:
    model = load_asr(model_name)
    long_audio = duration > LONG_AUDIO_SEC
    if long_audio:
        # Local attention lets Parakeet process up to ~3 h in one pass.
        model.change_attention_model("rel_pos_local_attn", [256, 256])
        model.change_subsampling_conv_chunking_factor(1)
    try:
        hyp = model.transcribe([str(wav)], timestamps=True, verbose=False)[0]
    finally:
        if long_audio:
            model.change_attention_model("rel_pos")
            model.change_subsampling_conv_chunking_factor(-1)

    words = []
    for w in hyp.timestamp.get("word", []):
        start = w.get("start")
        end = w.get("end")
        if start is None or end is None:
            continue
        words.append(Word(text=w["word"], start=float(start), end=float(end)))
    return words


def run(
    audio_path: str | Path,
    diar_model: str = DEFAULT_DIAR_MODEL,
    asr_model: str = DEFAULT_ASR_MODEL,
    max_pause: float = 2.0,
) -> Result:
    with tempfile.TemporaryDirectory() as tmp:
        wav = to_wav_16k_mono(audio_path, tmp)
        duration = _duration(wav)
        log.info("Audio: %.1f s", duration)

        log.info("Diarizing with %s ...", diar_model)
        segments = diarize(wav, diar_model)
        log.info("Found %d speaker(s)", len({s.speaker for s in segments}))

        log.info("Transcribing with %s ...", asr_model)
        words = transcribe_words(wav, duration, asr_model)

    assign_speakers(words, segments)
    turns = build_turns(words, max_pause=max_pause)
    return Result(turns=turns, segments=segments, words=words, duration=duration)
