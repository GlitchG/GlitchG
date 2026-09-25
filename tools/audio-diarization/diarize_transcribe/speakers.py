"""Fix over-split speakers: merge diarizer speakers that have the same voice.

The diarizer sometimes gives one person two labels (voice or mic changes, long
pauses). We compute a voice fingerprint (TitaNet speaker embedding) for each
label and merge labels whose voices match, or merge down to an exact speaker
count when the user knows it.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from .alignment import Segment

log = logging.getLogger(__name__)

SPEAKER_EMBEDDING_MODEL = "titanet_large"
SPEAKER_EMBEDDING_MODEL_HF = "nvidia/speakerverification_en_titanet_large"
# NeMo's own default for "same speaker" in EncDecSpeakerLabelModel.verify_speakers.
SAME_SPEAKER_SIMILARITY = 0.7

MIN_SEGMENT_SEC = 0.5  # shorter segments give unreliable fingerprints
MAX_SEGMENT_SEC = 10.0
SEGMENTS_PER_SPEAKER = 8


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def cluster_speakers(
    embeddings: dict[int, np.ndarray],
    num_speakers: int | None = None,
    threshold: float = SAME_SPEAKER_SIMILARITY,
) -> dict[int, int]:
    """Average-linkage agglomerative clustering on cosine similarity.

    Returns {original speaker: representative speaker}. With ``num_speakers`` it merges
    until that many clusters remain; otherwise it merges while the most similar pair of
    clusters is at least ``threshold`` alike.
    """
    clusters: list[list[int]] = [[s] for s in sorted(embeddings)]
    vecs = {s: _normalize(np.asarray(e, dtype=np.float64)) for s, e in embeddings.items()}

    def similarity(a: list[int], b: list[int]) -> float:
        return float(np.mean([vecs[x] @ vecs[y] for x in a for y in b]))

    while len(clusters) > 1:
        if num_speakers is not None and len(clusters) <= num_speakers:
            break
        best, pair = -2.0, (0, 1)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                sim = similarity(clusters[i], clusters[j])
                if sim > best:
                    best, pair = sim, (i, j)
        if num_speakers is None and best < threshold:
            break
        i, j = pair
        log.info("Merging speakers %s + %s (voice similarity %.2f)", clusters[i], clusters[j], best)
        clusters[i] = clusters[i] + clusters[j]
        del clusters[j]

    return {s: min(c) for c in clusters for s in c}


def relabel_by_arrival(segments: list[Segment], mapping: dict[int, int]) -> list[Segment]:
    """Apply a speaker mapping, then renumber 0..N-1 in order of first appearance."""
    merged = [Segment(s.start, s.end, mapping.get(s.speaker, s.speaker)) for s in segments]
    order: dict[int, int] = {}
    for s in sorted(merged, key=lambda s: s.start):
        order.setdefault(s.speaker, len(order))
    return [Segment(s.start, s.end, order[s.speaker]) for s in merged]


def _assign_to_nearest(speaker: int, segments: list[Segment], candidates: set[int]) -> int:
    """For a label with too little audio to fingerprint: use the closest-in-time other speaker."""
    own = [s for s in segments if s.speaker == speaker]
    others = [s for s in segments if s.speaker in candidates]
    if not own or not others:
        return speaker

    def gap(a: Segment, b: Segment) -> float:
        return max(0.0, max(a.start, b.start) - min(a.end, b.end))

    return min(others, key=lambda o: min(gap(o, s) for s in own)).speaker


@lru_cache(maxsize=1)
def load_speaker_model():
    from nemo.collections.asr.models import EncDecSpeakerLabelModel

    from .pipeline import _device, _quiet_nemo

    _quiet_nemo()
    try:
        model = EncDecSpeakerLabelModel.from_pretrained(SPEAKER_EMBEDDING_MODEL, map_location=_device())
    except Exception as exc:  # noqa: BLE001 - NGC download failed; same model is mirrored on Hugging Face
        log.warning("Could not load %s from NGC (%s); trying Hugging Face", SPEAKER_EMBEDDING_MODEL, exc)
        model = EncDecSpeakerLabelModel.from_pretrained(SPEAKER_EMBEDDING_MODEL_HF, map_location=_device())
    model.eval()
    return model


def speaker_embeddings(audio: np.ndarray, sr: int, segments: list[Segment]) -> dict[int, np.ndarray]:
    """Average fingerprint over each speaker's longest segments."""
    model = load_speaker_model()
    by_speaker: dict[int, list[Segment]] = {}
    for s in segments:
        if s.end - s.start >= MIN_SEGMENT_SEC:
            by_speaker.setdefault(s.speaker, []).append(s)

    result: dict[int, np.ndarray] = {}
    for spk, segs in by_speaker.items():
        segs = sorted(segs, key=lambda s: s.end - s.start, reverse=True)[:SEGMENTS_PER_SPEAKER]
        embs = []
        for s in segs:
            start = int(s.start * sr)
            end = int(min(s.end, s.start + MAX_SEGMENT_SEC) * sr)
            emb, _ = model.infer_segment(audio[start:end])
            embs.append(_normalize(emb.squeeze().cpu().numpy()))
        result[spk] = _normalize(np.mean(embs, axis=0))
    return result


def merge_speakers(
    audio: np.ndarray,
    sr: int,
    segments: list[Segment],
    num_speakers: int | None = None,
) -> list[Segment]:
    """Merge labels that belong to the same voice; optionally force an exact count."""
    speakers = {s.speaker for s in segments}
    if len(speakers) <= 1 or (num_speakers is not None and len(speakers) <= num_speakers):
        return relabel_by_arrival(segments, {})

    embeddings = speaker_embeddings(audio, sr, segments)
    mapping = cluster_speakers(embeddings, num_speakers) if len(embeddings) > 1 else {s: s for s in embeddings}
    # Labels with only tiny fragments can't be fingerprinted: fold them into the nearest
    # speaker in time (following that speaker's merge, if any).
    for spk in speakers - set(embeddings):
        target = _assign_to_nearest(spk, segments, set(embeddings))
        mapping[spk] = mapping.get(target, target)
    return relabel_by_arrival(segments, mapping)
