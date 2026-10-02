import numpy as np

from diarize_transcribe import speakers
from diarize_transcribe.alignment import Segment
from diarize_transcribe.speakers import cluster_speakers, merge_speakers, relabel_by_arrival

rng = np.random.default_rng(0)
VOICE_A, VOICE_B, VOICE_C, VOICE_D = (rng.normal(size=192) for _ in range(4))


def near(v, noise=0.1):
    return v + rng.normal(scale=noise * np.linalg.norm(v) / np.sqrt(v.size), size=v.size)


def segs(*triples):
    return [Segment(float(a), float(b), s) for a, b, s in triples]


def test_threshold_merges_same_voice_only():
    emb = {0: near(VOICE_A), 1: near(VOICE_B), 2: near(VOICE_A), 3: near(VOICE_C)}
    mapping = cluster_speakers(emb)
    assert mapping[2] == mapping[0]
    assert len({mapping[s] for s in emb}) == 3


def test_forced_count_merges_most_similar_first():
    # 6 labels, 4 real voices: labels 4 and 5 are repeats of voices A and B.
    emb = {0: near(VOICE_A), 1: near(VOICE_B), 2: near(VOICE_C), 3: near(VOICE_D),
           4: near(VOICE_A, 0.5), 5: near(VOICE_B, 0.5)}
    mapping = cluster_speakers(emb, num_speakers=4, threshold=1.1)
    assert mapping[4] == mapping[0] and mapping[5] == mapping[1]
    assert len({mapping[s] for s in emb}) == 4


def test_forced_count_never_splits():
    emb = {0: VOICE_A, 1: VOICE_B}
    assert len(set(cluster_speakers(emb, num_speakers=4).values())) == 2


def test_relabel_by_arrival():
    out = relabel_by_arrival(segs((5, 6, 7), (0, 1, 3), (2, 3, 9)), {9: 3})
    assert [(s.start, s.speaker) for s in out] == [(5.0, 1), (0.0, 0), (2.0, 0)]


def test_merge_speakers_end_to_end(monkeypatch):
    # Speaker 3 is a repeat of speaker 0; speaker 2 is a 0.2 s blip right before speaker 3.
    segments = segs((0, 5, 0), (5, 10, 1), (10.5, 10.7, 2), (10.7, 15, 3), (15, 20, 1))
    fake = {0: VOICE_A, 1: VOICE_B, 3: near(VOICE_A)}
    monkeypatch.setattr(speakers, "speaker_embeddings", lambda audio, sr, s: {k: fake[k] for k in {x.speaker for x in s} if k in fake})
    out = merge_speakers(np.zeros(16000 * 20, dtype=np.float32), 16000, segments)
    assert [s.speaker for s in out] == [0, 1, 0, 0, 1]


def test_merge_speakers_skips_when_already_few(monkeypatch):
    monkeypatch.setattr(speakers, "speaker_embeddings", lambda *a: (_ for _ in ()).throw(AssertionError("not called")))
    out = merge_speakers(np.zeros(10), 16000, segs((0, 1, 2), (1, 2, 5)), num_speakers=2)
    assert [s.speaker for s in out] == [0, 1]
