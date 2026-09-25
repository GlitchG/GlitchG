import json

from diarize_transcribe.alignment import Word, assign_speakers, build_turns, parse_segments, speaker_label
from diarize_transcribe.formatters import fmt_srt_time, render


def words(*items):
    return [Word(t, s, e) for t, s, e in items]


def test_parse_nemo_strings_and_sort():
    segs = parse_segments(["3.2 5.0 speaker_1", "0.00 3.12 speaker_0", "bad"])
    assert [(s.start, s.end, s.speaker) for s in segs] == [(0.0, 3.12, 0), (3.2, 5.0, 1)]


def test_parse_tuples():
    assert parse_segments([(1, 2, 3)])[0].speaker == 3


def test_assign_by_max_overlap_and_gap():
    segs = parse_segments(["0 2 speaker_0", "1.5 4 speaker_1"])
    ws = assign_speakers(words(("hi", 0.2, 0.6), ("both", 1.6, 2.4), ("gap", 4.5, 4.8)), segs)
    assert [w.speaker for w in ws] == [0, 1, 1]


def test_assign_without_segments_defaults_to_speaker_0():
    assert assign_speakers(words(("x", 0, 1)), [])[0].speaker == 0


def test_build_turns_groups_and_splits_on_pause():
    ws = words(("a", 0, 0.5), ("b", 0.6, 1), ("c", 1.2, 1.5), ("d", 5, 5.5))
    for w, spk in zip(ws, [0, 0, 1, 1]):
        w.speaker = spk
    turns = build_turns(ws, max_pause=2.0)
    assert [(t.speaker, t.text) for t in turns] == [(0, "a b"), (1, "c"), (1, "d")]


def test_roles_and_formats():
    assert speaker_label(0, ["Manager", "Client"]) == "Manager"
    assert speaker_label(2, ["Manager", "Client"]) == "Speaker 3"
    assert speaker_label(1, ["Manager", " "]) == "Speaker 2"

    ws = words(("Olá", 0, 0.5), ("Oi", 1, 1.4))
    ws[0].speaker, ws[1].speaker = 0, 1
    turns = build_turns(ws)
    assert render("txt", turns, ["Manager", "Client"]) == "[0:00:00] Manager: Olá\n\n[0:00:01] Client: Oi\n"
    assert "00:00:01,000 --> 00:00:01,400\nClient: Oi" in render("srt", turns, ["Manager", "Client"])
    data = json.loads(render("json", turns, ["Manager", "Client"]))
    assert data["turns"][0] == {"speaker": "Manager", "speaker_index": 0, "start": 0, "end": 0.5, "text": "Olá"}
    assert fmt_srt_time(3725.5) == "01:02:05,500"
