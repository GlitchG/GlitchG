import pytest

pytest.importorskip("telegram")
bot = pytest.importorskip("bot")


def test_chunks_respect_limit_and_keep_all_text():
    paras = [f"Speaker {i % 2 + 1}: " + "palavra " * 60 for i in range(40)]
    text = "\n\n".join(paras)
    chunks = list(bot._chunks(text, size=1000))
    assert all(len(c) <= 1000 for c in chunks)
    assert "\n\n".join(chunks) == text


def test_chunks_split_single_huge_paragraph():
    chunks = list(bot._chunks("x" * 2500, size=1000))
    assert [len(c) for c in chunks] == [1000, 1000, 500]
