import numpy as np
import pytest

from voxlocal._audio import AudioResult
from voxlocal.tts._supertonic import (
    DIRECT_STEPS,
    STREAMING_STEPS,
    SupertonicTTS,
    _split_progressive,
    _split_text,
)


def test_progressive_chunks_start_small_then_grow():
    text = " ".join(f"word{i}" for i in range(60))

    sizes = [len(chunk.split()) for chunk in _split_progressive(text)]

    assert sizes[:5] == [3, 5, 8, 12, 16]


def test_progression_groups_sentences_one_one_two_then_three():
    text = (
        "One two three. Second sentence. Third sentence. Fourth sentence. "
        "Fifth sentence. Sixth sentence. Seventh sentence. Eighth sentence."
    )

    chunks = _split_progressive(text)

    assert chunks == [
        "One two three.",
        "Second sentence.",
        "Third sentence.",
        "Fourth sentence. Fifth sentence.",
        "Sixth sentence. Seventh sentence. Eighth sentence.",
    ]


def test_japanese_without_spaces_is_split_progressively():
    text = (
        "これは非常に長い日本語の文章です。"
        "次の文も音声として生成されます。"
        "さらに別の文が続きます。"
    )

    chunks = _split_progressive(text, language="ja")

    assert len(chunks) >= 4
    assert "".join(chunks).replace(" ", "") == text


def test_invalid_chunk_strategy_is_rejected():
    with pytest.raises(ValueError, match="chunk_by"):
        _split_text("hello", "typo")


def test_direct_and_streaming_use_separate_step_counts():
    tts = SupertonicTTS("es")
    calls: list[tuple[str, int]] = []

    def synthesize(text: str, steps: int) -> AudioResult:
        calls.append((text, steps))
        return AudioResult(np.zeros(10, dtype=np.float32), 10)

    tts._synthesize = synthesize
    tts.speak("one two three")
    list(tts.speak_iter(" ".join(f"word{i}" for i in range(20))))

    assert calls[0][1] == DIRECT_STEPS
    assert all(steps == STREAMING_STEPS for _, steps in calls[1:])


def test_empty_text_fails_before_engine_use():
    tts = SupertonicTTS("es")

    with pytest.raises(ValueError, match="empty"):
        list(tts.speak_iter(""))
