from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from threading import Lock

from voxlocal._audio import AudioResult
from voxlocal._errors import DependencyMissingError

DIRECT_STEPS = 8
STREAMING_STEPS = 6
PROGRESSIVE_WORD_LIMITS = (3, 5, 8, 12, 16)
PROGRESSIVE_CJK_LIMITS = (8, 12, 20, 28, 36)
SENTENCE_GROUPS = (1, 1, 2)
CJK_LANGUAGES = frozenset({"ja", "zh"})


def _split_sentences(text: str) -> list[str]:
    """Split at common Latin and CJK sentence punctuation."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    sentences = re.findall(r".+?(?:[.!?。！？]+(?:[\"'”’）】」』]*)|$)", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _split_text(text: str, chunk_by: str) -> list[str]:
    """Split text by an explicit supported strategy."""
    if chunk_by == "sentence":
        parts = _split_sentences(text)
    elif chunk_by == "line":
        parts = text.splitlines()
    elif chunk_by == "paragraph":
        parts = re.split(r"\n\s*\n", text)
    else:
        raise ValueError(
            "chunk_by must be one of: progressive, sentence, line, paragraph"
        )
    return [part.strip() for part in parts if part.strip()]


def _split_units(text: str, language: str) -> list[str]:
    """Return words, or characters for languages without word whitespace."""
    if language in CJK_LANGUAGES and " " not in text.strip():
        return [character for character in text if not character.isspace()]
    return text.split()


def _join_units(units: list[str], language: str) -> str:
    separator = "" if language in CJK_LANGUAGES else " "
    return separator.join(units).strip()


def _split_first_sentence(sentence: str, language: str) -> list[str]:
    """Grow early chunks while never crossing the first sentence boundary."""
    units = _split_units(sentence, language)
    limits = (
        PROGRESSIVE_CJK_LIMITS
        if language in CJK_LANGUAGES and " " not in sentence
        else PROGRESSIVE_WORD_LIMITS
    )
    if len(units) <= limits[0]:
        return [sentence]

    chunks: list[str] = []
    cursor = 0
    for limit in limits:
        remaining = len(units) - cursor
        if remaining <= 0:
            break
        if remaining <= limit:
            chunks.append(_join_units(units[cursor:], language))
            cursor = len(units)
            break
        chunks.append(_join_units(units[cursor : cursor + limit], language))
        cursor += limit

    while cursor < len(units):
        end = min(cursor + limits[-1], len(units))
        chunks.append(_join_units(units[cursor:end], language))
        cursor = end
    return chunks


def _split_progressive(text: str, language: str = "en") -> list[str]:
    """Start tiny, finish the first sentence, then group 1, 1, 2, 3 sentences."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = _split_first_sentence(sentences[0], language)
    remaining = sentences[1:]
    cursor = 0
    for group_size in SENTENCE_GROUPS:
        if cursor >= len(remaining):
            break
        end = min(cursor + group_size, len(remaining))
        chunks.append(" ".join(remaining[cursor:end]))
        cursor = end

    while cursor < len(remaining):
        end = min(cursor + 3, len(remaining))
        chunks.append(" ".join(remaining[cursor:end]))
        cursor = end
    return chunks


class SupertonicTTS:
    """Supertonic TTS engine with separate direct and streaming quality."""

    def __init__(
        self, language: str = "en", model_dir: str | Path | None = None
    ):
        self.language = language
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self._tts = None
        self._style = None
        self._synthesis_lock = Lock()

    def _ensure_model(self) -> None:
        if self._tts is None:
            try:
                from supertonic import TTS
            except ImportError as error:
                raise DependencyMissingError("supertonic", "tts") from error

            if self.model_dir is None:
                raise RuntimeError("Supertonic model_dir is required")
            self._tts = TTS(
                model="supertonic-3",
                model_dir=self.model_dir,
                auto_download=False,
            )
            self._style = self._tts.get_voice_style(voice_name="M1")

    def warmup(self) -> None:
        """Pre-initialize Supertonic model resources."""
        self._ensure_model()

    @property
    def sample_rate(self) -> int:
        self._ensure_model()
        return self._tts.sample_rate

    def _synthesize(self, text: str, steps: int) -> AudioResult:
        if not text.strip():
            raise ValueError("text must not be empty")
        self._ensure_model()
        with self._synthesis_lock:
            wav, _duration = self._tts.synthesize(
                text=text,
                lang=self.language,
                voice_style=self._style,
                total_steps=steps,
                speed=1.0,
                silence_duration=0,
            )
        return AudioResult(numpy=wav, sample_rate=self._tts.sample_rate)

    def speak(self, text: str) -> AudioResult:
        """Generate one complete result at the higher-quality direct setting."""
        return self._synthesize(text, steps=DIRECT_STEPS)

    def speak_iter(
        self,
        text: str,
        chunk_by: str = "progressive",
    ) -> Iterator[AudioResult]:
        """Yield raw generated chunks at the lower-latency streaming setting."""
        chunks = (
            _split_progressive(text, self.language)
            if chunk_by == "progressive"
            else _split_text(text, chunk_by)
        )
        if not chunks:
            raise ValueError("text must not be empty")
        for chunk in chunks:
            yield self._synthesize(chunk, steps=STREAMING_STEPS)
