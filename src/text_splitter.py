"""Utilities for splitting long text into inference-friendly chunks."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？…])\s+|\n+")
_WHITESPACE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return len(text.split())


def _balanced_word_slices(words: Sequence[str], max_words: int) -> list[str]:
    """Split words into balanced chunks that never exceed ``max_words``."""
    if not words:
        return []

    chunk_count = math.ceil(len(words) / max_words)
    base_size, remainder = divmod(len(words), chunk_count)
    chunks: list[str] = []
    offset = 0

    for index in range(chunk_count):
        size = base_size + (1 if index < remainder else 0)
        chunks.append(" ".join(words[offset : offset + size]))
        offset += size

    return chunks


def _sentence_units(text: str, max_words: int) -> list[str]:
    sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    units: list[str] = []

    for sentence in sentences:
        words = sentence.split()
        if len(words) <= max_words:
            units.append(sentence)
        else:
            units.extend(_balanced_word_slices(words, max_words))

    return units


def split_text(text: str, min_words: int = 20, max_words: int = 40) -> list[str]:
    """Split text into chunks of approximately ``min_words`` to ``max_words``.

    Sentence endings are preferred. Oversized sentences are split into balanced
    word groups. A short final chunk is merged with or rebalanced against the
    preceding chunk whenever possible.

    Args:
        text: Input text to split.
        min_words: Preferred minimum number of words per chunk.
        max_words: Hard maximum number of words per chunk.

    Returns:
        Normalized, non-empty text chunks in their original order.

    Raises:
        ValueError: If chunk limits are invalid.
    """
    if min_words < 1:
        raise ValueError("min_words must be at least 1")
    if max_words < min_words:
        raise ValueError("max_words must be greater than or equal to min_words")

    normalized = _WHITESPACE.sub(" ", text).strip()
    if not normalized:
        return []
    if _word_count(normalized) <= max_words:
        return [normalized]

    units = _sentence_units(normalized, max_words)
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for unit in units:
        unit_words = unit.split()
        unit_size = len(unit_words)

        if current_size + unit_size <= max_words:
            current.extend(unit_words)
            current_size += unit_size
            continue

        if current:
            chunks.append(" ".join(current))
        current = unit_words
        current_size = unit_size

    if current:
        chunks.append(" ".join(current))

    if len(chunks) >= 2 and _word_count(chunks[-1]) < min_words:
        previous_words = chunks[-2].split()
        final_words = chunks[-1].split()

        if len(previous_words) + len(final_words) <= max_words:
            chunks[-2:] = [" ".join(previous_words + final_words)]
        else:
            needed = min_words - len(final_words)
            movable = max(0, len(previous_words) - min_words)
            transfer_count = min(needed, movable)
            if transfer_count:
                final_words = previous_words[-transfer_count:] + final_words
                previous_words = previous_words[:-transfer_count]
                chunks[-2:] = [" ".join(previous_words), " ".join(final_words)]

    # A very short sentence followed by a near-limit sentence can leave an
    # underfilled middle chunk. Fall back to balanced word boundaries only when
    # sentence-aware packing cannot satisfy the requested range.
    if any(_word_count(chunk) < min_words for chunk in chunks):
        balanced = _balanced_word_slices(normalized.split(), max_words)
        if all(_word_count(chunk) >= min_words for chunk in balanced):
            chunks = balanced

    return chunks
