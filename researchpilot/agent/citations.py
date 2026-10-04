"""Parse inline citation markers such as [1], [2][3], [1, 4] or [2-4] out of an answer."""

from __future__ import annotations

import re

_BRACKET = re.compile(r"\[(\s*\d+(?:\s*(?:,|-|–|—)\s*\d+)*\s*)\]")
_RANGE = re.compile(r"(\d+)\s*[-–—]\s*(\d+)")
MAX_RANGE = 50  # ignore absurd ranges like [1-9999]
# Some models (gpt-oss) write their native style 【1†L1-L4】 or 【1】 despite the prompt.
_LENTICULAR = re.compile(r"【\s*(\d+(?:\s*[,\-–—]\s*\d+)*)\s*(?:†[^】]*)?】")


def normalize_citations(text: str) -> str:
    """Rewrite lenticular-bracket markers such as 【1†L1-L4】 as [1], the format the UI and parser expect."""
    return _LENTICULAR.sub(r"[\1]", text or "")


def parse_citations(text: str) -> list[int]:
    """Citation numbers in order of first appearance, without duplicates."""
    numbers: list[int] = []
    for group in _BRACKET.findall(text or ""):
        for part in group.split(","):
            part = part.strip()
            match = _RANGE.fullmatch(part)
            if match:
                start, end = int(match.group(1)), int(match.group(2))
                if start <= end and end - start <= MAX_RANGE:
                    numbers.extend(range(start, end + 1))
            elif part.isdigit():
                numbers.append(int(part))
    return list(dict.fromkeys(numbers))


def split_valid(numbers: list[int], n_sources: int) -> tuple[list[int], list[int]]:
    """Split citation numbers into (valid, invalid) for a list of `n_sources` sources."""
    valid = [n for n in numbers if 1 <= n <= n_sources]
    invalid = [n for n in numbers if not 1 <= n <= n_sources]
    return valid, invalid
