"""Pull a batch number or a side out of a free-text reply."""

from __future__ import annotations

import re


def parse_number(text: str) -> int | None:
    """Last integer in the reply, if it is a single digit from 0 to 9."""
    found = re.findall(r"\d+", text or "")
    if not found:
        return None
    number = int(found[-1])
    if number > 9:
        return None
    return number


def parse_side(text: str) -> str | None:
    """Last ``left`` or ``right`` in the reply."""
    hits = list(re.finditer(r"left|right", (text or "").lower()))
    if not hits:
        return None
    return hits[-1].group(0)
