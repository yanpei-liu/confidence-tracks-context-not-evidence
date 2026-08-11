"""EgoLife timestamp parsing.

EgoLife encodes times as ``DAY{n}`` + ``HHMMSSFF`` where ``FF`` is a frame index at 30 fps.
``target_time`` (the evidence timestamp) appears in three encodings in the released A1 file:

1. ``{"date": "DAY1", "time": "11152408"}``                       -> one evidence point
2. ``{"date": "DAY1", "time_list": ["13394908", "13512718"]}``    -> several, same day
3. ``{"date": "DAY1", "time": "11153417DAY1_11181201"}``          -> several, day-qualified

One item uses lowercase ``Day1``. All 500 released items parse under these rules.
"""
from __future__ import annotations

import re
from typing import Any

from .config import SRC_FPS

_DAY_RE = re.compile(r"(?i)^\s*day\s*(\d+)\s*$")
_TIME_RE = re.compile(r"^(\d{8})$")
_MULTI_RE = re.compile(r"(?i)^(\d{8})((?:DAY\d+_\d{8})*)$")
_SUFFIX_RE = re.compile(r"(?i)DAY(\d+)_(\d{8})")


def day_number(date_field: Any) -> int:
    m = _DAY_RE.match(str(date_field))
    if not m:
        raise ValueError(f"unparseable date field: {date_field!r}")
    return int(m.group(1))


def abs_seconds(day: int, hhmmssff: str) -> float:
    """Absolute seconds since the start of DAY1 00:00:00.

    This is a *wall-clock* coordinate with the day boundaries intact; it is only used as
    a key into the stream index, never as a Δt measure (recording has overnight gaps).
    """
    s = str(hhmmssff)
    if not _TIME_RE.match(s):
        raise ValueError(f"unparseable time field: {hhmmssff!r}")
    hh, mm, ss, ff = int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])
    # FF is a 20 fps frame index; a handful of annotations exceed 19, so clamp to < 1 s
    # rather than letting them bleed into the next second.
    frac = min(ff / SRC_FPS, 0.999)
    return (day - 1) * 86400.0 + hh * 3600.0 + mm * 60.0 + ss + frac


def parse_time_field(field: dict) -> list[float]:
    """Parse a ``query_time`` or ``target_time`` dict into sorted absolute seconds."""
    day0 = day_number(field["date"])
    out: list[float] = []
    if "time_list" in field:
        for s in field["time_list"]:
            out.append(abs_seconds(day0, s))
    else:
        s = str(field["time"])
        m = _MULTI_RE.match(s)
        if not m:
            raise ValueError(f"unparseable time value: {s!r}")
        out.append(abs_seconds(day0, m.group(1)))
        for dd, tt in _SUFFIX_RE.findall(m.group(2)):
            out.append(abs_seconds(int(dd), tt))
    return sorted(out)


def encoding_of(field: dict) -> str:
    """Label the encoding variant, for the schema report."""
    if "time_list" in field:
        return "time_list"
    n = len(_SUFFIX_RE.findall(str(field["time"])))
    return "time_scalar" if n == 0 else f"time_concat_{n + 1}"
