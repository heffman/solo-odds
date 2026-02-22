# src/solo_odds/units.py
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


class UnitParseError(ValueError):
    pass


_SI_PREFIX_TO_MULTIPLIER = {
    "": 1.0,
    "k": 1e3,
    "K": 1e3,
    "m": 1e6,
    "M": 1e6,
    "g": 1e9,
    "G": 1e9,
    "t": 1e12,
    "T": 1e12,
    "p": 1e15,
    "P": 1e15,
    "e": 1e18,
    "E": 1e18,
}

# Examples matched:
#   9.4TH
#   1200 GH/s
#   500kH
#   9.4e12
#   .5 T
_HASHRATE_RE = re.compile(
    r"""
    ^\s*
    (?P<value>
        (?:\d+(?:\.\d*)?|\.\d+)      # 12, 12., 12.3, .5
        (?:[eE][+-]?\d+)?            # optional exponent
    )
    \s*
    (?P<prefix>[kKmMgGtTpPeE]?)      # optional SI prefix
    \s*
    (?:
        (?:h|H|hs|HS|h/s|H/s|hash|hashes)?   # optional unit token
    )
    \s*
    (?:
        (?:/s|/S|s|S|sec|second|seconds)?    # optional per-second token
    )
    \s*
    $
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Hashrate:
    hs: float  # hashes per second

    def __post_init__(self) -> None:
        if not math.isfinite(self.hs) or self.hs < 0:
            raise ValueError(f"Invalid hashrate hs={self.hs!r}")

    def format(self, precision: int = 3) -> str:
        return format_hashrate(self.hs, precision=precision)


def parse_hashrate(text: str) -> Hashrate:
    if text is None:
        raise UnitParseError("Hashrate text is None")

    cleaned = text.strip()
    if not cleaned:
        raise UnitParseError("Hashrate is empty")

    match = _HASHRATE_RE.match(cleaned)
    if not match:
        raise UnitParseError(
            f"Could not parse hashrate: {text!r}. "
            "Examples: '9.4TH', '1200 GH/s', '500kH', '9.4e12'."
        )

    value_str = match.group("value")
    prefix = match.group("prefix") or ""

    try:
        value = float(value_str)
    except ValueError as exc:
        raise UnitParseError(f"Invalid numeric value in hashrate: {text!r}") from exc

    if not math.isfinite(value) or value < 0:
        raise UnitParseError(f"Hashrate must be finite and non-negative: {text!r}")

    multiplier = _SI_PREFIX_TO_MULTIPLIER.get(prefix)
    if multiplier is None:
        raise UnitParseError(f"Unknown SI prefix {prefix!r} in hashrate: {text!r}")

    return Hashrate(hs=value * multiplier)


def format_hashrate(hs: float, precision: int = 3) -> str:
    if not math.isfinite(hs) or hs < 0:
        raise ValueError(f"Invalid hashrate hs={hs!r}")

    units = [
        ("E", 1e18),
        ("P", 1e15),
        ("T", 1e12),
        ("G", 1e9),
        ("M", 1e6),
        ("k", 1e3),
        ("", 1.0),
    ]

    prefix, scale = "", 1.0
    for p, s in units:
        if hs >= s:
            prefix, scale = p, s
            break

    value = hs / scale
    formatted = f"{value:.{precision}g}"
    return f"{formatted} {prefix}H/s".strip()


def maybe_parse_hashrate(text: Optional[str]) -> Optional[Hashrate]:
    if text is None:
        return None
    if not text.strip():
        return None
    return parse_hashrate(text)