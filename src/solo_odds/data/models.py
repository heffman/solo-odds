from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class SnapshotValidationError(ValueError):
    pass


def _require_finite_non_negative(name: str, value: float) -> float:
    if not isinstance(value, (int, float)):
        raise SnapshotValidationError(f'{name} must be a number, got {type(value).__name__}')
    value_f = float(value)
    if not math.isfinite(value_f) or value_f < 0:
        raise SnapshotValidationError(f'{name} must be finite and non-negative, got {value!r}')
    return value_f


def _require_non_empty_str(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnapshotValidationError(f'{name} must be a non-empty string')
    return value.strip()


def parse_iso8601_utc(ts: str) -> datetime:
    """
    Parse an ISO8601 timestamp. Accepts trailing 'Z' for UTC.
    Returns an aware datetime in UTC.
    """
    ts = _require_non_empty_str('timestamp', ts)
    # Allow "Z"
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError as exc:
        raise SnapshotValidationError(f'Invalid timestamp format: {ts!r}') from exc

    if dt.tzinfo is None:
        # Assume UTC if tz missing, but keep this strict: better to be explicit.
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def format_iso8601_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    # Use Z suffix
    return dt.replace(microsecond=0).isoformat().replace('+00:00', 'Z')


@dataclass(frozen=True)
class NetworkSnapshot:
    """
    Minimal network snapshot used by analytic and simulation layers.

    All rates are in base units:
      - network_hashrate_hs: H/s
      - difficulty: chain difficulty (dimensionless)
      - blocks_per_day: float (BTC/BCH ~144)
      - block_reward: coins per block (subsidy; fees excluded in v1)
    """
    coin: str  # 'btc' or 'bch'
    timestamp: datetime  # aware UTC
    network_hashrate_hs: float
    difficulty: float
    blocks_per_day: float
    block_reward: float
    source: Optional[str] = None  # e.g., 'mempool.space', 'blockchair', etc.

    def __post_init__(self) -> None:
        coin_norm = _require_non_empty_str('coin', self.coin).lower()
        if coin_norm not in ('btc', 'bch'):
            raise SnapshotValidationError(f"coin must be 'btc' or 'bch', got {self.coin!r}")

        ts = self.timestamp
        if not isinstance(ts, datetime):
            raise SnapshotValidationError('timestamp must be a datetime')
        if ts.tzinfo is None:
            raise SnapshotValidationError('timestamp must be timezone-aware (UTC preferred)')

        _require_finite_non_negative('network_hashrate_hs', self.network_hashrate_hs)
        _require_finite_non_negative('difficulty', self.difficulty)
        _require_finite_non_negative('blocks_per_day', self.blocks_per_day)
        _require_finite_non_negative('block_reward', self.block_reward)

        if self.source is not None and not isinstance(self.source, str):
            raise SnapshotValidationError('source must be a string or null')

    def to_dict(self) -> Dict[str, Any]:
        return {
            'coin': self.coin.lower(),
            'timestamp': format_iso8601_utc(self.timestamp),
            'network_hashrate_hs': float(self.network_hashrate_hs),
            'difficulty': float(self.difficulty),
            'blocks_per_day': float(self.blocks_per_day),
            'block_reward': float(self.block_reward),
            'source': self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkSnapshot':
        if not isinstance(data, dict):
            raise SnapshotValidationError('snapshot JSON must be an object')

        coin = _require_non_empty_str('coin', data.get('coin', ''))
        timestamp = parse_iso8601_utc(data.get('timestamp', ''))

        return cls(
            coin=coin.lower(),
            timestamp=timestamp,
            network_hashrate_hs=_require_finite_non_negative(
                'network_hashrate_hs', data.get('network_hashrate_hs', float('nan'))
            ),
            difficulty=_require_finite_non_negative(
                'difficulty', data.get('difficulty', float('nan'))
            ),
            blocks_per_day=_require_finite_non_negative(
                'blocks_per_day', data.get('blocks_per_day', float('nan'))
            ),
            block_reward=_require_finite_non_negative(
                'block_reward', data.get('block_reward', float('nan'))
            ),
            source=data.get('source'),
        )