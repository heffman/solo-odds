from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from solo_odds.data.models import NetworkSnapshot


class FetchError(RuntimeError):
    pass


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc).replace(microsecond=0)


def _get_text(url: str, *, timeout_s: float) -> str:
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        return resp.text.strip()
    except requests.RequestException as exc:
        raise FetchError(f"HTTP error fetching {url}: {exc}") from exc


def _get_json(url: str, *, timeout_s: float) -> Dict[str, Any]:
    try:
        resp = requests.get(url, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise FetchError(f"HTTP error fetching {url}: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"Invalid JSON from {url}: {exc}") from exc


def fetch_btc_snapshot(*, timeout_s: float = 10.0) -> NetworkSnapshot:
    """
    BTC snapshot using Blockchain.com Query API (plaintext).
    """
    # Query API doc includes endpoints like:
    # /q/getdifficulty, /q/hashrate (in gigahash), /q/interval (seconds), /q/bcperblock (BTC)
    base = "https://blockchain.info/q"

    difficulty = float(_get_text(f"{base}/getdifficulty", timeout_s=timeout_s))
    hashrate_gh = float(_get_text(f"{base}/hashrate", timeout_s=timeout_s))
    interval_s = float(_get_text(f"{base}/interval", timeout_s=timeout_s))
    reward_btc = float(_get_text(f"{base}/bcperblock", timeout_s=timeout_s))

    if interval_s <= 0:
        raise FetchError(f"Bad BTC interval_s={interval_s!r}")

    blocks_per_day = 86400.0 / interval_s
    network_hashrate_hs = hashrate_gh * 1e9

    return NetworkSnapshot(
        coin="btc",
        timestamp=_now_utc(),
        network_hashrate_hs=network_hashrate_hs,
        difficulty=difficulty,
        blocks_per_day=blocks_per_day,
        block_reward=reward_btc,
        source="blockchain.com query api",
        source_url=base,
    )


def _pick_first(d: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _hashrate_from_difficulty(difficulty: float, *, target_block_time_s: float) -> float:
    """
    Expected hashes per block ≈ difficulty * 2^32
    Hashrate ≈ expected_hashes_per_block / target_block_time
    """
    if target_block_time_s <= 0:
        raise FetchError(f"Bad target_block_time_s={target_block_time_s!r}")
    return float(difficulty) * 4294967296.0 / float(target_block_time_s)


def _fetch_bch_blockchair(*, timeout_s: float) -> NetworkSnapshot:
    """
    BCH snapshot using Blockchair stats JSON.

    Endpoint format is stable in practice, but field names may vary.
    We accept multiple likely keys.

    If reward is not provided, we fall back to 3.125 BCH (post-2024 era).
    """
    url = "https://api.blockchair.com/bitcoin-cash/stats"
    payload = _get_json(url, timeout_s=timeout_s)

    data = payload.get("data")
    if not isinstance(data, dict):
        raise FetchError(f"Unexpected Blockchair BCH stats payload shape from {url}")

    # Difficulty (usually present)
    difficulty = _pick_first(
        data,
        "difficulty",
        "difficulty_24h",
        "difficulty_current",
    )
    if difficulty is None:
        raise FetchError(f"Could not find difficulty in Blockchair BCH stats response keys={sorted(data.keys())}")

    # Hashrate (naming varies; try several)
    # Common variants seen across stats APIs:
    # - hashrate
    # - hash_rate
    # - hashrate_24h
    # - hash_rate_24h
    hashrate = _pick_first(
        data,
        "hashrate",
        "hash_rate",
        "hashrate_24h",
        "hash_rate_24h",
        "average_hashrate_24h",
        "average_hashrate",
    )
    if hashrate is None:
        raise FetchError(f"Could not find hashrate in Blockchair BCH stats response keys={sorted(data.keys())}")

    # Some APIs provide “hashrate” in H/s already; others provide in hashes/sec.
    # We assume H/s (most common). If you observe otherwise, adjust here.
    network_hashrate_hs = float(hashrate)

    # BCH targets ~144 blocks/day (10 min)
    blocks_per_day = 144.0

    # Reward: prefer API, else fallback.
    reward = _pick_first(
        data,
        "block_reward",
        "block_reward_bch",
        "next_block_reward",
        "next_block_reward_bch",
    )
    block_reward = float(reward) if reward is not None else 3.125

    return NetworkSnapshot(
        coin="bch",
        timestamp=_now_utc(),
        network_hashrate_hs=network_hashrate_hs,
        difficulty=float(difficulty),
        blocks_per_day=blocks_per_day,
        block_reward=block_reward,
        source="blockchair stats (plus fallback reward if missing)",
        source_url=url,
    )


def _fetch_bch_fullstack(*, timeout_s: float) -> NetworkSnapshot:
    """
    Fallback BCH provider: FullStack.cash getDifficulty (difficulty only).
    Hashrate is derived from difficulty assuming 600s target.
    """
    difficulty_url = "https://api.fullstack.cash/v5/blockchain/getDifficulty"
    difficulty = float(_get_text(difficulty_url, timeout_s=timeout_s))

    network_hashrate_hs = _hashrate_from_difficulty(difficulty, target_block_time_s=600.0)

    return NetworkSnapshot(
        coin="bch",
        timestamp=_now_utc(),
        network_hashrate_hs=network_hashrate_hs,
        difficulty=float(difficulty),
        blocks_per_day=144.0,
        block_reward=3.125,
        source="fullstack.cash getDifficulty (hashrate derived)",
        source_url=difficulty_url,
    )
 

def fetch_bch_snapshot(*, timeout_s: float = 10.0) -> NetworkSnapshot:
    """
    BCH snapshot with fallback:
      1) Blockchair (difficulty + hashrate)
      2) FullStack.cash getDifficulty (difficulty only; hashrate derived)
    """
    try:
        return _fetch_bch_blockchair(timeout_s=timeout_s)
    except FetchError:
        return _fetch_bch_fullstack(timeout_s=timeout_s)


def fetch_snapshot(coin: str, *, timeout_s: float = 10.0) -> NetworkSnapshot:
    coin_norm = coin.strip().lower()
    if coin_norm == "btc":
        return fetch_btc_snapshot(timeout_s=timeout_s)
    if coin_norm == "bch":
        return fetch_bch_snapshot(timeout_s=timeout_s)
    raise FetchError(f"Unsupported coin: {coin!r}")