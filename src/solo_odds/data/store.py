from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Optional

from solo_odds.data.models import NetworkSnapshot, SnapshotValidationError


class SnapshotStoreError(RuntimeError):
    pass


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + '.tmp')

    # Write to tmp first
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    # Atomic replace
    os.replace(tmp_path, path)


def _read_json(path: Path) -> dict:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise SnapshotStoreError(f'Snapshot file not found: {path}') from exc
    except json.JSONDecodeError as exc:
        raise SnapshotStoreError(f'Invalid JSON in snapshot file: {path}') from exc


@dataclass(frozen=True)
class SnapshotStore:
    """
    Local filesystem snapshot store.

    Layout:
      <root>/data/<coin>/latest.json
      <root>/data/<coin>/snapshots/YYYY-MM-DD.json
    """
    root_dir: Path

    @classmethod
    def from_repo_root(cls) -> 'SnapshotStore':
        """
        Default store rooted at ./data in current working directory.
        """
        return cls(root_dir=Path('data'))

    def _coin_dir(self, coin: str) -> Path:
        coin_norm = coin.strip().lower()
        if coin_norm not in ('btc', 'bch'):
            raise SnapshotStoreError(f"coin must be 'btc' or 'bch', got {coin!r}")
        return self.root_dir / coin_norm

    def latest_path(self, coin: str) -> Path:
        return self._coin_dir(coin) / 'latest.json'

    def dated_path(self, coin: str, snapshot_date: date) -> Path:
        return self._coin_dir(coin) / 'snapshots' / f'{snapshot_date.isoformat()}.json'

    def write_latest(self, snapshot: NetworkSnapshot) -> Path:
        path = self.latest_path(snapshot.coin)
        payload = snapshot.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        _atomic_write_text(path, text)
        return path

    def write_dated(self, snapshot: NetworkSnapshot, snapshot_date: Optional[date] = None) -> Path:
        """
        Write snapshot under snapshots/YYYY-MM-DD.json. Default uses snapshot timestamp date (UTC).
        """
        if snapshot_date is None:
            snapshot_date = snapshot.timestamp.date()
        path = self.dated_path(snapshot.coin, snapshot_date)
        payload = snapshot.to_dict()
        text = json.dumps(payload, indent=2, sort_keys=True) + '\n'
        _atomic_write_text(path, text)
        return path

    def read_latest(self, coin: str) -> NetworkSnapshot:
        path = self.latest_path(coin)
        data = _read_json(path)
        try:
            snapshot = NetworkSnapshot.from_dict(data)
        except SnapshotValidationError as exc:
            raise SnapshotStoreError(f'Invalid snapshot data in {path}: {exc}') from exc

        # Ensure coin matches requested coin (defensive)
        if snapshot.coin != coin.strip().lower():
            raise SnapshotStoreError(
                f'Latest snapshot coin mismatch: requested={coin!r} file_coin={snapshot.coin!r} path={path}'
            )
        return snapshot

    def read_dated(self, coin: str, snapshot_date: date) -> NetworkSnapshot:
        path = self.dated_path(coin, snapshot_date)
        data = _read_json(path)
        try:
            snapshot = NetworkSnapshot.from_dict(data)
        except SnapshotValidationError as exc:
            raise SnapshotStoreError(f'Invalid snapshot data in {path}: {exc}') from exc

        if snapshot.coin != coin.strip().lower():
            raise SnapshotStoreError(
                f'Dated snapshot coin mismatch: requested={coin!r} file_coin={snapshot.coin!r} path={path}'
            )
        return snapshot