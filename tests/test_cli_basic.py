from datetime import datetime, timezone
from pathlib import Path

from solo_odds.data.models import NetworkSnapshot
from solo_odds.data.store import SnapshotStore


def test_snapshot_store_latest_roundtrip(tmp_path: Path) -> None:
    store = SnapshotStore(root_dir=tmp_path / "data")
    snap = NetworkSnapshot(
        coin="bch",
        timestamp=datetime(2026, 2, 21, tzinfo=timezone.utc),
        network_hashrate_hs=3.0e18,
        difficulty=1.0,
        blocks_per_day=144.0,
        block_reward=3.125,
        source="test",
    )
    store.write_latest(snap)
    loaded = store.read_latest("bch")
    assert loaded.coin == "bch"