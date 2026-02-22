from datetime import datetime, timezone

from solo_odds.data.models import NetworkSnapshot
from solo_odds.data.store import SnapshotStore


def test_store_roundtrip(tmp_path) -> None:
    store = SnapshotStore(root_dir=tmp_path / 'data')

    snap = NetworkSnapshot(
        coin='bch',
        timestamp=datetime(2026, 2, 21, 0, 0, 0, tzinfo=timezone.utc),
        network_hashrate_hs=3.0e18,
        difficulty=1.23e12,
        blocks_per_day=144.0,
        block_reward=3.125,
        source='test',
    )

    store.write_latest(snap)
    loaded = store.read_latest('bch')
    assert loaded.coin == 'bch'
    assert loaded.network_hashrate_hs == snap.network_hashrate_hs