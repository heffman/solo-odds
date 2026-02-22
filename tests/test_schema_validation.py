import json
from pathlib import Path

from jsonschema import Draft202012Validator

from solo_odds.data.models import NetworkSnapshot
from solo_odds.math.analytic import analyze_constant_rate
from solo_odds.units import parse_hashrate


def load_schema() -> dict:
    schema_path = Path("schemas/report.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_validates_minimal_report(tmp_path) -> None:
    schema = load_schema()
    validator = Draft202012Validator(schema)

    # Minimal synthetic snapshot
    snapshot = NetworkSnapshot(
        coin="btc",
        timestamp=parse_iso("2026-02-21T00:00:00Z"),
        network_hashrate_hs=1e18,
        difficulty=1e12,
        blocks_per_day=144.0,
        block_reward=3.125,
        source="test",
        source_url="https://example.com",
    )

    hr = parse_hashrate("1TH")
    base_lambda = (hr.hs / snapshot.network_hashrate_hs) * snapshot.blocks_per_day
    analytic = analyze_constant_rate(base_lambda, horizon_days=30)
    snap_d = snapshot.to_dict()

    report = {
        "schema_version": 1,
        "generated_at": "2026-02-21T00:00:00Z",
        "input": {
            "coin": "btc",
            "hashrate_hs": hr.hs,
            "hashrate_display": hr.format(),
            "horizon_days": 30,
            "drift_model": {
                "type": "flat",
                "parameters": {
                    "step_pct": None,
                    "step_days": None,
                    "daily_pct": None,
                },
            },
            "monte_carlo_runs": 0,
        },
        "network_snapshot": {
            "timestamp": snap_d["timestamp"],
            "network_hashrate_hs": snap_d["network_hashrate_hs"],
            "difficulty": snap_d["difficulty"],
            "blocks_per_day": snap_d["blocks_per_day"],
            "block_reward": snap_d["block_reward"],
            "source": snap_d.get("source"),
            "source_url": snap_d.get("source_url"),
        },
        "analytic": {
            "lambda_per_day": analytic.lambda_per_day_effective,
            "expected_blocks": analytic.expected_blocks,
            "probability_at_least_one": analytic.probability_at_least_one,
            "probability_zero_blocks": analytic.probability_zero_blocks,
            "block_distribution": [
                {"k": k, "probability": p}
                for k, p in analytic.block_distribution
            ],
            "time_to_first_block_days": analytic.time_to_first_block_days,
        },
        "monte_carlo": {"enabled": False},
        "notes": ["test"],
    }

    errors = sorted(validator.iter_errors(report), key=lambda e: e.path)
    assert not errors, f"Schema validation errors: {errors}"


def parse_iso(s: str):
    from datetime import datetime, timezone
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt.astimezone(timezone.utc)