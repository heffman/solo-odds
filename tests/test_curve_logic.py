from solo_odds.math.analytic import analyze_segments
from solo_odds.math.drift import DriftModel, make_segments


def test_curve_monotonic_probability_flat() -> None:
    base_lambda = 0.01  # per day
    drift = DriftModel(type="flat")

    prev = 0.0
    for horizon in (1, 5, 10, 30, 60):
        res = analyze_segments(make_segments(base_lambda_per_day=base_lambda, horizon_days=horizon, drift=drift))
        assert res.probability_at_least_one >= prev
        prev = res.probability_at_least_one