from solo_odds.math.drift import DriftModel, make_segments

def test_flat_segments() -> None:
    segs = make_segments(base_lambda_per_day=1.0, horizon_days=10, drift=DriftModel(type="flat"))
    assert len(segs) == 1
    assert segs[0].duration_days == 10.0
    assert segs[0].lambda_per_day == 1.0

def test_step_segments_reduce_lambda() -> None:
    drift = DriftModel(type="step", step_pct=10.0, step_days=2)
    segs = make_segments(base_lambda_per_day=1.0, horizon_days=5, drift=drift)
    assert len(segs) == 3  # 2 + 2 + 1
    assert segs[0].lambda_per_day == 1.0
    assert segs[1].lambda_per_day < segs[0].lambda_per_day

def test_linear_segments_reduce_lambda() -> None:
    drift = DriftModel(type="linear", daily_pct=1.0)
    segs = make_segments(base_lambda_per_day=1.0, horizon_days=3, drift=drift)
    assert len(segs) == 3
    assert segs[1].lambda_per_day < segs[0].lambda_per_day