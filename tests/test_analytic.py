import math
from solo_odds.math.analytic import analyze_constant_rate, poisson_pmf

def test_poisson_pmf_mu0() -> None:
    assert poisson_pmf(0, 0.0) == 1.0
    assert poisson_pmf(1, 0.0) == 0.0

def test_analyze_constant_rate_basic() -> None:
    res = analyze_constant_rate(lambda_per_day=0.0, horizon_days=10)
    assert res.expected_blocks == 0.0
    assert res.probability_at_least_one == 0.0
    assert math.isinf(res.time_to_first_block_days["mean"])