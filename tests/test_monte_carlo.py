import math

from solo_odds.math.analytic import RateSegment, analyze_segments
from solo_odds.sim.monte_carlo import run_monte_carlo


def test_mc_matches_analytic_probability_flat_ish() -> None:
    # Use a moderate mu so probability isn't tiny, but still small enough for stability
    # lambda/day = 0.02 over 50 days => mu = 1.0 => P(>=1)=1-e^-1 ~ 0.632
    segments = [RateSegment(duration_days=50.0, lambda_per_day=0.02)]
    analytic = analyze_segments(segments)

    mc = run_monte_carlo(segments=segments, runs=20000, seed=123)

    p_analytic = analytic.probability_at_least_one
    p_mc = mc.blocks_over_horizon['probability_at_least_one']

    assert abs(p_mc - p_analytic) < 0.02


def test_mc_zero_rate() -> None:
    segments = [RateSegment(duration_days=10.0, lambda_per_day=0.0)]
    mc = run_monte_carlo(segments=segments, runs=5000, seed=1)

    assert mc.blocks_over_horizon['probability_zero'] == 1.0
    assert mc.blocks_over_horizon['probability_at_least_one'] == 0.0
    assert math.isinf(mc.time_to_first_block_days['mean'])