# src/solo_odds/sim/monte_carlo.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from solo_odds.math.analytic import RateSegment


class MonteCarloError(ValueError):
    pass


@dataclass(frozen=True)
class MonteCarloResult:
    runs: int
    blocks_over_horizon: Dict[str, float]
    time_to_first_block_days: Dict[str, float]


def _percentile(sorted_values: List[float], q: float) -> float:
    """
    Linear-interpolated percentile with q in [0, 1].
    """
    if not sorted_values:
        raise MonteCarloError('Cannot compute percentile of empty list')
    if q <= 0:
        return float(sorted_values[0])
    if q >= 1:
        return float(sorted_values[-1])

    n = len(sorted_values)
    pos = (n - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])

    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _poisson_sample(mu: float, rng: random.Random) -> int:
    """
    Sample from Poisson(mu).

    Uses:
      - Knuth for small mu
      - Normal approximation for larger mu (good enough for v1)

    This project’s typical mu (home mining) will usually be small.
    """
    if mu < 0 or not math.isfinite(mu):
        raise MonteCarloError(f'Invalid mu={mu!r}')
    if mu == 0.0:
        return 0

    # Small mu: Knuth
    if mu < 30.0:
        l = math.exp(-mu)
        k = 0
        p = 1.0
        while p > l:
            k += 1
            p *= rng.random()
        return k - 1

    # Larger mu: normal approx with continuity correction
    # Clamp to >= 0
    x = rng.gauss(mu, math.sqrt(mu))
    k = int(math.floor(x + 0.5))
    return k if k > 0 else 0


def _simulate_first_event_in_segment(
    lambda_per_day: float,
    duration_days: float,
    rng: random.Random,
) -> float | None:
    """
    If at least one event occurs in the segment, return time offset (days)
    to the first event within the segment, else None.

    For a homogeneous Poisson process within a segment:
      P(no events) = exp(-lambda * duration)
      Conditional on >=1 event, first arrival ~ Exponential(lambda) truncated.
    We can sample exponential and check if it lands within duration.
    """
    if lambda_per_day <= 0:
        return None

    # Sample exponential waiting time
    u = rng.random()
    wait = -math.log(1.0 - u) / lambda_per_day
    if wait <= duration_days:
        return wait
    return None


def run_monte_carlo(
    segments: Sequence[RateSegment],
    runs: int,
    seed: int | None = None,
) -> MonteCarloResult:
    if runs <= 0:
        raise MonteCarloError('runs must be > 0')
    if not segments:
        raise MonteCarloError('segments must not be empty')

    rng = random.Random(seed)

    horizon_days = sum(s.duration_days for s in segments)
    if horizon_days <= 0:
        raise MonteCarloError('horizon_days must be > 0')

    total_blocks: List[int] = []
    first_times: List[float] = []

    for _ in range(runs):
        blocks = 0
        t_cursor = 0.0
        first_time: float | None = None

        for seg in segments:
            mu = seg.lambda_per_day * seg.duration_days
            k = _poisson_sample(mu, rng)
            blocks += k

            if first_time is None:
                # Try to sample the first event time within this segment.
                # If k==0, skip quickly.
                if k > 0:
                    offset = _simulate_first_event_in_segment(
                        lambda_per_day=seg.lambda_per_day,
                        duration_days=seg.duration_days,
                        rng=rng,
                    )
                    if offset is not None:
                        first_time = t_cursor + offset

            t_cursor += seg.duration_days

        total_blocks.append(blocks)
        if first_time is not None:
            first_times.append(first_time)

    total_blocks_sorted = sorted(total_blocks)
    mean_blocks = float(sum(total_blocks) / runs)

    prob_zero = float(sum(1 for x in total_blocks if x == 0) / runs)
    prob_ge_one = 1.0 - prob_zero

    blocks_summary = {
        'mean': mean_blocks,
        'p10': _percentile([float(x) for x in total_blocks_sorted], 0.10),
        'p50': _percentile([float(x) for x in total_blocks_sorted], 0.50),
        'p90': _percentile([float(x) for x in total_blocks_sorted], 0.90),
        'probability_at_least_one': prob_ge_one,
        'probability_zero': prob_zero,
    }

    # Time-to-first-block percentiles:
    # - If a run has no block, it has no first_time. For reporting, we compute percentiles
    #   over the successful runs only, and set mean to inf if no successes.
    if first_times:
        first_times_sorted = sorted(first_times)
        mean_first = float(sum(first_times) / len(first_times))
        time_summary = {
            'p10': _percentile(first_times_sorted, 0.10),
            'p50': _percentile(first_times_sorted, 0.50),
            'p90': _percentile(first_times_sorted, 0.90),
            'mean': mean_first,
        }
    else:
        time_summary = {
            'p10': math.inf,
            'p50': math.inf,
            'p90': math.inf,
            'mean': math.inf,
        }

    return MonteCarloResult(
        runs=runs,
        blocks_over_horizon=blocks_summary,
        time_to_first_block_days=time_summary,
    )