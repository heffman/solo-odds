from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


class AnalyticError(ValueError):
    pass


@dataclass(frozen=True)
class RateSegment:
    """
    A segment of time with a constant block-discovery rate (lambda_per_day).

    duration_days: segment duration in days (float > 0)
    lambda_per_day: expected blocks per day during that segment (float >= 0)
    """
    duration_days: float
    lambda_per_day: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.duration_days) or self.duration_days <= 0:
            raise AnalyticError(f"duration_days must be finite and > 0, got {self.duration_days!r}")
        if not math.isfinite(self.lambda_per_day) or self.lambda_per_day < 0:
            raise AnalyticError(f"lambda_per_day must be finite and >= 0, got {self.lambda_per_day!r}")


@dataclass(frozen=True)
class AnalyticResult:
    """
    Analytic results for a horizon.

    Notes:
      - expected_blocks is mu (sum lambda_i * dt_i)
      - probability_at_least_one is 1 - exp(-mu)
      - time_to_first_block uses an effective constant rate:
          lambda_eff = mu / horizon_days
        This is exact only when lambda is constant; otherwise it is an approximation.
    """
    lambda_per_day_effective: float
    expected_blocks: float
    probability_at_least_one: float
    probability_zero_blocks: float
    block_distribution: List[Tuple[int, float]]
    time_to_first_block_days: dict


def compute_lambda_per_day(
    your_hashrate_hs: float,
    network_hashrate_hs: float,
    blocks_per_day: float,
) -> float:
    """
    lambda_per_day = your_share * blocks_per_day
    where your_share = your_hashrate / network_hashrate
    """
    if not (math.isfinite(your_hashrate_hs) and your_hashrate_hs >= 0):
        raise AnalyticError(f"Invalid your_hashrate_hs={your_hashrate_hs!r}")
    if not (math.isfinite(network_hashrate_hs) and network_hashrate_hs > 0):
        raise AnalyticError(f"Invalid network_hashrate_hs={network_hashrate_hs!r}")
    if not (math.isfinite(blocks_per_day) and blocks_per_day > 0):
        raise AnalyticError(f"Invalid blocks_per_day={blocks_per_day!r}")

    share = your_hashrate_hs / network_hashrate_hs
    return share * blocks_per_day


def expected_blocks(segments: Sequence[RateSegment]) -> float:
    mu = 0.0
    for seg in segments:
        mu += seg.lambda_per_day * seg.duration_days
    return mu


def poisson_pmf(k: int, mu: float) -> float:
    """
    Poisson PMF: P(K=k) = e^-mu * mu^k / k!
    Uses log form for numerical stability for moderate k.
    """
    if k < 0:
        return 0.0
    if mu < 0 or not math.isfinite(mu):
        raise AnalyticError(f"Invalid mu={mu!r}")
    if mu == 0.0:
        return 1.0 if k == 0 else 0.0

    # log(P) = -mu + k*log(mu) - log(k!)
    log_p = -mu + (k * math.log(mu)) - math.lgamma(k + 1)
    return math.exp(log_p)


def poisson_cdf(k: int, mu: float) -> float:
    """
    CDF up to k: sum_{i=0..k} pmf(i)
    For small mu / small k this is fine.
    v1 scope keeps mu small (home mining) and k small.
    """
    if k < 0:
        return 0.0
    total = 0.0
    for i in range(0, k + 1):
        total += poisson_pmf(i, mu)
    return min(1.0, total)


def block_distribution(
    mu: float,
    cumulative_target: float = 0.999,
    k_max: int = 1000,
) -> List[Tuple[int, float]]:
    """
    Returns [(k, Pk), ...] until cumulative probability reaches cumulative_target.

    Defaults:
      - cumulative_target=0.999 keeps output small but meaningful
      - k_max safety cap to avoid runaway in extreme mu scenarios
    """
    if not (0.0 < cumulative_target <= 1.0):
        raise AnalyticError("cumulative_target must be in (0, 1]")
    if k_max < 0:
        raise AnalyticError("k_max must be >= 0")
    if mu < 0 or not math.isfinite(mu):
        raise AnalyticError(f"Invalid mu={mu!r}")

    out: List[Tuple[int, float]] = []
    cumulative = 0.0
    k = 0
    while k <= k_max:
        pk = poisson_pmf(k, mu)
        out.append((k, pk))
        cumulative += pk
        if cumulative >= cumulative_target:
            break
        k += 1

    # Normalize minor floating drift, while keeping shape.
    # (This avoids cumulative slightly above 1 by rounding.)
    total = sum(p for _, p in out)
    if total > 0:
        out = [(k_i, p / total) for k_i, p in out]
    return out


def probability_at_least_one(mu: float) -> float:
    if mu < 0 or not math.isfinite(mu):
        raise AnalyticError(f"Invalid mu={mu!r}")
    # P(>=1) = 1 - e^-mu
    return 1.0 - math.exp(-mu)


def probability_zero(mu: float) -> float:
    if mu < 0 or not math.isfinite(mu):
        raise AnalyticError(f"Invalid mu={mu!r}")
    return math.exp(-mu)


def time_to_first_block_percentile_days(
    lambda_per_day: float,
    percentile: float,
) -> float:
    """
    For exponential waiting time:
      P(T <= t) = 1 - exp(-lambda * t)
      => t_p = -ln(1-p) / lambda
    """
    if not math.isfinite(lambda_per_day) or lambda_per_day < 0:
        raise AnalyticError(f"Invalid lambda_per_day={lambda_per_day!r}")
    if not (0.0 < percentile < 1.0):
        raise AnalyticError("percentile must be between 0 and 1 (exclusive)")

    if lambda_per_day == 0.0:
        return math.inf

    return -math.log(1.0 - percentile) / lambda_per_day


def analyze_segments(
    segments: Sequence[RateSegment],
    cumulative_target: float = 0.999,
) -> AnalyticResult:
    """
    Analyze one horizon represented by one or more constant-rate segments.

    This yields exact block-count probabilities under piecewise constant rate by
    using mu = sum(lambda_i * dt_i) and Poisson(K~Poisson(mu)).

    Note: For a non-homogeneous Poisson process with deterministic intensity,
    total count over horizon is Poisson with parameter mu (integrated intensity),
    so block-count distribution is still exact.

    Time-to-first-block calculations are approximated by an effective constant
    rate (lambda_eff = mu / horizon_days). For true non-homogeneous intensity,
    waiting time distribution differs. v1 keeps this approximation explicit.
    """
    if not segments:
        raise AnalyticError("segments must not be empty")

    horizon_days = sum(seg.duration_days for seg in segments)
    if horizon_days <= 0:
        raise AnalyticError("horizon_days must be > 0")

    mu = expected_blocks(segments)
    p0 = probability_zero(mu)
    p_ge_1 = 1.0 - p0

    dist = block_distribution(mu=mu, cumulative_target=cumulative_target)

    lambda_eff = mu / horizon_days if horizon_days > 0 else 0.0

    t_p10 = time_to_first_block_percentile_days(lambda_eff, 0.10)
    t_p50 = time_to_first_block_percentile_days(lambda_eff, 0.50)
    t_p90 = time_to_first_block_percentile_days(lambda_eff, 0.90)
    t_mean = (1.0 / lambda_eff) if lambda_eff > 0 else math.inf

    return AnalyticResult(
        lambda_per_day_effective=lambda_eff,
        expected_blocks=mu,
        probability_at_least_one=p_ge_1,
        probability_zero_blocks=p0,
        block_distribution=dist,
        time_to_first_block_days={
            "p10": t_p10,
            "p50": t_p50,
            "p90": t_p90,
            "mean": t_mean,
        },
    )


def analyze_constant_rate(
    lambda_per_day: float,
    horizon_days: float,
    cumulative_target: float = 0.999,
) -> AnalyticResult:
    """
    Convenience wrapper for the common constant-rate case.
    """
    if not math.isfinite(horizon_days) or horizon_days <= 0:
        raise AnalyticError(f"Invalid horizon_days={horizon_days!r}")
    seg = RateSegment(duration_days=float(horizon_days), lambda_per_day=float(lambda_per_day))
    return analyze_segments([seg], cumulative_target=cumulative_target)