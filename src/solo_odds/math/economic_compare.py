from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List


class EconomicCompareError(ValueError):
    pass


@dataclass(frozen=True)
class SoloDistributionPoint:
    k: int
    probability: float
    net_usd: float


@dataclass(frozen=True)
class EconomicCompareResult:
    mu: float
    revenue_per_block_usd: float
    electricity_cost_usd: float

    solo: Dict[str, float]
    pool: Dict[str, float]
    comparison: Dict[str, float]

    # optional detail for debugging / future UI
    solo_distribution: List[SoloDistributionPoint]


def _electricity_cost_usd(*, watts: float, days: float, cost_per_kwh: float) -> float:
    if watts < 0 or days < 0 or cost_per_kwh < 0:
        raise EconomicCompareError('electricity inputs must be non-negative')
    kwh = (watts / 1000.0) * 24.0 * days
    return kwh * cost_per_kwh


def _poisson_distribution(mu: float, cdf_target: float = 0.999999) -> List[float]:
    """
    Returns pmf list [p0, p1, ..., pkmax] such that sum(p) >= cdf_target.
    Uses stable recurrence; deterministic and dependency-free.
    """
    if mu < 0 or not math.isfinite(mu):
        raise EconomicCompareError(f'Invalid mu={mu!r}')
    if mu == 0.0:
        return [1.0]

    p0 = math.exp(-mu)
    pmf: List[float] = [p0]
    cdf = p0
    k = 0

    # Hard safety cap: mu + 10*sqrt(mu) + 50 (very conservative)
    hard_cap = int(mu + 10.0 * math.sqrt(mu) + 50.0)
    if hard_cap < 100:
        hard_cap = 100

    while cdf < cdf_target and k < hard_cap:
        pk = pmf[-1]
        k_next = k + 1
        p_next = pk * (mu / k_next)
        pmf.append(p_next)
        cdf += p_next
        k = k_next

    # Normalize (tiny truncation error)
    s = sum(pmf)
    if s <= 0:
        raise EconomicCompareError('Poisson PMF sum <= 0 (unexpected)')
    pmf = [p / s for p in pmf]
    return pmf


def _quantile_k_from_pmf(pmf: List[float], q: float) -> int:
    if not pmf:
        raise EconomicCompareError('Empty PMF')
    if q <= 0:
        return 0
    if q >= 1:
        return len(pmf) - 1

    cdf = 0.0
    for k, p in enumerate(pmf):
        cdf += p
        if cdf >= q:
            return k
    return len(pmf) - 1


def compare_solo_vs_pool(
    *,
    mu: float,
    block_reward: float,
    coin_price_usd: float,
    horizon_days: float,
    asic_power_watts: float,
    electricity_cost_per_kwh: float,
    pool_fee_pct: float,
) -> EconomicCompareResult:
    if mu < 0 or not math.isfinite(mu):
        raise EconomicCompareError('mu must be finite and non-negative')
    if block_reward < 0 or coin_price_usd < 0 or horizon_days < 0:
        raise EconomicCompareError('reward/price/days must be non-negative')
    if pool_fee_pct < 0 or pool_fee_pct >= 1:
        raise EconomicCompareError('pool_fee_pct must be in [0, 1)')

    r_block = block_reward * coin_price_usd
    c_elec = _electricity_cost_usd(
        watts=asic_power_watts,
        days=horizon_days,
        cost_per_kwh=electricity_cost_per_kwh,
    )

    # Pool: deterministic expectation (v1)
    pool_gross = mu * r_block
    pool_net = pool_gross * (1.0 - pool_fee_pct) - c_elec

    # Solo: Poisson distribution on blocks
    pmf = _poisson_distribution(mu)

    dist: List[SoloDistributionPoint] = []
    expected_net = 0.0
    p_negative = 0.0
    p_zero_blocks = pmf[0]

    for k, pk in enumerate(pmf):
        net_k = (k * r_block) - c_elec
        dist.append(SoloDistributionPoint(k=k, probability=pk, net_usd=net_k))
        expected_net += pk * net_k
        if net_k < 0:
            p_negative += pk

    # Quantiles (monotonic in k)
    k10 = _quantile_k_from_pmf(pmf, 0.10)
    k50 = _quantile_k_from_pmf(pmf, 0.50)
    k90 = _quantile_k_from_pmf(pmf, 0.90)

    if r_block <= 0:
        p10_net = -c_elec
        p50_net = -c_elec
        p90_net = -c_elec
    else:
        p10_net = (k10 * r_block) - c_elec
        p50_net = (k50 * r_block) - c_elec
        p90_net = (k90 * r_block) - c_elec

    # Probability solo underperforms pool:
    # net_solo(k) < pool_net  <=>  k*r_block - c_elec < pool_net
    # <=> k < (pool_net + c_elec) / r_block
    if r_block <= 0:
        p_under = 0.0 if abs(expected_net - pool_net) < 1e-12 else (1.0 if expected_net < pool_net else 0.0)
    else:
        # Under this pool model: pool_net + c_elec == mu*r_block*(1-fee)
        threshold = mu * (1.0 - pool_fee_pct)
        k_star = int(math.floor(threshold - 1e-12))
        if k_star < 0:
            p_under = 0.0
        elif k_star >= len(pmf) - 1:
            p_under = 1.0
        else:
            p_under = float(sum(pmf[: k_star + 1]))

    solo = {
        'expected_net_usd': float(expected_net),
        'p10_net_usd': float(p10_net),
        'p50_net_usd': float(p50_net),
        'p90_net_usd': float(p90_net),
        'k_p10': int(k10),
        'k_p50': int(k50),
        'k_p90': int(k90),
        'probability_negative_net': float(p_negative),
        'probability_zero_blocks': float(p_zero_blocks),
    }
    pool = {
        'expected_net_usd': float(pool_net),
        'expected_gross_usd': float(pool_gross * (1.0 - pool_fee_pct)),
        'electricity_cost_usd': float(c_elec),
    }
    comparison = {
        'probability_solo_underperforms_pool': float(p_under),
        'expected_value_delta_usd': float(expected_net - pool_net),
        'pool_blocks_threshold_k': float(threshold),
    }

    return EconomicCompareResult(
        mu=float(mu),
        revenue_per_block_usd=float(r_block),
        electricity_cost_usd=float(c_elec),
        solo=solo,
        pool=pool,
        comparison=comparison,
        solo_distribution=dist,
    )