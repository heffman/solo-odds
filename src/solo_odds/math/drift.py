from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from solo_odds.math.analytic import AnalyticError, RateSegment


class DriftError(ValueError):
    pass


@dataclass(frozen=True)
class DriftModel:
    """
    Drift model for network growth.

    type:
      - 'flat': no drift
      - 'step': network grows by step_pct every step_days (compounded)
      - 'linear': network grows by daily_pct per day (compounded daily)

    Percent inputs are in percent points (e.g. 2.0 means +2%).
    """
    type: str
    step_pct: Optional[float] = None
    step_days: Optional[int] = None
    daily_pct: Optional[float] = None

    def __post_init__(self) -> None:
        model_type = (self.type or "").strip().lower()
        if model_type not in ("flat", "step", "linear"):
            raise DriftError(f"Unknown drift model type: {self.type!r}")

        if model_type == "flat":
            return

        if model_type == "step":
            if self.step_pct is None or self.step_days is None:
                raise DriftError("step drift requires step_pct and step_days")
            if not math.isfinite(float(self.step_pct)):
                raise DriftError("step_pct must be finite")
            if int(self.step_days) <= 0:
                raise DriftError("step_days must be > 0")
            return

        if model_type == "linear":
            if self.daily_pct is None:
                raise DriftError("linear drift requires daily_pct")
            if not math.isfinite(float(self.daily_pct)):
                raise DriftError("daily_pct must be finite")
            return


def make_segments(
    *,
    base_lambda_per_day: float,
    horizon_days: int,
    drift: DriftModel,
) -> List[RateSegment]:
    """
    Convert a drift model into RateSegment list.

    base_lambda_per_day:
      lambda at time zero (no drift yet), derived from current network snapshot.

    horizon_days:
      integer horizon in days.

    Returns:
      list of RateSegment(duration_days, lambda_per_day) covering the horizon.

    Notes:
      - Drift is applied as growth in *network hashrate*, reducing your lambda as:
          lambda_t = base_lambda / growth_factor_t
      - The returned segments represent a piecewise-constant approximation of the
        time-varying lambda.

    For 'flat':
      single segment.

    For 'step':
      segments of length step_days, each with lambda adjusted by compounded growth.

    For 'linear':
      daily segments (1 day each), compounding by daily_pct each day.

    This keeps v1 simple and deterministic.
    """
    if not math.isfinite(base_lambda_per_day) or base_lambda_per_day < 0:
        raise AnalyticError(f"Invalid base_lambda_per_day={base_lambda_per_day!r}")
    if horizon_days <= 0:
        raise AnalyticError(f"Invalid horizon_days={horizon_days!r}")

    model_type = drift.type.strip().lower()

    if model_type == "flat":
        return [RateSegment(duration_days=float(horizon_days), lambda_per_day=float(base_lambda_per_day))]

    segments: List[RateSegment] = []
    remaining = float(horizon_days)

    if model_type == "step":
        step_days = int(drift.step_days or 0)
        step_pct = float(drift.step_pct or 0.0)

        step_factor = 1.0 + (step_pct / 100.0)
        if step_factor <= 0:
            raise DriftError("step_pct results in non-positive growth factor")

        step_index = 0
        while remaining > 0:
            duration = float(step_days) if remaining >= step_days else remaining
            growth_factor = step_factor ** step_index
            lambda_i = base_lambda_per_day / growth_factor
            segments.append(RateSegment(duration_days=duration, lambda_per_day=float(lambda_i)))
            remaining -= duration
            step_index += 1

        return segments

    if model_type == "linear":
        daily_pct = float(drift.daily_pct or 0.0)
        daily_factor = 1.0 + (daily_pct / 100.0)
        if daily_factor <= 0:
            raise DriftError("daily_pct results in non-positive growth factor")

        day_index = 0
        while remaining > 0:
            duration = 1.0 if remaining >= 1.0 else remaining
            growth_factor = daily_factor ** day_index
            lambda_i = base_lambda_per_day / growth_factor
            segments.append(RateSegment(duration_days=duration, lambda_per_day=float(lambda_i)))
            remaining -= duration
            day_index += 1

        return segments

    raise DriftError(f"Unhandled drift model type: {drift.type!r}")


def drift_model_from_cli(
    *,
    drift_type: str,
    step_pct: float,
    step_days: int,
    daily_pct: float,
) -> DriftModel:
    """
    Helper to map CLI options into a DriftModel with consistent defaults.

    - drift_type: 'flat' | 'step' | 'linear'
    - step_pct/step_days used only for 'step'
    - daily_pct used only for 'linear'
    """
    drift_type_norm = (drift_type or "").strip().lower()
    if drift_type_norm == "flat":
        return DriftModel(type="flat")

    if drift_type_norm == "step":
        return DriftModel(type="step", step_pct=step_pct, step_days=step_days, daily_pct=None)

    if drift_type_norm == "linear":
        return DriftModel(type="linear", step_pct=None, step_days=None, daily_pct=daily_pct)

    raise DriftError(f"Unknown drift_type: {drift_type!r}")