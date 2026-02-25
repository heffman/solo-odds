from __future__ import annotations

import json as _json
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import typer

from solo_odds.data.store import SnapshotStore, SnapshotStoreError
from solo_odds.data.fetch import FetchError, fetch_snapshot
from solo_odds.sim.monte_carlo import MonteCarloError, run_monte_carlo
from solo_odds.math.analytic import (
    AnalyticError,
    RateSegment,
    analyze_segments,
    compute_lambda_per_day,
)
from solo_odds.math.drift import DriftError, drift_model_from_cli, make_segments
from solo_odds.units import UnitParseError, parse_hashrate

app = typer.Typer(add_completion=False)


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _segments_to_effective_lambda(segments: List[RateSegment]) -> float:
    horizon = sum(s.duration_days for s in segments)
    if horizon <= 0:
        return 0.0
    mu = sum(s.duration_days * s.lambda_per_day for s in segments)
    return mu / horizon


def _network_snapshot_to_report(snapshot) -> Dict[str, Any]:
    d = snapshot.to_dict()
    return {
        "timestamp": d["timestamp"],
        "network_hashrate_hs": d["network_hashrate_hs"],
        "difficulty": d["difficulty"],
        "blocks_per_day": d["blocks_per_day"],
        "block_reward": d["block_reward"],
        "source": d.get("source"),
        "source_url": d.get("source_url"),
    }


def _analytic_to_report(result) -> Dict[str, Any]:
    return {
        "lambda_per_day": result.lambda_per_day_effective,
        "expected_blocks": result.expected_blocks,
        "probability_at_least_one": result.probability_at_least_one,
        "probability_zero_blocks": result.probability_zero_blocks,
        "block_distribution": [{"k": k, "probability": p} for k, p in result.block_distribution],
        "time_to_first_block_days": {
            "p10": result.time_to_first_block_days["p10"],
            "p50": result.time_to_first_block_days["p50"],
            "p90": result.time_to_first_block_days["p90"],
            "mean": result.time_to_first_block_days["mean"],
        },
    }

def _compute_curve_points(
    *,
    base_lambda_per_day: float,
    max_days: int,
    interval_days: int,
    drift_model,
) -> List[Dict[str, Any]]:
    """
    Compute a probability curve over horizons 1..max_days at interval_days steps.

    Returns a list of points:
      {day, expected_blocks, probability_at_least_one, probability_zero}
    """
    if max_days <= 0:
        raise ValueError("max_days must be > 0")
    if interval_days <= 0:
        raise ValueError("interval_days must be > 0")

    points: List[Dict[str, Any]] = []
    for horizon in range(interval_days, max_days + 1, interval_days):
        segments = make_segments(
            base_lambda_per_day=base_lambda_per_day,
            horizon_days=horizon,
            drift=drift_model,
        )
        res = analyze_segments(segments)
        points.append(
            {
                "day": horizon,
                "expected_blocks": res.expected_blocks,
                "probability_at_least_one": res.probability_at_least_one,
                "probability_zero": res.probability_zero_blocks,
            }
        )
    return points


def _render_text(coin: str, hashrate_display: str, days: int, report: Dict[str, Any]) -> str:
    analytic = report["analytic"]
    snap = report["network_snapshot"]
    lines = [
        f"Solo Odds ({coin.upper()})",
        f"Hashrate: {hashrate_display}",
        f"Horizon: {days} days",
        "",
        f"Network snapshot @ {snap['timestamp']}",
        f"  network_hashrate: {snap['network_hashrate_hs']:.6g} H/s",
        f"  difficulty:       {snap['difficulty']:.6g}",
        f"  blocks/day:       {snap['blocks_per_day']:.6g}",
        f"  block_reward:     {snap['block_reward']:.6g}",
        f"  source:           {snap.get('source') or ''}",
        f"  source_url:       {snap.get('source_url') or ''}",
        "",
        "Analytic (Poisson)",
        f"  lambda/day:       {analytic['lambda_per_day']:.6g}",
        f"  expected blocks:  {analytic['expected_blocks']:.6g}",
        f"  P(>=1 block):     {analytic['probability_at_least_one']:.6g}",
        f"  P(0 blocks):      {analytic['probability_zero_blocks']:.6g}",
        "",
        "Time to first block (days) [effective-rate approximation]",
        f"  p10: {analytic['time_to_first_block_days']['p10']:.6g}",
        f"  p50: {analytic['time_to_first_block_days']['p50']:.6g}",
        f"  p90: {analytic['time_to_first_block_days']['p90']:.6g}",
        f"  mean:{analytic['time_to_first_block_days']['mean']:.6g}",
    ]
    return "\n".join(lines)


@app.command()
def refresh(
    coin: str = typer.Option(..., help="btc or bch"),
    timeout_s: float = typer.Option(10.0, help="HTTP timeout (seconds)"),
) -> None:
    """
    Refresh cached network snapshot and write data/<coin>/latest.json
    """
    coin_norm = coin.strip().lower()
    if coin_norm not in ("btc", "bch"):
        typer.echo("coin must be 'btc' or 'bch'", err=True)
        raise typer.Exit(code=2)

    store = SnapshotStore.from_repo_root()

    try:
        snap = fetch_snapshot(coin_norm, timeout_s=timeout_s)
    except FetchError as exc:
        typer.echo(f"Refresh failed: {exc}", err=True)
        raise typer.Exit(code=2)

    store.write_latest(snap)
    typer.echo(f"Wrote data/{coin_norm}/latest.json @ {snap.timestamp.isoformat()}")


@app.command()
def odds(
    coin: str = typer.Option(..., help="btc or bch"),
    hashrate: str = typer.Option(..., help="e.g. 9.4TH, 1200 GH/s, 9.4e12"),
    days: int = typer.Option(..., help="Horizon in days"),
    drift: str = typer.Option("flat", help="flat|step|linear"),
    step_pct: float = typer.Option(2.0, help="Step drift percent per step (e.g. 2 for +2%)"),
    step_days: int = typer.Option(14, help="Days per step for step drift"),
    daily_pct: float = typer.Option(0.0, help="Daily drift percent for linear drift (e.g. 0.1)"),
    mc: int = typer.Option(0,help="Monte Carlo runs (0 disables)"),
    json: bool = typer.Option(False, help="Emit JSON output"),
) -> None:
    """
    Compute solo mining odds and variance-friendly analytics.

    """
    coin_norm = coin.strip().lower()
    if coin_norm not in ("btc", "bch"):
        typer.echo("coin must be 'btc' or 'bch'", err=True)
        raise typer.Exit(code=2)

    if days <= 0:
        typer.echo("days must be > 0", err=True)
        raise typer.Exit(code=2)

    try:
        hr = parse_hashrate(hashrate)
    except UnitParseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    store = SnapshotStore.from_repo_root()
    try:
        snapshot = store.read_latest(coin_norm)
    except SnapshotStoreError as exc:
        typer.echo(
            f"{exc}\n\nTip: run `solo-odds refresh --coin {coin_norm}` once fetch is implemented, "
            "or place a valid latest.json under data/<coin>/latest.json.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        base_lambda = compute_lambda_per_day(
            your_hashrate_hs=hr.hs,
            network_hashrate_hs=snapshot.network_hashrate_hs,
            blocks_per_day=snapshot.blocks_per_day,
        )
    except AnalyticError as exc:
        typer.echo(f"Error computing base rate: {exc}", err=True)
        raise typer.Exit(code=2)

    try:
        drift_model = drift_model_from_cli(
            drift_type=drift,
            step_pct=step_pct,
            step_days=step_days,
            daily_pct=daily_pct,
        )
        segments = make_segments(
            base_lambda_per_day=base_lambda,
            horizon_days=days,
            drift=drift_model,
        )
    except (DriftError, AnalyticError) as exc:
        typer.echo(f"Drift model error: {exc}", err=True)
        raise typer.Exit(code=2)

    try:
        analytic_res = analyze_segments(segments)
    except AnalyticError as exc:
        typer.echo(f"Analytic error: {exc}", err=True)
        raise typer.Exit(code=2)

    mc_block: Dict[str, Any] = {"enabled": False}
    if mc and mc > 0:
        try:
            mc_res = run_monte_carlo(segments=segments, runs=int(mc))
        except MonteCarloError as exc:
            typer.echo(f"Monte Carlo error: {exc}", err=True)
            raise typer.Exit(code=2)

        mc_block = {
            "enabled": True,
            "runs": mc_res.runs,
            "blocks_over_horizon": mc_res.blocks_over_horizon,
            "time_to_first_block_days": mc_res.time_to_first_block_days,
        }

    report: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _now_utc_iso(),
        "input": {
            "coin": coin_norm,
            "hashrate_hs": hr.hs,
            "hashrate_display": hr.format(),
            "horizon_days": days,
            "drift_model": {
                "type": drift_model.type,
                "parameters": {
                    "step_pct": drift_model.step_pct if drift_model.type == "step" else None,
                    "step_days": drift_model.step_days if drift_model.type == "step" else None,
                    "daily_pct": drift_model.daily_pct if drift_model.type == "linear" else None,
                },
            },
            "monte_carlo_runs": int(mc),
        },
        "network_snapshot": _network_snapshot_to_report(snapshot),
        "analytic": _analytic_to_report(analytic_res),
        "monte_carlo": mc_block,
        "notes": [
            "Block counts use a Poisson model with integrated intensity over the horizon.",
            "Time-to-first-block uses an effective constant-rate approximation when drift is enabled.",
        ],
    }

    if json:
        typer.echo(_json.dumps(report, indent=2, sort_keys=True))
        return

    typer.echo(_render_text(coin_norm, hr.format(), days, report))


@app.command()
def curve(
    coin: str = typer.Option(..., help="btc or bch"),
    hashrate: str = typer.Option(..., help="e.g. 9.4TH, 1200 GH/s, 9.4e12"),
    days: int = typer.Option(..., help="Max horizon in days"),
    interval_days: int = typer.Option(1, help="Sample every N days (1 => daily)"),
    drift: str = typer.Option("flat", help="flat|step|linear"),
    step_pct: float = typer.Option(2.0, help="Step drift percent per step (e.g. 2 for +2%)"),
    step_days: int = typer.Option(14, help="Days per step for step drift"),
    daily_pct: float = typer.Option(0.0, help="Daily drift percent for linear drift (e.g. 0.1)"),
    json: bool = typer.Option(False, help="Emit JSON output"),
    csv: bool = typer.Option(False, help="Emit CSV output"),
) -> None:
    """
    Produce a probability curve vs time horizon.

    Outputs a series of points with:
      - day
      - expected_blocks (mu)
      - probability_at_least_one
      - probability_zero
    """
    coin_norm = coin.strip().lower()
    if coin_norm not in ("btc", "bch"):
        typer.echo("coin must be 'btc' or 'bch'", err=True)
        raise typer.Exit(code=2)

    if days <= 0:
        typer.echo("days must be > 0", err=True)
        raise typer.Exit(code=2)

    if interval_days <= 0:
        typer.echo("interval_days must be > 0", err=True)
        raise typer.Exit(code=2)

    if json and csv:
        typer.echo("Choose only one: --json or --csv", err=True)
        raise typer.Exit(code=2)

    try:
        hr = parse_hashrate(hashrate)
    except UnitParseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    store = SnapshotStore.from_repo_root()
    try:
        snapshot = store.read_latest(coin_norm)
    except SnapshotStoreError as exc:
        typer.echo(
            f"{exc}\n\nTip: run `solo-odds refresh --coin {coin_norm}` to populate latest.json.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        base_lambda = compute_lambda_per_day(
            your_hashrate_hs=hr.hs,
            network_hashrate_hs=snapshot.network_hashrate_hs,
            blocks_per_day=snapshot.blocks_per_day,
        )
    except AnalyticError as exc:
        typer.echo(f"Error computing base rate: {exc}", err=True)
        raise typer.Exit(code=2)

    try:
        drift_model = drift_model_from_cli(
            drift_type=drift,
            step_pct=step_pct,
            step_days=step_days,
            daily_pct=daily_pct,
        )
    except DriftError as exc:
        typer.echo(f"Drift model error: {exc}", err=True)
        raise typer.Exit(code=2)

    # Build curve points
    try:
        points = _compute_curve_points(
            base_lambda_per_day=base_lambda,
            max_days=days,
            interval_days=interval_days,
            drift_model=drift_model,
        )
    except AnalyticError as exc:
        typer.echo(f"Analytic error computing curve: {exc}", err=True)
        raise typer.Exit(code=2)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _now_utc_iso(),
        "input": {
            "coin": coin_norm,
            "hashrate_hs": hr.hs,
            "hashrate_display": hr.format(),
            "max_days": days,
            "interval_days": interval_days,
            "drift_model": {
                "type": drift_model.type,
                "parameters": {
                    "step_pct": drift_model.step_pct if drift_model.type == "step" else None,
                    "step_days": drift_model.step_days if drift_model.type == "step" else None,
                    "daily_pct": drift_model.daily_pct if drift_model.type == "linear" else None,
                },
            },
        },
        "network_snapshot": _network_snapshot_to_report(snapshot),
        "points": points,
        "notes": [
            "Each point is computed independently for horizon=day (not cumulative simulation).",
            "Counts use Poisson with integrated intensity over the horizon.",
        ],
    }

    if json:
        typer.echo(_json.dumps(payload, indent=2, sort_keys=True))
        return

    if csv:
        # Simple CSV for plotting
        typer.echo("day,expected_blocks,probability_at_least_one,probability_zero")
        for p in points:
            typer.echo(
                f"{p['day']},{p['expected_blocks']:.12g},"
                f"{p['probability_at_least_one']:.12g},{p['probability_zero']:.12g}"
            )
        return

    # Text table
    typer.echo(f"Curve ({coin_norm.upper()}) hashrate={hr.format()} max_days={days} interval={interval_days}d")
    typer.echo("day  expected_blocks  P(>=1)       P(0)")
    for p in points:
        typer.echo(
            f"{p['day']:>3}  {p['expected_blocks']:<14.6g}  "
            f"{p['probability_at_least_one']:<11.6g}  {p['probability_zero']:.6g}"
        )
        

@app.command()
def plot(
    coin: str = typer.Option(..., help="btc or bch"),
    hashrate: str = typer.Option(..., help="e.g. 9.4TH, 1200 GH/s, 9.4e12"),
    days: int = typer.Option(..., help="Max horizon in days"),
    interval_days: int = typer.Option(1, help="Sample every N days (1 => daily)"),
    drift: str = typer.Option("flat", help="flat|step|linear"),
    step_pct: float = typer.Option(2.0, help="Step drift percent per step (e.g. 2 for +2%)"),
    step_days: int = typer.Option(14, help="Days per step for step drift"),
    daily_pct: float = typer.Option(0.0, help="Daily drift percent for linear drift (e.g. 0.1)"),
    y: str = typer.Option("p", help="y-axis: 'p' for P(>=1), 'mu' for expected blocks"),
    log_y: bool = typer.Option(False, help="Log-scale Y axis (only valid with --y mu)"),
    min_y: float = typer.Option(1e-12, help="Minimum Y value when using --log-y (mu only)"),
    out: Path = typer.Option(Path("curve.png"), help="Output PNG path"),
    title: Optional[str] = typer.Option(None, help="Override plot title"),
) -> None:
    """
    Plot a curve to a PNG file.

    y:
      - p  => probability_at_least_one
      - mu => expected_blocks
    """
    # Import matplotlib lazily so CLI works without it unless plot is used.
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
    except Exception as exc:
        typer.echo(f"matplotlib is required for plot: {exc}", err=True)
        raise typer.Exit(code=2)

    coin_norm = coin.strip().lower()
    if coin_norm not in ("btc", "bch"):
        typer.echo("coin must be 'btc' or 'bch'", err=True)
        raise typer.Exit(code=2)

    if days <= 0:
        typer.echo("days must be > 0", err=True)
        raise typer.Exit(code=2)

    if interval_days <= 0:
        typer.echo("interval_days must be > 0", err=True)
        raise typer.Exit(code=2)

    y_mode = (y or "").strip().lower()
    if y_mode not in ("p", "mu"):
        typer.echo("y must be 'p' or 'mu'", err=True)
        raise typer.Exit(code=2)

    if log_y and y_mode != "mu":
        typer.echo("--log-y is only valid with --y mu", err=True)
        raise typer.Exit(code=2)

    if log_y:
        if min_y <= 0:
            typer.echo("--min-y must be > 0 when using --log-y", err=True)
            raise typer.Exit(code=2)

    try:
        hr = parse_hashrate(hashrate)
    except UnitParseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    store = SnapshotStore.from_repo_root()
    try:
        snapshot = store.read_latest(coin_norm)
    except SnapshotStoreError as exc:
        typer.echo(
            f"{exc}\n\nTip: run `solo-odds refresh --coin {coin_norm}` to populate latest.json.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        base_lambda = compute_lambda_per_day(
            your_hashrate_hs=hr.hs,
            network_hashrate_hs=snapshot.network_hashrate_hs,
            blocks_per_day=snapshot.blocks_per_day,
        )
    except AnalyticError as exc:
        typer.echo(f"Error computing base rate: {exc}", err=True)
        raise typer.Exit(code=2)

    try:
        drift_model = drift_model_from_cli(
            drift_type=drift,
            step_pct=step_pct,
            step_days=step_days,
            daily_pct=daily_pct,
        )
    except DriftError as exc:
        typer.echo(f"Drift model error: {exc}", err=True)
        raise typer.Exit(code=2)

    # Build curve points
    try:
        points = _compute_curve_points(
            base_lambda_per_day=base_lambda,
            max_days=days,
            interval_days=interval_days,
            drift_model=drift_model,
        )
    except AnalyticError as exc:
        typer.echo(f"Analytic error computing curve: {exc}", err=True)
        raise typer.Exit(code=2)

    xs: List[int] = [int(p["day"]) for p in points]
    if y_mode == "p":
        ys: List[float] = [float(p["probability_at_least_one"]) for p in points]
    else:
        ys = [float(p["expected_blocks"]) for p in points]

    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    if y_mode == "p":
        ax.set_ylabel("P(>=1 block)")
        ax.set_ylim(0.0, 1.0)
        default_title = f"Solo mining probability ({coin_norm.upper()})"
    else:
        ax.set_ylabel("Expected blocks (mu)")
        default_title = f"Solo mining expected blocks ({coin_norm.upper()})"
        if log_y:
            ax.set_yscale("log")
            # Clamp values to avoid log(0)
            ys = [v if v > min_y else min_y for v in ys]

    ax.plot(xs, ys)

    ax.set_xlabel("Days")

    subtitle = f"hashrate={hr.format()} drift={drift_model.type} interval={interval_days}d"
    ax.set_title(title or f"{default_title}\n{subtitle}")

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    typer.echo(f"Wrote {out}")


if __name__ == "__main__":
    app()