from __future__ import annotations

import base64
import logging
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, ConfigDict

from solo_odds.data.store import SnapshotStore, SnapshotStoreError
from solo_odds.data.models import NetworkSnapshot
from solo_odds.math.analytic import AnalyticError, analyze_segments, compute_lambda_per_day
from solo_odds.math.drift import DriftError, drift_model_from_cli, make_segments
from solo_odds.math.economic_compare import compare_solo_vs_pool, EconomicCompareError
from solo_odds.sim.monte_carlo import MonteCarloError, run_monte_carlo, run_reinvest_curve
from solo_odds.units import UnitParseError, parse_hashrate

app = FastAPI(title='solo-odds web')
logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory='web/templates')
app.mount('/static', StaticFiles(directory='web/static'), name='static')


def _seed_from_token(token: str) -> int:
    # Stable 32-bit seed derived from token text
    digest = hashlib.sha256(token.encode('utf-8')).digest()
    return int.from_bytes(digest[:4], 'big', signed=False)


def _repo_root() -> Path:
    # WorkingDirectory for systemd should be repo root; this is a safe fallback.
    return Path(__file__).resolve().parents[1]


def _waitlist_path() -> Path:
    # Optional override via env; default to <repo>/var/waitlist.txt
    p = os.environ.get('SOLO_ODDS_WAITLIST_PATH')
    if p:
        return Path(p)
    return _repo_root() / 'var' / 'waitlist.txt'


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _metrics_path() -> Path:
    # Optional override via env; default to <repo>/var/metrics.jsonl
    p = os.environ.get('SOLO_ODDS_METRICS_PATH')
    if p:
        return Path(p)
    return _repo_root() / 'var' / 'metrics.jsonl'


class ShareParams(BaseModel):
    coin: str = Field(pattern='^(btc|bch)$')
    hashrate: str = Field(min_length=1)  # e.g. "9.4TH"
    days: int = Field(ge=1, le=3650)     # cap to avoid abuse
    interval_days: int = Field(ge=1, le=365)
    drift: str = Field(pattern='^(flat|step|linear)$')
    step_pct: float = 2.0
    step_days: int = 14
    daily_pct: float = 0.0
    mc: int = Field(ge=0, le=200000)    # cap MC to avoid abuse
    reinvest: bool = False
    reinvest_multiplier: float = Field(2.0, ge=1.0, le=20.0)


class TokenPayloadV2(BaseModel):
    v: int = Field(2)
    params: ShareParams
    # Store full snapshot dict (includes 'coin'); we validate via NetworkSnapshot.from_dict().
    snapshot: Dict[str, Any]


class ShareRequest(BaseModel):
    # JSON equivalent of the /share form
    model_config = ConfigDict(extra='forbid')
    coin: str = Field(pattern='^(btc|bch)$')
    hashrate: str = Field(min_length=1)
    days: int = Field(ge=1, le=3650)
    interval_days: int = Field(ge=1, le=365)
    drift: str = Field(pattern='^(flat|step|linear)$')
    step_pct: float = 2.0
    step_days: int = 14
    daily_pct: float = 0.0
    mc: int = Field(ge=0, le=200000)
    reinvest: bool = False
    reinvest_multiplier: float = Field(2.0, ge=1.0, le=20.0)


class CompareParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    coin: str = Field(pattern='^(btc|bch)$')
    hashrate: str = Field(min_length=1)
    horizon_days: int = Field(ge=1, le=3650)
    coin_price_usd: float = Field(gt=0)
    electricity_cost_per_kwh: float = Field(ge=0)
    asic_power_watts: float = Field(ge=0)
    pool_fee_pct: float = Field(ge=0, le=0.25)
    mc_runs: int = Field(0, ge=0, le=200000)


class CompareTokenPayloadV1(BaseModel):
    v: int = Field(1)
    params: CompareParams
    snapshot: Dict[str, Any]
    seed: int = Field(ge=0, le=2**32 - 1)


class CompareRequest(BaseModel):
    # For /api/v1/compare (non-tokenized endpoint)
    model_config = ConfigDict(extra='forbid')
    coin: str = Field(pattern='^(btc|bch)$')
    hashrate: str = Field(min_length=1)
    horizon_days: int = Field(ge=1, le=3650)
    coin_price_usd: float = Field(gt=0)
    electricity_cost_per_kwh: float = Field(ge=0)
    asic_power_watts: float = Field(ge=0)
    pool_fee_pct: float = Field(ge=0, le=0.25)
    mc_runs: int = Field(ge=0, le=200000)


class CompareHeatmapRequest(BaseModel):
    """
    Heatmap request for compare form.
    We intentionally keep it deterministic (no MC).
    """
    model_config = ConfigDict(extra='forbid')
    coin: str = Field(pattern='^(btc|bch)$')
    hashrate: str = Field(min_length=1)
    horizon_days: int = Field(ge=1, le=3650)
    coin_price_usd: float = Field(gt=0)
    electricity_cost_per_kwh: float = Field(ge=0)
    asic_power_watts: float = Field(ge=0)
    pool_fee_pct: float = Field(ge=0, le=0.25)

    # grid config (keep small to avoid abuse)
    price_span_pct: float = Field(0.5, ge=0.05, le=0.9)     # +/- 50% default
    elec_span_pct: float = Field(0.5, ge=0.05, le=0.9)      # +/- 50% default
    price_steps: int = Field(15, ge=5, le=31)
    elec_steps: int = Field(15, ge=5, le=31)


def _append_metrics(event: Dict[str, Any]) -> None:
    """
    Append one JSONL event to disk.
    Never throw back to the user if logging fails.
    """
    try:
        path = _metrics_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, separators=(',', ':'), sort_keys=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(line + '\n')
    except Exception:
        # Intentionally swallow: metrics must not break the app.
        return


def _request_meta(request: Request) -> Dict[str, Any]:
    """
    Minimal request metadata, best-effort. Useful for referrers/shares.
    """
    ip = None
    try:
        if request.client:
            ip = request.client.host
    except Exception:
        ip = None

    # If you’re behind nginx, you may want X-Forwarded-For; include it too.
    xff = request.headers.get('x-forwarded-for')
    ua = request.headers.get('user-agent')
    ref = request.headers.get('referer')

    return {
        'ip': ip,
        'xff': xff,
        'ua': ua,
        'ref': ref,
    }


def _token_fingerprint(token: str) -> str:
    # Avoid logging gigantic tokens raw. Stable fingerprint for joins.
    return hashlib.sha256(token.encode('utf-8')).hexdigest()[:16]


def _secret() -> bytes:
    s = os.environ.get('SOLO_ODDS_SECRET')
    if not s or len(s) < 16:
        raise RuntimeError('SOLO_ODDS_SECRET must be set (>=16 chars)')
    return s.encode('utf-8')


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')


def _b64url_decode(s: str) -> bytes:
    pad = '=' * ((4 - (len(s) % 4)) % 4)
    return base64.urlsafe_b64decode((s + pad).encode('ascii'))


def make_token(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, separators=(',', ':'), sort_keys=True).encode('utf-8')
    sig = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return f'{_b64url_encode(payload)}.{_b64url_encode(sig)}'


def make_token_from_obj(obj: Dict[str, Any]) -> str:
    # identical signing, explicit name so we can reuse for compare payloads too
    return make_token(obj)


def _normalize_coin(coin: str) -> str:
    return (coin or '').strip().lower()


def _normalize_hashrate_str(hashrate: str) -> str:
    # parse_hashrate likely handles case/spacing already, but normalize anyway
    return (hashrate or '').strip().replace(' ', '').upper()


def parse_token(token: str) -> Dict[str, Any]:
    try:
        payload_b64, sig_b64 = token.split('.', 1)
        payload = _b64url_decode(payload_b64)
        sig = _b64url_decode(sig_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid token format') from exc

    expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        raise HTTPException(status_code=400, detail='Invalid token signature')

    try:
        return json.loads(payload.decode('utf-8'))
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid token payload') from exc


def _parse_token_payload(raw: Dict[str, Any]) -> tuple[ShareParams, Optional[NetworkSnapshot]]:
    """
    Supports:
      - v1 tokens: payload is ShareParams dict (no snapshot; uses latest.json at render time)
      - v2 tokens: {"v":2,"params":{...},"snapshot":{...}}
    Returns (params, snapshot_or_none).
    """
    # v2
    if isinstance(raw, dict) and raw.get('v') == 2:
        try:
            payload = TokenPayloadV2(**raw)
        except Exception as exc:
            raise HTTPException(status_code=400, detail='Invalid token payload (v2)') from exc

        try:
            snap = NetworkSnapshot.from_dict(payload.snapshot)
        except Exception as exc:
            raise HTTPException(status_code=400, detail='Invalid snapshot embedded in token') from exc

        # Defensive: ensure snapshot coin matches params.coin
        if snap.coin != payload.params.coin:
            raise HTTPException(status_code=400, detail='Token snapshot coin mismatch')

        return payload.params, snap

    # v1 fallback: treat entire dict as ShareParams
    try:
        params = ShareParams(**raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid token payload (v1)') from exc
    return params, None


def _parse_compare_token_payload(raw: Dict[str, Any]) -> tuple[CompareParams, NetworkSnapshot, int]:
    if not isinstance(raw, dict) or raw.get('v') != 1:
        raise HTTPException(status_code=400, detail='Invalid compare token payload')

    try:
        payload = CompareTokenPayloadV1(**raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid compare token payload (v1)') from exc

    try:
        snap = NetworkSnapshot.from_dict(payload.snapshot)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid snapshot embedded in compare token') from exc

    if snap.coin != payload.params.coin:
        raise HTTPException(status_code=400, detail='Compare token snapshot coin mismatch')

    return payload.params, snap, int(payload.seed)


def _seed_for_payload(payload: Dict[str, Any]) -> int:
    # deterministic seed so MC outputs don’t change between refreshes
    b = json.dumps(payload, separators=(',', ':'), sort_keys=True).encode('utf-8')
    h = hashlib.sha256(b).digest()
    return int.from_bytes(h[:4], 'big', signed=False)


def _histogram_from_points(values: List[float], weights: List[float], *, bin_width: float) -> Dict[str, Any]:
    if not values:
        return {'bin_width': bin_width, 'bins': [], 'prob': []}
    vmin = min(values)
    vmax = max(values)

    # pad so 0 aligns nicely if possible
    start = math.floor(vmin / bin_width) * bin_width
    end = math.ceil(vmax / bin_width) * bin_width
    nbins = int(round((end - start) / bin_width))
    if nbins < 1:
        nbins = 1

    probs = [0.0] * nbins
    for v, w in zip(values, weights):
        idx = int((v - start) // bin_width)
        if idx < 0:
            idx = 0
        if idx >= nbins:
            idx = nbins - 1
        probs[idx] += float(w)

    bins = [start + i * bin_width for i in range(nbins)]
    # bins represent left edge; prob is probability mass in that bin
    return {'bin_width': float(bin_width), 'bins': bins, 'prob': probs}


def _compute_compare(params: CompareParams, snapshot: NetworkSnapshot, seed: int) -> Dict[str, Any]:
    """
    Compare math for tokenized compare pages.

    Requirements:
      - Use `snapshot` (frozen) rather than latest.json
      - Deterministic output per token (seed available if you later add MC)
    """
    hr = parse_hashrate(params.hashrate)

    lambda_per_day = compute_lambda_per_day(
        your_hashrate_hs=hr.hs,
        network_hashrate_hs=snapshot.network_hashrate_hs,
        blocks_per_day=snapshot.blocks_per_day,
    )
    mu = float(lambda_per_day) * float(params.horizon_days)

    # Deterministic compare (no MC for v1 compare page; mc_runs kept for future)
    res = compare_solo_vs_pool(
        mu=mu,
        block_reward=float(snapshot.block_reward),
        coin_price_usd=float(params.coin_price_usd),
        horizon_days=float(params.horizon_days),
        asic_power_watts=float(params.asic_power_watts),
        electricity_cost_per_kwh=float(params.electricity_cost_per_kwh),
        pool_fee_pct=float(params.pool_fee_pct),
    )

    dist = res.solo_distribution
    weights = [p.probability for p in dist]
    solo_net_values = [p.net_usd for p in dist]
    pool_net = float(res.pool['expected_net_usd'])
    delta_values = [p.net_usd - pool_net for p in dist]

    # Choose bin width heuristically: $50 for small ranges, else $100/$250
    span = max(solo_net_values) - min(solo_net_values) if solo_net_values else 0.0
    if span <= 2000:
        bin_width = 50.0
    elif span <= 10000:
        bin_width = 100.0
    else:
        bin_width = 250.0

    solo_hist = _histogram_from_points(solo_net_values, weights, bin_width=bin_width)
    delta_hist = _histogram_from_points(delta_values, weights, bin_width=bin_width)

    return {
        'schema_version': 1,
        'generated_at': _now_utc_iso(),
        'input': {
            'coin': params.coin,
            'hashrate_hs': hr.hs,
            'hashrate_display': hr.format(),
            'horizon_days': params.horizon_days,
            'coin_price_usd': params.coin_price_usd,
            'electricity_cost_per_kwh': params.electricity_cost_per_kwh,
            'asic_power_watts': params.asic_power_watts,
            'pool_fee_pct': params.pool_fee_pct,
            'mc_runs': params.mc_runs,
        },
        'network_snapshot': _network_snapshot_to_report(snapshot),
        'mu': res.mu,
        'revenue_per_block_usd': res.revenue_per_block_usd,
        'electricity_cost_usd': res.electricity_cost_usd,
        'solo': res.solo,
        'pool': res.pool,
        'comparison': res.comparison,
        'distributions': {
            'solo_net_usd': solo_hist,
            'delta_vs_pool_usd': delta_hist,
            'pool_expected_net_usd': pool_net,
            'solo_expected_net_usd': float(res.solo['expected_net_usd']),
        },
        'notes': [
            'Pool is modeled as expected value (deterministic) for v1 compare.',
            'Solo is modeled as Poisson variance on blocks over the horizon.',
            'Snapshot is frozen into the token for stable shared results.',
        ],
    }


def _network_snapshot_to_report(snapshot) -> Dict[str, Any]:
    d = snapshot.to_dict()
    # IMPORTANT: your schema disallows 'coin' here
    return {
        'timestamp': d['timestamp'],
        'network_hashrate_hs': d['network_hashrate_hs'],
        'difficulty': d['difficulty'],
        'blocks_per_day': d['blocks_per_day'],
        'block_reward': d['block_reward'],
        'source': d.get('source'),
        'source_url': d.get('source_url'),
    }


def _poisson_quantile(mu: float, q: float) -> int:
    """
    Small, dependency-free Poisson quantile (inverse CDF).

    Returns the smallest integer k such that P(X <= k) >= q for X ~ Poisson(mu).

    Uses stable recurrence:
      p0 = exp(-mu)
      p_{k+1} = p_k * mu / (k+1)

    This is plenty fast for typical home-mining mu and horizons.
    """
    if not math.isfinite(mu) or mu < 0:
        raise ValueError(f'Invalid mu={mu!r}')
    if q <= 0:
        return 0
    if q >= 1:
        # "infinite" quantile isn't meaningful; return a safe upper-ish bound
        if mu == 0:
            return 0
        return int(math.ceil(mu + 10.0 * math.sqrt(mu) + 50.0))
    if mu == 0:
        return 0

    # Start at k=0
    p = math.exp(-mu)
    cdf = p
    k = 0

    # Set a conservative max to avoid pathological loops if q is extreme.
    # For typical mu, this exits quickly.
    k_max = int(math.ceil(mu + 12.0 * math.sqrt(mu + 1.0) + 200.0))

    while cdf < q and k < k_max:
        k += 1
        p = p * mu / k
        cdf += p

    return k


def _compute_curve_points(
    *,
    base_lambda_per_day: float,
    max_days: int,
    interval_days: int,
    drift_model,
) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for horizon in range(interval_days, max_days + 1, interval_days):
        segments = make_segments(
            base_lambda_per_day=base_lambda_per_day,
            horizon_days=horizon,
            drift=drift_model,
        )
        res = analyze_segments(segments)

        mu = float(res.expected_blocks)
        p10_blocks = _poisson_quantile(mu, 0.10)
        p50_blocks = _poisson_quantile(mu, 0.50)
        p90_blocks = _poisson_quantile(mu, 0.90)

        points.append(
            {
                'day': horizon,
                'expected_blocks': res.expected_blocks,
                'probability_at_least_one': res.probability_at_least_one,
                'probability_zero': res.probability_zero_blocks,
                'blocks_p10': p10_blocks,
                'blocks_p50': p50_blocks,
                'blocks_p90': p90_blocks,
            }
        )
    return points


def build_report(
    params: ShareParams,
    snapshot: Optional[NetworkSnapshot] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    # Parse hashrate
    hr = parse_hashrate(params.hashrate)

    if snapshot is None:
        store = SnapshotStore.from_repo_root()
        snapshot = store.read_latest(params.coin)

    base_lambda = compute_lambda_per_day(
        your_hashrate_hs=hr.hs,
        network_hashrate_hs=snapshot.network_hashrate_hs,
        blocks_per_day=snapshot.blocks_per_day,
    )

    drift_model = drift_model_from_cli(
        drift_type=params.drift,
        step_pct=params.step_pct,
        step_days=params.step_days,
        daily_pct=params.daily_pct,
    )

    segments = make_segments(
        base_lambda_per_day=base_lambda,
        horizon_days=params.days,
        drift=drift_model,
    )
    analytic_res = analyze_segments(segments)

    # Time-to-first-block (mean/median) under effective constant-rate approximation.
    # This matches the semantics of your existing analytic time_to_first_block_days block.
    effective_lambda = float(analytic_res.lambda_per_day_effective)
    if effective_lambda > 0.0:
        t1_mean_days = 1.0 / effective_lambda
        t1_median_days = math.log(2.0) / effective_lambda
    else:
        t1_mean_days = math.inf
        t1_median_days = math.inf

    report: Dict[str, Any] = {
        'schema_version': 1,
        'generated_at': _now_utc_iso(),
        'input': {
            'coin': params.coin,
            'hashrate_hs': hr.hs,
            'hashrate_display': hr.format(),
            'horizon_days': params.days,
            'drift_model': {
                'type': drift_model.type,
                'parameters': {
                    'step_pct': drift_model.step_pct if drift_model.type == 'step' else None,
                    'step_days': drift_model.step_days if drift_model.type == 'step' else None,
                    'daily_pct': drift_model.daily_pct if drift_model.type == 'linear' else None,
                },
            },
            'monte_carlo_runs': int(params.mc),
            'reinvest': bool(params.reinvest),
            'reinvest_multiplier': float(params.reinvest_multiplier),
        },
        'network_snapshot': _network_snapshot_to_report(snapshot),
        'analytic': {
            'lambda_per_day': analytic_res.lambda_per_day_effective,
            'expected_blocks': analytic_res.expected_blocks,
            'probability_at_least_one': analytic_res.probability_at_least_one,
            'probability_zero_blocks': analytic_res.probability_zero_blocks,
            'block_distribution': [
                {'k': k, 'probability': p} for k, p in analytic_res.block_distribution
            ],
            'time_to_first_block_days': analytic_res.time_to_first_block_days,
            # Explicit mean/median callouts (more intuitive than "mean only")
            'time_to_first_block_mean_days': t1_mean_days,
            'time_to_first_block_median_days': t1_median_days,
        },
        # Structured summary block for downstream consumers
        'summary': {
            'time_to_first_block': {
                'median_days': t1_median_days,
                'mean_days': t1_mean_days,
                'interpretation': {
                    'median_probability': 0.5,
                    'description': (
                        'Median time represents the 50% probability threshold '
                        'for first block discovery under the effective-rate model.'
                    ),
                },
            },
        },
        'monte_carlo': {'enabled': False},
        'notes': [
            'Counts use a Poisson model with integrated intensity over the horizon.',
            'Time-to-first-block uses an effective constant-rate approximation when drift is enabled.',
        ],
    }

    mc_runs = int(params.mc)
    if params.mc and params.mc > 0:
        mc_res = run_monte_carlo(segments=segments, runs=mc_runs, seed=seed)
        report['monte_carlo'] = {
            'enabled': True,
            'runs': mc_res.runs,
            'blocks_over_horizon': mc_res.blocks_over_horizon,
            'time_to_first_block_days': mc_res.time_to_first_block_days,
        }

    # Add curve for charting (separate payload; not part of your report.schema.json)
    curve = _compute_curve_points(
        base_lambda_per_day=base_lambda,
        max_days=params.days,
        interval_days=params.interval_days,
        drift_model=drift_model,
    )
    report['_curve'] = curve

    # Reinvestment curve overlay (MC only)
    # We simulate the full horizon once and derive all cutoffs from that run.
    if params.reinvest:
        # If user didn't request MC, force a sane default so the feature works.
        runs_for_reinvest = mc_runs
        if runs_for_reinvest <= 0:
            runs_for_reinvest = 20000
        # Respect global cap (ShareParams caps mc, but reinvest can be enabled with mc==0)
        if runs_for_reinvest > 200000:
            runs_for_reinvest = 200000

        cutoffs = list(range(params.interval_days, params.days + 1, params.interval_days))
        reinvest_res = run_reinvest_curve(
            segments=segments,
            cutoffs_days=cutoffs,
            runs=runs_for_reinvest,
            multiplier=float(params.reinvest_multiplier),
            seed=seed,
        )
        report['_reinvest'] = {
            'enabled': True,
            'multiplier': reinvest_res.multiplier,
            'runs': reinvest_res.runs,
            'time_to_first_block_days': reinvest_res.time_to_first_block_days,
            'points': reinvest_res.points,
            'notes': [
                'Reinvestment assumes hashrate increases immediately after the first block is found.',
                'Reinvestment curve is Monte Carlo and may vary slightly run-to-run.',
            ],
        }

    return report


@app.get('/', response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        'index.html',
        {
            'request': request,
            'defaults': {
                'coin': 'bch',
                'hashrate': '9.4TH',
                'days': 365,
                'interval_days': 7,
                'drift': 'flat',
                'step_pct': 2.0,
                'step_days': 14,
                'daily_pct': 0.0,
                'mc': 0,
                'reinvest': False,
                'reinvest_multiplier': 2.0,
            },
        },
    )


@app.get('/methods', response_class=HTMLResponse)
def methods(request: Request) -> HTMLResponse:
    return templates.TemplateResponse('methods.html', {'request': request})


@app.get('/blog/solo-vs-pool-risk-first', response_class=HTMLResponse)
def methods(request: Request) -> HTMLResponse:
    return templates.TemplateResponse('solo-vs-pool-risk-first.html', {'request': request})


@app.get('/compare', response_class=HTMLResponse)
def compare_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        'compare_form.html',
        {
            'request': request,
            'defaults': {
                'coin': 'bch',
                'hashrate': '9.4TH',
                'horizon_days': 365,
                'coin_price_usd': 250.0,
                'electricity_cost_per_kwh': 0.12,
                'asic_power_watts': 3000,
                'pool_fee_pct': 0.01,
                'mc_runs': 0,
            },
        },
    )


@app.post('/waitlist')
def waitlist(
    request: Request,
    email: str = Form(...),
    company: str = Form('', alias='company'),  # honeypot field: should be empty
) -> RedirectResponse:
    # Honeypot: bots fill hidden fields
    if company and company.strip():
        return RedirectResponse(url='/?ok=1', status_code=303)

    e = (email or '').strip()
    if not e or len(e) > 254 or '@' not in e:
        return RedirectResponse(url='/?ok=0', status_code=303)

    ts = _now_utc_iso()
    path = _waitlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # One line per signup: timestamp<TAB>email
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'{ts}\t{e}\n')

    return RedirectResponse(url='/?ok=1', status_code=303)


@app.get('/robots.txt')
def robots() -> HTMLResponse:
    # Keep token pages out of search engines.
    body = "User-agent: *\nDisallow: /r/\nDisallow: /c/\nDisallow: /api/\n"
    return HTMLResponse(content=body, media_type='text/plain')


@app.post('/share')
def share(
    request: Request,
    coin: str = Form(...),
    hashrate: str = Form(...),
    days: int = Form(...),
    interval_days: int = Form(...),
    drift: str = Form(...),
    step_pct: float = Form(2.0),
    step_days: int = Form(14),
    daily_pct: float = Form(0.0),
    mc: int = Form(0),
    reinvest: str = Form(''),  # checkbox => "on" when checked
    reinvest_multiplier: float = Form(2.0),
) -> RedirectResponse:
    params = ShareParams(
        coin=coin.strip().lower(),
        hashrate=hashrate.strip(),
        days=int(days),
        interval_days=int(interval_days),
        drift=drift.strip().lower(),
        step_pct=float(step_pct),
        step_days=int(step_days),
        daily_pct=float(daily_pct),
        mc=int(mc),
        reinvest=(str(reinvest).strip().lower() in ('1', 'true', 'on', 'yes')),
        reinvest_multiplier=float(reinvest_multiplier),
    )

    meta = _request_meta(request)

    # Freeze snapshot into token (v2)
    store = SnapshotStore.from_repo_root()
    snapshot = store.read_latest(params.coin)
    token_payload = {
        'v': 2,
        'params': params.model_dump(),
        # store full snapshot dict (includes coin)
        'snapshot': snapshot.to_dict(),
    }
    token = make_token(token_payload)

    # Metrics: share link generated
    _append_metrics(
        {
            'ts': _now_utc_iso(),
            'event': 'share_created',
            'token_fp': _token_fingerprint(token),
            'coin': params.coin,
            'hashrate': params.hashrate,
            'days': params.days,
            'interval_days': params.interval_days,
            'drift': params.drift,
            'step_pct': params.step_pct,
            'step_days': params.step_days,
            'daily_pct': params.daily_pct,
            'mc': params.mc,
            'reinvest': params.reinvest,
            'reinvest_multiplier': params.reinvest_multiplier,
            'meta': meta,
        }
    )

    return RedirectResponse(url=f'/r/{token}', status_code=303)

@app.post('/api/share')
async def api_share(request: Request) -> JSONResponse:
    """
    JSON variant of /share.
    Returns {token, url} so the landing page can copy/share without a redirect.
    """
    try:
        raw = await request.json()
        if isinstance(raw, dict):
            raw['coin'] = _normalize_coin(raw.get('coin'))
            raw['hashrate'] = _normalize_hashrate_str(raw.get('hashrate'))
        req = ShareRequest(**raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid share inputs') from exc

    params = ShareParams(
        coin=req.coin,
        hashrate=req.hashrate,
        days=int(req.days),
        interval_days=int(req.interval_days),
        drift=req.drift,
        step_pct=float(req.step_pct),
        step_days=int(req.step_days),
        daily_pct=float(req.daily_pct),
        mc=int(req.mc),
        reinvest=bool(req.reinvest),
        reinvest_multiplier=float(req.reinvest_multiplier),
    )

    store = SnapshotStore.from_repo_root()
    snapshot = store.read_latest(params.coin)

    token_payload = {
        'v': 2,
        'params': params.model_dump(),
        'snapshot': snapshot.to_dict(),
    }
    token = make_token(token_payload)

    _append_metrics(
        {
            'ts': _now_utc_iso(),
            'event': 'share_created',
            'token_fp': _token_fingerprint(token),
            'coin': params.coin,
            'hashrate': params.hashrate,
            'days': params.days,
            'interval_days': params.interval_days,
            'drift': params.drift,
            'step_pct': params.step_pct,
            'step_days': params.step_days,
            'daily_pct': params.daily_pct,
            'mc': params.mc,
            'reinvest': params.reinvest,
            'reinvest_multiplier': params.reinvest_multiplier,
            'meta': _request_meta(request),
        }
    )

    return JSONResponse({'token': token, 'url': f'/r/{token}'})

@app.post('/compare/share')
def compare_share(
    request: Request,
    coin: str = Form(...),
    hashrate: str = Form(...),
    horizon_days: int = Form(...),
    coin_price_usd: float = Form(...),
    electricity_cost_per_kwh: float = Form(...),
    asic_power_watts: float = Form(...),
    pool_fee_pct: float = Form(...),
    mc_runs: int = Form(0),
) -> RedirectResponse:
    """
    Accepts FORM CompareParams and redirects to a shareable compare token page.
    """
    params = CompareParams(
        coin=coin.strip().lower(),
        hashrate=_normalize_hashrate_str(hashrate),
        horizon_days=int(horizon_days),
        coin_price_usd=float(coin_price_usd),
        electricity_cost_per_kwh=float(electricity_cost_per_kwh),
        asic_power_watts=float(asic_power_watts),
        pool_fee_pct=float(pool_fee_pct),
        mc_runs=int(mc_runs),
    )

    store = SnapshotStore.from_repo_root()
    snapshot = store.read_latest(params.coin)

    token_payload = {
        'v': 1,
        'params': params.model_dump(),
        'snapshot': snapshot.to_dict(),
    }
    token_payload['seed'] = _seed_for_payload(token_payload)

    token = make_token_from_obj(token_payload)
    
    _append_metrics(
        {
            'ts': _now_utc_iso(),
            'event': 'compare_share_created',
            'token_fp': _token_fingerprint(token),
            'coin': params.coin,
            'hashrate': params.hashrate,
            'horizon_days': params.horizon_days,
            'meta': _request_meta(request),
        }
    )

    return RedirectResponse(url=f'/c/{token}', status_code=303)


@app.post('/api/compare/share')
async def api_compare_share(request: Request) -> JSONResponse:
    """
    JSON variant for programmatic callers.
    """
    try:
        raw = await request.json()
         # normalize raw inputs before pydantic validates patterns, etc.
        if isinstance(raw, dict):
            raw['coin'] = _normalize_coin(raw.get('coin'))
            raw['hashrate'] = _normalize_hashrate_str(raw.get('hashrate'))
        params = CompareParams(**raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail='Invalid compare inputs') from exc

    store = SnapshotStore.from_repo_root()
    snapshot = store.read_latest(params.coin)

    token_payload = {
        'v': 1,
        'params': params.model_dump(),
        'snapshot': snapshot.to_dict(),
    }
    token_payload['seed'] = _seed_for_payload(token_payload)
    token = make_token_from_obj(token_payload)
    return JSONResponse({'token': token, 'url': f'/c/{token}'})


@app.get('/c/{token}', response_class=HTMLResponse)
def compare_from_token(token: str, request: Request) -> HTMLResponse:
    raw = parse_token(token)
    params, snap, seed = _parse_compare_token_payload(raw)
    result = _compute_compare(params=params, snapshot=snap, seed=seed)

    _append_metrics(
        {
            'ts': _now_utc_iso(),
            'event': 'compare_view',
            'token_fp': _token_fingerprint(token),
            'coin': params.coin,
            'horizon_days': params.horizon_days,
            'meta': _request_meta(request),
        }
    )

    return templates.TemplateResponse(
        'compare_report.html',
        {
            'request': request,
            'token': token,
            'result': result,
        },
    )


@app.get('/api/compare/{token}')
def api_compare_token(token: str) -> JSONResponse:
    raw = parse_token(token)
    params, snap, seed = _parse_compare_token_payload(raw)
    payload = _compute_compare(params=params, snapshot=snap, seed=seed)
    return JSONResponse(payload)


@app.get('/r/{token}', response_class=HTMLResponse)
def render_report(token: str, request: Request) -> HTMLResponse:
    raw = parse_token(token)
    try:
        params, snap = _parse_token_payload(raw)
        seed = _seed_from_token(token)
        report = build_report(params, snapshot=snap, seed=seed)
    except (UnitParseError, AnalyticError, DriftError, MonteCarloError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Internal error') from exc

    # Metrics: report viewed
    _append_metrics(
        {
            'ts': _now_utc_iso(),
            'event': 'report_view',
            'token_fp': _token_fingerprint(token),
            'coin': params.coin,
            'days': params.days,
            'drift': params.drift,
            'mc': params.mc,
            'meta': _request_meta(request),
        }
    )

    curve = report.pop('_curve', [])
    reinvest = report.pop('_reinvest', None)
    # Provide share URLs
    api_url = f'/api/report/{token}'
    return templates.TemplateResponse(
        'report.html',
        {
            'request': request,
            'token': token,
            'api_url': api_url,
            'report': report,
            'curve': curve,
            'reinvest': reinvest,
        },
    )


@app.get('/api/report/{token}')
def api_report(token: str) -> JSONResponse:
    raw = parse_token(token)
    try:
        params, snap = _parse_token_payload(raw)
        seed = _seed_from_token(token)
        report = build_report(params, snapshot=snap, seed=seed)
        report.pop('_curve', None)
        report.pop('_reinvest', None)
        return JSONResponse(report)
    except (UnitParseError, AnalyticError, DriftError, MonteCarloError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise


@app.post('/api/v1/compare')
def api_compare(req: CompareRequest) -> JSONResponse:
    try:
        coin = _normalize_coin(req.coin)
        hashrate = _normalize_hashrate_str(req.hashrate)

        params = CompareParams(
            coin=coin,
            hashrate=hashrate,
            horizon_days=req.horizon_days,
            coin_price_usd=req.coin_price_usd,
            electricity_cost_per_kwh=req.electricity_cost_per_kwh,
            asic_power_watts=req.asic_power_watts,
            pool_fee_pct=req.pool_fee_pct,
            mc_runs=req.mc_runs,
        )

        store = SnapshotStore.from_repo_root()
        snapshot = store.read_latest(coin)

        # deterministic seed for non-token API (so refresh is stable)
        seed_payload = {
            'coin': coin,
            'hashrate': hashrate,
            'horizon_days': req.horizon_days,
            'coin_price_usd': req.coin_price_usd,
            'electricity_cost_per_kwh': req.electricity_cost_per_kwh,
            'asic_power_watts': req.asic_power_watts,
            'pool_fee_pct': req.pool_fee_pct,
        }
        seed = _seed_for_payload(seed_payload)

        payload = _compute_compare(
            params=params,
            snapshot=snapshot,
            seed=seed,
        )

        return JSONResponse(payload)
    except (UnitParseError, AnalyticError, EconomicCompareError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

def _linspace(min_v: float, max_v: float, steps: int) -> List[float]:
    if steps <= 1:
        return [float(min_v)]
    if max_v < min_v:
        min_v, max_v = max_v, min_v
    span = float(max_v - min_v)
    return [float(min_v + (span * i) / float(steps - 1)) for i in range(steps)]
    

@app.post('/api/v1/compare/heatmap')
def api_compare_heatmap(req: CompareHeatmapRequest) -> JSONResponse:
    """
    Deterministic heatmap over:
    X: coin price (USD)
    Y: electricity cost ($/kWh)

    Cell value: probability_negative_net (solo).
    """
    try:
        coin = _normalize_coin(req.coin)
        hashrate = _normalize_hashrate_str(req.hashrate)
        hr = parse_hashrate(hashrate)

        store = SnapshotStore.from_repo_root()
        snapshot = store.read_latest(coin)

        lambda_per_day = compute_lambda_per_day(
            your_hashrate_hs=hr.hs,
            network_hashrate_hs=snapshot.network_hashrate_hs,
            blocks_per_day=snapshot.blocks_per_day,
        )
        mu = float(lambda_per_day) * float(req.horizon_days)

        # Grids (clamp to sensible mins)
        base_price = float(req.coin_price_usd)
        base_elec = float(req.electricity_cost_per_kwh)

        price_min = max(0.01, base_price * (1.0 - float(req.price_span_pct)))
        price_max = max(price_min, base_price * (1.0 + float(req.price_span_pct)))

        elec_min = max(0.0, base_elec * (1.0 - float(req.elec_span_pct)))
        elec_max = max(elec_min, base_elec * (1.0 + float(req.elec_span_pct)))

        price_grid = _linspace(price_min, price_max, int(req.price_steps))
        electricity_grid = _linspace(elec_min, elec_max, int(req.elec_steps))

        matrix: List[List[float]] = []
        for elec in electricity_grid:
            row: List[float] = []
            for price in price_grid:
                res = compare_solo_vs_pool(
                    mu=mu,
                    block_reward=float(snapshot.block_reward),
                    coin_price_usd=float(price),
                    horizon_days=float(req.horizon_days),
                    asic_power_watts=float(req.asic_power_watts),
                    electricity_cost_per_kwh=float(elec),
                    pool_fee_pct=float(req.pool_fee_pct),
                )
                row.append(float(res.solo['probability_negative_net']))
            matrix.append(row)

        probability_negative_net = matrix

        payload = {
            'schema_version': 1,
            'generated_at': _now_utc_iso(),
            'input': {
                'coin': coin,
                'hashrate_hs': hr.hs,
                'hashrate_display': hr.format(),
                'horizon_days': int(req.horizon_days),
                'asic_power_watts': float(req.asic_power_watts),
                'pool_fee_pct': float(req.pool_fee_pct),
                'coin_price_usd': base_price,
                'electricity_cost_per_kwh': base_elec,
                'price_span_pct': float(req.price_span_pct),
                'elec_span_pct': float(req.elec_span_pct),
                'price_steps': int(req.price_steps),
                'elec_steps': int(req.elec_steps),
            },
            'network_snapshot': _network_snapshot_to_report(snapshot),
            'mu': float(mu),
            'price_grid': price_grid,
            'electricity_grid': electricity_grid,
            'probability_negative_net': probability_negative_net,
            # Compatibility wrapper so clients can use `.grid.*`
            'grid': {
                'price_grid': price_grid,
                'electricity_grid': electricity_grid,
                'probability_negative_net': probability_negative_net,
            },
        }
        return JSONResponse(payload)
    except (UnitParseError, AnalyticError, EconomicCompareError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Heatmap error')
        raise HTTPException(status_code=500, detail='Internal error') from exc