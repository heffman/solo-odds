from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from solo_odds.data.store import SnapshotStore
from solo_odds.math.analytic import AnalyticError, analyze_segments, compute_lambda_per_day
from solo_odds.math.drift import DriftError, drift_model_from_cli, make_segments
from solo_odds.sim.monte_carlo import MonteCarloError, run_monte_carlo
from solo_odds.units import UnitParseError, parse_hashrate

app = FastAPI(title='solo-odds web')
templates = Jinja2Templates(directory='web/templates')
app.mount('/static', StaticFiles(directory='web/static'), name='static')


def _now_utc_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


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
        points.append(
            {
                'day': horizon,
                'expected_blocks': res.expected_blocks,
                'probability_at_least_one': res.probability_at_least_one,
                'probability_zero': res.probability_zero_blocks,
            }
        )
    return points


def build_report(params: ShareParams) -> Dict[str, Any]:
    # Parse hashrate
    hr = parse_hashrate(params.hashrate)

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
        },
        'monte_carlo': {'enabled': False},
        'notes': [
            'Counts use a Poisson model with integrated intensity over the horizon.',
            'Time-to-first-block uses an effective constant-rate approximation when drift is enabled.',
        ],
    }

    if params.mc and params.mc > 0:
        mc_res = run_monte_carlo(segments=segments, runs=int(params.mc))
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
            },
        },
    )


@app.post('/share')
def share(
    coin: str = Form(...),
    hashrate: str = Form(...),
    days: int = Form(...),
    interval_days: int = Form(...),
    drift: str = Form(...),
    step_pct: float = Form(2.0),
    step_days: int = Form(14),
    daily_pct: float = Form(0.0),
    mc: int = Form(0),
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
    )
    token = make_token(params.model_dump())
    return RedirectResponse(url=f'/r/{token}', status_code=303)


@app.get('/r/{token}', response_class=HTMLResponse)
def render_report(token: str, request: Request) -> HTMLResponse:
    raw = parse_token(token)
    try:
        params = ShareParams(**raw)
        report = build_report(params)
    except (UnitParseError, AnalyticError, DriftError, MonteCarloError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail='Internal error') from exc

    curve = report.pop('_curve', [])
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
        },
    )


@app.get('/api/report/{token}')
def api_report(token: str) -> JSONResponse:
    raw = parse_token(token)
    try:
        params = ShareParams(**raw)
        report = build_report(params)
        report.pop('_curve', None)
        return JSONResponse(report)
    except (UnitParseError, AnalyticError, DriftError, MonteCarloError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc