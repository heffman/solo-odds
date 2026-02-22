# Solo Odds (V1)

A rigorous **solo mining probability engine** for small miners.

This project models solo block discovery using a **Poisson process** (analytic results)
and optionally runs **Monte Carlo simulations** to produce variance bands and
percentile outcomes (time-to-first-block, blocks-in-window), including simple
network difficulty/hashrate drift scenarios.

V1 is intentionally narrow:
- **Coins:** BTC and BCH
- **Outputs:** probability of ≥1 block over a horizon, expected blocks, distributions, percentiles
- **Interfaces:** CLI first (web UI later)

This is not a mining dashboard and not a profitability calculator. It is a **risk/variance tool**.

---

## What It Computes

Given:
- your hashrate
- coin (btc/bch)
- horizon (days)
- network conditions (from cached snapshots)
- optional drift model

It returns:
- `P(>=1 block)` over N days
- expected blocks
- `P(k blocks)` distribution (0..K)
- time-to-first-block distribution + percentiles (p10/p50/p90)
- optional Monte Carlo estimates under drift

---

## Data Sources

The CLI uses a cached `data/<coin>/latest.json` snapshot by default.

You refresh snapshots explicitly:

```bash
solo-odds refresh --coin bch
solo-odds refresh --coin btc
```

## Install (Dev)

Requirements:

- Python 3.11+ recommended

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
```

- Run tests:

```bash
pytest -q
```

## CLI Usage

Examples (analytic mode):

```bash
solo-odds odds --coin bch --hashrate 9.4TH --days 90
solo-odds odds --coin btc --hashrate 9.4TH --days 365
```

With drift (step growth: +2% network hashrate every 14 days):

```bash
solo-odds odds --coin bch --hashrate 9.4TH --days 180 \
  --drift step --step-pct 2 --step-days 14
```

Monte Carlo (20k runs):

```bash
solo-odds odds --coin bch --hashrate 9.4TH --days 365 \
  --drift step --step-pct 2 --step-days 14 \
  --mc 20000
```

JSON output:

```bash
solo-odds odds --coin bch --hashrate 9.4TH --days 90 --json
```