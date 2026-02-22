# Solo Odds

A probabilistic tool for modeling solo Bitcoin (BTC) and Bitcoin Cash (BCH) mining outcomes.

Solo Odds answers a specific question:

> Given my hashrate and a time horizon, what are the actual odds?

It provides analytic Poisson modeling, drift-aware projections, Monte Carlo simulation, and exportable curves suitable for plotting or further analysis.

This is not a mining pool.
This is not a “profit calculator.”
This is a variance model.

---

## Why This Exists

Most mining calculators focus on expected value:

> “You’ll mine 0.004 blocks in 30 days.”

That number is meaningless without context.

Solo mining is dominated by variance. What matters is:

- Probability of at least one block
- Probability of zero blocks
- Distribution of outcomes
- Time-to-first-block percentiles
- Sensitivity to network growth

Solo Odds models those explicitly.

---

## Features

### Analytic Engine (Poisson Model)
- Integrated intensity over arbitrary time horizons
- Piecewise rate modeling
- Probability distribution of block counts
- Time-to-first-block percentiles
- Deterministic, fast, exact for homogeneous segments

### Drift Modeling

Supports network growth assumptions:

- `flat` – constant rate
- `step` – step increase every N days (e.g. +2% every 14 days)
- `linear` – continuous daily percent growth

This allows modeling realistic difficulty/network hashrate growth.

### Monte Carlo Simulation
- Independent simulation engine
- Validates analytic model
- Estimates:
    - mean blocks
    - p10 / p50 / p90 blocks
    - P(>=1)
    - P(0)
    - time-to-first-block distribution
- Useful when drift is non-trivial

### Curve Generation

Generate probability-vs-time curves:

- JSON output
- CSV output
- CLI table output

### Plot Generation

Render publication-ready PNG plots:

- Probability curves
- Expected blocks curves
- Log-scale support for low-rate scenarios

---

## Installation

```bash
pip install -e .
```

Or install from source:

```bash
git clone https://github.com/<yourname>/solo-odds.git
cd solo-odds
pip install .
```

--- 

## First Run

Fetch latest network snapshot:

```bash
solo-odds refresh --coin bch
```

Then compute odds:

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 180 \
  --json
```

---

## Example Output (Analytic)

```JSON
{
  "analytic": {
    "expected_blocks": 0.42,
    "probability_at_least_one": 0.344,
    "probability_zero_blocks": 0.656,
    "time_to_first_block_days": {
      "p10": 12.3,
      "p50": 110.4,
      "p90": 410.7,
      "mean": 238.1
    }
  }
}
```

Interpretation:

- You are more likely to mine zero blocks than one.
- Median time to first block exceeds your 180-day horizon.
- Expected value alone is misleading.

---

## Modeling Network Growth

Step growth:

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --drift step \
  --step-pct 2 \
  --step-days 14
```

Linear daily growth:

```bash
solo-odds odds \
  --coin btc \
  --hashrate 50TH \
  --days 365 \
  --drift linear \
  --daily-pct 0.15
```

---

## Monte Carlo

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --mc 20000
```

Monte Carlo confirms analytic probabilities and provides distribution summaries.

---

## Generate Curve (JSON)

```bash
solo-odds curve \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --json
```

---


## Plot Probability Curve

```bash
solo-odds plot \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --y p \
  --out p_curve.png
```

---

## Plot Expected Blocks (Log Scale)

```bash
solo-odds plot \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --y mu \
  --log-y \
  --min-y 1e-10 \
  --out mu_log.png
```

---

## Data Sources

Snapshots are fetched from public network APIs and cached locally:

- BTC: blockchain.com query API
- BCH: Blockchair (primary) with FullStack.cash fallback

Snapshots are stored under:

```
data/<coin>/latest.json
```

---

## Assumptions
- Blocks follow a Poisson process.
- Fees are excluded (subsidy-only modeling in v1).
- Network growth is modeled deterministically via drift.
- Mining hardware reliability is not modeled.

This tool models block-finding probability, not profitability.

---

## Non-Goals
- Mining pool functionality
- Hardware configuration
- Electricity cost modeling
- Tax reporting
- Financial advice


---

## Intended Audience
- Home solo miners
- ASIC hobbyists
- Statistically literate operators
- Anyone who wants to understand variance honestly

---

## Philosophy

Solo mining is a variance game.

Expected value without distribution context is incomplete.

This tool exists to make variance explicit.

--- 

## License

MIT License.
