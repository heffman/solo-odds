# Solo Odds

Quantitative risk modeling for solo mining decisions (BTC/BCH).

Most mining calculators emphasize expected revenue. Solo mining outcomes are probabilistic.
Expected value alone is insufficient for decision-making.

It answers questions like:

- What is the probability I mine zero blocks?
- What is the probability I mine at least one block?
- How long until my first block (p10 / p50 / p90 / mean)?
- How sensitive are outcomes to assumed network growth (drift)?
- (Web) What is the probability solo underperforms pool mining?

## Core Concepts
- Poisson block arrival modeling (integrated intensity over a horizon)
- Piecewise rate modeling with optional drift (flat/step/linear)
- Snapshot-frozen network state as deterministic inputs
- Deterministic shareable links in the web app (tokenized reports)
- Monte Carlo simulation (optional; CLI and web)

## Why This Exists

This tool is not an ROI marketing calculator.
It is a variance modeling engine.

Solo mining is dominated by variance. What matters is:

- Probability of at least one block
- Probability of zero blocks
- Distribution of outcomes (block counts and time-to-first-block)
- Sensitivity to network growth assumptions

Solo Odds models those explicitly.

## Installation

Editable install (local dev):

```bash
pip install -e .
```

Or install from source:

```bash
git clone https://github.com/<yourname>/solo-odds.git
cd solo-odds
pip install .
```

## CLI
### Refresh snapshot data

Fetch the latest cached network snapshot:

```bash
solo-odds refresh --coin bch
```

Writes:

- `data/bch/latest.json` (or `data/btc/latest.json`)

### Compute odds (analytic Poisson)

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 180 \
  --json
```

### Model network growth (drift)

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

Monte Carlo (optional)

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --mc 20000
```

### Generate a probability curve

JSON:

```bash
solo-odds curve \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --interval-days 7 \
  --json
```

CSV:

```bash
solo-odds curve \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --interval-days 7 \
  --csv
```

### Plot a curve to PNG

Probability curve:

```bash
solo-odds plot \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --interval-days 7 \
  --y p \
  --out p_curve.png
```

Expected blocks (log scale):

```bash
solo-odds plot \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --interval-days 7 \
  --y mu \
  --log-y \
  --min-y 1e-10 \
  --out mu_log.png
```

## Web API
- /api/report/{token} — solo variance report (tokenized, snapshot-frozen)
- /api/v1/compare — solo vs pool risk comparison (includes basic electricity + pool fee modeling)
- /api/compare/{token} — snapshot-frozen compare analysis (tokenized)

## Token Model (Web)

Share links freeze:

- All user inputs
- Full network snapshot at time of creation
- Deterministic seed (so Monte Carlo results remain stable)

This guarantees:

- Stable shared results
- No “moving target” disputes as network conditions change

## Intended Audience
- Serious hobby miners
- Small farm operators
- Data-driven decision makers

Not optimized for:

- GPU coin rotation strategies
- Short-term speculative mining
- Simple revenue-only calculators

This tool emphasizes probability distributions and risk boundaries over simple expected value.

## Example Output (Analytic)

`solo-odds odds --json` emits a report with fields like:

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

## Data Sources

Snapshots are fetched from public network APIs and cached locally:

- BTC: Blockchain.com Query API (https://blockchain.info/q)
- BCH: Blockchair stats API (https://api.blockchair.com/bitcoin-cash/stats)
  with FullStack.cash fallback (https://api.fullstack.cash/v5/blockchain/getDifficulty)

Snapshots are stored under:

- `data/<coin>/latest.json`

Each snapshot includes `source` and `source_url` fields.

## Assumptions
- Blocks follow a Poisson process.
- Inputs are taken from a snapshot and treated as fixed for a run.
- Drift models network growth deterministically (flat/step/linear).
- Hardware reliability and downtime are not modeled.
- Fees are excluded from the snapshot block_reward (subsidy-only in v1 snapshot data).
- Pool outcomes are modeled as deterministic expected value (v1); pool variance is not modeled.

This tool models block-finding probability and variance. It is not financial advice.

## Non-Goals
- Mining pool operation
- Hardware fleet management
- Tax reporting
- Financial advice

Notes:

- The CLI focuses on probability/variance outputs.
- The web compare endpoint includes basic electricity + pool fee modeling for risk framing.

## License

MIT