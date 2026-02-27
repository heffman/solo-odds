## Solo Odds

Quantitative risk modeling for solo mining decisions (BTC / BCH).

Most mining calculators focus on expected revenue.

Solo mining outcomes are probabilistic.
Expected value alone is insufficient.

Solo Odds models variance first.

It answers questions like:

- What is the probability I mine zero blocks?
- What is the probability I mine at least one block?
- What is the probability I lose money?
- What is the probability solo underperforms pool mining?
- How sensitive is risk to price and electricity assumptions?

## What This Is

Solo Odds is a deterministic variance modeling engine built around:
- Poisson block arrival modeling
- Snapshot-frozen network state
- Deterministic shareable links
- Explicit risk metrics (P(0), P(loss), regret probability)
- Sensitivity heatmaps

This is not an ROI marketing calculator.

It is a distribution modeling tool.

## Project Structure

The project has two primary surfaces:

1) Web App (Primary)

Deployed example:
https://solo-odds.hefftools.dev

Features:

- Solo variance report (tokenized)
- Solo vs Pool risk comparison
- Deterministic share links (snapshot-frozen)
- Probability histograms
- Risk heatmap: P(net < 0) vs price and electricity
- Deterministic Monte Carlo (seeded by token)

Key endpoints:

- `POST /api/v1/compare` — solo vs pool risk comparison
- `POST /api/v1/compare/heatmap` — deterministic sensitivity grid
- `POST /api/compare/share` — generate snapshot-frozen share link
- `GET /api/compare/{token}` — retrieve frozen comparison
- `GET /api/report/{token}` — retrieve frozen solo variance report

Share links freeze:
- All user inputs
- Full network snapshot
- Deterministic random seed

This guarantees reproducible shared results even as network conditions change.

2) CLI

The CLI focuses on analytic probability outputs and research workflows.

Install (editable):

```bash
pip install -e .
```

Refresh network snapshot:

```bash
solo-odds refresh --coin bch
```

Compute analytic Poisson odds:

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --json
```

Model network growth (drift):

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

Linear growth:

```bash
solo-odds odds \
  --coin btc \
  --hashrate 50TH \
  --days 365 \
  --drift linear \
  --daily-pct 0.15
```

Monte Carlo (optional):

```bash
solo-odds odds \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --mc 20000
```

Generate probability curves:

```bash
solo-odds curve \
  --coin bch \
  --hashrate 9.4TH \
  --days 365 \
  --interval-days 7 \
  --json
```

## Core Modeling Concepts
### Poisson Block Process

Block arrivals are modeled as a Poisson process with intensity:

μ = integrated block rate over the selected horizon

Outputs include:

- Expected blocks
- P(0 blocks)
- P(at least one block)
- Time-to-first-block distribution (p10 / p50 / p90 / mean)

### Drift Modeling

Network growth can be modeled deterministically:

- flat
- step (% every N days)
- linear (% daily)

Drift modifies integrated intensity across segments.

### Solo vs Pool (Web)

The compare model computes:

- Solo net distribution (Poisson block count × block reward − electricity)
- Pool net expected value (deterministic in v1)
- Regret probability:
  P(solo net < pool net)
- Probability of negative net
- Sensitivity heatmap across price and electricity

Pool variance is not modeled in v1 (treated as deterministic EV).

## Example Analytic Output
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

- BTC: Blockchain.com Query API
- BCH: Blockchair stats API
  with FullStack.cash fallback

Snapshots are stored at:

```
data/<coin>/latest.json
```

Each snapshot includes:
- network_hashrate
- difficulty
- blocks_per_day
- block_reward
- source
- source_url

Snapshots are frozen into web share tokens.

## Assumptions
- Blocks follow a Poisson process.
- Snapshot inputs are treated as fixed during a run.
- Drift is deterministic.
- Hardware downtime and reliability are not modeled.
- Electricity is modeled linearly from watts × $/kWh.
- Pool modeled as deterministic expected value (v1).
- Fees are not dynamically modeled beyond static pool_fee_pct.

This tool models probability distributions, not guaranteed financial outcomes.

Not financial advice.

## Intended Audience
- Serious hobby miners
- Small operators evaluating solo vs pool
- Engineers interested in variance modeling
- Data-driven decision makers

Not optimized for:

- GPU coin hopping
- Yield-maximization dashboards
- Short-term speculative mining
- Turnkey revenue calculators

## Non-Goals
- Mining pool operation
- Fleet management
- Hardware optimization
- Tax reporting
- Investment advice

## License

MIT