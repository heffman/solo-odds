# Notes:
---

# Project Structure

```
src/solo_odds/
  cli.py
  units.py              # parse 9.4TH, 500GH, etc.
  data/
    fetch.py            # network snapshot fetchers
    store.py            # read/write cached snapshots
    models.py           # Snapshot dataclasses
  math/
    poisson.py          # pmf/cdf utilities
    analytic.py         # closed-form results
    drift.py            # drift models => effective mu
  sim/
    monte_carlo.py      # optional simulation engine
  report/
    format_text.py
    format_json.py
tests/
data/                   # runtime snapshots (gitignored, samples optional)
```

## Non-Goals (V1)

- Pool profitability comparisons
- Miner telemetry monitoring
- Accounts / subscriptions
- Alerts / notifications
- Hardware ROI optimization
- Multi-coin beyond BTC/BCH

## Roadmap

V1:

- Correct analytic model (Poisson / exponential)
- Snapshot caching + refresh
- Drift model (flat, step, linear)
- Optional Monte Carlo + percentile outputs
- JSON report schema

V1.1:

- “curve” command: probability vs time curve output for plotting
- Better snapshot validation and source attribution

Later:

- Thin web UI wrapper
- Public hosted “live network snapshots” page

License: MIT for code. Data snapshots may have separate terms depending on sources.
