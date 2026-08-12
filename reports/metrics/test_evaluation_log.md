# Test-split evaluation log (§10)

The frozen test split is touched ONCE per model, after thresholds were fixed on validation
(reports/metrics/thresholds.json: 640 -> conf 0.21, 960 -> conf 0.28, max-F1 criterion).
Every invocation is recorded here as an audit trail.

| when | models | split | command |
|---|---|---|---|
| 2026-08-12 10:37 | 640, 960 | test | `python -m src.evaluate.eval_slices --split test --model 640=... --model 960=...` |
