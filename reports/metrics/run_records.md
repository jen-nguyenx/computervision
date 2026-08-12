# Training run records (§9.4)

One row per training run. Values are transcribed from each run's `args.yaml` and log, not from
intent — if a setting was overridden by Ultralytics (e.g. `optimizer=auto`), the *effective* value
is recorded. Data lineage is pinned by the manifest hashes in `manifests/HASHES.txt`.

## Shared data lineage (all COCO runs)

| Field | Value |
|---|---|
| Dataset | COCO5K — first 5,000 train2017 images by ascending numeric ID (D1) |
| Manifest | `manifests/coco5k.csv`, sha256 `80710a8fe9d440f31e1039ec2ca541e3bbc256cc42ae0096d14a233a3ace44cc` |
| Splits | `manifests/coco5k_splits.csv`, sha256 `c98e09be4afaec339036d6695e6dbd9751d99dccfb5dc53bf38b77e98473f705` — 4000/500/500, seed 0, disjoint |
| Labels | 35,765 converted annotations; 448 iscrowd excluded; ledger `reports/metrics/conversion_detect.json` (reconciliation_ok: true) |
| Initial weights | **None — random init** (`yolo11n.yaml`, `pretrained=False`), per §9.2 leakage policy |
| Git commit | _(to fill: repo not yet committed at time of these runs)_ |

## Run 1 — `coco5k-detect-40ep` (superseded, kept as evidence)

| Field | Value |
|---|---|
| Run ID | coco5k-detect-baseline (40 epochs) |
| Purpose | First attempt; superseded by Run 2 after being diagnosed as undertrained |
| Architecture | YOLO11n, 2,624,080 params, 6.7 GFLOPs |
| Image size | 640 |
| Epochs | 40 |
| Batch | 32 |
| Optimizer | **AdamW, lr0 0.000119, momentum 0.9** (auto-selected; `lr0=0.01` was ignored) |
| LR schedule | annealed to 4.1e-6 by the final epoch |
| Augmentation | Ultralytics defaults (mosaic 1.0, close_mosaic 10, fliplr 0.5, hsv h/s/v 0.015/0.7/0.4, scale 0.5, erasing 0.4, copy_paste 0.0) |
| Seed / determinism | 0 / True |
| Early stopping | patience 100 (not triggered) |
| Hardware | RunPod RTX 4090 24GB; Python 3.12.3, torch 2.8.0+cu128, CUDA 12.8, ultralytics 8.4.117 |
| Duration | 0.115 h (6.9 min), 414 s total |
| Peak GPU memory | 5.02 GB |
| Best-checkpoint criterion | Ultralytics default (max fitness = 0.1·mAP50 + 0.9·mAP50-95 on val) |
| **Val result** | mAP50 **0.022**, mAP50-95 **0.0095**, P 0.604, R 0.025 |
| Artifacts | `artifacts/coco5k-detect-40ep/` (best.pt, last.pt, results.csv, args.yaml, curves, env_record.txt) |
| Note | mAP was still rising at the final epoch and the LR had annealed to ~0 → a complete but too-short schedule. Diagnosis in AI_DECISIONS Entry 13. |

## Run 2 — `detect-300ep-640` (principal detect baseline)

| Field | Value |
|---|---|
| Run ID | detect-300ep-640 |
| Purpose | Principal COCO detection baseline, using the standard from-scratch recipe |
| Architecture | YOLO11n, `pretrained=False` |
| Image size | 640 |
| Epochs | 300 |
| Batch | 32 |
| Optimizer | **SGD, lr0 0.01, momentum 0.937, weight_decay 0.0005** (explicit override of `auto`) |
| LR schedule | linear to `lr0 × lrf` = 0.0001; warmup 3 epochs (warmup_momentum 0.8, warmup_bias_lr 0.1) |
| Augmentation | identical defaults to Run 1 (copy_paste 0.0 — imbalance deliberately not treated, D13) |
| Seed / determinism | 0 / True |
| AMP | True |
| Early stopping | patience 100 |
| Hardware | RunPod RTX 4090 24GB; Python 3.12.3, torch 2.8.0+cu128, CUDA 12.8, ultralytics 8.4.117 |
| Duration | 45 min (2707 s) — **early-stopped at epoch 269**, best epoch 169 |
| Peak GPU memory | 6.52 GB |
| **Val result (Ultralytics)** | mAP50 **0.2329**, mAP50-95 **0.1486**, P 0.334, R 0.255 |
| **Val result (COCOeval)** | mAP50 0.2263, mAP50-95 0.1431; AP small 0.0702 / medium 0.1668 / large 0.2334 |
| Median latency (MPS, imgsz 640) | 15.7 ms |
| Artifacts | `artifacts/coco5k-detect-300ep/runs/detect/runs/detect-300ep-640/` |
| Note | Converged: 100 epochs with no improvement triggered early stopping, so this baseline is **not** compute-limited at 640. |

## Run 3 — `detect-300ep-960` (controlled experiment, §9.5)

| Field | Value |
|---|---|
| Run ID | detect-300ep-960 |
| Purpose | Controlled experiment: single change vs Run 2, `imgsz 640 → 960` |
| Changed vs Run 2 | **image size only** — architecture, data, splits, seed, epochs, batch, optimizer, LR, and augmentation all identical |
| Hypothesis | Pre-registered in `reports/metrics/experiment_imgsz_hypothesis.md` before either run finished — **refuted** by the result |
| Duration | 110 min (6581 s) — ran the full 300 epochs, best epoch 217, early stopping never triggered |
| **Val result (Ultralytics)** | mAP50 **0.2566**, mAP50-95 **0.1655**, P 0.376, R 0.266 |
| **Val result (COCOeval)** | mAP50 0.2575, mAP50-95 0.1651; AP small 0.0664 / medium 0.1940 / large 0.2555 |
| Median latency (MPS, imgsz 960) | 18.6 ms (+18% vs 640, far less than the +125% pixel count) |
| Artifacts | `artifacts/coco5k-detect-300ep/runs/detect/runs/detect-300ep-960/` |
| Outcome | +15.4% overall mAP50-95, but **AP small fell 5.4%** while AP medium rose 16.3% — the hypothesis predicted the opposite pattern. See the experiment file for the full analysis. |

## Comparison caveat (for the report)

Run 2 differs from Run 1 in **two** variables (optimizer and epoch budget), so the ~9× improvement is
an **observation, not a controlled result** — and the comparison is further confounded because Run 1's
learning rate had fully annealed by its final epoch while Run 2 was mid-schedule at the compared point.
Only Run 2 vs Run 3 is a controlled single-variable comparison.
