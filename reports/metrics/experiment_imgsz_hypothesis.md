# Controlled experiment (§9.5) — pre-registered BEFORE running

Written and committed before either run finished. Nothing below was edited after seeing results;
the measured outcome goes in a separate section appended afterwards.

## 1. Hypothesis

Increasing the training/inference image size from 640 to 960 px will improve detection of small
objects substantially more than it improves detection of large objects.

**Why I expect this (evidence, not guesswork).** The COCO5K audit
(`reports/metrics/coco5k_audit.json`) measured the object-area distribution over the 5,000-image
subset using the COCO 32²/96² convention:

- small (< 32² px): **14,896 annotations (41%)**
- medium: 12,490
- large: 8,827

YOLO predicts on feature maps at strides 8/16/32. A 32 px object occupies 4×4 cells at stride 8 and
a single cell at stride 32, so it is close to the resolution floor of the architecture. Scaling the
input by 1.5× scales every object by 1.5× in pixels, moving small objects into a size regime the
network handles better. Large objects are already well above that floor, so they should gain little.

## 2. The single primary change

`imgsz: 640 -> 960`. Everything else identical: same architecture (yolo11n.yaml, `pretrained=False`),
same frozen data and splits, same seed (0), same optimizer (SGD, lr0 0.01), same epochs (300),
same batch (32), same augmentation defaults, same hardware.

## 3. Expected effect (stated in advance)

- Overall mAP50-95 on the **validation** split: increase, roughly +20–60% relative.
- Small-object slice: the largest relative gain.
- Large-object slice: little change (within a few percent relative).
- Inference latency: increase roughly with pixel count (~2.25×), so this is a genuine
  accuracy-vs-latency trade-off, not a free win.
- Training cost: ~2.25× the baseline's ~52 minutes.

## 4. How it will be judged

Both models evaluated with `model.val()` on the **validation** split only (the frozen test split is
not touched for this comparison, per §10.4). Primary metric: mAP50-95 overall, plus per-size slices
computed with the same 32²/96² definition used in the audit. Latency measured on identical hardware,
median over >= 100 steady-state inferences.

**The hypothesis is supported only if** the small-object slice improves by a clearly larger relative
margin than the large-object slice. An overall mAP gain alone does NOT confirm it — that could come
from anywhere.

## 5. Measured result

Both runs trained from scratch on the frozen COCO5K train split; evaluated on the **validation**
split with the official `pycocotools` COCOeval (same 32²/96² size thresholds as the audit), each
model at its own training resolution, detections at conf 0.001. Raw numbers:
`reports/metrics/slices_val.json`. Command: `python -m src.evaluate.eval_slices --split val --model 640=... --model 960=...`

| metric | 640 baseline | 960 experiment | change |
|---|---|---|---|
| mAP50-95 | 0.1431 | 0.1651 | **+15.4%** |
| mAP50 | 0.2263 | 0.2575 | +13.8% |
| **AP small** | **0.0702** | **0.0664** | **−5.4%** |
| AP medium | 0.1668 | 0.1940 | **+16.3%** |
| AP large | 0.2334 | 0.2555 | +9.5% |
| AR small | 0.1307 | 0.1495 | +14.3% |
| AR medium | 0.3625 | 0.3604 | −0.6% |
| AR large | 0.4714 | 0.4374 | −7.2% |
| median latency (MPS) | 15.7 ms | 18.6 ms | +18.1% |
| training time | 45 min | 110 min | +144% |

### Is the hypothesis supported? **No — refuted on its own criterion.**

Section 4 stated the test explicitly: support requires the small-object slice to improve by a
clearly larger relative margin than the large-object slice. What actually happened is the opposite
of the prediction's core claim — **AP for small objects went DOWN 5.4%**, while the gain came from
**medium** objects (+16.3%). The overall +15.4% mAP50-95 improvement is real, but it does not come
from where the hypothesis said it would. Had I judged the experiment on the headline number alone,
I would have wrongly declared the hypothesis confirmed; the pre-registered slice criterion is what
prevented that.

### The informative detail: recall and precision moved in opposite directions

Small objects were **found more often** (AR small +14.3%) but **scored worse** (AP small −5.4%). More
pixels did make small objects more detectable — the mechanism in section 1 was not wrong — but the
extra detections are poorly localized and/or low-confidence. Because AP averages over IoU thresholds
from 0.50 to 0.95, loose boxes on tiny objects are punished hard: a few pixels of error is a large
fraction of a 20 px object's area. So the resolution increase bought *detection* of small objects
without buying *localization* of them.

The large-object row shows the mirror image (AP +9.5% with AR −7.2%): fewer large objects recovered,
but the ones found are cleaner.

### Confounds and caveats (stated, not hidden)

1. **The two models are not equally converged.** The 640 run early-stopped at epoch 269 (best 169)
   after 100 epochs without improvement; the 960 run used all 300 epochs (best 217) and never
   triggered early stopping, so it was likely still improving. Both used the identical `patience=100`
   policy, so this is a *consequence* of the change rather than an introduced confound — but the 960
   model may be under-trained relative to its own ceiling, which if anything understates its result.
2. **Latency scaled far less than pixel count** (+18% for +125% pixels). At nano scale on MPS,
   fixed overheads (preprocessing, NMS, Python dispatch) dominate the convolution work, so the
   accuracy/latency trade-off is much cheaper than the 2.25× I predicted. Training cost, however,
   did scale roughly as expected (+144%).
3. **COCOeval vs Ultralytics numbers differ slightly** (640: 0.1431 vs 0.1486 mAP50-95). Different
   implementations, different crowd handling; the comparison above is internally consistent because
   both models were scored by the same evaluator.

## 6. What I would test next

1. **Attack localization, not resolution.** Since small-object recall improved but precision did not,
   the bottleneck looks like box regression on tiny objects, not visibility. The cheap test: keep
   imgsz 640 and raise the DFL loss weight (`dfl`, default 1.5), or evaluate at a higher resolution
   than training (train 640 / infer 960) to separate "seeing" from "learning".
2. **Give 960 a fair convergence budget.** Rerun at 960 with `patience` unchanged but more epochs
   (or until early stopping fires) so both arms are compared at convergence rather than at a budget
   cutoff.
3. **Copy-paste augmentation for the tail.** Unrelated to resolution, but the per-class table shows
   rare classes near zero; `copy_paste=0.3` uses the segmentation masks already produced by the
   pipeline and is the most promising single change for minority-class recall.

**Decision for the report:** the 960 model is the better overall detector and costs 2.4× the training
time for ~15% mAP; the 640 model remains the principal baseline (converged, cheaper, and the honest
reference point), with the 960 run reported as a completed controlled experiment whose hypothesis
was refuted.
