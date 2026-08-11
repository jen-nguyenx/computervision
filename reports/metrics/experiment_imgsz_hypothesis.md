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

_(to be appended after both runs complete — do not edit anything above)_

## 6. What I would test next

_(to be written with the result)_
