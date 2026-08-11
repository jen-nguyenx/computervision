# Initial Plan

## 1. Problem Definition

Three models, each with a different output geometry.

- coco5k-detect. Natural images in, a list of (class, axis-aligned box, confidence) out.
  TODO: one sentence on when a plain box is enough. Hint: counting or locating objects.
- coco5k-segment. Same images in, a mask and a box for each object out.
  TODO: one sentence on when a box is not enough and you need the shape.
  Hint: measuring area, cutting an object out of an image.
- dota5k-obb. 1024 x 1024 aerial patches in, rotated four-corner boxes out.
  TODO: one sentence on why axis-aligned boxes fail on aerial images.
  Hint: a diagonal ship inside an upright box is mostly background.

Separate concerns: classification (what is it), localisation (where is it),
shape estimation (its outline), rotation-aware localisation (where plus orientation).
Also: reproducibility (same data and seed give the same result), task-appropriate
evaluation (box mAP, mask mAP, oriented mAP are different things), and deployment
(latency and usable JSON output).

## 2. Ambiguities and Assumptions

| Question | My assumption | Why |
|---|---|---|
| Deployment hardware? |  | |
| Precision vs recall priority? | TODO.  | |
| COCO crowd annotations (iscrowd=1)? | Exclude from training labels. Count and log every exclusion. | Saw image 153344: one crowd box spans nearly the whole image labeled person. Matches official COCO evaluation, which ignores crowd regions. |
| COCO category IDs? | Build an explicit 80-row mapping table (COCO id to 0-79). | Verified myself: IDs run 1 to 90 with gaps (12, 26, 29, and more), so subtracting 1 from the id is silently wrong. |
| Disconnected multi-polygon instances? | To decide during conversion. Provisional: keep all polygons belonging to an instance. | Have not hit one yet. Will inspect examples when converting. |
| DOTA difficult instances? | Provisional: keep in training, count them, revisit after DOTA exploration. | Have not explored DOTA yet. Will confirm and revise (see section 5). |
| DOTA patch splitting? | Split by source image, never by patch. | Patches from one image overlap by 200 pixels. The same pixels in train and test would fake good results (leakage). |
| Pretrained weights? | Train from scratch (yolo11n architecture, no pretrained weights). | Standard checkpoints were trained on COCO and DOTA. My test images are inside their training data, so generalisation results would be fake. |
| Exported-model parity? | TODO. Hint: pick a tolerance, for example same classes detected and confidences within about 0.02. Refine during the export step. | |

## 3. Task Breakdown

| # | Task | Produces | Done when |
|---|---|---|---|
| 1 | Environment setup | .venv, requirements.txt, environment report | Done. yolo runs on MPS. |
| 2 | COCO download | datasets/coco/ | Done. 118,287 images verified. |
| 3 | COCO exploration | notebook findings | Done. Bbox format verified visually (pixels, top-left plus width and height). Crowd and category-gap traps found. Built an image_id lookup dictionary for fast annotation access. |
| 4 | DOTA download and exploration | datasets/dota/train/ | 1,411 images and 1,411 labels present. One image and label pair inspected. |
| 5 | COCO5K manifest script | manifests/coco5k.csv | Exactly 5,000 rows, sorted by image id. Identical file on re-run. |
| 6 | DOTA5K patching and manifest | 5,000 patches plus manifest | Patch count is 5,000. Every patch maps to its source image. |
| 7 | Dataset audits | reports/ tables and figures | Covers the checklists in brief section 7. |
| 8 | Splits (frozen) | manifests for train, val, test | TODO: pick ratios, for example 80/10/10, and justify. Same COCO split for both COCO models. DOTA split by source image. Overlap-check test passes. |
| 9 | Label converters (detect, segment, obb) | YOLO-format label files | 12 visualised examples per task look correct. Exclusions counted and logged. |
| 10 | Automated tests (at least 5) | tests/ | All pass. |
| 11 | Smoke tests, one per task | tiny 1-epoch runs | Loss finite, checkpoint saves, predict works. |
| 12 | Baselines, one per task | trained yolo11n models with logged configs | Overnight runs complete. Metrics recorded. |
| 13 | Controlled experiment (one change) | comparison table | TODO: pick one idea now, for example image size 640 vs 960 on DOTA (small aerial objects). State the hypothesis. |
| 14 | Evaluation on frozen test sets | metrics and per-class tables | Thresholds chosen on validation data only. |
| 15 | Slice and error analysis | at least 15 failures, taxonomy | Two slices per dataset analysed. |
| 16 | predict.py CLI | JSON and annotated images | Works for all three tasks on a folder of images. |
| 17 | ONNX export, parity check, Docker, benchmark | exported model, Dockerfile, latency table | Exported model actually runs. Parity criterion met. |
| 18 | FINAL_REPORT.md and slides | report under 2,500 words, at most 8 slides | Checklist in brief section 16 all ticked. |

## 4. Risks, Fallbacks, Time Budget

| Risk | Fallback |
|---|---|
| Training too slow on Mac (MPS) | Time one epoch in the smoke test first. If the projection is more than about 6 hours per model: fewer epochs, smaller image size, or Kaggle T4. |
| DOTA patching tool behaves unexpectedly | Pin the exact ultralytics version. Visually verify 12 patches before trusting it. |
| A converter bug discovered late | Visual validation before training. Automated tests run before every training launch. |
| TODO: add one or two more from your own worries. Download corruption? Running out of evenings? | |

Time budget (target about 10 hours of active work; training runs overnight and does not count):
TODO: split roughly. Data and audit: __ h. Conversion and validation: __ h.
Training setup and babysitting: __ h. Evaluation and error analysis: __ h.
Deployment: __ h. Report and slides: __ h.
Hint: brief section 13 says data correctness outranks everything, so weight it accordingly.

## 5. Plan Revisions

- 2026-08-11: Initial plan.
- (Add dated entries here when evidence changes a decision, for example after DOTA exploration.)
