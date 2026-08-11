# COCO5K Pipeline — Decision Registry

Every policy below was decided explicitly (exploration findings → discussion → decision) and is
enforced by code that reads `configs/pipeline_coco.yaml`. Numbers come from the exploration
notebook's verified sweep of the first 5,000 images (35,765 polygon annotations checked against
pycocotools with 0 mismatches).

## Decisions

| # | Finding | Decision | Enforced in |
|---|---------|----------|-------------|
| D1 | Assignment mandates deterministic subset | COCO5K = first 5,000 images sorted by ascending numeric ID; the 49 empty images are kept and get empty label files | `build_coco5k.py`, `convert_coco.py` |
| D2 | Category IDs run 1–90 with 10 gaps (80 classes) | Explicit sorted-original-ID → 0..79 map saved to `configs/coco_category_map.json`; never `id − 1` | `build_coco5k.py` |
| D3 | Reproducibility requirement (§4) | Manifest `image_id,file_name,width,height,source_split`; sha256 of every manifest recorded in `manifests/HASHES.txt` | `build_coco5k.py` |
| D4 | From-scratch nano needs data; rare classes (toaster 13, hair drier 7) may vanish from partitions | 80/10/10 split (4000/500/500), seed 0, on image_id; ONE split shared by detect & segment; frozen + hashed; per-partition class coverage reported with rare classes flagged | `make_splits.py` |
| D5 | COCO bbox is top-left `[x,y,w,h]` (verified visually + by round-trip) | Convert to normalized center `cxcywh` | `convert_coco.py` |
| D6 | 448 `iscrowd=1` annotations (RLE regions covering many objects) | Excluded from training labels + counted; matches official COCO evaluation semantics | `convert_coco.py` |
| D7 | 3,455 multipolygon instances (~10%) — one object, several pieces | Merge with Ultralytics `merge_multi_segment` (pinned 8.4.117; internal API — logged in AI_DECISIONS); merges counted; instances with ≥8 parts flagged as suspicious in the ledger but NEVER removed (includes the 17-part collage-dog, image 16950) | `convert_coco.py` |
| D8 | 0 degenerate polygons observed here, but they exist elsewhere in train2017 | Defensive policy: exclude invalid part (<3 points / odd coord count), count under a named reason, keep going — no fail-fast, nothing silent | `convert_coco.py` |
| D9 | 635 crude polygons (≤4 vertices on objects >800px²) — valid but imprecise labels | Pass through untouched; flagged in the audit as a label-noise limitation (GT noise caps measurable mask quality) | `audit_coco5k.py` |
| D10 | COCO occasionally has coords slightly past image bounds | Clip to [0,1] after normalization; count clipped annotations | `convert_coco.py` |
| D11 | Silent data loss is the classic converter failure | **Reconciliation identity** asserted at the end of every conversion: `input_annotations == converted + sum(excluded by reason)` | `convert_coco.py`, `test_reconciliation.py` |
| D12 | 14,896 tiny objects (<32²px, 41% of annotations) | Keep all — ground truth is ground truth; only zero/negative-area boxes dropped (counted). Small-object weakness gets exposed by slice analysis, not hidden by filtering | `convert_coco.py` |
| D13 | 154 grayscale images (91% RGB-encoded), 0 EXIF rotations, 0 size mismatches, person=31.2% | Measure + report only. No exclusions, no resampling, no conversion. Imbalance handled at evaluation (per-class metrics, common-vs-minority slice) | `audit_coco5k.py` |
| D14 | 5,000 images already on disk; duplication is waste | Symlink trees into `data/yolo/<task>/images/<split>/`; `data/` gitignored. Caveat: links break if `datasets/` moves (documented) | `convert_coco.py` |
| D15 | Ultralytics caches parsed labels by path+size, not content — stale caches silently resurrect old labels | Every label rewrite deletes `*.cache` under the task root | `convert_coco.py` |
| D16 | Relative data.yaml paths resolve against `~/datasets` and stock yamls auto-download full datasets | Generated data.yamls use absolute `path:` | `convert_coco.py` |
| D17 | Assignment §4.3: no manual edits | Every exclusion/correction is implemented in code, counted, logged in the ledger, explained here | all modules |

## Interfaces

- **Manifest** `manifests/coco5k.csv`: `image_id,file_name,width,height,source_split` (source_split = `train2017`).
- **Splits** `manifests/coco5k_splits.csv`: `image_id,split` with split ∈ {train,val,test}.
- **Category map** `configs/coco_category_map.json`: `{"map": {"<original_id>": contiguous_id}, "names": {"<contiguous_id>": "class name"}}`.
- **Conversion ledger** `reports/metrics/conversion_<task>.json` (task ∈ detect,segment):
  `{"input_annotations": int, "converted": int, "excluded": {"crowd": int, "degenerate_parts": int, "empty_segmentation": int, "zero_area": int}, "clipped_coord_count": int, "merged_multipolygon": int, "suspicious_flagged": [[image_id, ann_id, n_parts], ...], "reconciliation_ok": true, "ultralytics_version": "8.4.117"}`
  — `excluded` counts whole annotations; `degenerate_parts` counts dropped parts separately without breaking the identity (an annotation is "converted" if ≥1 valid part remains).
- **YOLO layout** `data/yolo/coco5k-{detect,segment}/{images,labels}/{train,val,test}/`; images are absolute symlinks into `datasets/coco/train2017/`; every image has a label file (empty file for empty images).
- **Label lines**: detect `cls cx cy w h`; segment `cls x1 y1 x2 y2 …` — normalized floats, 6-decimal formatting.
- **Data yamls** `configs/coco5k-detect.yaml`, `configs/coco5k-segment.yaml`: absolute `path:`, `train/val/test: images/<split>`, `names:` from the category map.

## Expected numbers (verification targets, known from exploration)

manifest rows = 5,000 · empty images = 49 · crowd excluded = 448 per task · merged multipolygon ≈ 3,455 ·
suspicious flagged ≈ dozens (incl. image 16950 ann 18863, 17 parts) · splits exactly 4000/500/500, disjoint ·
detect converted = 35,765 = segment converted (same retained annotations) · reconciliation balances exactly ·
audit: grayscale 154, person share 31.15%, toaster 13 / hair drier 7 flagged in coverage.
