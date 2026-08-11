# AI Decisions Log

This log records meaningful AI interactions per assignment §6: what the AI proposed, what I challenged,
how claims were verified, and what was finally decided. Entries were drafted with AI assistance from the
session transcripts and reviewed by me; I own every decision recorded here.

Format per entry: the 8 required fields, with "Risks or assumptions" expanded to include **my critique**,
since several decisions changed because of it.

---

## Entry 1 — Problem decomposition and execution plan

| Field | Content |
|---|---|
| Stage | Problem decomposition (§6 workstream 1) |
| Prompt | "Read the requirement and advise a step-by-step plan on how to execute the work and how to start." |
| AI recommendation | 4-session plan: Mac never trains full models (no CUDA), 3 from-scratch nano baselines on free Kaggle GPU ≤6 GPU-h, everything else local; INITIAL_PLAN.md committed before training code; pre-committed cut list; conf-threshold study as the zero-GPU controlled experiment. |
| Risks / my critique | AI's internal audit of its own drafts surfaced real errors before I saw them: one draft evaluated the test set *before* selecting thresholds on val (violates §10 preamble), and all drafts' hour budgets were optimistic (13–16h realistic vs 10h claimed). I required the plan to be advice-only until I approved scope. |
| Verification | Plan cross-checked against the brief's own text (§5 commit-before-code, §9.2 leakage policy, §13 priority order, §16 checklist); hardware claims (no CUDA, MPS present) checked on my machine. |
| Final decision | Accepted the corrected plan structure; adopted commit-plan-first, from-scratch weights, Kaggle-for-training. |
| Evidence | INITIAL_PLAN.md (committed before any training code); this repo's session structure. |

## Entry 2 — Why verify against pycocotools at all (methodology challenge)

| Field | Content |
|---|---|
| Stage | Dataset exploration |
| Prompt | "Why do we need to verify our annotation against another official library annotation?" |
| AI recommendation | Verify derived values (polygon-extent bbox, shoelace area) against independently stored fields via pycocotools, because self-checks share the same wrong assumption ("correlated error"); an independent oracle validates *understanding*, not just execution. |
| Risks / my critique | I questioned whether the step was necessary at all before accepting it — the justification (correlated errors) had to be argued, not asserted. Also: the AI assumed stored `area` could differ from shoelace area by up to ~10% due to rasterization; measurement showed the real difference is **<0.05% everywhere** in the subset — the AI's tolerance assumption was empirically over-cautious. Kept the tolerance anyway as defensive slack. |
| Verification | Mechanized sweep over the first 5,000 images: **0 mismatches across 35,765 polygon annotations** on all three checks (index, bbox-from-polygon, area-from-shoelace). |
| Final decision | Verification-by-independent-oracle adopted as the standard for all AI-generated conversion code in this project. |
| Evidence | exploration.ipynb `verify_and_audit` cell output; sweep summary (0/0/0 mismatches). |

## Entry 3 — COCO bbox format (the top-left vs center trap)

| Field | Content |
|---|---|
| Stage | Label-conversion design |
| Prompt | Explanation of COCO format and conversion traps requested during exploration planning. |
| AI recommendation | COCO bbox is top-left `[x,y,w,h]`; Ultralytics wants normalized **center** `cxcywh`; the naive misread trains a model with half-shifted boxes that looks merely "weak", not broken. |
| Risks / my critique | Claim was plausible but unverified — exactly the kind of API detail §6 says not to assume. |
| Verification | Drew the boxes both ways on a real image (the deliberately-wrong center interpretation visibly shifts every box); hand-computed round-trip example (bbox [10,20,30,40] in 100×200 → 0.25, 0.20, 0.30, 0.20) reserved as a unit test. |
| Final decision | Converter uses top-left→center with the hand-computed example as `tests/test_geometry.py`. |
| Evidence | exploration.ipynb Part-2 cell; NOTES.md glossary entry ("Verified this visually in the notebook"); tests/test_geometry.py. |

## Entry 4 — Edge-case visualization: I rejected the AI's deliverable twice ★

| Field | Content |
|---|---|
| Stage | Exploration tooling |
| Prompt | "Need visuals … so I can print the image … for example when the polygon formation is off." |
| AI recommendation | First: a concept diagram of the verification sweep. Second: an auto-generated static gallery of edge-case examples. |
| Risks / my critique | **Both missed what I needed.** A diagram explains the method but shows no data; a pre-baked gallery shows the AI's picks, not mine. I rejected both and specified the actual requirement: a function **I can run myself**, per bucket, to browse real images and learn what each edge case looks like. |
| Verification | The replacement (`show_caught("<bucket>")`) was executed against every bucket before delivery (disconnected, crude, extreme-aspect, tiny, crowd-RLE decode, empty, pagination) — all paths render real images. |
| Final decision | `show_caught()` browse function with wildest-first sorting and vertex rendering; the diagram kept only as secondary documentation. |
| Evidence | exploration.ipynb browse cell; the 17-part "dog" (image 16950) surfaced on my first run of it. |

## Entry 5 — Grayscale detection method

| Field | Content |
|---|---|
| Stage | Dataset audit |
| Prompt | "Grayscale — is there a library or a way to detect using RGB scale?" (my question) |
| AI recommendation | Two checks needed: PIL `mode == "L"` for stored-grayscale, plus per-pixel channel-spread == 0 for grayscale *stored as RGB* — mode alone misses the second kind. |
| Risks / my critique | Whether the second check mattered was an empirical question, not a given. |
| Verification | Full pixel scan of all 5,000 images: 154 grayscale total, of which **140 are RGB-encoded** — the naive mode-only check would have missed 91% of them. EXIF rotations: 0. JSON-vs-file size mismatches: 0. |
| Final decision | Both checks in the audit script; no pipeline action (grayscale images stay in training; YOLO reads 3-channel regardless); domain note for the report. |
| Evidence | exploration.ipynb pixel-scan output; reports/metrics/coco5k_audit.json (pixel section). |

## Entry 6 — Multipolygon merge: my "what if the label was wrong to begin with?" challenge ★

| Field | Content |
|---|---|
| Stage | Label-conversion policy |
| Prompt | "Is merge-with-bridges really best? What if the polygon was wrong from the beginning — then you're merging something that should not be there. Should we remove that from the training set?" (my challenge) |
| AI recommendation | Initially: merge disconnected instances with connecting bridges (Ultralytics semantics). After my challenge, the recommendation was **refined, not just defended**: merging is a *representation* decision (information-preserving encoding of what the label claims); label *quality* is a separate axis that applies to every annotation. Removal fails three ways: no coded rule separates the collage-dog from legitimately occluded objects; sparse random noise is nearly harmless while systematic policies (largest-part-only) bias everything; §4.3 forbids manual edits. |
| Risks / my critique | My concern was legitimate — the sweep proved it: the most-fragmented instance (dog, 17 parts, image 16950) **is** a mislabeled photo collage. Deleting it blindly, however, would require an arbitrary threshold that also deletes valid occluded objects. |
| Verification | Visual inspection of the top fragmented instances via `show_caught("disconnected")`; count check: 3,455 multipolygon instances ≈ 10% of objects — far too many to be mostly errors. |
| Final decision | **Merge + flag, never remove**: merge via Ultralytics `merge_multi_segment` (pinned 8.4.117; internal-API dependency accepted and recorded); instances with ≥8 parts listed in the ledger's `suspicious_flagged`; collage-dog kept, flagged, and cited in audit + error analysis. Optional future §9.5 experiment: train with/without flagged records (prediction: no measurable difference). |
| Evidence | configs/pipeline_coco.yaml (`suspicious_parts_threshold: 8`), docs/COCO_PIPELINE_DECISIONS.md D7, reports/metrics/conversion_segment.json `suspicious_flagged`. |

## Entry 7 — Class imbalance: my oversampling proposal, revised after argument ★

| Field | Content |
|---|---|
| Stage | Training design |
| Prompt | "I try oversampling all the classes that are imbalanced, or get a representative person class — certain zoom, in a group — but I don't know how to segment this." (my proposal) |
| AI recommendation | Against oversampling here: (1) augmentation and sampling are different mechanisms — I had merged them; (2) the arithmetic fails at the tail (hair drier = 7 instances — repetition creates no diversity); (3) COCO5K construction is mandated deterministic; (4) my "representative person sub-groups" are not training-curation targets but **evaluation slices derivable from annotations I already have**: zoom = bbox/image area ratio, groups = person count per image, occlusion = multipolygon parts, truncation = border contact. |
| Risks / my critique | My instinct (ensure good representation of person conditions) was correct in *goal* but aimed at the wrong stage; the assignment's §10.5 slice list turned out to match my sub-groups almost one-for-one. |
| Verification | Measured severity first: person 31.2%, top-5 = 45.6%, max/min ratio 1,612×, toaster/hair-drier at 13/7 — small enough to vanish from a 10% partition (checked in split coverage). |
| Final decision | **Expose, don't fight**: no resampling; per-class metrics; common-vs-minority and density/scale slices at evaluation; rare-class partition coverage flagged by the split script. Changed my plan based on the argument and the numbers. |
| Evidence | NOTES.md "Class imbalance" section; reports/metrics/split_class_coverage.json `rare_watch`; imbalance chart in exploration.ipynb. |

## Entry 8 — Small objects and "zoom": can we upscale without destroying pixels?

| Field | Content |
|---|---|
| Stage | Training design |
| Prompt | "Is there a way to zoom the image without destroying the pixels? Does it even matter for object detection?" (my question) |
| AI recommendation | Interpolation adds no information but **still helps detection** (objects move into feature-map scale regimes the stride-8/16/32 heads handle); the only pixel-preserving zoom is tiling — which is exactly what DOTA patching is; COCO at imgsz=640 is essentially native resolution (verified: long side is 640 or 500 for nearly every image). |
| Risks / my critique | Accepted after the geometry evidence; noted 41% of annotations are sub-32²px, so small-object recall will dominate the error taxonomy. |
| Verification | Image-geometry audit (dimension scatter shows the 640/500 caps; median letterbox padding 27%); effective-size arithmetic in NOTES.md. |
| Final decision | Baseline imgsz=640; imgsz 640→960/1280 kept as the GPU-budget-permitting §9.5 candidate; conf-threshold study remains the zero-GPU default experiment. |
| Evidence | Geometry cell in exploration.ipynb; NOTES.md "What image size affects". |

## Entry 9 — Pipeline architecture: I rejected a single proposal and required alternatives ★

| Field | Content |
|---|---|
| Stage | Implementation planning |
| Prompt | "I need a plan first … we work as partners … there should be multiple proposed pipelines, and ask about all the findings and edge cases so we can decide." (my requirement) |
| AI recommendation | Initially presented one pipeline design. On my rejection, restructured into three architectures (A: scripts+constants, B: scripts+policy-config, C: single orchestrator) and turned **every** edge-case finding into an explicit decision question. |
| Risks / my critique | A single "recommended" plan hides the decision space; I wanted the trade-offs on the table before code existed. |
| Verification | Each policy option was presented with its counts and consequences (e.g., excluding crude polygons = deleting 2% of real objects). |
| Final decision | **Architecture B** — config-driven scripts, every policy a visible line in configs/pipeline_coco.yaml. Eleven policies decided explicitly by me (see registry below). |
| Evidence | configs/pipeline_coco.yaml; docs/COCO_PIPELINE_DECISIONS.md; approved plan file. |

## Entry 10 — Pipeline implementation

| Field | Content |
|---|---|
| Stage | Dataset preparation implementation |
| Prompt | "Propose an implementation plan with everything we discussed and all decision points, then put it into code." |
| AI recommendation | Staged multi-agent build (core → converter+audit → visuals+tests) with an adversarial reviewer checking code against the decision registry, then a fresh end-to-end run + pytest against known expected numbers. |
| Risks / my critique | AI-generated code is only trusted after execution against real data and the independent test suite; the reviewer explicitly hunts for the traps from this log (top-left/center, silent drops, split leakage, stale caches). |
| Verification | Every expected number matched on the real data: manifest 5,000 rows (sha256 recorded); splits exactly 4000/500/500, disjoint, toaster/hair-drier flagged absent from test; both ledgers reconcile exactly (36,213 = 35,765 converted + 448 crowd); 3,455 multipolygon merges; 14 suspicious instances flagged incl. the collage dog [16950, 18863, 17] — flagged, not removed; audit matched exploration (154 grayscale, 635 crude, person 31.15%); 24 label overlays rendered from CONVERTED labels and visually checked; pytest 25/25 green in ~6s; 1-epoch MPS smoke run: Ultralytics scanned the generated layout with 0 corrupt labels. |
| Final decision | Pipeline accepted after fixes. The adversarial reviewer confirmed all load-bearing policies clean and found 7 minor issues — none a data defect: 3 weak tests (no absolute-symlink assert; empty-file floor instead of exact recomputed equality; ledger identity not cross-checked against an independent recount of the raw json), 2 latent converter counter issues (clip counted before zero-area exclusion; suspicious flag nested in the multi-part branch), 1 hard-coded grayscale id in the visualizer, 1 mixed-units "usable" figure in the audit. All 7 fixed; chain re-run; results byte-identical where expected; tests re-passed. |
| Evidence | reports/metrics/conversion_detect.json + conversion_segment.json (reconciliation_ok: true); manifests/HASHES.txt; reports/metrics/split_class_coverage.json; reports/figures/label_checks/*.png; pytest output (25 passed); docs/COCO_PIPELINE_DECISIONS.md. |

## Entry 11 — Training compute: I overrode the AI's Kaggle-first recommendation

| Field | Content |
|---|---|
| Stage | Training configuration (§6 workstream 3) |
| Prompt | "I need one model first and earlier — renting a GPU online, would that be fine and faster, like 1 hour max?" (my override, from an outside suggestion) |
| AI recommendation | Original ladder was Kaggle free tier first (free, ~30 GPU-h/week), rented pod as fallback. When I prioritized time-to-first-model, the AI agreed renting is faster (~30–40 min training on an RTX 4090-class GPU, <$1) and reordered: local MPS smoke test first (never ship unverified labels to a paid pod), then pod, with Kaggle demoted to fallback. Local overnight MPS training remains a documented third option (measured decision rule: time one epoch, train locally if ≥20 epochs fit in ~10h). |
| Risks / my critique | The AI's first plan optimized for cost; my constraint was calendar time. Both are valid — the point is the choice was mine and explicit. Billing gotcha noted: terminate (not stop) the pod. Cloud-side §9.1 env pinning is scripted so the checkpoint's actual machine is recorded. |
| Verification | 1-epoch MPS smoke run on the generated dataset: 0 corrupt labels, checkpoint saved, val ran — layout proven before any upload. On the pod: RTX 4090 confirmed in the env record, 4000 train / 500 val images scanned with 0 corrupt, finite losses from epoch 1, ~20 s/epoch measured (40 epochs ≈ 15 min, well under the ≤6 GPU-h budget). |
| Final decision | Detect model first, from scratch on a rented RTX 4090 (40 epochs, imgsz 640, batch 32, seed 0). Kaggle demoted to fallback; local MPS overnight kept as the last resort. Recorded in INITIAL_PLAN.md revision 2. |
| Evidence | scripts/pod_train_detect.sh; env_record.txt (pod Python/torch/CUDA/GPU); train.log; INITIAL_PLAN.md revision 2. |

## Entry 12 — The AI's data-transfer design failed in the target environment ★

| Field | Content |
|---|---|
| Stage | Deployment of training data to remote compute |
| Prompt | "What's happening?" / "so i have to upload again?" — my questions while the pod stalled |
| AI recommendation | Original: package all 5,000 images + labels into one ~850 MB tar, upload through Jupyter, untar on the pod. It worked in local testing (tar built and inspected on the Mac) — but was never tested end-to-end on a pod. |
| Risks / my critique | It failed in three ways I hit in sequence: (1) I extracted before the upload finished, and the AI's `2>/dev/null` had silenced tar's errors, so the real cause stayed invisible for several rounds — a debugging anti-pattern the AI itself introduced; (2) macOS silently added AppleDouble `._*` sidecar files that appear as bogus label files on Linux; (3) untarring ~15,000 small files onto the pod's network volume crawled at ~1 MB/min (22 MB after ten minutes). I pushed back on repeated "just wait" answers and asked for an estimate, which forced the pivot. |
| Verification | Byte-exact upload confirmed (888,730,112 bytes) — proving the upload was NOT the problem and re-uploading would have wasted more time; `df` ruled out disk space (network volume 77% used); `du` showed extraction progress flatlining on both the network volume and local disk. After the pivot: 5,000 images fetched and verified in seconds, Ultralytics scanned them with 0 corrupt. |
| Final decision | Replaced the design: ship labels + manifests only (7 MB) and let the pod fetch images from the COCO CDN in parallel (`scripts/pod_fetch_images.sh`). Also hardened it beyond the AI's version — the fetch script now decodes every image and checks its dimensions against `manifests/coco5k.csv`, because "file exists and is non-empty" would not catch a truncated or substituted download. Packaging is now a committed script (`scripts/package_labels_for_pod.sh`) that fails if AppleDouble junk survives; previously it was an ad-hoc command that existed nowhere in the repo — an unnoticed §12.1 reproducibility gap. |
| Evidence | scripts/package_labels_for_pod.sh; scripts/pod_fetch_images.sh (manifest verification block); INITIAL_PLAN.md revision 3; pod session log (byte-count check, df, du flatline, successful fetch). |

## Entry 13 — Undertrained first baseline: diagnosis and the fix

| Field | Content |
|---|---|
| Stage | Result interpretation (§6 workstream 6) and training configuration |
| Prompt | "Now it is undertrained, look at the result and let's discuss how to make it better." / "maybe more epochs with better optimiser. and increase pxl" |
| AI recommendation | Diagnosis from `results.csv`: mAP50 was still climbing at epoch 40 (0.018 → 0.022 over the last 10 epochs), losses still falling, and the learning rate had annealed to 4.1e-6 — i.e. a *complete* schedule that was simply too short, not an interrupted run. Also flagged that `optimizer=auto` had selected AdamW at lr0 = 0.000119, ~80× lower than the reference from-scratch recipe (SGD @ 0.01). Recommended 300 epochs + SGD as the baseline, and — against my initial instinct to change all three knobs at once — keeping imgsz as a *separate* controlled experiment. |
| Risks / my critique | I wanted more epochs + better optimizer + bigger images in one run. The AI pushed back: bundling them would make the result unattributable and waste the §9.5 experiment. I accepted the split, but only after checking the reasoning — epochs and optimizer are the standard from-scratch recipe (baseline configuration, not experimentation), while imgsz is a genuine hypothesis to test. I also asked whether class imbalance was the cause; the evidence said no, and I kept my existing "expose, don't fight" policy. |
| Verification | The imbalance question was settled with data, not opinion: `person` has 11,281 training instances (31% of the dataset) and still only reached mAP50 0.214. If imbalance were the binding constraint the head class would be strong and the tail weak; instead everything was weak. Conclusion: training budget, not distribution. Optimizer override confirmed in the new run's log (`optimizer: SGD(lr=0.01, momentum=0.937)`), replacing the auto-selected AdamW. |
| Final decision | Baseline = 300 epochs, SGD lr0 0.01, imgsz 640, seed 0, from scratch. Controlled experiment = the same run at imgsz 960, single variable, **hypothesis pre-registered and committed before either run finished** (`reports/metrics/experiment_imgsz_hypothesis.md`), grounded in my own audit finding that 41% of COCO5K annotations are under 32² px. The 40-epoch run is kept as honest evidence about training duration, described as an observation rather than a pre-registered experiment. Class imbalance stays unaddressed in training by design; copy-paste augmentation (using the segment masks I already produced) is recorded as the recommended next experiment. |
| Evidence | artifacts/coco5k-detect-40ep/ (results.csv, args.yaml, per-class metrics); reports/metrics/experiment_imgsz_hypothesis.md; train_both.log optimizer line; INITIAL_PLAN.md revisions. |

## Entry 14 — DOTA label format checked against the assignment's warnings

| Field | Content |
|---|---|
| Stage | Dataset preparation (DOTA5K) |
| Prompt | First inspection of the DOTA v1.0 train split after extracting images and labels. |
| AI recommendation | Earlier in the session the AI listed DOTA parsing traps to expect: metadata header lines before the annotations, exactly 10 tokens per annotation line (8 coordinates + class + difficulty), and the risk of accidentally holding v1.5/v2.0 labels, which add a 16th class (container-crane) and would silently break a 15-class mapping. |
| Risks / my critique | These were claims about a dataset neither of us had opened yet — worth checking before writing any converter, since a wrong class list or a naive line parser corrupts everything downstream. |
| Verification | Inspected `labelTxt/P0000.txt` directly: the first two lines are indeed `imagesource:GoogleEarth` and `gsd:0.146343590398`, followed by 10-token annotation lines. Tallied class names across all 1,411 label files: **exactly 15 distinct names** — ship 28,068, small-vehicle 26,126, large-vehicle 16,969, plane 8,055, harbor 5,983, storage-tank 5,029, tennis-court 2,367, bridge 2,047, swimming-pool 1,736, helicopter 630, basketball-court 515, baseball-diamond 415, roundabout 399, soccer-ball-field 326, ground-track-field 325. No `container-crane`, confirming this is genuinely v1.0. |
| Final decision | Parser will skip non-numeric-leading lines explicitly (never a bare try/except, which would swallow real annotations), assert 10 tokens per line, and assert the observed class set equals the 15 official v1.0 names. Also noted for the audit: DOTA is imbalanced too (ship ≈ 86× ground-track-field), so the same "measure and slice, don't resample" policy carries over. |
| Evidence | datasets/dota/labelTxt/P0000.txt (header lines visible); class tally over all 1,411 label files (15 names, counts above); 1,411 images / 1,411 label files paired. |

---

## Substantive corrections and rejections (assignment requires ≥3)

1. **Entry 4** — rejected the AI's deliverable twice (diagram, then auto-gallery) and replaced it with a simpler, more useful runnable function. *(Replaced an overcomplicated proposal with a simpler solution.)*
2. **Entry 6** — challenged the merge recommendation's hidden assumption (that the source label is trustworthy); the policy was amended to merge **+ flag-never-remove**, and the sweep proved the concern real (collage-dog). *(Identified an unsupported assumption.)*
3. **Entry 7** — my own oversampling plan was revised after argument and measurement; representation concerns were redirected into §10.5 slice analysis. *(Changed a plan after evidence contradicted the initial idea.)*
4. **Entry 9** — rejected the single-plan presentation; required multiple pipeline proposals and explicit per-finding decisions. *(Rejected an AI recommendation on process.)*
5. **Entry 2** — the AI's rasterization-noise assumption (~10% tolerance needed) measured as <0.05% in practice. *(Found an assumption to be empirically unfounded; tolerance kept as defensive slack.)*
6. **Entry 13** — I proposed changing epochs, optimizer, and image size in one run; after arguing it through I split them into a properly-configured baseline plus a single-variable experiment, and pre-registered the hypothesis before any result existed. I also questioned whether class imbalance was to blame and accepted the evidence that it was not. *(Changed my own plan after argument and measurement.)*
7. **Entry 12** — the AI's data-transfer design worked locally but failed on the pod (network-volume untar at ~1 MB/min, macOS sidecar files, errors hidden by its own `2>/dev/null`). I refused "just wait", demanded an estimate, and the design was replaced with a 7 MB labels upload plus CDN fetch — then hardened further with manifest verification the AI's version lacked. *(Corrected AI-generated code after it failed in the real target environment.)*

## Decided vs. still open

**Decided (all confirmed by me, enforced in configs/pipeline_coco.yaml):** COCO-only round first · scripts+policy-config architecture (B) · 80/10/10 seed-0 split shared by both COCO tasks · iscrowd exclude+count · multipolygon merge via Ultralytics + suspicious flag ≥8 parts, never removed · crude polygons pass-through + audit flag · degenerate exclude+count (no fail-fast) · tiny keep-all · out-of-bounds clip+count · grayscale/imbalance measure-report-only · symlink image layout · from-scratch weights (§9.2).

**Still open (deliberately deferred):** DOTA difficulty-field policy and background-patch policy (after DOTA exploration) · final controlled-experiment choice (conf-threshold default vs imgsz if GPU budget allows) · export parity tolerance numbers (draft: ≥95% matched at IoU≥0.9, |Δconf|≤0.02 — to finalize at export time) · confidence-threshold selection criterion (likely max-F1 on val) · deployment hardware assumption (CPU/ONNX assumed; logged as an INITIAL_PLAN ambiguity).

## Provenance note (§6 requirement)

The assignment requires at least one **training** component and one **deployment** component to begin from an AI-assisted draft. Planned: `src/train/train.py` and `src/inference/predict.py` will start as AI drafts in later rounds; their review-and-correction will be logged here when it happens. The exploration sweep and the data pipeline in this round also began as AI drafts, verified by execution against real data, the pycocotools oracle, and the pytest suite.
