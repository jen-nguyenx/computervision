# Internal Notes: Terms Used in This Project

Personal glossary. Plain-language definitions of every term this assignment uses.
Not a deliverable, just my working reference.

## Data and annotation terms

- **Annotation**: one labeled object in one image. Example: "there is a person at this box location". One image can have many annotations.
- **Bounding box (bbox)**: a rectangle around an object. COCO stores it as [x, y, width, height] in pixels, where x and y are the top-left corner. Verified this visually in the notebook.
- **Axis-aligned**: a rectangle whose sides are parallel to the image edges. It cannot tilt.
- **Oriented bounding box (OBB)**: a rectangle that can rotate. Stored as four corner points. Used for aerial images where objects appear at any angle.
- **Quadrilateral**: any four-cornered shape. DOTA labels are quadrilaterals. They are usually close to rotated rectangles but not exactly.
- **Segmentation / mask**: the exact outline of an object, not just a rectangle. COCO stores it as a polygon, a list of points along the object's edge.
- **Polygon**: a shape defined by a list of corner points. COCO segmentation polygons can have hundreds of points.
- **RLE (run-length encoding) / encoded mask**: a compressed way to store a mask as numbers instead of polygon points. COCO uses it for crowd annotations. Looks like a dict with a "counts" field.
- **iscrowd**: a flag on every COCO annotation. 0 means one individual object. 1 means one annotation lazily covering many objects at once (a crowd of people, a pile of fruit). My policy: exclude iscrowd=1 from training, count and log every exclusion.
- **Category ID / class ID**: the number that says what kind of object it is. COCO category IDs run 1 to 90 with gaps but there are only 80 classes. Ultralytics needs gap-free IDs 0 to 79, so I need an explicit mapping table.
- **Patch**: a small tile cut out of a huge image. DOTA images are too big to train on directly, so they are cut into 1024 x 1024 patches with 200 pixels of overlap.
- **Background patch**: a patch that contains no objects at all. Kept in the dataset unless I justify otherwise.
- **Difficulty (DOTA)**: a per-object flag in DOTA marking objects the annotators considered hard (tiny, blurry). I need a policy for whether to train and evaluate on them.
- **Manifest**: a small file (CSV) listing exactly which images are in my dataset, so anyone can rebuild it. The opposite of "whatever files happened to be in the folder".
- **Deterministic**: running the same script twice gives byte-identical output. Required for the manifests.

## Dataset split terms

- **Train / validation / test split**: three separate piles of data. Train teaches the model. Validation is for my decisions (which epoch, which threshold). Test is touched once at the very end to report honest numbers.
- **Held-out**: kept away from training and tuning. The test set is held out.
- **Frozen**: written to a file before training and never changed afterwards.
- **Seed**: a number that fixes randomness. Same seed, same "random" result every run. Makes shuffling reproducible.
- **Leakage**: when information from the test set sneaks into training, making results look better than they really are. My main risk: overlapping DOTA patches from the same source image landing in both train and test. Prevented by splitting by source image.

## Training terms

- **Epoch**: one full pass through all training images. Training runs many epochs.
- **Batch / batch size**: how many images the model looks at in one step. Bigger batches use more memory.
- **Loss**: the number the model tries to make small during training. If it goes down, learning is happening. If it is NaN or infinite, something is broken.
- **Optimizer**: the algorithm that updates the model weights each step (SGD, AdamW).
- **Learning rate**: how big each update step is. Too big diverges, too small crawls.
- **Checkpoint**: the model's weights saved to a file (best.pt, last.pt). Can be loaded later to continue or to run predictions.
- **Pretrained weights**: a model that someone already trained on a big dataset, used as a starting point. Trap in this assignment: the standard checkpoints were trained on COCO and DOTA, which contain my test images. That is why I train from scratch.
- **From scratch**: starting with random weights instead of pretrained ones. Needs more epochs but avoids the contamination problem.
- **Fine-tuning**: continuing training from pretrained weights. Not my main approach here.
- **Augmentation**: random changes to training images (flips, color shifts, scaling) so the model sees more variety. Applied only during training, never during evaluation.
- **Smoke test**: a tiny fast run (few images, one epoch) to prove the pipeline works before spending hours on a real run.
- **Early stopping**: stopping training when validation performance stops improving.
- **imgsz**: Ultralytics name for input image size. Images are resized to this before entering the model, for example 640.
- **Device (mps, cuda, cpu)**: which chip does the math. mps is the Apple GPU on my Mac. cuda is NVIDIA. cpu is the slow fallback.
- **Nano model (yolo11n)**: the smallest YOLO size. Fastest to train, least accurate. The brief recommends it.

## Evaluation terms

- **Inference / prediction**: running the trained model on an image to get outputs. No learning happens.
- **Confidence**: the model's score (0 to 1) for how sure it is about a prediction.
- **Confidence threshold**: the cutoff below which predictions are thrown away. I must choose this using validation data and justify it, not just accept the default.
- **IoU (intersection over union)**: how much a predicted box overlaps the true box. 1.0 is a perfect match, 0 is no overlap. Used to decide whether a prediction counts as correct.
- **Precision**: of everything the model predicted, what fraction was right. Low precision means many false alarms.
- **Recall**: of everything that was actually there, what fraction the model found. Low recall means many misses.
- **False positive**: the model predicted an object that is not there.
- **False negative**: the model missed an object that is there.
- **mAP (mean average precision)**: the standard single-number score for detection. Averages precision across all classes and confidence levels. Higher is better.
- **mAP50**: mAP where a prediction counts as correct if IoU is at least 0.50. The lenient version.
- **mAP50-95**: mAP averaged over IoU thresholds from 0.50 to 0.95. The strict version. Always lower than mAP50.
- **Box mAP vs mask mAP vs oriented mAP**: the same idea measured on different geometry. A segmentation model must be judged on mask mAP, an OBB model on oriented mAP. Reporting the wrong one is a listed mistake in the brief.
- **Per-class metrics**: the score broken down by class, to see which classes are weak.
- **NMS (non-maximum suppression)**: post-processing that removes duplicate overlapping predictions of the same object.
- **Slice analysis**: measuring performance on subgroups (small objects vs large, dense images vs sparse) instead of one overall number.
- **Error taxonomy**: sorting failures into named categories (missed small object, wrong class, duplicate box) with rough counts.

## Deployment terms

- **Export**: converting the trained PyTorch model (.pt) into a portable format that runs without PyTorch.
- **ONNX**: the most common such portable format. Runs via onnxruntime.
- **Runtime**: the software that executes the exported model (onnxruntime).
- **Parity check**: comparing the original model and the exported model on the same images to confirm they give (nearly) the same answers. I define the tolerance before measuring.
- **Latency**: how long one inference takes, usually milliseconds.
- **Median latency**: the middle value over many runs. Better than the average because it ignores outliers.
- **P95 latency**: 95 percent of runs are faster than this. Captures the bad cases.
- **Throughput**: images processed per second.
- **Warm-up**: the first few inferences are slow (loading, caching). They are run and discarded before measuring latency.
- **CLI (command-line interface)**: a script run from the terminal with arguments, like predict.py --task detect --source images/.
- **Docker / Dockerfile**: a recipe that packages the code and its environment into a container so it runs the same on any machine.

## Process terms

- **Reproducible**: another person with the code and the data gets the same results by following the README.
- **Pinning versions**: recording exact library versions (ultralytics 8.4.117) so the environment can be recreated.
- **Audit**: systematic counting and summarising of the dataset (class counts, image sizes, oddities) before trusting it.
- **Visual validation**: drawing labels on images and checking with my own eyes that they line up. Code running without an error proves nothing about correctness.
- **Controlled experiment**: changing exactly one thing, predicting the effect beforehand, and measuring whether the prediction held.
- **Baseline**: the first honest, simple version of a model. Improvements are measured against it.

## COCO exploration plan (guide for exploration.ipynb)

Goal: understand the data before writing pipeline code. Budget ~1.5 hours. Done when Part 7's policy cell is written.

- **Part 0 — Setup**: download only the annotations zip (~241 MB). No images yet — every image record has a `coco_url`, so I can fetch single images on demand. Install pycocotools, matplotlib, pillow, pandas.
- **Part 1 — Read the JSON first**: top-level keys; print one entry each from `images`, `annotations`, `categories`; count them (~118k images, ~860k annotations); print sorted category IDs to see the 1–90-with-gaps problem myself. Note the `width`/`height` fields — checked against real pixels in Part 5.
- **Part 2 — One image by hand**: lowest-ID image, downloaded via `coco_url`. Draw its boxes treating bbox as top-left [x, y, w, h]. Deliberately draw them wrong once (pretend x,y is the center) to see the half-box shift. Draw the polygons, label class names via my own mapping.
- **Part 3 — Verify against pycocotools**: same image through `COCO()` / `showAnns()`; confirm my hand-drawn version matches. See "Why verify against an independent library" below.
- **Part 4 — Edge-case hunt**: find and display one of each, keep a tally:
  - Geometry: tiny object (area < 32²), object touching the border, extreme aspect ratio.
  - Occlusion: a multipolygon instance (one object, disconnected parts), two touching same-class instances.
  - Density: an empty image; the most crowded image in the first 5,000 IDs.
  - Label pathologies: an iscrowd=1 annotation (its segmentation is an RLE dict, not polygons — decode and display), a degenerate polygon (<6 coordinate values) or zero-area box.
  - File-level: check a handful of images for grayscale and EXIF orientation flags; verify JSON width/height against actual pixels.
  - Every find becomes a counted, documented converter policy later.
- **Part 5 — Statistics over the sorted-first-5,000** (this IS COCO5K, prototyped):
  - Imbalance: class distribution (person dominates — sanity anchor); minority-class census — which classes have <~20 instances, would a 10% split lose them entirely?
  - Scale: area distribution (log scale); small/medium/large at the 32²/96² thresholds; effective size after a 640 letterbox = area × (640 / long side)² — count objects below ~10 px, the model will never see those.
  - Image geometry: width×height scatter, aspect-ratio histogram, how much letterbox padding the extremes imply.
- **Part 6 — One conversion, round-tripped**: hand-compute one bbox→YOLO example on paper (bbox [10,20,30,40] in 100×200 → cx 0.25, cy 0.20, w 0.30, h 0.20), write the function + inverse, assert round-trip within a pixel. Becomes one of my ≥5 required tests verbatim.
- **Part 7 — Findings → policies**: closing markdown cell translating findings into draft INITIAL_PLAN.md assumptions: iscrowd policy, multipolygon policy, degenerate-annotation policy, empty-image handling, imbalance stance (report and slice, don't resample), imgsz rationale.

## Why verify my drawings against pycocotools

- My own eyes can't catch a bug that is consistent with itself. If I misread the bbox format, my visualizer, converter, and audit would all share the same wrong assumption and agree with each other — a **correlated error**. The image still looks plausible.
- pycocotools is an **independent oracle**: written by the dataset authors, it encodes the official interpretation of the format. If my overlay matches `showAnns()`, my *understanding* is validated, not just "the code ran".
- Same habit applies to AI-generated code later: agreement with an authoritative independent source is verification; "it looked reasonable" is not.

## What counts as an edge case in imaging (general, not just COCO)

An edge case = any input where the pipeline's or model's typical assumptions stop holding. Five families:

- **Geometry/size extremes**: tiny objects (invisible after resize), objects filling the frame, extreme aspect ratios (pole, train), objects cut by the image border.
- **Occlusion and instance confusion**: hidden objects; one object split into disconnected visible parts; touching same-class instances (merge risk); reflections and pictures-of-pictures (is a person on a poster a "person"?).
- **Density extremes**: zero-object images (teach the model what NOT to detect) and extremely crowded scenes (NMS and annotation quality both strain).
- **Appearance conditions**: low light, motion blur, glare, weather, low contrast, odd viewpoints, huge within-class scale variation.
- **Data pathologies** (edge cases of files/labels, not scenes): wrong or ambiguous labels, loose boxes, missing/duplicate annotations, degenerate geometry (zero-area, <3 points, self-intersections), out-of-bounds coordinates, corrupt files, grayscale images in an RGB pipeline, **EXIF rotation** (pixels stored unrotated + a metadata flag; tools disagree, annotations can silently misalign).

Why they matter: models fail disproportionately here while average metrics hide it; pipelines crash (best case) or mishandle silently (worst case). Habit: enumerate, count, write a policy per category.

## Class imbalance (person dominates COCO)

- Imbalance mirrors the real world; it is not automatically a disease. Detection is gentler than classification — each object is predicted independently. Real effects: rare classes get few gradient updates (worse recall, worse calibration), confusions drift toward frequent classes.
- General toolbox (mostly NOT used here): more rare-class data; oversampling images with rare classes; loss reweighting; copy-paste augmentation; per-class thresholds.
- **This assignment: expose it, don't fight it.** COCO5K is mandated and deterministic, so no resampling. What scores: class distribution in the audit; check rare classes survive into the 10% partitions (the brief requires explaining absent/poorly-represented classes); per-class metrics; common-vs-minority classes as one slice analysis. Honest analysis beats clumsy correction.

## What image size affects

- **Everything is letterboxed to imgsz** (e.g. 640): scale until the long side fits, pad with grey. What matters is object size AFTER resize: effective size ≈ original × (imgsz / longest image side). A 50 px car in a 4000 px aerial image becomes ~8 px — invisible. This is exactly why DOTA is patched into 1024 tiles instead of resized whole.
- **Aspect ratio = wasted pixels**: very wide images letterboxed to a square are mostly padding. Hence the aspect-ratio audit.
- **Normalized coordinates exist because of resizing** — but only work if normalized by the correct per-image dimensions (DOTA edge-patch trap: actual patch size, never hardcoded 1024).
- **Metadata can lie**: assert JSON width/height == actual pixel dimensions in the audit; a mismatch corrupts every normalized coordinate for that image.
- **Cost scales ~quadratically with imgsz** for training and inference — the classic accuracy-vs-latency trade-off, and a candidate controlled experiment.
