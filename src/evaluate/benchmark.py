"""Dedicated inference efficiency benchmark (assignment §10.7).

§10.7 asks for latency, throughput and model size reported next to accuracy, so the
report can argue about the accuracy/cost trade-off. This script produces those numbers
properly; it deliberately does NOT reuse the latency figures already sitting in
reports/metrics/slices_val.json.

**Why the existing numbers cannot be quoted as steady state.** Those were measured
inside the mAP inference pass: no warm-up, no device synchronisation, and a different
image on every call. The first call therefore carries model-load, weight transfer and
Metal graph compilation, which is visible as the 960 model's P95 of ~162 ms against a
median of ~17.5 ms. A P95 an order of magnitude above the median is not a tail, it is
one contaminated sample.

What this script does instead, per model:

  1. **Load time, measured and reported separately.** Two numbers, because they are two
     different costs: `YOLO(path)` construction (read checkpoint, build the module) and
     the first inference, which is what actually triggers the device transfer and the
     lazy Metal graph build. Their sum is the cold-start cost a service pays once.
  2. **>= 20 warm-up inferences, discarded** — and at least one full second of them,
     whichever is longer. The count alone is not sufficient: at ~6 ms a call, 20
     iterations is 0.12 s, too short for the GPU to leave its idle power state, and the
     measurement then captures the clock ramping rather than steady state.
  3. **>= 100 measured inferences** on ONE fixed image from the validation split, so
     input size, object count and NMS load are held constant across every sample and
     across both models. That block is repeated N_ATTEMPTS times and the least-contended
     steady block is reported — this machine shares its GPU with the window compositor,
     and a single block can land in a visibly slower regime through no fault of the
     model. See measure_latency_stable() for why that selection is not cherry-picking.
  4. **torch.mps.synchronize() before every clock read.** MPS dispatch is asynchronous:
     `predict()` can return before the GPU has finished, so without an explicit
     synchronisation the measured time is silently too small and the cost simply
     reappears in a later sample. Synchronising both before starting and before stopping
     the clock makes each sample a complete, self-contained unit of work.

Training-side facts (duration, peak GPU memory) are read out of
reports/metrics/run_records.md, never restated from memory. A field that is not in that
file is reported as "not recorded" rather than guessed. Note that those were measured on
the training machine (RunPod RTX 4090), while the latency here is measured on the local
Apple Silicon machine — the two are not comparable and the output says so.

VALIDATION ONLY, structurally. There is no --split flag: the split is pinned to "val" by
BENCHMARK_SPLIT so this script cannot be pointed at the frozen test partition (§10
preamble). Latency does not depend on which partition an image came from, so there is
nothing to gain from test anyway.

Usage:
    python -m src.evaluate.benchmark \
        --model 640=artifacts/.../detect-300ep-640/weights/best.pt \
        --model 960=artifacts/.../detect-300ep-960/weights/best.pt
"""

import argparse
import json
import math
import platform
import statistics
import time
from pathlib import Path

from src.data.common import DEFAULT_CONFIG, load_config
from src.evaluate.coco_eval import PRED_CONF, PRED_IOU, PRED_MAX_DET, rows_for_split
from src.evaluate.eval_slices import IMGSZ_FROM_LABEL

# The split is a constant, not a flag: test must not be touched (§10 preamble).
BENCHMARK_SPLIT = "val"

# Protocol minimums from the approved plan. The CLI may raise them, never lower them.
MIN_WARMUP = 20
MIN_MEASURED = 100
# Warm-up also has a wall-clock floor. A count alone does not guarantee the GPU has left
# its idle power state on a fast model — see the comment in measure_latency().
WARMUP_MIN_SECONDS = 1.0
# The measured block has a wall-clock floor for the same reason: a block that lasts well
# under a second is too short to distinguish the model's cost from desktop interference.
MEASURE_MIN_SECONDS = 1.5
# A measured block is accepted when the medians of its first, middle and last third
# agree to within this fraction of the overall median. Quiet runs on this machine land
# at 0.5-4%, so 5% is comfortably achievable and anything above it means the GPU was
# being shared with something else (on a Mac, usually the window compositor).
STEADY_MAX_SPREAD = 0.05
# How many times the whole measured block is repeated. All of them always run; the
# least-contended steady block is reported. See measure_latency_stable().
N_ATTEMPTS = 5

# Default hardware description for the report. Verified against sysctl at run time and
# the detected values are stored alongside, so a wrong string cannot pass unnoticed.
DEFAULT_HARDWARE = "Apple M5 Pro, 64 GB, MPS backend"

RUN_RECORDS_FILE = "run_records.md"
SLICES_FILE = f"slices_{BENCHMARK_SPLIT}.json"


def pick_benchmark_image(cfg, image_id=None):
    """Choose the one fixed validation image every measurement runs on.

    Default is the LOWEST image id in the val split — an arbitrary but deterministic
    choice, so the benchmark is reproducible and both models are timed on identical
    input. Returns the manifest row as a plain dict plus the absolute image path.
    """
    rows = rows_for_split(cfg, BENCHMARK_SPLIT).sort_values("image_id")
    if image_id is None:
        row = rows.iloc[0]
    else:
        selected = rows[rows["image_id"] == image_id]
        if selected.empty:
            raise SystemExit(f"image_id {image_id} is not in the {BENCHMARK_SPLIT} split")
        row = selected.iloc[0]
    return {
        "image_id": int(row.image_id),
        "file_name": str(row.file_name),
        "width": int(row.width),
        "height": int(row.height),
        "path": str(cfg["paths"]["images_dir"] / row.file_name),
    }


def run_id_from_checkpoint(model_path):
    """Training run directory name for a checkpoint, e.g. '.../detect-300ep-640/weights/best.pt'
    -> 'detect-300ep-640'. That name is the key used to find the run in run_records.md.
    """
    return Path(model_path).resolve().parent.parent.name


def parse_run_records(path):
    """Parse reports/metrics/run_records.md into a list of {field: value} dicts.

    The file is a sequence of '## Run N — `run-id`' sections, each holding a two-column
    markdown table of Field | Value. It is parsed rather than transcribed so the
    benchmark output cannot drift from the training record. Field names are lower-cased
    and stripped of markdown bold so lookups are stable.
    """
    if not Path(path).exists():
        return []

    records = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("## "):
            records.append({})
            continue
        if not records or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2 or set(cells[0]) <= set("-: "):
            continue                              # separator row
        field = cells[0].strip("*").strip().lower()
        records[-1][field] = cells[1].strip()
    return records


def training_facts(records, run_id, source_path):
    """Training duration and peak GPU memory for one run, straight from run_records.md.

    The run is found by matching `run_id` (the checkpoint's run directory name) against
    the 'Run ID' row of each section. Every field is either the recorded string or the
    literal "not recorded" — this function never fills a gap with an estimate. Run 3
    (the 960 model) genuinely has no peak-memory row, so "not recorded" is the correct
    output there, not a bug.
    """
    file_name = Path(source_path).name
    record = next((r for r in records if run_id in r.get("run id", "")), None)
    if record is None:
        missing = f"not recorded (no section for run id '{run_id}' in {file_name})"
        return {
            "run_id": run_id,
            "source": str(source_path),
            "training_duration": missing,
            "peak_gpu_memory": missing,
            "training_hardware": missing,
            "training_median_latency_quoted": missing,
        }

    def field(name):
        value = record.get(name)
        return value if value else f"not recorded (no '{name}' row in {file_name})"

    # The latency row in run_records.md is the CONTAMINATED in-pass figure; it is carried
    # through only so the comparison table can show what changed.
    latency_key = next((k for k in record if k.startswith("median latency")), None)
    return {
        "run_id": run_id,
        "source": str(source_path),
        "training_duration": field("duration"),
        "peak_gpu_memory": field("peak gpu memory"),
        "training_hardware": field("hardware"),
        "training_median_latency_quoted": (
            record[latency_key] if latency_key else f"not recorded (no latency row in {file_name})"
        ),
    }


def in_pass_reference(cfg, label):
    """The contaminated latency for this model from slices_{split}.json, for comparison.

    Those numbers were produced inside the mAP pass with no warm-up and no
    synchronisation. They are quoted here purely so the report can state, with both
    numbers side by side, how much proper warm-up moved the measurement.
    """
    path = Path(cfg["paths"]["metrics_dir"]) / SLICES_FILE
    if not path.exists():
        return None
    with open(path) as f:
        doc = json.load(f)
    entry = doc.get("models", {}).get(label)
    if not entry:
        return None
    return {
        "source": str(path),
        "median_ms": entry.get("latency_median_ms"),
        "p95_ms": entry.get("latency_p95_ms"),
        "caveat": "measured inside the mAP pass: no warm-up, no torch.mps.synchronize(), "
                  "a different image every call — first-call cost is included",
    }


def init_device_context(device, torch):
    """Create the MPS context once, before any model is loaded.

    The very first tensor operation in a process pays for building the Metal device
    context. Without this, whichever model happened to be benchmarked first would carry
    that one-off process cost in its load time and would look slower than an identical
    model benchmarked second. Paying it here up front makes the per-model load times
    comparable to each other.
    """
    if device == "mps":
        torch.zeros(1, device="mps") + 1
        torch.mps.synchronize()


def load_model(model_path, torch):
    """Time YOLO(path): read the checkpoint off disk and build the nn.Module.

    Nothing has touched the GPU at this point — Ultralytics constructs on CPU and only
    moves to the device on the first predict() — so this is pure CPU and disk time. The
    synchronize() beforehand just drains anything already queued so t0 is a clean start.
    """
    from ultralytics import YOLO

    torch.mps.synchronize()
    started = time.perf_counter()
    model = YOLO(model_path)
    return model, (time.perf_counter() - started) * 1000


def time_first_inference(model, image_path, imgsz, device, torch):
    """Time the first predict() call — the other half of cold start, measured separately.

    This is where the weights move to the MPS device, AutoBackend is built and Metal
    compiles the graph for this input shape. It is far slower than steady state, which is
    exactly why it must be reported on its own and must not sit inside the measured
    samples. torch.mps.synchronize() brackets it because MPS dispatch is asynchronous:
    without it the call returns before the GPU has finished and the compile cost would
    leak into the first warm-up iteration instead of being attributed here.
    """
    started = time.perf_counter()
    model.predict(image_path, imgsz=imgsz, conf=PRED_CONF, iou=PRED_IOU,
                  max_det=PRED_MAX_DET, device=device, verbose=False)
    torch.mps.synchronize()
    return (time.perf_counter() - started) * 1000


def count_parameters(model):
    """Parameter count of the live nn.Module.

    Call this BEFORE the first predict(): Ultralytics fuses Conv+BatchNorm on the first
    inference, which folds the BatchNorm parameters away and lowers the count. The
    as-trained number is the one that matches run_records.md and the published YOLO11n
    figure; the post-fusion number is what actually executes. Both are reported.
    """
    return sum(p.numel() for p in model.model.parameters())


def measure_latency(model, image_path, imgsz, device, n_warmup, n_measured, torch):
    """Steady-state single-image latency: n_warmup discarded, then n_measured timed.

    One fixed image, batch size 1, the same predict() call path the mAP pass uses, so the
    only differences from the numbers in slices_val.json are the warm-up and the
    synchronisation. Each sample is end-to-end: image read + letterbox preprocess +
    forward pass + NMS + postprocess — what a caller actually waits for.

    torch.mps.synchronize() is called immediately before the clock starts (so no earlier
    queued work is charged to this sample) and again before the clock stops (so the GPU
    has genuinely finished). Without the second call MPS returns early and every timing
    here is silently under-measured.
    """
    # Warm up for at least n_warmup iterations AND at least WARMUP_MIN_SECONDS of wall
    # clock, whichever takes longer. The count alone is not enough: at ~6 ms per call, 20
    # iterations is 0.12 s, which is too short for the GPU to leave its idle power state,
    # and the run then measures the clock ramping up rather than steady-state latency.
    # This was not a hypothetical — with a count-only warm-up the 640 model's stability
    # check failed at an 11% spread across the run.
    warmup_started = time.perf_counter()
    n_warmup_done = 0
    while (n_warmup_done < n_warmup
           or time.perf_counter() - warmup_started < WARMUP_MIN_SECONDS):
        model.predict(image_path, imgsz=imgsz, conf=PRED_CONF, iou=PRED_IOU,
                      max_det=PRED_MAX_DET, device=device, verbose=False)
        n_warmup_done += 1
    torch.mps.synchronize()

    # As with the warm-up, the measured block runs for at least n_measured iterations AND
    # at least MEASURE_MIN_SECONDS. The count floor is the protocol requirement; the time
    # floor is what makes the block long enough to be judged. At ~6 ms a call, 100
    # iterations is 0.6 s, short enough that one hiccup from the window compositor shifts
    # the median of an entire third and the stability check rejects an otherwise fine run.
    samples = []
    n_detections = None
    measure_started = time.perf_counter()
    while (len(samples) < n_measured
           or time.perf_counter() - measure_started < MEASURE_MIN_SECONDS):
        torch.mps.synchronize()
        started = time.perf_counter()
        result = model.predict(image_path, imgsz=imgsz, conf=PRED_CONF, iou=PRED_IOU,
                               max_det=PRED_MAX_DET, device=device, verbose=False)[0]
        torch.mps.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
        n_detections = len(result.boxes)

    samples_sorted = sorted(samples)
    median_ms = statistics.median(samples_sorted)
    mean_ms = statistics.fmean(samples_sorted)
    # Nearest-rank P95: the smallest sample at or above the 95th percentile position.
    # With 100 samples that is simply the 95th value — no interpolation to explain.
    p95_index = math.ceil(0.95 * len(samples_sorted)) - 1
    p95_ms = samples_sorted[p95_index]
    return {
        "n_warmup_requested": n_warmup,
        "n_warmup": n_warmup_done,
        "warmup_seconds": time.perf_counter() - warmup_started,
        "n_measured_requested": n_measured,
        "n_measured": len(samples),
        "median_ms": median_ms,
        "p95_ms": p95_ms,
        "mean_ms": mean_ms,
        "std_ms": statistics.stdev(samples_sorted) if len(samples_sorted) > 1 else 0.0,
        "min_ms": samples_sorted[0],
        "max_ms": samples_sorted[-1],
        # Batch size 1 and one image at a time, so sustained throughput is simply the
        # reciprocal of the mean latency. Not a batched-throughput figure.
        "throughput_img_per_s": 1000.0 / mean_ms if mean_ms else 0.0,
        "detections_on_benchmark_image": n_detections,
        "stability": stability_check(samples, median_ms),
        # Every raw sample, in measurement order, so the summary statistics above can be
        # recomputed or re-plotted without re-running the benchmark.
        "samples_ms": samples,
    }


def measure_latency_stable(model, image_path, imgsz, device, n_warmup, n_measured, torch):
    """Repeat the measured block N_ATTEMPTS times and report the least-contended one.

    This is a laptop whose GPU is shared with the macOS window compositor, not a
    dedicated benchmarking host, and it shows: the same model measured minutes apart
    produced medians of 6.0 ms and 10.0 ms, each internally consistent to within 1%. A
    single block therefore reports whichever regime the machine happened to be in, and
    the within-run stability check cannot tell the difference — both blocks look steady.

    The fix is the standard one for latency benchmarking on a shared machine: repeat the
    whole block and take the **minimum median** across the repeats. Interference is
    strictly additive — another process competing for the GPU can only make this model
    slower, never faster — so the fastest steady block is the one with the least foreign
    work in it, and is the best available estimate of the model's own cost. Blocks that
    fail the within-run stability check are discarded first, because a block that drifted
    mid-run has no single meaningful median.

    This is not cherry-picking: the rule is fixed in advance, every attempt is run (no
    early exit on a lucky first block), and every attempt's median is written to the
    output so the spread is visible to the reader.
    """
    attempts = []
    for i in range(N_ATTEMPTS):
        attempts.append(measure_latency(model, image_path, imgsz, device,
                                        n_warmup, n_measured, torch))
        stability = attempts[-1]["stability"]
        print(f"    attempt {i + 1}/{N_ATTEMPTS}: median {attempts[-1]['median_ms']:.2f} ms, "
              f"spread {stability['spread_fraction_of_median']:.1%} "
              f"{'' if stability['steady'] else '(drifted, discarded)'}")

    steady = [a for a in attempts if a["stability"]["steady"]] or attempts
    best = min(steady, key=lambda a: a["median_ms"])
    best["attempts"] = len(attempts)
    best["attempt_medians_ms"] = [a["median_ms"] for a in attempts]
    best["attempts_discarded_as_drifted"] = len(attempts) - len(
        [a for a in attempts if a["stability"]["steady"]])
    best["selection_rule"] = (
        "minimum median across the attempts that passed the within-run stability check; "
        "GPU contention is additive, so the fastest steady block is the least contended"
    )
    return best


def stability_check(samples, median_ms):
    """Split the samples into thirds and compare their medians.

    A benchmark can be spoiled in a way no summary statistic reveals: if the machine is
    busy, or if the warm-up was too short, the whole run shifts and the median moves with
    it — it still looks like a tidy number. Comparing the first, middle and last third
    makes that visible. A run whose thirds agree to within a few percent is steady; a
    large spread means the measurement should be repeated on a quiet machine.
    """
    size = len(samples) // 3
    thirds = [statistics.median(samples[i * size:(i + 1) * size]) for i in range(3)]
    spread = (max(thirds) - min(thirds)) / median_ms if median_ms else 0.0
    return {
        "third_medians_ms": thirds,
        "spread_fraction_of_median": spread,
        "steady": spread <= STEADY_MAX_SPREAD,
        "max_spread_accepted": STEADY_MAX_SPREAD,
        "note": "medians of the first, middle and last third of the measured samples; "
                "steady=false means the run drifted or the machine was contended and the "
                "numbers should be re-measured",
    }


def static_facts(model, model_path, imgsz, device, torch, parameters_as_trained):
    """Size and precision facts read off the loaded model, not off the config file.

    parameters      the as-trained count captured before Conv+BatchNorm fusion, which is
                    the number run_records.md reports. The post-fusion count that
                    actually executes is carried alongside it.
    precision       determined two independent ways: the dtype of every parameter tensor,
                    and AutoBackend's fp16 flag on the predictor built during load. Both
                    must say fp32 for the report to claim fp32. Ultralytics only enables
                    half precision when half=True is requested, which this script does
                    not do; nothing here is quantised or exported.
    checkpoint_size the .pt file on disk, in MiB (1024^2 bytes), with raw bytes kept.
    """
    parameters_fused = sum(p.numel() for p in model.model.parameters())
    dtypes = sorted({str(p.dtype) for p in model.model.parameters()})
    fp16_flag = getattr(getattr(model, "predictor", None), "model", None)
    fp16_flag = getattr(fp16_flag, "fp16", None)
    precision = "fp32" if dtypes == ["torch.float32"] and fp16_flag is False else "/".join(dtypes)

    size_bytes = Path(model_path).stat().st_size
    return {
        "checkpoint": str(model_path),
        "parameters": parameters_as_trained,
        "parameters_millions": parameters_as_trained / 1e6,
        "parameters_fused_for_inference": parameters_fused,
        "parameters_note": (
            f"{parameters_as_trained:,} as trained; Ultralytics fuses Conv+BatchNorm on the "
            f"first inference, which folds away {parameters_as_trained - parameters_fused:,} "
            f"BatchNorm parameters and leaves {parameters_fused:,} executing at inference"
        ),
        "checkpoint_size_mb": size_bytes / (1024 * 1024),
        "checkpoint_size_bytes": size_bytes,
        "imgsz": imgsz,
        "batch_size": 1,
        "precision": precision,
        "precision_evidence": (
            f"every parameter tensor has dtype {', '.join(dtypes)} and the predictor's "
            f"AutoBackend reports fp16={fp16_flag}; half precision was never requested "
            f"and no export/quantisation step is involved"
        ),
        "device": device,
        "torch_version": torch.__version__,
    }


def detect_hardware(torch):
    """Machine facts read from the OS, so the hardware string in the report is checkable."""
    import subprocess

    def sysctl(key):
        result = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    memory_bytes = sysctl("hw.memsize")
    return {
        "cpu_brand": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(memory_bytes) / (1024 ** 3)) if memory_bytes.isdigit() else "unknown",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mps_available": bool(torch.backends.mps.is_available()),
    }


def write_markdown(path, report):
    """Three tables ready to paste into the report: latency, static/training, comparison."""
    protocol = report["protocol"]
    image = report["benchmark_image"]
    labels = list(report["models"])

    lines = []
    lines.append("# Inference efficiency benchmark (§10.7)")
    lines.append("")
    lines.append("Generated by `src/evaluate/benchmark.py`. Dedicated benchmark — these numbers "
                 "replace the in-pass latency in `slices_val.json`, which had no warm-up.")
    lines.append("")
    lines.append(f"- hardware: **{report['hardware']['description']}** "
                 f"(detected: {report['hardware']['detected']['cpu_brand']}, "
                 f"{report['hardware']['detected']['memory_gb']} GB, "
                 f"{report['hardware']['detected']['platform']})")
    lines.append(f"- torch {report['versions']['torch']} · ultralytics "
                 f"{report['versions']['ultralytics']} · device `{protocol['device']}` · "
                 f"batch size 1 · precision fp32")
    lines.append(f"- fixed input: val image id **{image['image_id']}** "
                 f"(`{image['file_name']}`, {image['width']}x{image['height']}) — one image for "
                 f"every sample and both models, so object count and NMS load are constant")
    lines.append(f"- per call: conf {protocol['conf']}, NMS IoU {protocol['iou']}, "
                 f"max_det {protocol['max_det']} (the mAP-protocol settings, so the comparison "
                 f"with the in-pass figures is like for like)")
    lines.append(f"- protocol: at least {protocol['n_warmup_min']} discarded warm-up inferences "
                 f"*and* at least {protocol['warmup_min_seconds']} s of them (per-model counts in "
                 f"the table below), then at least {protocol['n_measured_min']} measured *and* at "
                 f"least {protocol['measure_min_seconds']} s of them — and that whole block "
                 f"repeated {protocol['n_attempts']} times; `torch.mps.synchronize()` before "
                 f"every clock read because MPS dispatch is asynchronous")
    lines.append(f"- latency is end-to-end per image: read + letterbox preprocess + forward + "
                 f"NMS + postprocess")
    lines.append("")

    lines.append("## Steady-state latency (batch 1, single image)")
    lines.append("")
    lines.append("| model | imgsz | median ms | P95 ms | mean ms | std ms | throughput img/s | "
                 "n warm-up | n measured |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        m = report["models"][label]
        lat, static = m["latency"], m["static"]
        lines.append(
            f"| {label} | {static['imgsz']} | {lat['median_ms']:.2f} | {lat['p95_ms']:.2f} | "
            f"{lat['mean_ms']:.2f} | {lat['std_ms']:.2f} | {lat['throughput_img_per_s']:.1f} | "
            f"{lat['n_warmup']} | {lat['n_measured']} |"
        )
    lines.append("")
    for label in labels:
        stability = report["models"][label]["latency"]["stability"]
        thirds = ", ".join(f"{t:.2f}" for t in stability["third_medians_ms"])
        lat = report["models"][label]["latency"]
        warning = "" if stability["steady"] else (
            " **No block passed the stability gate on this machine, so this row is the "
            "least-bad of the five — treat it as an upper bound with roughly +/-10% "
            "uncertainty, not a precise figure.**"
        )
        lines.append(f"- {label}: {lat['attempts']} repeated blocks with medians "
                     f"{', '.join(f'{m:.2f}' for m in lat['attempt_medians_ms'])} ms "
                     f"({lat['attempts_discarded_as_drifted']} discarded as drifted); the "
                     f"reported block has first/middle/last-third medians of {thirds} ms, a "
                     f"spread of {stability['spread_fraction_of_median']:.1%} of its median."
                     f"{warning}")
    lines.append("")

    lines.append("Measurement caveat: this is a laptop whose GPU is shared with the macOS window "
                 "compositor, so any block can be slowed by unrelated desktop activity. The whole "
                 f"block is therefore repeated {report['protocol']['n_attempts']} times; blocks "
                 f"that drift mid-run (thirds disagreeing by more than "
                 f"{100 * report['protocol']['steady_max_spread']:.0f}% of the median) are "
                 "discarded, and the fastest of the survivors is reported, because contention is "
                 "additive and can only ever slow a block down. Every attempt's median is listed "
                 "above so the run-to-run spread is visible.")
    lines.append("")
    lines.append("## Model size, load cost and training cost")
    lines.append("")
    lines.append("| model | params | checkpoint MB | precision | load: construct ms | "
                 "load: first inference ms | cold start ms | training duration | peak GPU mem |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---|---|")
    for label in labels:
        m = report["models"][label]
        static, load, train = m["static"], m["load"], m["training"]
        lines.append(
            f"| {label} | {static['parameters']:,} | {static['checkpoint_size_mb']:.2f} | "
            f"{static['precision']} | {load['construct_ms']:.1f} | "
            f"{load['first_inference_ms']:.1f} | {load['total_cold_start_ms']:.1f} | "
            f"{train['training_duration']} | {train['peak_gpu_memory']} |"
        )
    lines.append("")
    lines.append("Parameter counts are the as-trained figures (matching `run_records.md`). "
                 "Ultralytics fuses Conv+BatchNorm on the first inference, so the count actually "
                 "executing is slightly lower — see `parameters_fused_for_inference` in "
                 "`efficiency.json`. Checkpoint size is MiB (1024^2 bytes) of the `best.pt` file.")
    lines.append("")
    lines.append("Cold start is measured with the MPS device context already created, so it is "
                 "not charged to whichever model is benchmarked first. Metal's compiled-kernel "
                 "cache is still process-wide: a model loaded second in the same process reuses "
                 "some of those kernels and its `first inference` column is therefore a lower "
                 "bound on a genuinely cold process.")
    lines.append("")
    lines.append("Training duration and peak GPU memory are transcribed from "
                 "`reports/metrics/run_records.md` and were measured on the **training** machine "
                 "(RunPod RTX 4090 24 GB), not on the Apple Silicon machine used for the latency "
                 "columns. The two are not comparable. `not recorded` means the value is absent "
                 "from the run record — it has not been estimated.")
    lines.append("")

    lines.append("## Effect of warm-up: in-pass figures vs this benchmark")
    lines.append("")
    lines.append("| model | in-pass median ms | steady median ms | change | in-pass P95 ms | "
                 "steady P95 ms | change |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for label in labels:
        m = report["models"][label]
        ref, lat = m["in_pass_reference"], m["latency"]
        if not ref:
            lines.append(f"| {label} | n/a | {lat['median_ms']:.2f} | n/a | n/a | "
                         f"{lat['p95_ms']:.2f} | n/a |")
            continue
        d_median = 100 * (lat["median_ms"] - ref["median_ms"]) / ref["median_ms"]
        d_p95 = 100 * (lat["p95_ms"] - ref["p95_ms"]) / ref["p95_ms"]
        lines.append(
            f"| {label} | {ref['median_ms']:.2f} | {lat['median_ms']:.2f} | {d_median:+.1f}% | "
            f"{ref['p95_ms']:.2f} | {lat['p95_ms']:.2f} | {d_p95:+.1f}% |"
        )
    lines.append("")
    for label in labels:
        quoted = report["models"][label]["training"]["training_median_latency_quoted"]
        lines.append(f"- `run_records.md` quotes **{quoted}** as the median latency for model "
                     f"{label}; that is the same contaminated in-pass measurement and should be "
                     f"superseded by the steady-state column above.")
    lines.append("")
    lines.append("The in-pass columns come from `slices_val.json`: measured inside the mAP loop "
                 "with no warm-up, no device synchronisation, and a different image on every "
                 "call. Two separate effects are collapsed in that number — the uncompiled first "
                 "call, and a fresh letterbox shape per image forcing repeated Metal graph "
                 "builds. Neither is steady state, so only the columns in this file should be "
                 "quoted as inference latency.")
    lines.append("")

    Path(path).write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Inference efficiency benchmark (§10.7).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model", action="append", required=True, metavar="LABEL=PATH",
                        help="repeatable, e.g. --model 640=path/to/best.pt")
    parser.add_argument("--image-id", type=int, default=None,
                        help=f"which {BENCHMARK_SPLIT} image to time on; default is the "
                             f"lowest {BENCHMARK_SPLIT} image id")
    parser.add_argument("--warmup", type=int, default=MIN_WARMUP,
                        help=f"discarded warm-up inferences (minimum {MIN_WARMUP})")
    parser.add_argument("--iters", type=int, default=MIN_MEASURED,
                        help=f"measured inferences (minimum {MIN_MEASURED})")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--hardware", default=DEFAULT_HARDWARE,
                        help="hardware string for the report; the detected values are stored "
                             "next to it either way")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    # The plan fixes these minimums; the CLI may raise them but must not undercut them.
    if args.warmup < MIN_WARMUP:
        parser.error(f"--warmup must be at least {MIN_WARMUP} (§10.7 protocol)")
    if args.iters < MIN_MEASURED:
        parser.error(f"--iters must be at least {MIN_MEASURED} (§10.7 protocol)")

    import torch                                  # imported here so --help needs no torch
    import ultralytics

    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("device 'mps' requested but torch reports MPS is unavailable")

    cfg = load_config(args.config)
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    image = pick_benchmark_image(cfg, args.image_id)
    records = parse_run_records(metrics_dir / RUN_RECORDS_FILE)
    hardware = detect_hardware(torch)

    print(f"benchmark image: {BENCHMARK_SPLIT} image_id {image['image_id']} "
          f"({image['file_name']}, {image['width']}x{image['height']})")
    print(f"protocol: {args.warmup} warm-up discarded, {args.iters} measured, batch 1, "
          f"device {args.device}, torch.mps.synchronize() around every sample")

    report = {
        "assignment_section": "10.7",
        "split": BENCHMARK_SPLIT,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {"description": args.hardware, "detected": hardware},
        "versions": {"torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "protocol": {
            "device": args.device,
            "batch_size": 1,
            "n_warmup_min": args.warmup,
            "warmup_min_seconds": WARMUP_MIN_SECONDS,
            "n_measured_min": args.iters,
            "measure_min_seconds": MEASURE_MIN_SECONDS,
            "steady_max_spread": STEADY_MAX_SPREAD,
            "n_attempts": N_ATTEMPTS,
            "conf": PRED_CONF,
            "iou": PRED_IOU,
            "max_det": PRED_MAX_DET,
            "synchronisation": "torch.mps.synchronize() before starting and before stopping "
                               "the clock on every sample; MPS dispatch is asynchronous, so "
                               "without it the timings are silently under-measured",
            "latency_scope": "end-to-end per image: read + letterbox preprocess + forward + "
                             "NMS + postprocess",
            "why_not_reuse_slices": "the latency in slices_val.json was measured inside the mAP "
                                    "pass with no warm-up and a different image per call, so it "
                                    "includes model-load and Metal graph-compile cost",
        },
        "benchmark_image": image,
        "models": {},
    }

    # Pay the one-off process/device setup before any model is loaded, so the first
    # model measured is not charged for something the second one gets for free.
    init_device_context(args.device, torch)

    for position, spec in enumerate(args.model, start=1):
        label, model_path = spec.split("=", 1)
        imgsz = IMGSZ_FROM_LABEL.get(label, 640)
        run_id = run_id_from_checkpoint(model_path)
        print(f"\n[{label}] imgsz {imgsz}, run record '{run_id}'")

        model, construct_ms = load_model(model_path, torch)
        parameters_as_trained = count_parameters(model)     # before fusion; see the docstring
        first_inference_ms = time_first_inference(model, image["path"], imgsz, args.device, torch)
        load = {
            "construct_ms": construct_ms,
            "first_inference_ms": first_inference_ms,
            "total_cold_start_ms": construct_ms + first_inference_ms,
            "process_load_order": position,
            "note": "construct = YOLO(path) checkpoint read + module build (CPU only); "
                    "first_inference = weight transfer to MPS + AutoBackend setup + lazy "
                    "Metal graph compile. Reported separately, never summed into latency. "
                    "The MPS device context is created before any model is loaded so this "
                    "is not charged to whichever model happens to go first; Metal's kernel "
                    "cache is still process-wide, so a model loaded second (process_load_order "
                    "> 1) can reuse compiled kernels and shows a lower first_inference_ms "
                    "than it would in a fresh process.",
        }
        print(f"[{label}] load: construct {load['construct_ms']:.1f} ms + first inference "
              f"{load['first_inference_ms']:.1f} ms = {load['total_cold_start_ms']:.1f} ms cold start")

        static = static_facts(model, model_path, imgsz, args.device, torch, parameters_as_trained)
        print(f"[{label}] {static['parameters']:,} params as trained "
              f"({static['parameters_fused_for_inference']:,} after Conv+BN fusion), "
              f"{static['checkpoint_size_mb']:.2f} MB checkpoint, {static['precision']}")

        print(f"[{label}] warming up (>= {args.warmup} iters, >= {WARMUP_MIN_SECONDS}s) then "
              f"measuring (>= {args.iters} iters, >= {MEASURE_MIN_SECONDS}s) x {N_ATTEMPTS} ...")
        latency = measure_latency_stable(model, image["path"], imgsz, args.device,
                                         args.warmup, args.iters, torch)
        print(f"[{label}] warm-up: {latency['n_warmup']} iterations in "
              f"{latency['warmup_seconds']:.2f} s (discarded); reporting the fastest of "
              f"{latency['attempts']} steady blocks")
        print(f"[{label}] median {latency['median_ms']:.2f} ms, P95 {latency['p95_ms']:.2f} ms, "
              f"mean {latency['mean_ms']:.2f} +/- {latency['std_ms']:.2f} ms, "
              f"{latency['throughput_img_per_s']:.1f} img/s")
        stability = latency["stability"]
        thirds = ", ".join(f"{t:.2f}" for t in stability["third_medians_ms"])
        print(f"[{label}] stability: thirds {thirds} ms "
              f"(spread {stability['spread_fraction_of_median']:.1%} of median, "
              f"{'steady' if stability['steady'] else 'NOT STEADY — re-measure on a quiet machine'})")

        report["models"][label] = {
            "label": label,
            "static": static,
            "load": load,
            "latency": latency,
            "training": training_facts(records, run_id, metrics_dir / RUN_RECORDS_FILE),
            "in_pass_reference": in_pass_reference(cfg, label),
        }

    out_json = Path(args.out_json) if args.out_json else metrics_dir / "efficiency.json"
    out_md = Path(args.out_md) if args.out_md else metrics_dir / "efficiency.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    write_markdown(out_md, report)

    print(f"\n{'model':<8}{'median ms':>11}{'P95 ms':>9}{'mean ms':>9}{'std ms':>8}"
          f"{'img/s':>8}{'load ms':>10}{'params':>10}{'MB':>7}")
    for label, m in report["models"].items():
        lat, static, load = m["latency"], m["static"], m["load"]
        print(f"{label:<8}{lat['median_ms']:>11.2f}{lat['p95_ms']:>9.2f}{lat['mean_ms']:>9.2f}"
              f"{lat['std_ms']:>8.2f}{lat['throughput_img_per_s']:>8.1f}"
              f"{load['total_cold_start_ms']:>10.1f}{static['parameters']:>10,}"
              f"{static['checkpoint_size_mb']:>7.2f}")

    print("\nwarm-up effect (in-pass mAP-loop latency -> steady state):")
    for label, m in report["models"].items():
        ref, lat = m["in_pass_reference"], m["latency"]
        if not ref:
            print(f"  {label}: no in-pass reference found in {SLICES_FILE}")
            continue
        print(f"  {label}: median {ref['median_ms']:.2f} -> {lat['median_ms']:.2f} ms "
              f"({100 * (lat['median_ms'] - ref['median_ms']) / ref['median_ms']:+.1f}%), "
              f"P95 {ref['p95_ms']:.2f} -> {lat['p95_ms']:.2f} ms "
              f"({100 * (lat['p95_ms'] - ref['p95_ms']) / ref['p95_ms']:+.1f}%)")

    print(f"\nwritten: {out_json}")
    print(f"written: {out_md}")


if __name__ == "__main__":
    main()
