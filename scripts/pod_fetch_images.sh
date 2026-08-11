#!/usr/bin/env bash
# Fetch the COCO5K images straight from the COCO CDN, ON THE POD.
#
# Why: uploading + untarring ~850MB of small files across a network volume is
# painfully slow, while a datacenter link pulls the same images from source in
# a couple of minutes. Only the labels (~1MB) need to travel from the laptop.
#
# WHICH images: derived from the label file names, so the set is exactly the
# frozen COCO5K manifest — same deterministic subset, same splits (D1/D4).
# Nothing here re-decides membership; it just fetches what the labels reference.
#
# Usage (from the folder containing coco5k/):  bash pod_fetch_images.sh
set -uo pipefail
ROOT="$(pwd)/coco5k"
BASE_URL="http://images.cocodataset.org/train2017"
PARALLEL=32

[ -d "$ROOT/labels-detect" ] || { echo "coco5k/labels-detect not found — extract the labels tarball first" >&2; exit 1; }

for split in train val test; do
  mkdir -p "$ROOT/images/$split"
  ls "$ROOT/labels-detect/$split" | sed 's/\.txt$/.jpg/' > "/tmp/$split.list"
  want=$(wc -l < "/tmp/$split.list")
  echo "[$split] fetching $want images with $PARALLEL parallel connections..."
  # -f fail on HTTP error, -s silent, --retry survives transient blips.
  xargs -a "/tmp/$split.list" -P "$PARALLEL" -I{} \
    curl -fsS --retry 3 --retry-delay 1 -o "$ROOT/images/$split/{}" "$BASE_URL/{}"
  have=$(ls "$ROOT/images/$split" | wc -l)
  echo "[$split] have $have / $want"
done

# Retry pass: anything missing or zero-byte gets one more attempt.
echo "verifying..."
missing=0
for split in train val test; do
  while read -r name; do
    f="$ROOT/images/$split/$name"
    if [ ! -s "$f" ]; then
      curl -fsS --retry 3 -o "$f" "$BASE_URL/$name" || true
      [ -s "$f" ] || { echo "STILL MISSING: $split/$name"; missing=$((missing+1)); }
    fi
  done < "/tmp/$split.list"
done

echo
for split in train val test; do
  echo "$split: $(ls "$ROOT/images/$split" | wc -l) images"
done
[ "$missing" -eq 0 ] || { echo "ERROR: $missing images could not be fetched" >&2; exit 1; }

# Integrity check against the frozen manifest: every fetched file must decode,
# and its pixel dimensions must equal the width/height recorded when the subset
# was built (D3). This proves the pod holds the SAME images as the laptop —
# a byte-count check alone would not catch a truncated or substituted file.
if [ -f "$ROOT/coco5k.csv" ]; then
  python3 - "$ROOT" <<'PY'
import csv, sys
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
by_name = {}
with open(root / "coco5k.csv") as f:
    for row in csv.DictReader(f):
        by_name[row["file_name"]] = (int(row["width"]), int(row["height"]))

checked = bad = 0
for split in ("train", "val", "test"):
    for path in (root / "images" / split).iterdir():
        expected = by_name.get(path.name)
        if expected is None:
            print(f"NOT IN MANIFEST: {split}/{path.name}"); bad += 1; continue
        try:
            with Image.open(path) as im:
                im.verify()                      # catches truncated/corrupt files
            with Image.open(path) as im:
                size = im.size
        except Exception as exc:
            print(f"UNREADABLE: {split}/{path.name}: {exc}"); bad += 1; continue
        if size != expected:
            print(f"SIZE MISMATCH: {split}/{path.name} {size} != {expected}"); bad += 1
        checked += 1

print(f"verified {checked} images against the frozen manifest, {bad} problems")
sys.exit(1 if bad else 0)
PY
else
  echo "NOTE: coco5k.csv not in the tarball — skipping manifest verification."
fi
echo "All images present and verified. Next: bash pod_train_detect.sh"
