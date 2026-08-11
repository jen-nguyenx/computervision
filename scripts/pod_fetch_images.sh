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
echo "All images present. Next: bash pod_train_detect.sh"
