#!/usr/bin/env bash
# Package the LABELS ONLY for a GPU pod (~1 MB). Run on the laptop, repo root.
#
# Preferred over scripts/package_for_pod.sh (which ships ~850MB of images):
# uploading thousands of small image files through a browser and untarring them
# onto a pod's network volume is slow and fragile — measured at ~1 MB/min in
# practice. The pod fetches the pixels itself from the COCO CDN in ~1 minute
# (scripts/pod_fetch_images.sh), so only the labels need to travel.
#
# The tarball also carries the frozen manifest + splits + hashes so the pod can
# VERIFY it reconstructed exactly the committed COCO5K subset (D1, D3, D4).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=coco5k_labels.tgz
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/coco5k"
cp -R data/yolo/coco5k-detect/labels  "$STAGE/coco5k/labels-detect"
cp -R data/yolo/coco5k-segment/labels "$STAGE/coco5k/labels-segment"
cp configs/coco_category_map.json     "$STAGE/coco5k/"
cp manifests/coco5k.csv manifests/coco5k_splits.csv manifests/HASHES.txt "$STAGE/coco5k/"

# macOS writes AppleDouble ._* sidecars for extended attributes; on Linux they
# would look like extra label files. Strip them, and stop tar from adding more.
find "$STAGE" -name '._*' -delete
COPYFILE_DISABLE=1 tar -czf "$OUT" -C "$STAGE" coco5k

if tar -tzf "$OUT" | grep -q '/\._'; then
  echo "ERROR: AppleDouble junk survived into $OUT" >&2
  exit 1
fi
echo "$OUT  $(du -h "$OUT" | cut -f1)  ($(tar -tzf "$OUT" | wc -l | tr -d ' ') entries, 0 junk files)"
echo "Upload $OUT + scripts/pod_fetch_images.sh + scripts/pod_train_detect.sh to the pod."
