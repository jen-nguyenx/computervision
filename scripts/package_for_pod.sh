#!/usr/bin/env bash
# Package COCO5K for a rented GPU pod (RunPod/Lambda/Vast).
#
# Produces coco5k_pod.tar (~1 GB) containing:
#   coco5k/images/{train,val,test}/   real image bytes — the local layout uses
#                                     symlinks (D14), which would arrive broken,
#                                     so they are DEREFERENCED here with cp -RL
#   coco5k/labels-detect/…            detect labels
#   coco5k/labels-segment/…           segment labels (same upload serves both tasks)
#   coco5k/coco_category_map.json
#
# Upload the tar to the pod (runpodctl send / scp), extract with:  tar -xf coco5k_pod.tar
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=coco5k_pod.tar
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

echo "Staging (dereferencing 5,000 image symlinks, ~1 GB copy)..."
mkdir -p "$STAGE/coco5k"
cp -RL data/yolo/coco5k-detect/images "$STAGE/coco5k/images"
cp -R  data/yolo/coco5k-detect/labels "$STAGE/coco5k/labels-detect"
cp -R  data/yolo/coco5k-segment/labels "$STAGE/coco5k/labels-segment"
cp configs/coco_category_map.json "$STAGE/coco5k/"

# refuse to ship a symlink by accident
if find "$STAGE/coco5k/images" -type l | grep -q .; then
  echo "ERROR: symlinks survived the copy — aborting." >&2
  exit 1
fi

echo "Creating $OUT ..."
tar -cf "$OUT" -C "$STAGE" coco5k
du -h "$OUT"
echo "Done. Upload $OUT to the pod, then run scripts/pod_train_detect.sh there."
