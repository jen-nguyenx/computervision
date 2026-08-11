#!/usr/bin/env bash
# Train coco5k-detect FROM SCRATCH on a rented GPU pod. Run ON THE POD.
#
# Pod checklist (RunPod: PyTorch template, one RTX 4090/L4-class GPU):
#   1. upload coco5k_pod.tar (runpodctl send locally / receive on pod, or scp)
#   2. tar -xf coco5k_pod.tar          (creates ./coco5k)
#   3. bash pod_train_detect.sh        (defaults: 40 epochs, imgsz 640, batch 32)
#   4. download coco5k-detect-outputs.tgz
#   5. TERMINATE the pod (terminated, not stopped — stopped pods keep billing storage)
#
# Back home: extract into artifacts/coco5k-detect/ ; env_record.txt is the
# cloud-side environment pin the assignment (§9.1) requires for the machine
# that produced the checkpoint.
set -euo pipefail
EPOCHS="${1:-40}"
DATA_ROOT="$(pwd)/coco5k"
[ -d "$DATA_ROOT/images/train" ] || { echo "coco5k/ not found — extract coco5k_pod.tar first" >&2; exit 1; }

pip install --quiet ultralytics==8.4.117   # same pin as the local pipeline

# ultralytics expects labels/ mirroring images/ — point it at the detect tree
ln -sfn "$DATA_ROOT/labels-detect" "$DATA_ROOT/labels"

# data yaml with pod-absolute path (same D16 rule as local: no auto-download traps)
python3 - <<EOF
import json
names = json.load(open("$DATA_ROOT/coco_category_map.json"))["names"]
with open("coco5k-detect-pod.yaml", "w") as f:
    f.write("path: $DATA_ROOT\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n")
    for k in sorted(names, key=int):
        f.write(f"  {k}: {names[k]}\n")
EOF

# cloud-side environment record (§9.1: pin the machine that MADE the checkpoint)
python3 - <<'EOF' | tee env_record.txt
import platform, torch, ultralytics
print("python      ", platform.python_version())
print("torch       ", torch.__version__)
print("cuda        ", torch.version.cuda)
print("gpu         ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — wrong pod template?")
print("ultralytics ", ultralytics.__version__)
EOF

# From-scratch baseline (D: §9.2 leakage policy — no pretrained weights).
yolo detect train model=yolo11n.yaml pretrained=False data=coco5k-detect-pod.yaml \
  epochs="$EPOCHS" imgsz=640 batch=32 seed=0 deterministic=True device=0 workers=8 \
  project=runs name=coco5k-detect-baseline exist_ok=True

tar -czf coco5k-detect-outputs.tgz env_record.txt coco5k-detect-pod.yaml runs/coco5k-detect-baseline
echo
echo "DONE — download coco5k-detect-outputs.tgz, then TERMINATE the pod."
