#!/usr/bin/env bash
set -euo pipefail

mode="${1:-smoke}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="outputs/experiments/${timestamp}-${mode}"
mkdir -p "$out_dir"

checkpoint="${out_dir}/checkpoint.pt"

case "$mode" in
  smoke)
    class_preset="basic"
    scale_mode="unit"
    canonical_units="24"
    samples_per_class="3"
    val_samples_per_class="1"
    epochs="1"
    batch_size="32"
    model="small"
    snr_min=""
    snr_max=""
    eval_samples_per_class="2"
    segmentation_text="CQ TEST"
    ;;
  reference)
    class_preset="real"
    scale_mode="unit"
    canonical_units="32"
    samples_per_class="45"
    val_samples_per_class="12"
    epochs="8"
    batch_size="64"
    model="glyph"
    snr_min="5"
    snr_max="25"
    eval_samples_per_class="20"
    segmentation_text="CQ TEST 599"
    ;;
  *)
    echo "usage: $0 [smoke|reference]" >&2
    exit 2
    ;;
esac

device="${CW_GLYPH_DEVICE:-auto}"
if command -v nvidia-smi >/dev/null 2>&1; then
  device="${CW_GLYPH_DEVICE:-cuda}"
fi

{
  echo "mode=${mode}"
  echo "created_at=${timestamp}"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "python=$(command -v python)"
  python --version
  echo "device=${device}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  fi
  python - <<'PY'
import platform
import torch

print(f"platform={platform.platform()}")
print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch_device={torch.cuda.get_device_name(0)}")
PY
} | tee "${out_dir}/environment.txt"

train_cmd=(
  python scripts/train_cnn.py
  --class-preset "$class_preset"
  --scale-mode "$scale_mode"
  --canonical-units "$canonical_units"
  --samples-per-class "$samples_per_class"
  --val-samples-per-class "$val_samples_per_class"
  --epochs "$epochs"
  --batch-size "$batch_size"
  --device "$device"
  --model "$model"
  --out "$checkpoint"
  --show-confusions 12
)
if [[ -n "$snr_min" && -n "$snr_max" ]]; then
  train_cmd+=(--snr-min "$snr_min" --snr-max "$snr_max")
fi

printf '%q ' "${train_cmd[@]}" | tee "${out_dir}/train.command.txt"
echo | tee -a "${out_dir}/train.command.txt" >/dev/null
"${train_cmd[@]}" 2>&1 | tee "${out_dir}/train.log"

eval_cmd=(
  python scripts/eval_cnn.py
  "$checkpoint"
  --samples-per-class "$eval_samples_per_class"
  --device "$device"
  --json-out "${out_dir}/eval_cnn.json"
)
if [[ "$mode" == "reference" ]]; then
  eval_cmd+=(--sweep-snr 25 15 10 5 0)
fi

printf '%q ' "${eval_cmd[@]}" | tee "${out_dir}/eval_cnn.command.txt"
echo | tee -a "${out_dir}/eval_cnn.command.txt" >/dev/null
"${eval_cmd[@]}" 2>&1 | tee "${out_dir}/eval_cnn.log"

seg_cmd=(
  python scripts/eval_segmentation_synth.py
  --text "$segmentation_text"
  --wpm 15 20 25
  --seeds 0 1 2
  --snr-db 12
  --qsb-rate-hz 0.25
  --qsb-depth-db 12
  --json-out "${out_dir}/segmentation_synth.json"
)

printf '%q ' "${seg_cmd[@]}" | tee "${out_dir}/segmentation_synth.command.txt"
echo | tee -a "${out_dir}/segmentation_synth.command.txt" >/dev/null
"${seg_cmd[@]}" 2>&1 | tee "${out_dir}/segmentation_synth.log"

echo "experiment_dir=${out_dir}"
