#!/usr/bin/env bash
# Evaluate on the *standard LIBERO* benchmark (not LIBERO-Pro).
#
# One suite ↔ one Pro checkpoint:
#   libero_spatial → outputs/LIBERO-Spatial-Pro
#   libero_object  → outputs/LIBERO-Object-Pro
#   libero_goal    → outputs/LIBERO-Goal-Pro
#   libero_10      → outputs/LIBERO-Long-Pro
#
# Naming note:
#   *-Pro ckpt = VLA-Adapter Pro model weights (unrelated to LIBERO-Pro benchmark).
#   For LIBERO-Pro eval, use: vla-scripts/run_libero_pro_eval.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Standard LIBERO package (original benchmark)
export PYTHONPATH="${ROOT_DIR}/LIBERO:${PYTHONPATH:-}"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
NUM_TRIALS="${NUM_TRIALS:-20}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ALL_SUITES=(libero_spatial libero_object libero_goal libero_10)

resolve_ckpt() {
  case "$1" in
    libero_spatial) echo "outputs/LIBERO-Spatial-Pro" ;;
    libero_object)  echo "outputs/LIBERO-Object-Pro" ;;
    libero_goal)    echo "outputs/LIBERO-Goal-Pro" ;;
    libero_10)      echo "outputs/LIBERO-Long-Pro" ;;
    *) return 1 ;;
  esac
}

run_one_suite() {
  local suite="$1"
  local ckpt="${ADAPTER_CKPT:-}"
  if [[ -z "${ckpt}" ]]; then
    ckpt="$(resolve_ckpt "${suite}")" || {
      echo "ERROR: no Pro checkpoint mapping for suite '${suite}'"
      echo "Supported: ${ALL_SUITES[*]} | all"
      exit 1
    }
  fi
  if [[ ! -d "${ckpt}" ]]; then
    cat <<EOF
ERROR: checkpoint missing: ${ckpt}

Download all Pro checkpoints:
  bash vla-scripts/download_libero_pro_ckpts.sh
EOF
    exit 1
  fi

  echo "Eval: suite=${suite}  ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" uv run \
    experiments/robot/libero/run_libero_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint "${ckpt}" \
    --task_suite_name "${suite}" \
    --use_pro_version True \
    --num_trials_per_task "${NUM_TRIALS}"
}

if [[ "${TASK_SUITE}" == "all" || "${TASK_SUITE}" == "ALL" ]]; then
  if [[ -n "${ADAPTER_CKPT:-}" ]]; then
    echo "ERROR: ADAPTER_CKPT cannot be set with TASK_SUITE=all (each suite needs its own ckpt)"
    exit 1
  fi
  for suite in "${ALL_SUITES[@]}"; do
    run_one_suite "${suite}"
  done
else
  run_one_suite "${TASK_SUITE}"
fi

# Examples:
#   bash vla-scripts/run_libero_eval.sh
#   TASK_SUITE=libero_object bash vla-scripts/run_libero_eval.sh
#   TASK_SUITE=all bash vla-scripts/run_libero_eval.sh
#   ADAPTER_CKPT=outputs/my-ckpt TASK_SUITE=libero_spatial bash vla-scripts/run_libero_eval.sh
