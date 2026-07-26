#!/usr/bin/env bash
# Evaluate on the *LIBERO-plus* robustness benchmark (7 perturbation dimensions).
#
# One suite ↔ one Pro checkpoint:
#   libero_spatial → outputs/LIBERO-Spatial-Pro
#   libero_object  → outputs/LIBERO-Object-Pro
#   libero_goal    → outputs/LIBERO-Goal-Pro
#   libero_10      → outputs/LIBERO-Long-Pro
#
# Naming note:
#   LIBERO-plus = evaluation benchmark (camera/robot/language/light/bg/noise/layout)
#   *-Pro ckpt  = VLA-Adapter Pro model weights
#   Standard LIBERO eval: vla-scripts/run_libero_eval.sh
#   LIBERO-Pro eval:      vla-scripts/run_libero_pro_eval.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# Use third_party/LIBERO-plus instead of ./LIBERO
export PYTHONPATH="${ROOT_DIR}/third_party/LIBERO-plus:${ROOT_DIR}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${ROOT_DIR}/experiments/robot/libero/.libero_plus_config"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
# Official Plus protocol: 1 trial per perturbed task
NUM_TRIALS="${NUM_TRIALS:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Optional: e.g. "Camera Viewpoints" / "Language Instructions" / ""
PERTURBATION_CATEGORY="${PERTURBATION_CATEGORY:-}"
# Optional smoke: MAX_TASKS=2 bash vla-scripts/run_libero_plus_eval.sh
MAX_TASKS="${MAX_TASKS:-0}"
START_TASK_ID="${START_TASK_ID:-0}"
# Rollout MP4s off by default; set SAVE_VIDEOS=1 to enable
SAVE_VIDEOS="${SAVE_VIDEOS:-0}"

ALL_SUITES=(libero_spatial libero_object libero_goal libero_10)

ASSETS_DIR="${ROOT_DIR}/third_party/LIBERO-plus/libero/libero/assets"
# HF assets.zip nests files under inspire/.../LIBERO-plus-0/assets; relocate if needed.
if [[ ! -d "${ASSETS_DIR}" ]]; then
  NESTED="$(find "${ROOT_DIR}/third_party/LIBERO-plus/libero/libero" -type d -path '*/LIBERO-plus-0/assets' 2>/dev/null | head -1 || true)"
  if [[ -n "${NESTED}" ]]; then
    mv "${NESTED}" "${ASSETS_DIR}"
    rm -rf "${ROOT_DIR}/third_party/LIBERO-plus/libero/libero/inspire"
  fi
fi
if [[ ! -d "${ASSETS_DIR}" ]]; then
  cat <<EOF
ERROR: LIBERO-plus assets missing at:
  ${ASSETS_DIR}

Download via mirror (4 workers, no rate-limit):
  bash vla-scripts/download_libero_plus.sh
EOF
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${PERTURBATION_CATEGORY}" ]]; then
  EXTRA_ARGS+=(--perturbation_category "${PERTURBATION_CATEGORY}")
fi
if [[ "${MAX_TASKS}" != "0" ]]; then
  EXTRA_ARGS+=(--max_tasks "${MAX_TASKS}")
fi
if [[ "${START_TASK_ID}" != "0" ]]; then
  EXTRA_ARGS+=(--start_task_id "${START_TASK_ID}")
fi
if [[ "${SAVE_VIDEOS}" == "1" || "${SAVE_VIDEOS}" == "true" || "${SAVE_VIDEOS}" == "True" ]]; then
  EXTRA_ARGS+=(--save_videos True)
fi

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
    experiments/robot/libero/run_libero_plus_eval.py \
    --use_proprio True \
    --num_images_in_input 2 \
    --use_film False \
    --pretrained_checkpoint "${ckpt}" \
    --task_suite_name "${suite}" \
    --use_pro_version True \
    --num_trials_per_task "${NUM_TRIALS}" \
    "${EXTRA_ARGS[@]}"
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
#   MAX_TASKS=2 bash vla-scripts/run_libero_plus_eval.sh
#   PERTURBATION_CATEGORY="Camera Viewpoints" MAX_TASKS=10 bash vla-scripts/run_libero_plus_eval.sh
#   TASK_SUITE=libero_object bash vla-scripts/run_libero_plus_eval.sh
#   TASK_SUITE=all bash vla-scripts/run_libero_plus_eval.sh
