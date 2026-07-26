#!/usr/bin/env bash
# Run Harness VLA (sde_harness) on the *LIBERO-plus* robustness benchmark.
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
#   Standard LIBERO harness:   vla-scripts/run_libero_eval_harness.sh
#   LIBERO-Pro harness:        vla-scripts/run_libero_pro_eval_harness.sh
#   LIBERO-plus eval:          vla-scripts/run_libero_plus_eval.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="${ROOT_DIR}/sde_harness"
cd "${ROOT_DIR}"

# Use third_party/LIBERO-plus instead of ./LIBERO
export PYTHONPATH="${HARNESS_DIR}:${ROOT_DIR}/third_party/LIBERO-plus:${ROOT_DIR}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH="${ROOT_DIR}/experiments/robot/libero/.libero_plus_config"
export SDE_HARNESS_ROOT="${HARNESS_DIR}"
export VLA_ADAPTER_ROOT="${ROOT_DIR}"
export LIBERO_TYPE=plus
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

TASK_SUITE="${TASK_SUITE:-libero_spatial}"
TASK="${TASK:-0}"
SEED="${SEED:-0}"
PLANNER="${PLANNER:-cursor}"
MODEL="${MODEL:-composer-2.5}"

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

base_suite() {
  local suite="$1"
  for b in "${ALL_SUITES[@]}"; do
    if [[ "$suite" == "$b" || "$suite" == "$b"_* ]]; then
      echo "$b"
      return 0
    fi
  done
  echo "$suite"
}

resolve_ckpt() {
  case "$(base_suite "$1")" in
    libero_spatial) echo "outputs/LIBERO-Spatial-Pro" ;;
    libero_object)  echo "outputs/LIBERO-Object-Pro" ;;
    libero_goal)    echo "outputs/LIBERO-Goal-Pro" ;;
    libero_10)      echo "outputs/LIBERO-Long-Pro" ;;
    *) return 1 ;;
  esac
}

ckpt="${ADAPTER_CKPT:-}"
if [[ -z "${ckpt}" ]]; then
  ckpt="$(resolve_ckpt "${TASK_SUITE}")" || {
    echo "ERROR: no Pro checkpoint mapping for suite '${TASK_SUITE}'"
    echo "Supported: ${ALL_SUITES[*]}"
    echo "Set ADAPTER_CKPT explicitly to override."
    exit 1
  }
fi
if [[ ! -d "${ckpt}" && ! -d "${ROOT_DIR}/${ckpt}" ]]; then
  cat <<EOF
ERROR: checkpoint missing: ${ckpt}

Download all Pro checkpoints:
  bash vla-scripts/download_libero_pro_ckpts.sh
EOF
  exit 1
fi
if [[ -d "${ckpt}" && "${ckpt}" = /* ]]; then
  export ADAPTER_CHECKPOINT_PATH="${ckpt}"
elif [[ -d "${ROOT_DIR}/${ckpt}" ]]; then
  export ADAPTER_CHECKPOINT_PATH="${ROOT_DIR}/${ckpt}"
else
  export ADAPTER_CHECKPOINT_PATH="${ckpt}"
fi

echo "Harness: suite=${TASK_SUITE} task=${TASK} seed=${SEED}"
echo "  LIBERO_TYPE=${LIBERO_TYPE}"
echo "  ADAPTER_CHECKPOINT_PATH=${ADAPTER_CHECKPOINT_PATH}"
echo "  planner=${PLANNER} model=${MODEL}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" uv run \
  python -m rpent.cli.main \
  --env libero \
  --libero-type "${LIBERO_TYPE}" \
  --suite "${TASK_SUITE}" \
  --task "${TASK}" \
  --seed "${SEED}" \
  --adapter-checkpoint "${ADAPTER_CHECKPOINT_PATH}" \
  --planner "${PLANNER}" \
  --model "${MODEL}" \
  "$@"

# Examples:
#   bash vla-scripts/run_libero_plus_eval_harness.sh
#   TASK_SUITE=libero_object TASK=0 bash vla-scripts/run_libero_plus_eval_harness.sh
#   ADAPTER_CKPT=outputs/LIBERO-Spatial-Pro TASK_SUITE=libero_spatial bash vla-scripts/run_libero_plus_eval_harness.sh
