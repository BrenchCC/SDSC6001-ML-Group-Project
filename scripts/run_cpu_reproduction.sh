#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-cs_hw}"
RESULT_DIR="${RESULT_DIR:-results}"
DATA_DIR="${DATA_DIR:-data}"

if [ "$#" -gt 0 ]; then
  SEEDS=("$@")
else
  SEEDS=(0 1)
fi

echo "Running CPU reproduction with conda env: ${CONDA_ENV}"
echo "Result dir: ${RESULT_DIR}"
echo "Data dir: ${DATA_DIR}"
echo "Seeds: ${SEEDS[*]}"

run_experiment() {
  local experiment="$1"
  echo "================================================================"
  echo "Running ${experiment}"
  echo "================================================================"
  conda run -n "${CONDA_ENV}" python main.py \
    --experiment "${experiment}" \
    --stage all \
    --preset cpu \
    --device cpu \
    --result-dir "${RESULT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --seeds "${SEEDS[@]}"
}

run_experiment rotation_adam_p0
run_experiment rotation_adam
run_experiment variable_n

echo "================================================================"
echo "Building report artifact index"
echo "================================================================"
conda run -n "${CONDA_ENV}" python experiments/smooth_report_curves.py \
  --result-dir "${RESULT_DIR}" \
  --preset cpu

conda run -n "${CONDA_ENV}" python experiments/report_artifacts.py \
  --result-dir "${RESULT_DIR}" \
  --preset cpu \
  --seeds "${SEEDS[@]}"

echo "================================================================"
echo "Report-ready artifacts"
echo "================================================================"
echo "${RESULT_DIR}/report_artifacts.md"
echo "${RESULT_DIR}/smoothed_curve_artifacts.json"
echo "${RESULT_DIR}/rotation_adam_p0/cpu/figures"
echo "${RESULT_DIR}/rotation_adam/cpu/figures"
echo "${RESULT_DIR}/variable_n/cpu/figures"
