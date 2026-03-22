#!/bin/bash
#SBATCH --job-name=reveng-compile
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/compile_%j.out
#SBATCH --error=logs/compile_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ichters.fe@gmail.com

set -euo pipefail
mkdir -p logs

WORK_ROOT="${WORK}/reveng-data"
BUILD_ROOT="${WORK_ROOT}/build"
BINARIES="${WORK_ROOT}/binaries"
MANIFESTS="${WORK_ROOT}/manifests"
PACKAGE_LIST="${MANIFESTS}/package_list.txt"

mkdir -p "${BUILD_ROOT}" "${BINARIES}" "${WORK_ROOT}/logs/compile"

if [ ! -f "${PACKAGE_LIST}" ]; then
    echo "ERROR: ${PACKAGE_LIST} not found. Run slurm_download.sh first."
    exit 1
fi

# Each make -j2 inside compile_package, so use cpus/2 workers
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} / 2 ))

# Compile all packages with all 8 configs
echo "Starting compilation"
uv run python scripts/pipeline/compile_all.py \
    --package-list "${PACKAGE_LIST}" \
    --build-root "${BUILD_ROOT}" \
    --workers "${WORKERS}" \
    --resume \
    --status-file "${MANIFESTS}/compile_status.json" \
    2>&1 | tee "${WORK_ROOT}/logs/compile/compile_all.log"

# Collect validated binaries
echo "Collecting binaries"
uv run python scripts/pipeline/collect_binaries.py \
    --build-root "${BUILD_ROOT}" \
    --output-dir "${BINARIES}" \
    --min-text-size 64 \
    --manifest "${MANIFESTS}/binary_manifest.csv" \
    2>&1 | tee "${WORK_ROOT}/logs/compile/collect.log"

echo "Binaries in: ${BINARIES}"
ls -1 "${BINARIES}" | wc -l
du -sh "${BINARIES}"

# Optionally clean build trees to save space
# echo "Cleaning build trees"
# rm -rf "${BUILD_ROOT}"
