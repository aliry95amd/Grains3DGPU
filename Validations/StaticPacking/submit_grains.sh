#!/bin/bash
# Submit GRAINS3D-GPU to ARC SLURM. Optional: sbatch submit_grains.sh path/to/input.xml

#SBATCH --job-name=grains3dgpu
# ARC gpu partition is V100-based (not A100).
#SBATCH --partition=gpu
#SBATCH --account=st-wachs-1-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-00:10:00
#SBATCH --output=/scratch/st-wachs-1/aliry95/grains_%j.out
#SBATCH --error=/scratch/st-wachs-1/aliry95/grains_%j.err
# ARC rejects /arc/home as job cwd; use scratch.
#SBATCH --chdir=/scratch/st-wachs-1/aliry95
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@ubc.ca

GRAINS_HOME="$HOME/Grains3DGPU"
source "${GRAINS_HOME}/Env/grainsGPU_arc.env.sh"

SCRATCH_BASE="/scratch/st-wachs-1/aliry95"
RUN_DIR="${SCRATCH_BASE}/grains_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

INPUT_XML="${1:-${GRAINS_HOME}/Validations/StaticPacking/Grains/Init/insert.xml}"
INPUT_XML="$(readlink -f "${INPUT_XML}")"
SIM_DIR="$(dirname "${INPUT_XML}")"
# Working dir must contain ./Grains/Init relative to case root.
CASE_DIR="$(cd "${SIM_DIR}/../.." && pwd)"

cp -a "${CASE_DIR}/." "${RUN_DIR}/"
REL_XML="${INPUT_XML#${CASE_DIR}/}"

echo "========================================"
echo "GRAINS3D-GPU -- SLURM Job ${SLURM_JOB_ID}"
echo "========================================"
echo "Date       : $(date)"
echo "Node       : ${SLURM_NODELIST}"
echo "GPUs       : ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Input XML  : ${INPUT_XML}"
echo "Run dir    : ${RUN_DIR}"
echo "Binary     : ${GRAINS_HOME}/Main/bin${GRAINS_FULL_EXT}/grains"
echo "========================================"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

cd "${RUN_DIR}"
"${GRAINS_HOME}/Main/bin${GRAINS_FULL_EXT}/grains" "${RUN_DIR}/${REL_XML}"
EXIT_CODE=$?

echo "========================================"
echo "Job finished at $(date) with exit code ${EXIT_CODE}"
echo "Results in : ${RUN_DIR}"
echo "========================================"

exit ${EXIT_CODE}
