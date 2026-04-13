#!/bin/bash
# =============================================================================
# SLURM Job Submission Script for GRAINS3D-GPU on ARC
# =============================================================================
# This script submits a GRAINS3D-GPU simulation to the ARC SLURM scheduler.
# Usage:
#   sbatch submit_grains.sh                     (uses default XML input)
#   sbatch submit_grains.sh path/to/input.xml   (uses a custom XML file)
# =============================================================================

# ---------------------------------------------------------------------------
# SLURM directives — these tell the scheduler what resources to allocate
# ---------------------------------------------------------------------------

# Job name shown in squeue output (change to something meaningful for you)
#SBATCH --job-name=grains3dgpu

# Allocate resources from the "gpu" partition (V100 GPUs on ARC)
# Note: ARC does not have A100 GPUs; the gpu partition has 4× V100-32GB/node
#SBATCH --partition=gpu

# Charge the job to your GPU-enabled allocation account
#SBATCH --account=st-wachs-1-gpu

#SBATCH --nodes=1

# Request 1 GPU on a single node.  Increase to e.g. --gres=gpu:2 for more.
#SBATCH --gres=gpu:1

# Number of CPU cores for the job — 1 is usually enough for a single-GPU run;
# increase if you need OpenMP threads or pre/post-processing parallelism.
#SBATCH --cpus-per-task=1

# Memory per node.  Adjust based on your simulation size. 
# Each GPU node has ~180 GB total.
#SBATCH --mem=16G

# Wall-clock time limit.  Format: D-HH:MM:SS.  The job is killed when exceeded.
# Maximum allowed on the gpu partition is 7 days (7-00:00:00).
#SBATCH --time=0-00:10:00

# Standard output and error logs.  %j is replaced by the SLURM job ID.
# Logs are written to your scratch directory (ARC forbids jobs in /arc/home).
#SBATCH --output=/scratch/st-wachs-1/aliry95/grains_%j.out
#SBATCH --error=/scratch/st-wachs-1/aliry95/grains_%j.err

# Tell SLURM to start the job in /scratch (required — ARC rejects /arc/home).
#SBATCH --chdir=/scratch/st-wachs-1/aliry95

# Optional: receive an email when the job starts, ends, or fails.
# Uncomment and set your address to enable.
##SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=your.email@ubc.ca

# ---------------------------------------------------------------------------
# Environment setup — load modules and source the GRAINS environment file
# ---------------------------------------------------------------------------

# GRAINS_HOME is the top-level directory of the Grains3DGPU repository
GRAINS_HOME="$HOME/Grains3DGPU"

# Source the environment file.  This loads the nvhpc/23.9 module (providing
# nvcc & CUDA 12.2), adds GCC 9.4.0 to PATH, sets compiler flags, library
# paths, and all GRAINS_* variables needed at runtime.
source "${GRAINS_HOME}/Env/grainsGPU_arc.env.sh"

# ---------------------------------------------------------------------------
# Scratch directory — ARC requires jobs to run from /scratch, not /arc/home
# ---------------------------------------------------------------------------

# Your allocation's scratch space on the Lustre parallel filesystem.
SCRATCH_BASE="/scratch/st-wachs-1/aliry95"

# Create a unique run directory under scratch named after the job ID.
# This keeps each run isolated and easy to find later.
RUN_DIR="${SCRATCH_BASE}/grains_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"

# ---------------------------------------------------------------------------
# Input file — either passed as the first argument or default
# ---------------------------------------------------------------------------

# Use the first sbatch argument ($1) if provided, otherwise fall back to the
# StaticPacking validation case that ships with the repository.
INPUT_XML="${1:-${GRAINS_HOME}/Validations/StaticPacking/Grains/Init/insert.xml}"

# Resolve to absolute path.  SIM_DIR is the "case directory" — the parent that
# contains subdirectories the simulation expects (e.g. Grains/Init).
INPUT_XML="$(readlink -f "${INPUT_XML}")"
SIM_DIR="$(dirname "${INPUT_XML}")"

# Walk up from the XML file to the case root.  The simulation expects to find
# ./Grains/Init relative to its working directory, so we go two levels above
# the XML (Init → Grains → case root).
CASE_DIR="$(cd "${SIM_DIR}/../.." && pwd)"

# Copy the entire case directory tree into the scratch run directory so the
# simulation can find all relative paths (Grains/Init, output dirs, etc.).
cp -a "${CASE_DIR}/." "${RUN_DIR}/"

# The XML path relative to the case root (e.g. Grains/Init/insert.xml)
REL_XML="${INPUT_XML#${CASE_DIR}/}"

# ---------------------------------------------------------------------------
# Runtime information — printed to the .out log for debugging
# ---------------------------------------------------------------------------

echo "========================================"
echo "GRAINS3D-GPU — SLURM Job ${SLURM_JOB_ID}"
echo "========================================"
echo "Date       : $(date)"
echo "Node       : ${SLURM_NODELIST}"
echo "GPUs       : ${SLURM_GPUS_ON_NODE:-1}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Input XML  : ${INPUT_XML}"
echo "Run dir    : ${RUN_DIR}"
echo "Binary     : ${GRAINS_HOME}/Main/bin${GRAINS_FULL_EXT}/grains"
echo "========================================"

# Show which GPU(s) CUDA will use — helpful for debugging multi-GPU issues
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null

# ---------------------------------------------------------------------------
# Run the simulation
# ---------------------------------------------------------------------------

# Change into the scratch run directory so SLURM is happy and any output files
# produced by the simulation land in scratch (not /arc/home).
cd "${RUN_DIR}"

# Run GRAINS.  The binary takes a single argument: the path to the XML file.
"${GRAINS_HOME}/Main/bin${GRAINS_FULL_EXT}/grains" "${RUN_DIR}/${REL_XML}"

# Capture the exit code so SLURM logs it correctly
EXIT_CODE=$?

echo "========================================"
echo "Job finished at $(date) with exit code ${EXIT_CODE}"
echo "Results in : ${RUN_DIR}"
echo "========================================"

exit ${EXIT_CODE}
