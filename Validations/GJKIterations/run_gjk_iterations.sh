#!/usr/bin/env bash
# =============================================================================
# run_gjk_iterations.sh
#
# Build (if needed) and run the GJK-iterations-vs-distance benchmark, then
# invoke the plot script to generate the figures.
#
# Usage:
#   ./run_gjk_iterations.sh [--trials N] [--seed S] [--no-build] [--no-plot]
#
# Options:
#   --trials N    Number of random-orientation trials per (pair, gap) sample
#                 Default: 100
#   --seed S      Random seed for reproducibility.  Default: 42
#   --no-build    Skip the build step (requires a pre-built binary)
#   --no-plot     Skip the plotting step
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- defaults ---------------------------------------------------------------
TRIALS=100
SEED=42
DO_BUILD=1
DO_PLOT=1

# ---- argument parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)  TRIALS="$2";  shift 2 ;;
        --seed)    SEED="$2";    shift 2 ;;
        --no-build) DO_BUILD=0;  shift   ;;
        --no-plot)  DO_PLOT=0;   shift   ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--trials N] [--seed S] [--no-build] [--no-plot]"
            exit 1
            ;;
    esac
done

CSV_FILE="data/gjk_iterations.csv"
EXECUTABLE="./build/GJKIterationsTest"

# ---- build ------------------------------------------------------------------
if [[ $DO_BUILD -eq 1 ]]; then
    echo "============================================================"
    echo "Building GJKIterationsTest..."
    echo "============================================================"
    make all
fi

# ---- run --------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Running GJK iterations benchmark"
echo "  Trials : $TRIALS"
echo "  Seed   : $SEED"
echo "  Output : $CSV_FILE"
echo "============================================================"

mkdir -p data
"$EXECUTABLE" "$TRIALS" "$SEED" "$CSV_FILE"

# ---- plot -------------------------------------------------------------------
if [[ $DO_PLOT -eq 1 ]]; then
    echo ""
    echo "============================================================"
    echo "Generating plots..."
    echo "============================================================"
    python3 plot.py --csv "$CSV_FILE" --plot-dir data
    echo "Plots written to data/"
fi

echo ""
echo "Done."
