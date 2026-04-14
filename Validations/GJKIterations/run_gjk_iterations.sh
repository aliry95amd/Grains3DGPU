#!/usr/bin/env bash
# Build/run GJK-iterations benchmark and optional plots.
# Usage: ./run_gjk_iterations.sh [--trials N] [--seed S] [--no-build] [--no-plot]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TRIALS=100
SEED=42
DO_BUILD=1
DO_PLOT=1

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

if [[ $DO_BUILD -eq 1 ]]; then
    echo "Building GJKIterationsTest..."
    make all
fi

echo "Running GJK iterations benchmark (trials=$TRIALS seed=$SEED -> $CSV_FILE)"
mkdir -p data
"$EXECUTABLE" "$TRIALS" "$SEED" "$CSV_FILE"

if [[ $DO_PLOT -eq 1 ]]; then
    echo "Generating plots..."
    python3 plot.py --csv "$CSV_FILE" --plot-dir data
    echo "Plots written to data/"
fi

echo "Done."
