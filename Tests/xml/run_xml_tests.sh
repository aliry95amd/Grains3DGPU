#!/bin/bash
# Run XML tests under Tests/xml/* using the built grains binary,
# then run any Python plotting scripts in each test directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root is two levels above Tests/xml
GRAINS_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"

BINARY="${BINARY:-$GRAINS_HOME/Main/bin${GRAINS_FULL_EXT}/grains}"
TIMEOUT_CMD="${TIMEOUT_CMD:-timeout}"
# Create root logs, plots and results directories next to this script
LOG_ROOT="$SCRIPT_DIR/logs"
PLOTS_ROOT="$SCRIPT_DIR/plots"
RESULTS_ROOT="$SCRIPT_DIR/results"
mkdir -p "$LOG_ROOT" "$PLOTS_ROOT" "$RESULTS_ROOT"

# Color codes for final message
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Track test counts
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

if [ ! -x "$BINARY" ]; then
    printf "Error: grains binary not found or not executable at %s\n" "$BINARY" >&2
    exit 1
fi

echo
echo "=========================================="
echo "  GrainsGPU XML Tests"
echo "=========================================="
echo

# Iterate subdirectories of Tests/xml
# If this script lives inside Tests/xml, use SCRIPT_DIR directly, otherwise append /xml
if [ "$(basename "$SCRIPT_DIR")" = "xml" ]; then
    XML_ROOT="$SCRIPT_DIR"
fi

for testdir in "$XML_ROOT"/*/; do
    [ -d "$testdir" ] || continue
    testname="$(basename "$testdir")"
    # Skip runner-created directories
    if [ "$testname" = "logs" ] || [ "$testname" = "plots" ] || [ "$testname" = "results" ]; then
        continue
    fi
    printf "Running: %s\n" "$testname"

    # find xml files in the directory (non-recursive)
    shopt -s nullglob
    # compute column width so .xml and .py lines align
    maxlen=0
    for f in "$testdir"/*.xml; do
        [ -f "$f" ] || continue
        name="$(basename "$f")"
        l=${#name}
        if [ $l -gt $maxlen ]; then
            maxlen=$l
        fi
    done
    for p in "$testdir"/*.py; do
        [ -f "$p" ] || continue
        name="$(basename "$p")"
        l=${#name}
        if [ $l -gt $maxlen ]; then
            maxlen=$l
        fi
    done
    if [ $maxlen -lt 20 ]; then
        maxlen=20
    fi

    for xml in "$testdir"/*.xml; do
        xmlbase="$(basename "$xml")"
        # per-test directories (logs, plots, results)
        TEST_LOG_DIR="$LOG_ROOT/$testname"
        TEST_PLOTS_DIR="$PLOTS_ROOT/$testname"
        TEST_RESULTS_DIR="$RESULTS_ROOT/$testname"
        mkdir -p "$TEST_LOG_DIR" "$TEST_PLOTS_DIR" "$TEST_RESULTS_DIR"
        log_file="$TEST_LOG_DIR/${xmlbase%.xml}.log"
        # Run the test and capture output to log_file (both stdout and stderr).
        # Run binary with RESULTS_DIR pointing to the per-test results directory
        TESTS_RUN=$((TESTS_RUN+1))
        if RESULTS_DIR="$TEST_RESULTS_DIR" $TIMEOUT_CMD 300 "$BINARY" "$xml" > "$log_file" 2>&1; then
            printf "  %-*s: %bSIMULATION FINISHED%b\n" "$maxlen" "$xmlbase" "$GREEN" "$NC"
            TESTS_PASSED=$((TESTS_PASSED+1))
        else
            printf "  %-*s: %bFAIL%b (log: %s)\n" "$maxlen" "$xmlbase" "$RED" "$NC" "$log_file"
            TESTS_FAILED=$((TESTS_FAILED+1))
            # skip plotting for failed runs
            continue
        fi

        # Run Python plotting scripts in the same test directory
        for py in "$testdir"/*.py; do
            pybase="$(basename "$py")"
            plot_log="$TEST_LOG_DIR/${pybase%.py}.plot.log"
            # Run plotting script from the test directory and export RESULTS_DIR and PLOTS_DIR
            if (cd "$testdir" && RESULTS_DIR="$TEST_RESULTS_DIR" PLOTS_DIR="$TEST_PLOTS_DIR" python3 "$py") > "$plot_log" 2>&1; then
                printf "  %-*s: %bPLOT GENERATED%b\n" "$maxlen" "$pybase" "$GREEN" "$NC"
            else
                printf "  %-*s: %bFAILED%b (log: %s)\n" "$maxlen" "$pybase" "$RED" "$NC" "$plot_log" >&2
            fi
        done
    done
    shopt -u nullglob
done

echo
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo "Run:    $TESTS_RUN"
echo -e "Passed: ${GREEN}$TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "Failed: ${RED}$TESTS_FAILED${NC}"
else
    echo "Failed: 0"
fi
