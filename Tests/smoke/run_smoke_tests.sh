#!/bin/bash
# Smoke tests for GrainsGPU - quick verification that basic functionality works

set +e  # Don't exit on errors - run all tests

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Setup paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GRAINS_HOME="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load environment
if [ -f "$GRAINS_HOME/Env/grainsGPU.env.sh" ]; then
    source "$GRAINS_HOME/Env/grainsGPU.env.sh" > /dev/null 2>&1
fi

# Configuration
BINARY="${BINARY:-$GRAINS_HOME/Main/bin${GRAINS_FULL_EXT}/grains}"
TIMEOUT=300
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

print_header() {
    echo ""
    echo "=========================================="
    echo "  GrainsGPU Smoke Tests"
    echo "=========================================="
    echo ""
}

run_test() {
    local test_name=$1
    local xml_file=$2
    
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -n "Running test: $test_name ... "
    
    mkdir -p logs
    
    if timeout $TIMEOUT "$BINARY" "$xml_file" > "logs/${test_name}.log" 2>&1; then
        echo -e "${GREEN}PASSED${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}FAILED${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        echo "  See logs/${test_name}.log for details"
        return 1
    fi
}

print_summary() {
    echo ""
    echo "=========================================="
    echo "  Test Summary"
    echo "=========================================="
    echo "Tests run:    $TESTS_RUN"
    echo -e "Tests passed: ${GREEN}$TESTS_PASSED${NC}"
    echo -e "Tests failed: $([ $TESTS_FAILED -gt 0 ] && echo "${RED}$TESTS_FAILED${NC}" || echo "0")"
    echo ""
    
    if [ $TESTS_FAILED -gt 0 ]; then
        echo -e "${RED}SMOKE TESTS FAILED${NC}"
        exit 1
    else
        echo -e "${GREEN}ALL SMOKE TESTS PASSED${NC}"
        exit 0
    fi
}

main() {
    print_header
    
    # Check binary
    if [ ! -f "$BINARY" ]; then
        echo -e "${RED}ERROR: Binary not found at $BINARY${NC}"
        echo "Please compile the project first or set BINARY environment variable"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    
    # Generate tests
    echo "Generating parametric tests..."
    if [ -f "test_params.json" ] && [ -f "generate_tests.py" ]; then
        if python3 generate_tests.py; then
            echo -e "${GREEN}Test generation complete${NC}\n"
        else
            echo -e "${RED}Test generation failed${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Error: test_params.json or generate_tests.py not found${NC}"
        exit 1
    fi
    
    # Run generated tests
    if [ -d "generated" ] && [ -f "generated/test_list.txt" ]; then
        echo "Running parametric tests..."
        while IFS= read -r test_name; do
            if [ -f "generated/${test_name}.xml" ]; then
                run_test "$test_name" "generated/${test_name}.xml"
            fi
        done < "generated/test_list.txt"
    else
        echo -e "${RED}No generated tests found${NC}"
        exit 1
    fi
    
    print_summary
}

main
