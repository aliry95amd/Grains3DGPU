#!/bin/bash

# GrainsGPU Test Runner Script

set -e  # Exit on any error

echo "=== GrainsGPU Test Suite ==="

# Check for required dependencies
check_dependencies() {
    echo "Checking dependencies..."
    
    if ! command -v nvcc &> /dev/null; then
        echo "Error: CUDA compiler (nvcc) not found"
        exit 1
    fi
    
    if ! command -v cmake &> /dev/null; then
        echo "Error: CMake not found"
        exit 1
    fi
    
    echo "Dependencies OK"
}

# Build tests
build_tests() {
    echo "Building test suite..."
    
    mkdir -p build
    cd build
    
    cmake -DCMAKE_BUILD_TYPE=Debug \
          -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
          -DGTEST_ROOT=/usr/local \
          ..
    
    make -j$(nproc)
    cd ..
    
    echo "Build complete"
}

# Run different test categories
run_unit_tests() {
    echo "Running unit tests..."
    ./build/grains_tests --gtest_filter="*Test.*" --gtest_output=xml:unit_test_results.xml
}

run_integration_tests() {
    echo "Running integration tests..."
    ./build/grains_tests --gtest_filter="*IntegrationTest.*" --gtest_output=xml:integration_test_results.xml
}

run_cuda_tests() {
    echo "Running CUDA tests..."
    ./build/grains_tests --gtest_filter="*CudaTest.*" --gtest_output=xml:cuda_test_results.xml
}

run_performance_tests() {
    echo "Running performance tests..."
    ./build/grains_tests --gtest_filter="*PerformanceTest.*" --gtest_output=xml:performance_test_results.xml
}

# Generate coverage report
generate_coverage() {
    echo "Generating coverage report..."
    
    if command -v gcov &> /dev/null; then
        gcov -r build/*.gcno
        lcov --capture --directory . --output-file coverage.info
        genhtml coverage.info --output-directory coverage_report
        echo "Coverage report generated in coverage_report/"
    else
        echo "gcov not available, skipping coverage report"
    fi
}

# Main execution
main() {
    local test_type="${1:-all}"
    
    check_dependencies
    build_tests
    
    case $test_type in
        "unit")
            run_unit_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "cuda")
            run_cuda_tests
            ;;
        "performance")
            run_performance_tests
            ;;
        "all")
            run_unit_tests
            run_integration_tests
            run_cuda_tests
            run_performance_tests
            ;;
        *)
            echo "Usage: $0 [unit|integration|cuda|performance|all]"
            exit 1
            ;;
    esac
    
    generate_coverage
    
    echo "=== Test Suite Complete ==="
}

main "$@"
