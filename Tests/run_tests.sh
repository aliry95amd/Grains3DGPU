#!/bin/bash
# GrainsGPU test runner.

set -e

echo "GrainsGPU tests"

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

run_geometry_tests() {
    echo "Running geometry tests..."
    ./build/grains_tests --gtest_filter="*GeometryTest.*" --gtest_output=xml:geometry_test_results.xml
}

run_collision_tests() {
    echo "Running collision tests..."
    ./build/grains_tests --gtest_filter="*CollisionTest.*" --gtest_output=xml:collision_test_results.xml
}

run_smoke_tests() {
    echo "Running smoke tests..."
    cd smoke
    if [ -f "run_smoke_tests.sh" ]; then
        ./run_smoke_tests.sh
        local exit_code=$?
        cd ..
        return $exit_code
    else
        echo "Error: smoke/run_smoke_tests.sh not found"
        cd ..
        return 1
    fi
}

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

main() {
    local test_type="${1:-all}"
    case $test_type in
        "geometry")
            check_dependencies
            build_tests
            run_geometry_tests
            ;;
        "collision")
            check_dependencies
            build_tests
            run_collision_tests
            ;;
        "smoke")
            run_smoke_tests
            ;;
        "all")
            check_dependencies
            build_tests
            run_geometry_tests
            run_collision_tests
            run_smoke_tests
            generate_coverage
            ;;
        *)
            echo "Usage: $0 [geometry|collision|smoke|all]"
            exit 1
            ;;
    esac
    echo "Test suite complete"
}

main "$@"
