# GrainsGPU Testing Strategy

## Overview

This document outlines the comprehensive testing strategy for GrainsGPU. Our testing approach ensures reliability, performance, and correctness across CPU and GPU implementations.

## Testing Philosophy

### 1. **Component-Wise Testing**
- Each major component (Geometry, Collision Detection, Physics Integration) has dedicated test suites
- Tests are isolated and don't depend on external systems
- Mock objects are used when testing interactions between components

### 2. **Multi-Platform Testing**
- **CPU Tests**: Standard C++ unit tests using Google Test
- **GPU Tests**: CUDA kernel tests with device memory management
- **Cross-Platform**: Tests that verify CPU-GPU result consistency

### 3. **Test Categories**

#### Unit Tests
- Test individual functions and classes
- Fast execution (< 1ms per test)
- No external dependencies
- Located in: `Tests/geometry/`, `Tests/math/`, `Tests/collision/`

#### Integration Tests  
- Test component interactions
- End-to-end simulation scenarios
- Located in: `Tests/integration/`

#### Performance Tests
- Benchmark critical algorithms (GJK, quaternion operations)
- Memory usage verification
- Located in: `Tests/performance/`

#### CUDA Tests
- Device function testing
- Memory management verification
- Located in: `Tests/cuda/`

## Running Tests

### Prerequisites
**Important**: Before building tests, ensure the Grains library is compiled:
```bash
cd ../Grains
make
```

This will generate the required object files in `Grains/objGNU-*/` that the test build system uses.

### Quick Start (Optimized Build)
The test system is optimized to use pre-compiled object files from the Grains library, eliminating the need to recompile everything:

```bash
cd Tests
chmod +x run_tests.sh
./run_tests.sh all
```

### Manual Build (Using CMake with Pre-compiled Objects)
```bash
cd Tests
mkdir build && cd build
source ../../Env/grainsGPU.env.sh
cmake ..
make -j
./grains_tests
```

**Performance Note**: The optimized build system uses existing object files from `Grains/objGNU-*` and include files from `Grains/include`, which significantly reduces build time compared to compiling everything from scratch.

### Automatic Dependency Tracking

The build system delegates to the Grains Makefile for dependency tracking and compilation:

- **Makefile Integration**: CMake calls `make` in the Grains directory to handle all source dependencies
- **Automatic Rebuilds**: The Grains Makefile determines what needs recompilation based on file timestamps
- **Minimal CMake**: CMake focuses only on test compilation and linking, leaving Grains compilation to its native Makefile

### Manual Control

```bash
# Manually rebuild Grains library
make rebuild-grains
```

### Specific Test Categories
```bash
./run_tests.sh unit          # Unit tests only
./run_tests.sh integration   # Integration tests only
./run_tests.sh cuda         # CUDA tests only
./run_tests.sh performance  # Performance tests only
```

### Manual CMake Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
./grains_tests
```

## Writing New Tests

### Test Naming Convention
- Test files: `test_<component>.cpp`
- Test classes: `<Component>Test`
- Test methods: `TEST_F(<Component>Test, <SpecificFeature>)`

### Example Test Structure
```cpp
#include <gtest/gtest.h>
#include "YourComponent.hh"

class YourComponentTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize test data
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
    
    // Test data members
    const double EPSILON = 1e-10;
};

TEST_F(YourComponentTest, SpecificFeatureTest) {
    // Arrange
    YourComponent component(/* parameters */);
    
    // Act
    auto result = component.doSomething();
    
    // Assert
    EXPECT_NEAR(result, expected_value, EPSILON);
}
```

### CUDA Test Structure
```cpp
#include <gtest/gtest.h>
#include <cuda_runtime.h>

__global__ void test_kernel(/* parameters */) {
    // Your CUDA kernel code
}

TEST(CudaTest, KernelFunctionality) {
    // Setup device memory
    // Launch kernel
    // Copy results back
    // Verify results
    // Cleanup device memory
}
```

## Test Coverage Goals

- **Unit Tests**: > 90% line coverage for core algorithms
- **Integration Tests**: Cover all major use cases
- **Performance Tests**: Regression testing for critical paths
- **CUDA Tests**: Device function coverage matching CPU tests

## Continuous Integration

### GitHub Actions Pipeline
- **CPU Tests**: Run on every commit and PR
- **GPU Tests**: Run on dedicated GPU runners
- **Static Analysis**: cppcheck and clang-tidy
- **Coverage Reporting**: Automated coverage reports

### Quality Gates
- All tests must pass before merge
- No decrease in test coverage
- Performance regressions flagged
- Static analysis warnings addressed

## Testing Best Practices

### 1. **Numerical Precision**
- Use appropriate epsilon values for floating-point comparisons
- Test with both single and double precision
- Consider accumulated numerical errors in iterative algorithms

### 2. **Physics Validation**
- Test conservation laws (energy, momentum)
- Verify physical constraints (non-penetration, friction)
- Use analytical solutions where available

### 3. **Edge Cases**
- Test boundary conditions
- Handle degenerate cases (zero vectors, singular matrices)
- Test with extreme values (very large/small numbers)

### 4. **Memory Management**
- Verify CUDA memory allocation/deallocation
- Test for memory leaks in repeated operations
- Validate host-device memory transfers

### 5. **Determinism**
- Ensure reproducible results with fixed seeds
- Test parallel algorithm consistency
- Verify GPU vs CPU result matching

## Tools and Dependencies

### Required Tools
- **Google Test**: Unit testing framework
- **Google Benchmark**: Performance testing
- **CUDA Toolkit**: GPU testing
- **CMake**: Build system
- **lcov/gcov**: Coverage reporting

### Optional Tools
- **Valgrind**: Memory error detection
- **NVIDIA Nsight**: GPU profiling
- **cppcheck**: Static analysis
- **clang-tidy**: Code quality

## Debugging Failed Tests

### Common Issues
1. **Floating-point precision**: Adjust epsilon values
2. **CUDA context**: Ensure proper device initialization
3. **Memory alignment**: Check for proper CUDA memory alignment
4. **Race conditions**: Verify thread safety in parallel code

### Debugging Commands
```bash
# Run specific test with verbose output
./grains_tests --gtest_filter="*SpecificTest*" --gtest_verbose

# Run with memory checking
valgrind --leak-check=full ./grains_tests

# CUDA debugging
cuda-gdb ./grains_tests
```

## Future Enhancements

1. **Property-Based Testing**: Generate random test cases
2. **Fuzzing**: Test with random/malformed inputs
3. **Hardware-in-the-Loop**: Test with real sensor data
4. **Cross-Platform**: Test on different GPU architectures
5. **Regression Testing**: Automated performance baseline comparison

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure both CPU and GPU implementations are tested
3. Add performance benchmarks for critical algorithms
4. Update documentation with new test procedures
5. Verify CI pipeline passes completely

For questions or issues with testing, please open an issue in the repository.
