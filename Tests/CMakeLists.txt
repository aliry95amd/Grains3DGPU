# Testing CMakeLists.txt for GrainsGPU - Complete build from scratch with dependency tracking
cmake_minimum_required(VERSION 3.18)

# Set policies
cmake_policy(SET CMP0144 NEW)  # Use upper-case package root variables
cmake_policy(SET CMP0146 OLD)  # Suppress FindCUDA deprecation warning

# Project declaration
project(GrainsGPU_Tests 
    VERSION 1.0.0
    DESCRIPTION "Test suite for GrainsGPU - builds everything from scratch"
    LANGUAGES CXX CUDA)

# Enable multithreaded compilation (like Makefile)
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set_property(GLOBAL PROPERTY JOB_POOLS "compilation=${N}")
    set(CMAKE_JOB_POOL_COMPILE compilation)
    message(STATUS "Enabled parallel compilation with ${N} jobs")
endif()

# Set C++ standard from environment
if(DEFINED ENV{CMAKE_CXX_STANDARD})
    set(CMAKE_CXX_STANDARD $ENV{CMAKE_CXX_STANDARD})
    message(STATUS "Using C++ standard from environment: $ENV{CMAKE_CXX_STANDARD}")
else()
    set(CMAKE_CXX_STANDARD 20)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type from environment
if(DEFINED ENV{CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE})
    message(STATUS "Using build type from environment: $ENV{CMAKE_BUILD_TYPE}")
else()
    set(CMAKE_BUILD_TYPE Release)
endif()

# Verify environment variables are set
if(NOT DEFINED ENV{GRAINS_ROOT})
    message(FATAL_ERROR "GRAINS_ROOT environment variable not set. Please source grainsGPU.env.sh")
endif()

if(NOT DEFINED ENV{GRAINS_FULL_EXT})
    message(FATAL_ERROR "GRAINS_FULL_EXT environment variable not set. Please source grainsGPU.env.sh")
endif()

# Set Grains paths from environment variables
set(GRAINS_ROOT "$ENV{GRAINS_ROOT}")
set(GRAINS_FULL_EXT "$ENV{GRAINS_FULL_EXT}")

message(STATUS "Using Grains root from environment: ${GRAINS_ROOT}")
message(STATUS "Using Grains extension: ${GRAINS_FULL_EXT}")

# Find CUDA using environment variables
if(DEFINED ENV{GRAINS_GPU_COMPILER_ROOT})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{GRAINS_GPU_COMPILER_ROOT})
    message(STATUS "Using CUDA from environment: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

find_package(CUDA REQUIRED)
enable_language(CUDA)
message(STATUS "CUDA found: ${CUDA_VERSION}")

# Set CUDA architecture from environment
if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
    message(STATUS "Using CUDA architectures from environment: $ENV{CMAKE_CUDA_ARCHITECTURES}")
endif()

# Set CUDA standard from environment
if(DEFINED ENV{CMAKE_CUDA_STANDARD})
    set(CMAKE_CUDA_STANDARD $ENV{CMAKE_CUDA_STANDARD})
    message(STATUS "Using CUDA standard from environment: $ENV{CMAKE_CUDA_STANDARD}")
else()
    set(CMAKE_CUDA_STANDARD 20)
endif()

# Define compiler from environment variables
set(GRAINS_GPU_COMPILER "$ENV{GRAINS_GPU_COMPILER}")
message(STATUS "Using CUDA compiler: ${GRAINS_GPU_COMPILER}")

# Define linker from environment variables (use the compiler path)
set(GRAINS_GPU_LINKER "${GRAINS_GPU_COMPILER}")
message(STATUS "Using CUDA linker: ${GRAINS_GPU_LINKER}")

# Set nvcc as the CUDA compiler (CMake will handle linking with the same compiler)
set(CMAKE_CUDA_COMPILER "${GRAINS_GPU_COMPILER}")

# Parse compiler flags from environment variables with conservative filtering
if(DEFINED ENV{GRAINS_GPU_COMPILER_FLAGS})
    string(REPLACE " " ";" GRAINS_CUDA_FLAGS_LIST "$ENV{GRAINS_GPU_COMPILER_FLAGS}")
    
    # Use only essential safe flags that are known to work with CMake's CUDA
    set(SAFE_CUDA_FLAGS)
    foreach(flag ${GRAINS_CUDA_FLAGS_LIST})
        # Only include flags that are safe and don't conflict with CMake's CUDA handling
        if(flag MATCHES "^(-O[0-3]|--use_fast_math|-use_fast_math|--extra-device-vectorization|-extra-device-vectorization|--restrict|-restrict|--extended-lambda|--expt-relaxed-constexpr|-g|--generate-line-info|-lineinfo)$" OR
           flag MATCHES "^-Xcompiler")
            list(APPEND SAFE_CUDA_FLAGS ${flag})
        endif()
    endforeach()
    
    message(STATUS "Using safe CUDA compiler flags: ${SAFE_CUDA_FLAGS}")
    string(REPLACE ";" " " CUDA_FLAGS_STRING "${SAFE_CUDA_FLAGS}")
endif()

# Parse linker flags from environment variables with conservative filtering
if(DEFINED ENV{GRAINS_GPU_LINKER_FLAGS})
    string(REPLACE " " ";" GRAINS_CUDA_LINKER_FLAGS_LIST "$ENV{GRAINS_GPU_LINKER_FLAGS}")
    
    # Use only essential safe linker flags
    set(SAFE_CUDA_LINKER_FLAGS)
    foreach(flag ${GRAINS_CUDA_LINKER_FLAGS_LIST})
        # Only include flags that are safe for linking and exclude CUDA-only compiler flags
        if((flag MATCHES "^(-O[0-3]|--use_fast_math|-use_fast_math|-g|--generate-line-info|-lineinfo)$" OR
           flag MATCHES "^-l" OR
           flag MATCHES "^-L") AND
           NOT flag MATCHES "^(--extra-device-vectorization|-extra-device-vectorization|--restrict|-restrict|--extended-lambda|--expt-relaxed-constexpr)$")
            list(APPEND SAFE_CUDA_LINKER_FLAGS ${flag})
        endif()
    endforeach()
    
    message(STATUS "Using safe CUDA linker flags: ${SAFE_CUDA_LINKER_FLAGS}")
    string(REPLACE ";" " " CUDA_LINKER_FLAGS_STRING "${SAFE_CUDA_LINKER_FLAGS}")
endif()

# Define all Grains source directories (following Makefile structure)
set(GRAINS_SOURCE_DIRS
    "${GRAINS_ROOT}/Base/src"
    "${GRAINS_ROOT}/CollisionDetection/src"
    "${GRAINS_ROOT}/Component/src"
    "${GRAINS_ROOT}/Geometry/src"
    "${GRAINS_ROOT}/PostProcessing/src"
    "${GRAINS_ROOT}/TimeIntegration/src"
    "${GRAINS_ROOT}/XML/src"
)

# Define all Grains include directories (following Makefile structure)
set(GRAINS_INCLUDE_DIRS
    "${GRAINS_ROOT}/Base/include"
    "${GRAINS_ROOT}/CollisionDetection/include"
    "${GRAINS_ROOT}/Component/include"
    "${GRAINS_ROOT}/Geometry/include"
    "${GRAINS_ROOT}/PostProcessing/include"
    "${GRAINS_ROOT}/TimeIntegration/include"
    "${GRAINS_ROOT}/XML/include"
)

# Add include directories
include_directories(${GRAINS_INCLUDE_DIRS})

# Add CUDA include directories from environment
if(DEFINED ENV{GRAINS_GPU_COMPILER_INCDIR})
    include_directories($ENV{GRAINS_GPU_COMPILER_INCDIR})
    message(STATUS "Using CUDA includes from environment: $ENV{GRAINS_GPU_COMPILER_INCDIR}")
endif()

# Add Xerces include directories from environment
if(DEFINED ENV{GRAINS_XERCES_INCDIR})
    include_directories($ENV{GRAINS_XERCES_INCDIR})
    message(STATUS "Using Xerces include directory: $ENV{GRAINS_XERCES_INCDIR}")
else()
    message(FATAL_ERROR "GRAINS_XERCES_INCDIR environment variable not set")
endif()

# Find all Grains source files (following Makefile pattern)
set(ALL_GRAINS_SOURCES)
foreach(src_dir ${GRAINS_SOURCE_DIRS})
    file(GLOB sources "${src_dir}/*.cpp")
    if(sources)
        list(APPEND ALL_GRAINS_SOURCES ${sources})
        list(LENGTH sources src_count)
        get_filename_component(dir_name ${src_dir} NAME_WE)
        message(STATUS "Found ${src_count} source files in ${dir_name}")
    endif()
endforeach()

list(LENGTH ALL_GRAINS_SOURCES total_sources)
message(STATUS "Total Grains source files found: ${total_sources}")

if(NOT ALL_GRAINS_SOURCES)
    message(FATAL_ERROR "No Grains source files found! Please check that source files exist in: ${GRAINS_SOURCE_DIRS}")
endif()

# Find all Grains header files for dependency tracking
set(ALL_GRAINS_HEADERS)
foreach(inc_dir ${GRAINS_INCLUDE_DIRS})
    file(GLOB headers "${inc_dir}/*.hh")
    if(headers)
        list(APPEND ALL_GRAINS_HEADERS ${headers})
    endif()
endforeach()

list(LENGTH ALL_GRAINS_HEADERS total_headers)
message(STATUS "Tracking ${total_headers} Grains header files for dependency changes")

# Create Grains library from all sources
add_library(grains_lib STATIC ${ALL_GRAINS_SOURCES})

# Set properties for CUDA compilation
set_target_properties(grains_lib PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Apply CUDA compiler flags to Grains library using target_compile_options
if(DEFINED CUDA_FLAGS_STRING)
    # Convert string back to list for target_compile_options
    string(REPLACE " " ";" CUDA_FLAGS_LIST "${CUDA_FLAGS_STRING}")
    target_compile_options(grains_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS_LIST}>)
    message(STATUS "Applied CUDA compiler flags to grains_lib: ${CUDA_FLAGS_LIST}")
endif()

# Add preprocessor definitions
target_compile_definitions(grains_lib PRIVATE 
    CUDA_AVAILABLE=1 
    _XML
    __CUDACC__
)

# Set CUDA as the language for all source files with proper device code handling
set_source_files_properties(${ALL_GRAINS_SOURCES} PROPERTIES 
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Create dependency tracking target for headers (like Makefile .d files)
add_custom_target(grains_headers
    DEPENDS ${ALL_GRAINS_HEADERS}
    COMMENT "Tracking Grains header dependencies"
)

# Make Grains library depend on headers
add_dependencies(grains_lib grains_headers)

# Find GTest
if(DEFINED ENV{GTEST_ROOT})
    set(GTEST_ROOT $ENV{GTEST_ROOT})
    message(STATUS "Using GTest root from environment: $ENV{GTEST_ROOT}")
endif()

find_package(GTest REQUIRED)
if(NOT GTest_FOUND)
    message(FATAL_ERROR "Google Test not found. Please install libgtest-dev or set GTEST_ROOT")
endif()

# Collect test source files
set(TEST_SOURCES main.cpp)

# Check for test files and add them if they exist
set(POTENTIAL_TEST_FILES
    geometry/test_vector3.cpp
    geometry/test_matrix3.cpp
    geometry/test_quaternion.cpp
    geometry/test_transform3.cpp
    collision/test_gjk.cpp
    collision/test_obb.cpp
    collision/test_collision_detection.cpp
    component/test_rigid_body.cpp
    component/test_convex.cpp
    integration/test_physics_integration.cpp
    performance/test_benchmarks.cpp
    cuda/test_cuda_kernels.cpp
)

foreach(test_file ${POTENTIAL_TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
        list(APPEND TEST_SOURCES ${test_file})
        message(STATUS "Added test file: ${test_file}")
    else()
        message(STATUS "Test file not found, skipping: ${test_file}")
    endif()
endforeach()

# Create test executable
add_executable(grains_tests ${TEST_SOURCES})

# Set CUDA properties for test executable
set_target_properties(grains_tests PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Set test files as CUDA to match Grains library compilation
set_source_files_properties(${TEST_SOURCES} PROPERTIES 
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Add preprocessor definitions for tests
target_compile_definitions(grains_tests PRIVATE 
    CUDA_AVAILABLE=1
    TEST_BUILD=1
    __CUDACC__
)

# Apply CUDA compiler flags only to CUDA compilation using target_compile_options
if(DEFINED CUDA_FLAGS_STRING)
    # Convert string back to list for target_compile_options
    string(REPLACE " " ";" CUDA_FLAGS_LIST "${CUDA_FLAGS_STRING}")
    target_compile_options(grains_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_FLAGS_LIST}>)
    message(STATUS "Applied CUDA compiler flags to grains_tests: ${CUDA_FLAGS_LIST}")
endif()

# Link libraries
target_link_libraries(grains_tests 
    grains_lib
    GTest::GTest 
    GTest::Main
)

# Link pthread (required by GTest)
find_package(Threads REQUIRED)
target_link_libraries(grains_tests Threads::Threads)

# Link CUDA libraries
if(DEFINED ENV{GRAINS_GPU_COMPILER_LIBDIR})
    link_directories($ENV{GRAINS_GPU_COMPILER_LIBDIR})
    message(STATUS "Using CUDA libraries from environment: $ENV{GRAINS_GPU_COMPILER_LIBDIR}")
endif()
target_link_libraries(grains_tests ${CUDA_LIBRARIES})

# Link Xerces library using environment variables
if(DEFINED ENV{GRAINS_XERCES_LIBDIR})
    find_library(XERCES_LIB xerces-c 
        HINTS $ENV{GRAINS_XERCES_LIBDIR}
        NO_DEFAULT_PATH)
    
    if(XERCES_LIB)
        target_link_libraries(grains_tests ${XERCES_LIB})
        message(STATUS "Linked Xerces library: ${XERCES_LIB}")
    else()
        message(FATAL_ERROR "Xerces library not found in $ENV{GRAINS_XERCES_LIBDIR}")
    endif()
else()
    message(FATAL_ERROR "GRAINS_XERCES_LIBDIR environment variable not set")
endif()

# Apply only essential CUDA linker flags that are compatible with g++ final linking
if(DEFINED CUDA_LINKER_FLAGS_STRING AND CMAKE_BUILD_TYPE MATCHES "Debug")
    # Only apply optimization and debug flags that g++ understands for final linking
    string(REPLACE " " ";" CUDA_LINKER_FLAGS_LIST "${CUDA_LINKER_FLAGS_STRING}")
    set(COMPATIBLE_LINKER_FLAGS)
    foreach(flag ${CUDA_LINKER_FLAGS_LIST})
        if(flag MATCHES "^(-O[0-3]|-g)$")
            list(APPEND COMPATIBLE_LINKER_FLAGS ${flag})
        endif()
    endforeach()
    
    if(COMPATIBLE_LINKER_FLAGS)
        string(REPLACE ";" " " COMPATIBLE_FLAGS_STRING "${COMPATIBLE_LINKER_FLAGS}")
        set_target_properties(grains_tests PROPERTIES
            LINK_FLAGS "${COMPATIBLE_FLAGS_STRING}"
        )
        message(STATUS "Applied compatible linker flags: ${COMPATIBLE_FLAGS_STRING}")
    endif()
endif()

# Enable testing
enable_testing()

# Set test timeout from environment
if(DEFINED ENV{GRAINS_TEST_TIMEOUT})
    set(TEST_TIMEOUT $ENV{GRAINS_TEST_TIMEOUT})
    message(STATUS "Using test timeout from environment: ${TEST_TIMEOUT} seconds")
else()
    set(TEST_TIMEOUT 300)
endif()

add_test(NAME grains_unit_tests COMMAND grains_tests)
set_tests_properties(grains_unit_tests PROPERTIES TIMEOUT ${TEST_TIMEOUT})

# Custom targets for maintenance (like Makefile)
add_custom_target(update-tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target grains_tests
    DEPENDS grains_headers ${ALL_GRAINS_SOURCES} ${ALL_GRAINS_HEADERS}
    COMMENT "Force rebuild tests when Grains sources or headers change"
)

add_custom_target(clean-tests
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMENT "Clean all test build artifacts"
)

# Custom targets information
message(STATUS "Custom targets available:")
message(STATUS "  - update-tests: Force rebuild tests when Grains sources change")
message(STATUS "  - clean-tests: Clean all test build artifacts") 
message(STATUS "  - grains_headers: Dependency target for all Grains headers")

message(STATUS "Test executable 'grains_tests' configured successfully with ${total_sources} Grains sources")
