# GrainsGPU Test Suite - Uses Grains Makefile for dependency tracking
cmake_minimum_required(VERSION 3.18)

cmake_policy(SET CMP0144 NEW)
cmake_policy(SET CMP0146 OLD)

project(GrainsGPU_Tests VERSION 1.0.0 LANGUAGES CXX CUDA)

# Parallel compilation
include(ProcessorCount)
ProcessorCount(N)
if(NOT N EQUAL 0)
    set(CMAKE_BUILD_PARALLEL_LEVEL ${N} CACHE STRING "Number of parallel build jobs")
    set_property(GLOBAL PROPERTY JOB_POOLS "compilation=${N}")
    set(CMAKE_JOB_POOL_COMPILE compilation)
    message(STATUS "Parallel compilation enabled with ${N} jobs")
else()
    set(CMAKE_BUILD_PARALLEL_LEVEL 4 CACHE STRING "Number of parallel build jobs")
    message(STATUS "Parallel compilation enabled with 4 jobs (default)")
endif()

# Standards and build type
set(CMAKE_CXX_STANDARD $ENV{CMAKE_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE})

# Environment validation
if(NOT DEFINED ENV{GRAINS_ROOT})
    message(FATAL_ERROR "GRAINS_ROOT not set. Source grainsGPU.env.sh")
endif()
if(NOT DEFINED ENV{GRAINS_FULL_EXT})
    message(FATAL_ERROR "GRAINS_FULL_EXT not set. Source grainsGPU.env.sh")
endif()

# Paths and configuration
set(GRAINS_ROOT "$ENV{GRAINS_ROOT}")
set(GRAINS_FULL_EXT "$ENV{GRAINS_FULL_EXT}")
set(GRAINS_OBJ_DIR "${GRAINS_ROOT}/obj${GRAINS_FULL_EXT}")
set(GRAINS_INCLUDE_DIR "${GRAINS_ROOT}/include")

# CUDA setup
find_package(CUDA REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_COMPILER "$ENV{GRAINS_GPU_COMPILER}")
set(CMAKE_CUDA_STANDARD $ENV{CMAKE_CUDA_STANDARD})
set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})

# Compiler flags
string(REPLACE " " ";" GRAINS_CUDA_FLAGS_LIST "$ENV{GRAINS_GPU_COMPILER_FLAGS}")
set(SAFE_CUDA_FLAGS)
set(HAS_DEVICE_DEBUG FALSE)
set(HAS_OPTIMIZE FALSE)

# First pass: check if device-debug flag is present
foreach(flag ${GRAINS_CUDA_FLAGS_LIST})
    if(flag MATCHES "^(-g|-G|--device-debug)$")
        set(HAS_DEVICE_DEBUG TRUE)
    endif()
    if(flag MATCHES "^-O[0-3]$")
        set(HAS_OPTIMIZE TRUE)
    endif()
endforeach()

# Second pass: add flags, skip --generate-line-info if device-debug is present
foreach(flag ${GRAINS_CUDA_FLAGS_LIST})
    if(flag MATCHES "^(-O[0-3]|--use_fast_math|--extra-device-vectorization|--restrict|--extended-lambda|--expt-relaxed-constexpr|-g|-G|--device-debug)$" OR flag MATCHES "^-Xcompiler")
        list(APPEND SAFE_CUDA_FLAGS ${flag})
    elseif(flag MATCHES "^(--generate-line-info|-lineinfo)$" AND NOT HAS_DEVICE_DEBUG)
        # Only add line-info flags if device-debug is not present
        list(APPEND SAFE_CUDA_FLAGS ${flag})
    endif()
endforeach()

# Override CMake's default optimization flags to prevent conflicts
if(HAS_OPTIMIZE)
    set(CMAKE_CUDA_FLAGS_DEBUG "")
    set(CMAKE_CUDA_FLAGS_RELEASE "")
    set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "")
    set(CMAKE_CUDA_FLAGS_MINSIZEREL "")
endif()

# Include directories
include_directories(${GRAINS_INCLUDE_DIR})
include_directories($ENV{GRAINS_GPU_COMPILER_INCDIR})
include_directories($ENV{GRAINS_XERCES_INCDIR})

# Delegate to Grains Makefile for dependency tracking and compilation
add_custom_target(grains_build
    COMMAND ${CMAKE_COMMAND} -E chdir ${GRAINS_ROOT} make grains
    COMMENT "Building Grains library using Makefile (target: grains)"
    VERBATIM
)

# Find object files after Grains build
file(GLOB GRAINS_OBJECT_FILES "${GRAINS_OBJ_DIR}/*.o")
if(NOT GRAINS_OBJECT_FILES)
    message(FATAL_ERROR "No object files found. Run 'make' in Grains directory first.")
endif()

# Create Grains library from object files
add_library(grains_lib STATIC ${GRAINS_OBJECT_FILES})
set_target_properties(grains_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    LINKER_LANGUAGE CUDA
)
target_compile_definitions(grains_lib PRIVATE CUDA_AVAILABLE=1 _XML __CUDACC__)
add_dependencies(grains_lib grains_build)

# GTest setup
find_package(GTest REQUIRED)

# Test files
set(TEST_SOURCES main.cpp)
set(POTENTIAL_TEST_FILES
    geometry/test_vector3.cpp
    geometry/test_matrix3.cpp
    geometry/test_quaternion.cpp
    geometry/RotationMathTest.cpp
    geometry/test_transform3.cpp
    geometry/test_transform.cpp
    collision/test_gjk.cpp
    collision/test_obb.cpp
    collision/test_obc.cpp
    collision/test_collision_detection.cpp
    collision/test_neighborlist.cpp
    component/test_rigid_body.cpp
    component/test_convex.cpp
    contactForce/test_contact_table_host.cpp
    contactForce/test_contact_table_device.cpp
)

foreach(test_file ${POTENTIAL_TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
        list(APPEND TEST_SOURCES ${test_file})
    endif()
endforeach()

# Test executable
add_executable(grains_tests ${TEST_SOURCES})
set_target_properties(grains_tests PROPERTIES
    CUDA_STANDARD ${CMAKE_CUDA_STANDARD}
    CUDA_STANDARD_REQUIRED ON
    LINKER_LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
set_source_files_properties(${TEST_SOURCES} PROPERTIES 
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_compile_definitions(grains_tests PRIVATE CUDA_AVAILABLE=1 TEST_BUILD=1 __CUDACC__)
target_compile_options(grains_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${SAFE_CUDA_FLAGS}>)

# Libraries
find_package(Threads REQUIRED)
find_library(XERCES_LIB xerces-c HINTS $ENV{GRAINS_XERCES_LIBDIR} NO_DEFAULT_PATH)
link_directories($ENV{GRAINS_GPU_COMPILER_LIBDIR})

target_link_libraries(grains_tests 
    grains_lib
    GTest::GTest 
    GTest::Main
    Threads::Threads
    ${CUDA_LIBRARIES}
    ${XERCES_LIB}
)

# Testing
enable_testing()
add_test(NAME grains_unit_tests COMMAND grains_tests)
set_tests_properties(grains_unit_tests PROPERTIES TIMEOUT $ENV{GRAINS_TEST_TIMEOUT})

# Custom targets
add_custom_target(rebuild-grains
    COMMAND ${CMAKE_COMMAND} -E chdir ${GRAINS_ROOT} make
    COMMENT "Rebuild Grains library"
)
