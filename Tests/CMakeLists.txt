# Testing CMakeLists.txt for GrainsGPU
cmake_minimum_required(VERSION 3.18)

# Set policies
cmake_policy(SET CMP0144 NEW)  # Use upper-case package root variables

# Project declaration
project(GrainsGPU_Tests 
    VERSION 1.0.0
    DESCRIPTION "Test suite for GrainsGPU"
    LANGUAGES CXX CUDA)

# Set C++ standard - use environment variable if available
if(DEFINED ENV{CMAKE_CXX_STANDARD})
    set(CMAKE_CXX_STANDARD $ENV{CMAKE_CXX_STANDARD})
    message(STATUS "Using C++ standard from environment: $ENV{CMAKE_CXX_STANDARD}")
else()
    set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type from environment
if(DEFINED ENV{CMAKE_BUILD_TYPE})
    set(CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE})
    message(STATUS "Using build type from environment: $ENV{CMAKE_BUILD_TYPE}")
endif()

# Find required packages - use environment variables for GTest
if(DEFINED ENV{GTEST_ROOT})
    set(GTEST_ROOT $ENV{GTEST_ROOT})
    message(STATUS "Using GTest root from environment: $ENV{GTEST_ROOT}")
endif()

find_package(GTest QUIET)
if(NOT GTest_FOUND)
    # Try to find using PkgConfig as fallback
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(GTEST gtest>=1.10.0)
        pkg_check_modules(GTEST_MAIN gtest_main>=1.10.0)
    endif()
    
    # If still not found, use system installation or environment variables
    if(NOT GTEST_FOUND)
        message(STATUS "GTest not found via find_package, trying environment variables...")
        
        # Use environment variables for paths
        set(GTEST_SEARCH_PATHS)
        if(DEFINED ENV{GTEST_INCLUDE_DIR})
            list(APPEND GTEST_SEARCH_PATHS $ENV{GTEST_INCLUDE_DIR})
            message(STATUS "Using GTest include from environment: $ENV{GTEST_INCLUDE_DIR}")
        else()
            list(APPEND GTEST_SEARCH_PATHS /usr/include /usr/local/include)
        endif()
        
        find_path(GTEST_INCLUDE_DIR gtest/gtest.h
            HINTS ${GTEST_SEARCH_PATHS})
        
        # Search for libraries using environment variables
        set(GTEST_LIB_SEARCH_PATHS)
        if(DEFINED ENV{GTEST_LIBRARY_DIR})
            list(APPEND GTEST_LIB_SEARCH_PATHS $ENV{GTEST_LIBRARY_DIR})
            message(STATUS "Using GTest library path from environment: $ENV{GTEST_LIBRARY_DIR}")
        else()
            list(APPEND GTEST_LIB_SEARCH_PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)
        endif()
        
        find_library(GTEST_LIBRARY gtest
            HINTS ${GTEST_LIB_SEARCH_PATHS})
        find_library(GTEST_MAIN_LIBRARY gtest_main
            HINTS ${GTEST_LIB_SEARCH_PATHS})
        
        if(GTEST_INCLUDE_DIR AND GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
            set(GTEST_FOUND TRUE)
            # Create imported targets
            add_library(GTest::GTest UNKNOWN IMPORTED)
            set_target_properties(GTest::GTest PROPERTIES
                IMPORTED_LOCATION "${GTEST_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}")
            
            add_library(GTest::Main UNKNOWN IMPORTED)
            set_target_properties(GTest::Main PROPERTIES
                IMPORTED_LOCATION "${GTEST_MAIN_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GTEST_INCLUDE_DIR}")
        endif()
    endif()
endif()

# Handle case where GTest is still not found
if(NOT GTEST_FOUND AND NOT GTest_FOUND)
    message(FATAL_ERROR "Google Test not found. Please install libgtest-dev or set GTEST_ROOT")
endif()

# Set policy to suppress the FindCUDA deprecation warning
cmake_policy(SET CMP0146 OLD)

# Find CUDA using environment variables if available
if(DEFINED ENV{GRAINS_GPU_COMPILER_ROOT})
    set(CUDA_TOOLKIT_ROOT_DIR $ENV{GRAINS_GPU_COMPILER_ROOT})
    message(STATUS "Using CUDA from environment: ${CUDA_TOOLKIT_ROOT_DIR}")
endif()

find_package(CUDA QUIET)
if(CUDA_FOUND)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CUDA_VERSION}")
    
    # Set CUDA architecture from environment
    if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES $ENV{CMAKE_CUDA_ARCHITECTURES})
        message(STATUS "Using CUDA architectures from environment: $ENV{CMAKE_CUDA_ARCHITECTURES}")
    endif()
    
    # Add CUDA include directories from environment or found paths
    if(DEFINED ENV{GRAINS_GPU_COMPILER_INCDIR})
        include_directories($ENV{GRAINS_GPU_COMPILER_INCDIR})
        message(STATUS "Using CUDA includes from environment: $ENV{GRAINS_GPU_COMPILER_INCDIR}")
    else()
        include_directories(${CUDA_INCLUDE_DIRS})
    endif()
    
    # Define CUDA availability for preprocessor
    add_definitions(-DCUDA_AVAILABLE=1)
else()
    message(WARNING "CUDA not found. GPU tests will be disabled.")
    add_definitions(-DCUDA_AVAILABLE=0)
endif()

# Add flag to disable CUDA in test compilation for compatibility
add_definitions(-DTEST_BUILD=1)

# Include directories - use environment variables where available
set(GRAINS_INCLUDE_DIRS)
include_directories(${GRAINS_INCLUDE_DIRS})

# Use environment variable for Xerces include directory
if(DEFINED ENV{GRAINS_XERCES_INCDIR})
    include_directories($ENV{GRAINS_XERCES_INCDIR})
    message(STATUS "Using Xerces include directory: $ENV{GRAINS_XERCES_INCDIR}")
else()
    message(WARNING "GRAINS_XERCES_INCDIR environment variable not set")
endif()

# Collect test source files
set(TEST_SOURCES
    # Test runner
    main.cpp
)

# Check for test files and add them if they exist
set(POTENTIAL_TEST_FILES
    # Geometry tests
    geometry/test_vector3.cpp
    geometry/test_matrix3.cpp
    geometry/test_quaternion.cpp
    geometry/test_transform3.cpp
    
    # Collision detection tests
    collision/test_gjk.cpp
    collision/test_obb.cpp
    collision/test_collision_detection.cpp
    
    # Component tests
    component/test_rigid_body.cpp
    component/test_convex.cpp
    
    # Integration tests
    integration/test_physics_integration.cpp
    
    # Performance tests
    performance/test_benchmarks.cpp
)

# Add existing test files to sources
foreach(test_file ${POTENTIAL_TEST_FILES})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${test_file}")
        list(APPEND TEST_SOURCES ${test_file})
        message(STATUS "Added test file: ${test_file}")
    else()
        message(STATUS "Test file not found, skipping: ${test_file}")
    endif()
endforeach()

# Add CUDA tests if CUDA is available
if(CUDA_FOUND)
    set(CUDA_TEST_FILES
        cuda/test_cuda_kernels.cpp
    )
    
    foreach(cuda_test_file ${CUDA_TEST_FILES})
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${cuda_test_file}")
            list(APPEND TEST_SOURCES ${cuda_test_file})
            message(STATUS "Added CUDA test file: ${cuda_test_file}")
        endif()
    endforeach()
endif()

# Create test executable only if we have test sources
if(TEST_SOURCES)
    add_executable(grains_tests ${TEST_SOURCES})
    
    # Link libraries
    if(GTest_FOUND)
        target_link_libraries(grains_tests GTest::GTest GTest::Main)
    else()
        target_link_libraries(grains_tests ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
        target_include_directories(grains_tests PRIVATE ${GTEST_INCLUDE_DIR})
    endif()
    
    # Link pthread (required by GTest on Linux)
    find_package(Threads REQUIRED)
    target_link_libraries(grains_tests Threads::Threads)
    
    # Link Grains library using environment variables - with object files option
    if(DEFINED ENV{GRAINS_ROOT} AND DEFINED ENV{GRAINS_FULL_EXT})
        # Check if we should use object files or shared library
        if(DEFINED ENV{GRAINS_OBJDIR})
            set(GRAINS_OBJ_DIR "$ENV{GRAINS_OBJDIR}")
            if(EXISTS ${GRAINS_OBJ_DIR})
                # Link against object files for better CUDA device linking
                file(GLOB GRAINS_OBJECTS "${GRAINS_OBJ_DIR}/*.o")
                # Filter out any main.o to avoid conflicts
                list(FILTER GRAINS_OBJECTS EXCLUDE REGEX "main\\.o$")
                target_link_libraries(grains_tests ${GRAINS_OBJECTS})
                message(STATUS "Linked Grains object files: ${GRAINS_OBJ_DIR} (${list_length} objects)")
            endif()
        else()
            # Fallback to shared library
            set(GRAINS_LIB_DIR "$ENV{GRAINS_ROOT}/lib$ENV{GRAINS_FULL_EXT}")
            set(GRAINS_LIB_PATH "${GRAINS_LIB_DIR}/libgrains.so")
            
            if(EXISTS ${GRAINS_LIB_PATH})
                target_link_libraries(grains_tests ${GRAINS_LIB_PATH})
                message(STATUS "Linked Grains library: ${GRAINS_LIB_PATH}")
            else()
                message(WARNING "Grains library not found at ${GRAINS_LIB_PATH}")
            endif()
        endif()
    else()
        message(WARNING "GRAINS_ROOT or GRAINS_FULL_EXT environment variables not set")
    endif()
    
    # Link CUDA libraries if available
    if(CUDA_FOUND)
        # Set CUDA standard from environment
        if(DEFINED ENV{CMAKE_CUDA_STANDARD})
            set_property(TARGET grains_tests PROPERTY CUDA_STANDARD $ENV{CMAKE_CUDA_STANDARD})
            message(STATUS "Using CUDA standard from environment: $ENV{CMAKE_CUDA_STANDARD}")
        else()
            set_property(TARGET grains_tests PROPERTY CUDA_STANDARD 14)
        endif()
        set_property(TARGET grains_tests PROPERTY CUDA_STANDARD_REQUIRED ON)
        
        # Use environment variables for CUDA library path if available
        if(DEFINED ENV{GRAINS_GPU_COMPILER_LIBDIR})
            link_directories($ENV{GRAINS_GPU_COMPILER_LIBDIR})
            message(STATUS "Using CUDA libraries from environment: $ENV{GRAINS_GPU_COMPILER_LIBDIR}")
        endif()
        
        target_link_libraries(grains_tests ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})
    endif()
    
    # Link Xerces XML library using environment variables
    if(DEFINED ENV{GRAINS_XERCES_LIBDIR})
        # Find the actual Xerces library files
        find_library(XERCES_LIB xerces-c 
            HINTS $ENV{GRAINS_XERCES_LIBDIR}
            NO_DEFAULT_PATH)
        
        if(XERCES_LIB)
            target_link_libraries(grains_tests ${XERCES_LIB})
            message(STATUS "Linked Xerces library: ${XERCES_LIB}")
        else()
            # Fallback to direct path if find_library doesn't work
            set(XERCES_LIB_PATH "$ENV{GRAINS_XERCES_LIBDIR}/libxerces-c.so")
            if(EXISTS ${XERCES_LIB_PATH})
                target_link_libraries(grains_tests ${XERCES_LIB_PATH})
                message(STATUS "Linked Xerces library (direct): ${XERCES_LIB_PATH}")
            else()
                message(WARNING "Xerces library not found at ${XERCES_LIB_PATH}")
            endif()
        endif()
    else()
        message(WARNING "GRAINS_XERCES_LIBDIR environment variable not set")
    endif()
    
    # Enable testing with environment-based configuration
    enable_testing()
    
    # Set test timeout from environment
    if(DEFINED ENV{GRAINS_TEST_TIMEOUT})
        set_tests_properties(grains_unit_tests PROPERTIES TIMEOUT $ENV{GRAINS_TEST_TIMEOUT})
        message(STATUS "Using test timeout from environment: $ENV{GRAINS_TEST_TIMEOUT} seconds")
    endif()
    
    add_test(NAME grains_unit_tests COMMAND grains_tests)
    
    message(STATUS "Test executable 'grains_tests' configured successfully")
else()
    message(WARNING "No test source files found. Please create test files first.")
endif()
