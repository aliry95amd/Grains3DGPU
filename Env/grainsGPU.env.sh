# Definition
# CPU
export GRAINS_CPP_COMPILER=g++
export GRAINS_CPP_COMPILER_DIST="GNU"
export GRAINS_CPP_COMPILER_VERSION="13.2.0"
# End CPU


# GPU
export GRAINS_GPU_COMPILER=nvcc
export GRAINS_GPU_COMPILER_DIST="CUDA"
export GRAINS_GPU_COMPILER_VERSION="12.6.0"
export GRAINS_GPU_COMPILER_ROOT=/usr/local/cuda-12.6
export GRAINS_GPU_COMPILER_INCDIR="${GRAINS_GPU_COMPILER_ROOT}/include"
export GRAINS_GPU_COMPILER_BINDIR="${GRAINS_GPU_COMPILER_ROOT}/bin"
export GRAINS_GPU_COMPILER_LIBDIR="${GRAINS_GPU_COMPILER_ROOT}/lib64"
# End GPU


# Grains
export GRAINS_FULL_EXT=${GRAINS_CPP_COMPILER_DIST}-${GRAINS_CPP_COMPILER_VERSION}-${GRAINS_GPU_COMPILER_DIST}-${GRAINS_GPU_COMPILER_VERSION}
export GRAINS_HOME=${HOME}/Desktop/Work/Codes/GrainsGPU
export GRAINS_ROOT=${GRAINS_HOME}/Grains
export GRAINS_INCDIR=${GRAINS_ROOT}/include
export GRAINS_OBJDIR=${GRAINS_ROOT}/obj${GRAINS_FULL_EXT}
export GRAINS_LIBDIR=${GRAINS_ROOT}/lib${GRAINS_FULL_EXT}
# End Grains


# Xerces
export GRAINS_XERCES_ROOT=${GRAINS_HOME}/XERCES-2.8.0
export GRAINS_XERCES_INCDIR="${GRAINS_XERCES_ROOT}/include"
export GRAINS_XERCES_BINDIR="${GRAINS_XERCES_ROOT}/bin"
export GRAINS_XERCES_LIBDIR="${GRAINS_XERCES_ROOT}/lib64-${GRAINS_CPP_COMPILER_DIST}-${GRAINS_CPP_COMPILER_VERSION}"
# End Xerces


# Grains Test
export GTEST_ROOT=/usr
export GTEST_INCLUDE_DIR=/usr/include
export GTEST_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu
export GRAINS_TEST_TIMEOUT=300
export GRAINS_TEST_PARALLEL_JOBS=8
# End Testing


# Display
echo -e '\033[31mGRAINS_HOME\033[0m =' $GRAINS_HOME
echo -e '\033[31mGRAINS_CPP_COMPILER\033[0m =' $GRAINS_CPP_COMPILER
echo -e '\033[31mGRAINS_CPP_COMPILER_DIST\033[0m =' $GRAINS_CPP_COMPILER_DIST
echo -e '\033[31mGRAINS_CPP_COMPILER_VERSION\033[0m =' $GRAINS_CPP_COMPILER_VERSION
echo -e '\033[31mGRAINS_GPU_COMPILER\033[0m =' $GRAINS_GPU_COMPILER
echo -e '\033[31mGRAINS_GPU_COMPILER_DIST\033[0m =' $GRAINS_GPU_COMPILER_DIST
echo -e '\033[31mGRAINS_GPU_COMPILER_VERSION\033[0m =' $GRAINS_GPU_COMPILER_VERSION
echo -e '\033[31mGRAINS_GPU_COMPILER_ROOT\033[0m =' $GRAINS_GPU_COMPILER_ROOT
echo -e '\033[31mGRAINS_FULL_EXT\033[0m =' $GRAINS_FULL_EXT
echo -e '\033[31mXERCES_ROOT\033[0m =' $GRAINS_XERCES_ROOT
# End Display


# Compilers
export GRAINS_CPP_COMPILER_FLAGS="-m64 -O3 -fPIC -std=c++20 \
    -Wno-ctor-dtor-privacy \
    -Wall -Wextra -Wconversion -Wshadow -Wpedantic -Wwrite-strings \
    -fmax-errors=8 \
    -g"
export GRAINS_CPP_LINKER_FLAGS="${GRAINS_CPP_COMPILER_FLAGS} -shared"
###########
export GRAINS_GPU_COMPILER="${GRAINS_GPU_COMPILER_BINDIR}/${GRAINS_GPU_COMPILER}"
export GRAINS_GPU_LINKER="${GRAINS_GPU_COMPILER_BINDIR}/${GRAINS_GPU_COMPILER}"
# Release mode flags (optimized for performance)
export GRAINS_GPU_COMPILER_FLAGS_RELEASE="-t=8 -x cu -m64 \
    -O3 -dlto -dc \
    -std=c++20 -arch=sm_75 \
    -cudart static -cudadevrt static \
    -use_fast_math -extra-device-vectorization -restrict \
    --extended-lambda --expt-relaxed-constexpr \
    -Xcompiler \"-rdynamic,-fPIC,-fopenmp\" \
    -g"
export GRAINS_GPU_LINKER_FLAGS_RELEASE="-O3 -dlto \
    -arch=sm_75 -lcudart \
    -use_fast_math -extra-device-vectorization -restrict \
    -lcudart -lcudadevrt \
    -lgomp \
    -g"

# Debug mode flags (full debug info, no optimization)
export GRAINS_GPU_COMPILER_FLAGS_DEBUG="-t=8 -x cu -m64 \
    -O0 -G -dc \
    -std=c++20 -arch=sm_75 \
    -cudart static -cudadevrt static \
    --extended-lambda --expt-relaxed-constexpr \
    -Xcompiler "-rdynamic,-fPIC,-fopenmp" \
    -g -DDEBUG"
export GRAINS_GPU_LINKER_FLAGS_DEBUG="-O0 -G \
    -arch=sm_75 -lcudart \
    -lcudart -lcudadevrt \
    -lgomp \
    -g"

# Set active flags based on MODE (default: release)
export MODE=${MODE:-release}
if [ "$MODE" = "debug" ]; then
    export GRAINS_GPU_COMPILER_FLAGS="$GRAINS_GPU_COMPILER_FLAGS_DEBUG"
    export GRAINS_GPU_LINKER_FLAGS="$GRAINS_GPU_LINKER_FLAGS_DEBUG"
else
    export GRAINS_GPU_COMPILER_FLAGS="$GRAINS_GPU_COMPILER_FLAGS_RELEASE"
    export GRAINS_GPU_LINKER_FLAGS="$GRAINS_GPU_LINKER_FLAGS_RELEASE"
fi
###########
export GRAINS_XERCES_FLAGS="-L${GRAINS_XERCES_LIBDIR} -lxerces-c -lxerces-depdom"
###########
export GRAINS_Z_LIB="/usr/lib64"
export GRAINS_Z_FLAGS="-L${GRAINS_Z_LIB} -lz"
# End Flags


# CMake Configuration
export CMAKE_CXX_STANDARD=20
export CMAKE_BUILD_TYPE=Release
export CMAKE_CUDA_ARCHITECTURES=75
export CMAKE_CUDA_STANDARD=20
export CMAKE_PREFIX_PATH="${GRAINS_XERCES_ROOT}:${GRAINS_GPU_COMPILER_ROOT}:${CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="${GRAINS_XERCES_LIBDIR}/pkgconfig:${PKG_CONFIG_PATH}"
# End CMake


# LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GRAINS_XERCES_LIBDIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GRAINS_ROOT}/lib${GRAINS_FULL_EXT}
# End LD_LIBRARY_PATH


# Compatibilty for Xerces
source $GRAINS_HOME/Env/grains_xerces.env.sh