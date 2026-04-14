# ARC (Lmod): nvhpc/23.9 bundles CUDA 12.2; GCC 9.4.0 is the host compiler for nvcc and Xerces.
# Lmod is not always initialized in non-interactive shells (e.g. SLURM jobs).
if ! command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
    source /etc/profile.d/zz-60-lmod.sh 2>/dev/null || true
fi
module load nvhpc/23.9 2>/dev/null || true

_GCC940=/arc/software/spack-2024/opt/spack/linux-rocky9-skylake_avx512/gcc-11.4.1/gcc-9.4.0-xraorchustmpt5xtpv3f7z2mw4wdkpef
if [ -d "${_GCC940}" ]; then
    export PATH="${_GCC940}/bin:${PATH}"
fi
unset _GCC940

export GRAINS_CPP_COMPILER=$(command -v g++ 2>/dev/null || echo g++)
export GRAINS_CPP_COMPILER_DIST="GNU"
export GRAINS_CPP_COMPILER_VERSION="9.4.0"

export NVHPC_BASE=${NVHPC_ROOT}/Linux_x86_64/23.9
export GRAINS_GPU_COMPILER=nvcc
export GRAINS_GPU_COMPILER_DIST="CUDA"
export GRAINS_GPU_COMPILER_VERSION="12.2"
export GRAINS_GPU_COMPILER_ROOT=${NVHPC_BASE}/cuda/12.2
export GRAINS_GPU_COMPILER_INCDIR="${GRAINS_GPU_COMPILER_ROOT}/include"
export GRAINS_GPU_COMPILER_BINDIR="${GRAINS_GPU_COMPILER_ROOT}/bin"
export GRAINS_GPU_COMPILER_LIBDIR="${GRAINS_GPU_COMPILER_ROOT}/lib64"

export GRAINS_FULL_EXT=${GRAINS_CPP_COMPILER_DIST}-${GRAINS_CPP_COMPILER_VERSION}-${GRAINS_GPU_COMPILER_DIST}-${GRAINS_GPU_COMPILER_VERSION}
export GRAINS_HOME=/home/aliry95/Grains3DGPU
export GRAINS_ROOT=${GRAINS_HOME}/Grains
export GRAINS_INCDIR=${GRAINS_ROOT}/include
export GRAINS_OBJDIR=${GRAINS_ROOT}/obj${GRAINS_FULL_EXT}
export GRAINS_LIBDIR=${GRAINS_ROOT}/lib${GRAINS_FULL_EXT}

export GRAINS_XERCES_ROOT=${GRAINS_HOME}/XERCES-2.8.0
export GRAINS_XERCES_INCDIR="${GRAINS_XERCES_ROOT}/include"
export GRAINS_XERCES_BINDIR="${GRAINS_XERCES_ROOT}/bin"
export GRAINS_XERCES_LIBDIR="${GRAINS_XERCES_ROOT}/lib64-${GRAINS_CPP_COMPILER_DIST}-${GRAINS_CPP_COMPILER_VERSION}"

export GTEST_ROOT=/usr
export GTEST_INCLUDE_DIR=/usr/include
export GTEST_LIBRARY_DIR=/usr/lib64
export GRAINS_TEST_TIMEOUT=300
export GRAINS_TEST_PARALLEL_JOBS=8

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

export GRAINS_CPP_COMPILER_FLAGS="-m64 -O3 -fPIC -std=c++17 \
    -Wno-deprecated-declarations \
    -Wall -Wextra -Wconversion -Wshadow -Wpedantic \
    -fmax-errors=8 \
    -g"
export GRAINS_CPP_LINKER_FLAGS="${GRAINS_CPP_COMPILER_FLAGS} -shared"
export GRAINS_GPU_COMPILER="${GRAINS_GPU_COMPILER_BINDIR}/${GRAINS_GPU_COMPILER}"
export GRAINS_GPU_LINKER="${GRAINS_GPU_COMPILER_BINDIR}/nvcc"
_GCC940_BIN=$(command -v g++)
export GRAINS_GPU_COMPILER_FLAGS_RELEASE="-x cu -m64 \
    -O3 -dlto -dc \
    -std=c++17 -arch=sm_70 \
    -cudart static -cudadevrt static \
    -use_fast_math -extra-device-vectorization -restrict \
    --extended-lambda --expt-relaxed-constexpr \
    -ccbin ${_GCC940_BIN} \
    -Xcompiler \"-rdynamic,-fPIC,-fopenmp\" \
    -I${NVHPC_BASE}/math_libs/12.2/include \
    -D_PSTL_PAR_BACKEND_SERIAL \
    -DPSTL_USE_PARALLEL_POLICIES=0 \
    -g"
export GRAINS_GPU_LINKER_FLAGS_RELEASE="-O3 -dlto \
    -arch=sm_70 \
    -use_fast_math -extra-device-vectorization -restrict \
    -ccbin ${_GCC940_BIN} \
    -lcudart -lcudadevrt \
    -lgomp \
    -g"

export GRAINS_GPU_COMPILER_FLAGS_DEBUG="-x cu -m64 \
    -O0 -G -dc \
    -std=c++17 -arch=sm_70 \
    -cudart static -cudadevrt static \
    --extended-lambda --expt-relaxed-constexpr \
    -ccbin ${_GCC940_BIN} \
    -Xcompiler \"-rdynamic,-fPIC,-fopenmp\" \
    -I${NVHPC_BASE}/math_libs/12.2/include \
    -D_PSTL_PAR_BACKEND_SERIAL \
    -DPSTL_USE_PARALLEL_POLICIES=0 \
    -g -DDEBUG"
export GRAINS_GPU_LINKER_FLAGS_DEBUG="-O0 -G \
    -arch=sm_70 \
    -ccbin ${_GCC940_BIN} \
    -lcudart -lcudadevrt \
    -lgomp \
    -g"
unset _GCC940_BIN

export MODE=${MODE:-release}
if [ "$MODE" = "debug" ]; then
    export GRAINS_GPU_COMPILER_FLAGS="$GRAINS_GPU_COMPILER_FLAGS_DEBUG"
    export GRAINS_GPU_LINKER_FLAGS="$GRAINS_GPU_LINKER_FLAGS_DEBUG"
else
    export GRAINS_GPU_COMPILER_FLAGS="$GRAINS_GPU_COMPILER_FLAGS_RELEASE"
    export GRAINS_GPU_LINKER_FLAGS="$GRAINS_GPU_LINKER_FLAGS_RELEASE"
fi
export GRAINS_XERCES_FLAGS="-L${GRAINS_XERCES_LIBDIR} -lxerces-c -lxerces-depdom"
export GRAINS_Z_LIB="/home/aliry95/local/lib"
export GRAINS_Z_FLAGS="-L${GRAINS_Z_LIB} -lz"

export CMAKE_CXX_STANDARD=17
export CMAKE_BUILD_TYPE=Release
export CMAKE_CUDA_ARCHITECTURES=75
export CMAKE_CUDA_STANDARD=17
export CMAKE_PREFIX_PATH="${GRAINS_XERCES_ROOT}:${GRAINS_GPU_COMPILER_ROOT}:${CMAKE_PREFIX_PATH}"
export PKG_CONFIG_PATH="${GRAINS_XERCES_LIBDIR}/pkgconfig:${PKG_CONFIG_PATH}"

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GRAINS_XERCES_LIBDIR}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GRAINS_ROOT}/lib${GRAINS_FULL_EXT}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GRAINS_GPU_COMPILER_ROOT}/targets/x86_64-linux/lib

source $GRAINS_HOME/Env/grains_xerces.env.sh
