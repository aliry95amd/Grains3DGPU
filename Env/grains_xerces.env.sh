#!/bin/bash
export GRAINS_BITS_DEFAULT=64
# Prefer a standard g++ for Xerces (Xerces 2.8.0 configure scripts were written for GCC);
# If g++ is in PATH (e.g. added manually in the ARC env file), use it and derive SERCOMPIL from it.
# Fall back to GRAINS_CPP_COMPILER otherwise.
if command -v g++ >/dev/null 2>&1; then
    _XERCES_CXX=$(command -v g++)
    _XERCES_ENV=GNU
    _XERCES_VER=$(g++ -dumpfullversion 2>/dev/null)
else
    _XERCES_CXX=${GRAINS_CPP_COMPILER}
    _XERCES_ENV=${GRAINS_CPP_COMPILER_DIST}
    _XERCES_VER=${GRAINS_CPP_COMPILER_VERSION}
fi
export GRAINS_SERCOMPIL_ENV=${_XERCES_ENV}
export GRAINS_SERCOMPIL_VERSION=${_XERCES_VER}
export GRAINS_C=gcc
export GRAINS_CPP=${_XERCES_CXX}
unset _XERCES_CXX _XERCES_ENV _XERCES_VER
export XERCESCROOT=${GRAINS_XERCES_ROOT}
export XERCESC_ROOT=${XERCESCROOT}
# Keep GRAINS_XERCES_LIBDIR consistent with the compiler that will actually be
# used to build Xerces (may differ from the main GRAINS_CPP_COMPILER_DIST).
export GRAINS_XERCES_LIBDIR="${GRAINS_XERCES_ROOT}/lib64-${GRAINS_SERCOMPIL_ENV}-${GRAINS_SERCOMPIL_VERSION}"
# We basically rename some variables here. The bash comes from the previous
# version of Grains, and we do this to make it compatible with XERSESC which is
# configured for the previous version of Grains
