#ifndef _MISC_KERNELS_HH_
#define _MISC_KERNELS_HH_

#include "Basic.hh"

// =============================================================================
/** @brief Miscellaneous kernels for Grains simulation.

    This header file contains miscellaneous kernels that are used in the Grains
    simulation. These kernels are used for various purposes such as initializing
    buffers, computing hashes, and other utility functions.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name Miscellaneous kernels */
//@{
/** @brief Fills a buffer to default 
    @param buffer the buffer to be initialized
    @param size size of the buffer
    @param value the default value to be set (default is T()) */
template <typename T>
__GLOBAL__ void fill_Kernel(T* buffer, const size_t size, const T& value = T())
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;
    buffer[idx] = value;
}

// -----------------------------------------------------------------------------
/** @brief fills a buffer to incremental unsigned i
    @param cells pointer to the Cells object
    @param transforms buffer of transformations
    @param size size of the buffer
    @param particleHash output buffer for particle hashes */
template <typename T>
__GLOBAL__ void fillIncremental_Kernel(T* buffer, const size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;
    buffer[idx] = static_cast<T>(idx);
}
//@}

#endif