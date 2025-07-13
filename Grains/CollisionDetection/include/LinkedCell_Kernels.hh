#ifndef _LINKEDCELL_KERNELS_HH_
#define _LINKEDCELL_KERNELS_HH_

// =============================================================================
/** @brief The class LinkedCell_Kernels.

    This header file contains the declarations of the various kernels used for
    updating the linked cells in the simulation.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name LinkedCell_Kernels: External Kernels */
//@{
/** @brief Zeros out the array
    @param array array to be zero-ed out
    @param numElements number of elements in the array */
__GLOBAL__ void zeroOut_Kernel(uint* array, uint numElements);

/** @brief Kernel to find the start of each cell
    The cellStart array will contain the start index for each cell hash,
    @param particleHash Array of particle hashes
    @param numParticles Number of particles
    @param cellStart Output array to store start indices for each cell hash */
__GLOBAL__ void computeCellStart_Kernel(const uint* particleHash,
                                        uint        numParticles,
                                        uint*       cellStart);
//@}

#endif