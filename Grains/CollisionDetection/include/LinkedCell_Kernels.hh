#ifndef _LINKEDCELL_KERNELS_HH_
#define _LINKEDCELL_KERNELS_HH_

#include <cooperative_groups.h>

#include "Basic.hh"
#include "Cells.hh"

// =============================================================================
/** @brief The class LinkedCell_Kernels.

    This header file contains the declarations of the various kernels used for
    updating the linked cells in the simulation.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name LinkedCell_Kernels: External Kernels */
//@{
/** @brief Resizes the cells
    @param cells pointer to the Cells object
    @param cellSize new size of the cell
    @param numCells number of cells */
template <typename T>
__GLOBAL__ void
    resizeCells_Device(Cells<T>** cells, const T cellSize, uint* numCells)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID > 0)
        return;

    cells[0]->resize(cellSize);
    *numCells = cells[0]->getNumCells();
}

// -----------------------------------------------------------------------------
/** @brief Gets the neighbor cells array
    @param cells pointer to the Cells object
    @param numCells number of cells
    @param tr transformations */
template <typename T>
__GLOBAL__ void generateNeighborCells_Device(const Cells<T>* const* cells,
                                             const uint             numCells,
                                             uint* neighborCells)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numCells)
        return;
    // Generate neighbor cells for the cell with index tID
    // Each thread is responsible for one cell
    cells[0]->generateNeighborCells(neighborCells, tID, tID + 1);
}

// -----------------------------------------------------------------------------
/** @brief Computes the cell hash for a given point
    @param cells pointer to the Cells object
    @param positions buffer of positions
    @param numParticles number of particles
    @param particleHash particle hash */
template <typename T>
__GLOBAL__ void computeHash_Device(const Cells<T>* const* cells,
                                   const Vector3<T>*      positions,
                                   uint                   numParticles,
                                   uint*                  particleHash)
{
    // TODO: Load cells to shared memory if needed
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numParticles)
        return;

    particleHash[tID] = cells[0]->computeCellHash(positions[tID]);
}

// -----------------------------------------------------------------------------
/** @brief Finds the start of each cell
    The cellStart array will contain the start index for each cell hash,
    @param particleHash Array of particle hashes
    @param numParticles Number of particles
    @param cellStart Output array to store start indices for each cell hash */
static __GLOBAL__ void computeCellStart_Kernel(const uint* particleHash,
                                               uint        numParticles,
                                               uint*       cellStart)
{
    using namespace cooperative_groups;
    // Handle to thread block group
    thread_block           cta = this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint                   tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;
    if(tid < numParticles)
    {
        hash = particleHash[tid];
        // Load hash data into shared memory so that we can look at neighboring
        // particle's hash value without loading two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;
        // first thread in block must load neighboring particle hash as well
        if(tid > 0 && threadIdx.x == 0)
            sharedHash[0] = particleHash[tid - 1];
    }
    sync(cta);

    if(tid < numParticles)
    {
        // If this particle has a different cell hash value to the previous
        // particle then it must be the first particle in the cell.
        // As it isn't the first particle, it must also be the end of the
        // previous particle's cell.
        if(tid == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = tid;
        }
    }
}
//@}

#endif