#ifndef _LINKEDCELL_KERNELS_HH_
#define _LINKEDCELL_KERNELS_HH_

#include <cooperative_groups.h>

#include "Basic.hh"
#include "Cells.hh"
#include "OBB.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"

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
/** @brief Fills a buffer with component IDs in a way that the first few
    elements correspond to obstacles and the rest to particles
    @param componentID Output buffer for component IDs
    @param obstacleBufferSize Size of the obstacle buffer (the first portion of
    the array)
    @param nObstacles Number of obstacles
    @param nParticles Number of particles */
static __GLOBAL__ void fillComponentID_Device(uint*      componentID,
                                              const uint obstacleBufferSize,
                                              const uint nObstacles,
                                              const uint nParticles)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nParticles)
        return;

    componentID[obstacleBufferSize + tID] = nObstacles + tID;
}

// -----------------------------------------------------------------------------
/** @brief Fills a buffer with cell IDs in a way that the first few
    elements correspond to obstacles and the rest to particles all set to 
    UINT_MAX
    @param componentID Output buffer for component IDs
    @param obstacleBufferSize Size of the obstacle buffer (the first portion of
    the array)
    @param nParticles Number of particles */
static __GLOBAL__ void fillCellID_Device(uint*      cellID,
                                         const uint obstacleBufferSize,
                                         const uint nParticles)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nParticles)
        return;

    cellID[obstacleBufferSize + tID] = UINT_MAX;
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
    @param obstaclesBufferSize size of the obstacles buffer
    @param numParticles number of particles
    @param cellIDs particle hash */
template <typename T>
__GLOBAL__ void computeHash_Device(const Cells<T>* const* cells,
                                   const Vector3<T>*      positions,
                                   const uint             obstaclesBufferSize,
                                   uint                   numParticles,
                                   uint*                  cellIDs)
{
    // TODO: Load cells to shared memory if needed
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numParticles)
        return;

    cellIDs[obstaclesBufferSize + tID]
        = cells[0]->computeCellHash(positions[tID]);
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

// -----------------------------------------------------------------------------
/** @brief Links obstacles to cells
    @param rb pointer to the rigid bodies (obstacles)
    @param positions world-space centers of obstacles
    @param quaternions world-space orientations of obstacles
    @param cells pointer to the Cells object (grid definition)
    @param nObstacles number of obstacles
    @param nCells number of cells
    @param maxPerObstacle maximum number of cells an obstacle can occupy
    @param obstacleIDs list of obstacle IDs
    @param obstacleCellHash list of cells obstacles belong to */
template <typename T>
__GLOBAL__ void linkObstacles_Device(const RigidBody<T>* const* rb,
                                     const Vector3<T>*          positions,
                                     const Quaternion<T>*       quaternions,
                                     const Cells<T>* const*     cells,
                                     const uint                 nObstacles,
                                     const uint                 nCells,
                                     const uint                 maxPerObstacle,
                                     uint*                      obstacleIDs,
                                     uint* obstacleCellHash)
{
    const uint r = blockIdx.x;
    if(r >= nObstacles)
        return;

    // Per-obstacle (per-block) append counter in shared memory
    __shared__ uint index;
    if(threadIdx.x == 0)
        index = 0u;
    __syncthreads();

    const T          cellSize     = cells[0]->getCellSize();
    const T          halfCellSize = T(0.5) * cellSize;
    const Vector3<T> cellBBox(halfCellSize, halfCellSize, halfCellSize);
    const Vector3<T> minCorner = cells[0]->getMinCornerLinkedCell();

    // Obstacle info
    const Vector3<T> BBox = rb[r]->getConvex()->computeBoundingBox();
    const uint       base = r * maxPerObstacle;

    // Iterate all cells in a grid-stride fashion
    for(uint c = threadIdx.x; c < nCells; c += blockDim.x)
    {
        const uint3 id = cells[0]->computeCellID(c);
        Vector3<T>  cellCenter(minCorner[0] + (T(id.x) + T(0.5)) * cellSize,
                              minCorner[1] + (T(id.y) + T(0.5)) * cellSize,
                              minCorner[2] + (T(id.z) + T(0.5)) * cellSize);

        if(intersectOrientedBoundingBox(cellBBox,
                                        BBox,
                                        positions[r] - cellCenter,
                                        quaternions[r]))
        {
            const uint idx               = atomicAdd(&index, 1u);
            obstacleIDs[base + idx]      = r;
            obstacleCellHash[base + idx] = c;
        }
    }
}
//@}

#endif