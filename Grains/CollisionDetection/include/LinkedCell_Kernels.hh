#ifndef _LINKEDCELL_KERNELS_HH_
#define _LINKEDCELL_KERNELS_HH_

#include <cooperative_groups.h>

#include "Basic.hh"
#include "Cells.hh"
#include "Quaternion.hh"
#include "QuaternionMath.hh"
#include "RigidBody.hh"

// =============================================================================
/** @brief The class LinkedCell_Kernels.

    This header file contains the declarations of the various kernels used for
    updating the linked cells in the simulation.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name LinkedCell_Kernels: External Kernels */
//@{
/** @brief Extracts radii from rigid bodies into an array
    @param rb array of rigid body pointers
    @param startID starting index in the rigid body array
    @param endID ending index in the rigid body array  
    @param radii output array for storing individual radii */
template <typename T>
__GLOBAL__ void computeMaxRadius_Device(const RigidBody<T>* const* rb,
                                        const uint                 startID,
                                        const uint                 endID,
                                        T*                         radii)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    uint idx = tID + startID;

    if(idx >= endID)
        return;

    // Extract radius and store in output array
    radii[tID] = rb[idx]->getCircumscribedRadius();
}

// -----------------------------------------------------------------------------
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
    @param cellIDs particle hash */
template <typename T>
__GLOBAL__ void computeHash_Device(const Cells<T>* const* cells,
                                   const Vector3<T>*      positions,
                                   uint                   numParticles,
                                   uint*                  cellIDs)
{
    // TODO: Load cells to shared memory if needed
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numParticles)
        return;

    cellIDs[tID] = cells[0]->computeCellHash(positions[tID]);
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
/** @brief Links obstacles to cells using support function and 1-ring expansion
    @param rb pointer to the rigid bodies (obstacles)
    @param positions world-space centers of obstacles
    @param quaternions world-space orientations of obstacles
    @param cells pointer to the Cells object (grid definition)
    @param nObstacles number of obstacles
    @param maxPerObstacle maximum number of cells an obstacle can occupy
    @param obstacleID buffer for obstacle IDs and counts (uint2)
    @param obstacleCellID buffer for cell IDs that obstacles occupy */
template <typename T>
static __GLOBAL__ void linkObstacles_Device(const RigidBody<T>* const* rb,
                                            const Vector3<T>*      positions,
                                            const Quaternion<T>*   quaternions,
                                            const Cells<T>* const* cells,
                                            const uint             nObstacles,
                                            const uint maxPerObstacle,
                                            uint2*     obstacleID,
                                            uint*      obstacleCellID)
{
    const uint obstacleIdx = blockIdx.x;
    if(obstacleIdx >= nObstacles)
        return;

    // Only use one thread per block for simplicity
    if(threadIdx.x != 0)
        return;

    // Lambda-like device function for support computation
    auto support = [&](const Vector3<T>& worldDirection) -> Vector3<T> {
        // Transform world direction to local coordinates using inverse rotation
        const Quaternion<T>& q              = quaternions[obstacleIdx];
        const Vector3<T>     localDirection = q << worldDirection;
        Vector3<T>           supPt
            = rb[obstacleIdx]->getConvex()->support(localDirection);
        transform(q, positions[obstacleIdx], supPt);
        return supPt;
    };

    // Get grid dimensions
    const uint4 numCells = cells[0]->getNumCellsPerDirection();

    // Compute AABB by querying support in all 6 axis directions
    const Vector3<T> minExt(support(Vector3<T>(-1, 0, 0))[0],
                            support(Vector3<T>(0, -1, 0))[1],
                            support(Vector3<T>(0, 0, -1))[2]);
    const Vector3<T> maxExt(support(Vector3<T>(1, 0, 0))[0],
                            support(Vector3<T>(0, 1, 0))[1],
                            support(Vector3<T>(0, 0, 1))[2]);

    // Convert world coordinates to cell coordinates
    const uint3 minCell = cells[0]->computeCellID(minExt, false);
    const uint3 maxCell = cells[0]->computeCellID(maxExt, false);

    int minX = std::max((int)minCell.x - 1, 0);
    int maxX = std::min((int)maxCell.x + 1, (int)numCells.x - 1);
    int minY = std::max((int)minCell.y - 1, 0);
    int maxY = std::min((int)maxCell.y + 1, (int)numCells.y - 1);
    int minZ = std::max((int)minCell.z - 1, 0);
    int maxZ = std::min((int)maxCell.z + 1, (int)numCells.z - 1);

    uint cellCount = 0;
    // Offset in the obstacleCellID buffer
    const uint offset = obstacleIdx * maxPerObstacle;
    // Nested loops with 1-ring expansion
    for(int x = minX; x <= maxX; ++x)
    {
        for(int y = minY; y <= maxY; ++y)
        {
            for(int z = minZ; z <= maxZ; ++z)
            {
                uint cellHash = cells[0]->computeCellHash(
                    make_uint3((uint)x, (uint)y, (uint)z));
                obstacleCellID[offset + cellCount] = cellHash;
                ++cellCount;
            }
        }
    }

    // Update the obstacle buffer
    obstacleID[obstacleIdx].x = obstacleIdx;
    obstacleID[obstacleIdx].y = cellCount;
}
//@}

#endif