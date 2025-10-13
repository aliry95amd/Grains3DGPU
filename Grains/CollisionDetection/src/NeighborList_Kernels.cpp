#include <algorithm>
#include <set>

#include "Transform3.hh"

// -----------------------------------------------------------------------------
// Updates the neighbor list on host using an O(n^2) algorithm
__HOST__ void updateNeighborList_Nsq_Host(const uint nObstacles,
                                          const uint nParticles,
                                          uint2*     pairList)
{
    for(uint i = 0; i < nObstacles; ++i)
        for(uint j = 0; j < nParticles; ++j)
            pairList[nParticles * i + j] = make_uint2(i, nObstacles + j);

    // Offset for p-p interactions.
    uint offset = nObstacles * nParticles;
    for(uint i = 0; i < nParticles; ++i)
        for(uint j = i + 1; j < nParticles; ++j)
            pairList[offset + i + j * (j - 1) / 2]
                = make_uint2(nObstacles + i, nObstacles + j);
}

// -----------------------------------------------------------------------------
// Updates the neighbor list on device using an O(n^2) algorithm
__GLOBAL__ void updateNeighborList_Nsq_Device(const uint nObstacles,
                                              const uint nParticles,
                                              uint2*     pairList)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID < nObstacles)
    {
        // Obstacle to particle pairs
        for(uint j = 0; j < nParticles; ++j)
            pairList[nParticles * tID + j] = make_uint2(tID, nObstacles + j);
    }
    else if(tID < nObstacles + nParticles)
    {
        // offset
        const uint offset = nObstacles * nParticles;
        // adjust tID to start from 0 for particles
        tID -= nObstacles;
        // Particle to obstacle pairs
        for(uint j = tID + 1; j < nParticles; ++j)
            pairList[offset + tID + j * (j - 1) / 2]
                = make_uint2(nObstacles + tID, nObstacles + j);
    }
    else
        return;
}

// -----------------------------------------------------------------------------
// Updates the neighbor list on host using a linked cell approach
__HOST__ void updateNeighborList_LC_Host(
    const uint*                         cellNeighborsList,
    const uint2*                        obstacleIDs,
    const uint*                         obstacleCellIDs,
    const uint*                         particleIDs,
    const uint*                         cellIDs,
    const std::vector<std::list<uint>>& cellParticles,
    const uint                          maxCellsPerObstacle,
    const uint                          numObstacles,
    const uint                          numParticles,
    uint2*                              pairList,
    uint*                               pairCount)
{
    constexpr uint NUM_NEIGHBOR_CELLS = 27; // Number of neighboring cells
    uint           counter            = 0;

    // FIRST PASS: Loop over all obstacles
    for(uint i = 0; i < numObstacles; ++i)
    {
        const uint offset = i * maxCellsPerObstacle;

        const uint obstacleIndex      = obstacleIDs[i].x;
        const uint numCellsToTraverse = obstacleIDs[i].y;

        for(uint c = 0; c < numCellsToTraverse; ++c)
        {
            const uint  cell                = obstacleCellIDs[offset + c];
            const auto& targetCellParticles = cellParticles[cell];

            // Check against all particles in the target cell
            for(uint particleID : targetCellParticles)
            {
                pairList[counter++] = make_uint2(obstacleIndex, particleID);
            }
        }
    }

    // SECOND PASS: Cell-centric approach for particle-particle pairs
    for(uint c = 0; c < cellParticles.size(); ++c)
    {
        if(cellParticles[c].empty())
            continue; // Skip empty cells

        // Get neighbor cells for this cell (includes own cell)
        const uint* neighborCells = &cellNeighborsList[NUM_NEIGHBOR_CELLS * c];

        // Check interactions with neighboring cells
        for(uint cc = 0; cc < NUM_NEIGHBOR_CELLS; ++cc)
        {
            uint targetCell = neighborCells[cc];
            if(targetCell == UINT_MAX || targetCell < c)
                continue; // Skip invalid cells and cells with lower indices

            const auto& neighborCellParticles = cellParticles[targetCell];
            if(neighborCellParticles.empty())
                continue; // Skip empty target cells

            // Process particle pairs between cells
            for(uint primaryParticle : cellParticles[c])
            {
                for(uint otherParticle : neighborCellParticles)
                {
                    // ordering to avoid duplicates
                    if(primaryParticle >= otherParticle)
                        continue;

                    pairList[counter++]
                        = make_uint2(primaryParticle, otherParticle);
                }
            }
        }
    }

    *pairCount = counter;
}

// -----------------------------------------------------------------------------
// Generate obstacle-particle pairs on device
__GLOBAL__ void
    generateObstacleParticlePairs_Device(const uint2* obstacleIDs,
                                         const uint*  obstacleCellIDs,
                                         const uint*  cellStartIDs,
                                         const uint*  particleIDs,
                                         const uint   maxCellsPerObstacle,
                                         const uint   numObstacles,
                                         const uint   numParticles,
                                         const uint   numCells,
                                         uint2*       pairList,
                                         uint*        pairCount)
{
    uint obstacleIdx = blockIdx.x;
    if(obstacleIdx >= numObstacles)
        return;

    // Inintialize pair count to zero by the first thread
    if(obstacleIdx == 0 && threadIdx.x == 0)
        *pairCount = 0;
    __syncthreads();

    const uint offset             = obstacleIdx * maxCellsPerObstacle;
    const uint obstacleIndex      = obstacleIDs[obstacleIdx].x;
    const uint numCellsToTraverse = obstacleIDs[obstacleIdx].y;

    // Each thread handles one cell for this obstacle
    for(uint c = threadIdx.x; c < numCellsToTraverse; c += blockDim.x)
    {
        const uint cell      = obstacleCellIDs[offset + c];
        const uint cellStart = cellStartIDs[cell];

        if(cellStart == UINT_MAX)
            continue; // Empty cell

        // Find cell end
        uint cellEnd;
        uint k = cell;
        do
        {
            ++k;
            cellEnd = (k < numCells) ? cellStartIDs[k] : numParticles;
        } while(cellEnd == UINT_MAX && k < numCells);

        // Add pairs for all particles in this cell
        for(uint p = cellStart; p < cellEnd; ++p)
        {
            uint particleID       = particleIDs[p];
            uint globalIndex      = atomicAdd(pairCount, 1);
            pairList[globalIndex] = make_uint2(obstacleIndex, particleID);
        }
    }
}

// -----------------------------------------------------------------------------
// Updates the neighbor list on device using a linked cell approach
__GLOBAL__ void updateNeighborList_LC_Device(const uint* cellNeighborsList,
                                             const uint* particleIDs,
                                             const uint* cellIDs,
                                             const uint* cellStartIDs,
                                             const uint  numObstacles,
                                             const uint  numParticles,
                                             const uint  numCells,
                                             uint2*      pairList,
                                             uint*       pairCount)
{
    // constexpr variables
    // constexpr uint MAX_PAIRS_PER_PARTICLE = 64; // Maximum pairs per particle
    constexpr uint NUM_NEIGHBOR_CELLS = 27; // Number of neighboring cells

    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numParticles)
        return;

    // initialize pair count to zero by the first thread only if there is no
    // obstacles
    if(tID == 0 && numObstacles == 0)
        *pairCount = 0;
    __syncthreads();

    const uint  i             = particleIDs[tID];
    const uint  cell          = cellIDs[i];
    const uint* neighborCells = &cellNeighborsList[NUM_NEIGHBOR_CELLS * cell];
    uint        c, cellStart, cellEnd, numParticlesInCell, j;

    // Loop over all neighboring cells
    for(uint cID = 0; cID < NUM_NEIGHBOR_CELLS; ++cID)
    {
        // Get the neighboring cell hash
        c = neighborCells[cID];

        // Check if the neighboring cell is valid (not a boundary cell)
        if(c == UINT_MAX || c < cell)
            continue;

        // Get the particle IDs in the cell
        cellStart = cellStartIDs[c];

        // Skip empty cells
        if(cellStart == UINT_MAX)
            continue;

        // Get the end of the cell
        uint k = c;
        do
        {
            ++k;
            cellEnd = cellStartIDs[k];
        } while(cellEnd == UINT_MAX && k < numCells);
        // Last cell case
        if(k == numCells)
            cellEnd = numParticles;

        // Number of particles in the cell
        numParticlesInCell = cellEnd - cellStart;
        for(uint p = 0; p < numParticlesInCell; ++p)
        {
            j = particleIDs[cellStart + p];
            if(i >= j)
                continue; // Avoid duplicates and self-pairs
            // Use atomic operation to get unique index
            uint globalIndex      = atomicAdd(pairCount, 1);
            pairList[globalIndex] = make_uint2(i, j);
        }
    }
}