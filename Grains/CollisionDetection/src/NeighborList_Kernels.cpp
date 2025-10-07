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
    const uint*                         componentID,
    const uint*                         cellID,
    const std::vector<std::list<uint>>& cellComponents,
    const uint*                         cellNeighborsList,
    const uint                          maxObstacleID,
    const uint                          numObstacles,
    const uint                          numParticles,
    uint2*                              pairList,
    uint*                               pairCount)
{
    constexpr uint NUM_NEIGHBOR_CELLS = 27; // Number of neighboring cells
    uint           counter            = 0;

    // Use a set to track unique obstacle-particle pairs (to handle multi-cell obstacles)
    std::set<std::pair<uint, uint>> uniqueObstacleParticlePairs;

    // FIRST PASS: Loop over all obstacles
    for(uint obstacleID = 0; obstacleID < maxObstacleID; ++obstacleID)
    {
        const uint obstacleIndex = componentID[obstacleID];
        const uint obstacleCell  = cellID[obstacleID];
        if(obstacleCell == UINT_MAX)
            continue; // Obstacle not in any cell

        // Get neighbor cells for this obstacle (includes own cell)
        const uint* neighborCells
            = &cellNeighborsList[NUM_NEIGHBOR_CELLS * obstacleCell];

        // Check all neighboring cells (including own cell)
        for(uint nCellID = 0; nCellID < NUM_NEIGHBOR_CELLS; ++nCellID)
        {
            uint targetCell = neighborCells[nCellID];
            if(targetCell == UINT_MAX)
                continue;

            const auto& targetCellComponents = cellComponents[targetCell];

            // Check against all components in the target cell
            for(uint otherComponentID : targetCellComponents)
            {
                // Only add obstacle-particle pairs (particles have ID >= numObstacles)
                // Skip self-pairs for own cell
                if(otherComponentID < numObstacles)
                    continue;
                uniqueObstacleParticlePairs.insert(
                    std::make_pair(obstacleIndex, otherComponentID));
            }
        }
    }

    // Add unique obstacle-particle pairs to the output
    for(const auto& pair : uniqueObstacleParticlePairs)
    {
        pairList[counter++] = make_uint2(pair.first, pair.second);
    }

    // SECOND PASS: Cell-centric approach for particle-particle pairs
    for(uint cellID = 0; cellID < cellComponents.size(); ++cellID)
    {
        const auto& cellComps = cellComponents[cellID];
        if(cellComps.empty())
            continue; // Skip empty cells

        // Get neighbor cells for this cell (includes own cell)
        const uint* neighborCells
            = &cellNeighborsList[NUM_NEIGHBOR_CELLS * cellID];

        // Check interactions with neighboring cells
        for(uint nCellID = 0; nCellID < NUM_NEIGHBOR_CELLS; ++nCellID)
        {
            uint targetCell = neighborCells[nCellID];
            if(targetCell == UINT_MAX || targetCell < cellID)
                continue; // Skip invalid cells and cells with lower indices

            const auto& targetCellComps = cellComponents[targetCell];
            if(targetCellComps.empty())
                continue; // Skip empty target cells

            // Process particle pairs between cells
            for(uint comp1 : cellComps)
            {
                // Skip obstacles in second pass (already handled in first pass)
                if(comp1 < numObstacles)
                    continue;

                for(uint comp2 : targetCellComps)
                {
                    // Skip obstacles
                    if(comp2 < numObstacles)
                        continue;

                    // Always use ordering to avoid duplicates
                    if(comp1 >= comp2)
                        continue;

                    pairList[counter++] = make_uint2(comp1, comp2);
                }
            }
        }
    }

    *pairCount = counter;
}

// -----------------------------------------------------------------------------
// Updates the neighbor list on device using a linked cell approach
__GLOBAL__ void updateNeighborList_LC_Device(const uint* componentID,
                                             const uint* cellID,
                                             const uint* cellNeighborsList,
                                             const uint* cellStartID,
                                             const uint  maxObstacleID,
                                             const uint  numObstacles,
                                             const uint  numParticles,
                                             const uint  numCells,
                                             uint2*      pairList,
                                             uint*       pairCount)
{
    constexpr uint NUM_NEIGHBOR_CELLS = 27; // Number of neighboring cells

    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize pair count
    if(tID == 0)
        pairCount[0] = 0;

    __syncthreads();

    // FIRST PASS: Each thread handles one obstacle for obstacle-particle pairs
    if(tID < maxObstacleID)
    {
        const uint obstacleIndex = componentID[tID];
        const uint obstacleCell  = cellID[tID];

        if(obstacleCell != UINT_MAX)
        {
            // Get neighbor cells for this obstacle
            const uint* neighborCells
                = &cellNeighborsList[NUM_NEIGHBOR_CELLS * obstacleCell];

            // Check all neighboring cells
            for(uint nCellID = 0; nCellID < NUM_NEIGHBOR_CELLS; ++nCellID)
            {
                uint targetCell = neighborCells[nCellID];
                if(targetCell == UINT_MAX)
                    continue;

                // Get the range of components in target cell
                uint cellStart = cellStartID[targetCell];
                if(cellStart == UINT_MAX)
                    continue;

                uint cellEnd;
                uint k = targetCell;
                do
                {
                    ++k;
                    cellEnd = (k < numCells) ? cellStartID[k]
                                             : (maxObstacleID + numParticles);
                } while(cellEnd == UINT_MAX && k < numCells);

                // Check all components in this cell
                for(uint compIdx = cellStart; compIdx < cellEnd; ++compIdx)
                {
                    uint otherComponentID = componentID[compIdx];

                    // Only add obstacle-particle pairs (avoid obstacle-obstacle pairs)
                    if(otherComponentID >= numObstacles)
                    {
                        uint globalIndex = atomicAdd(pairCount, 1);
                        pairList[globalIndex]
                            = make_uint2(obstacleIndex, otherComponentID);
                    }
                }
            }
        }
    }

    __syncthreads();

    // SECOND PASS: Each thread handles one cell for particle-particle pairs
    if(tID < numCells)
    {
        uint cellID_current = tID;

        // Get the range of components in current cell
        uint cellStart = cellStartID[cellID_current];
        if(cellStart == UINT_MAX)
            return;

        uint cellEnd;
        uint k = cellID_current;
        do
        {
            ++k;
            cellEnd = (k < numCells) ? cellStartID[k]
                                     : (maxObstacleID + numParticles);
        } while(cellEnd == UINT_MAX && k < numCells);

        // Get neighbor cells
        const uint* neighborCells
            = &cellNeighborsList[NUM_NEIGHBOR_CELLS * cellID_current];

        // Check interactions with neighboring cells
        for(uint nCellID = 0; nCellID < NUM_NEIGHBOR_CELLS; ++nCellID)
        {
            uint targetCell = neighborCells[nCellID];
            if(targetCell == UINT_MAX || targetCell < cellID_current)
                continue;

            // Get target cell range
            uint targetCellStart = cellStartID[targetCell];
            if(targetCellStart == UINT_MAX)
                continue;

            uint targetCellEnd;
            uint kt = targetCell;
            do
            {
                ++kt;
                targetCellEnd = (kt < numCells)
                                    ? cellStartID[kt]
                                    : (maxObstacleID + numParticles);
            } while(targetCellEnd == UINT_MAX && kt < numCells);

            // Process particle pairs between cells
            for(uint comp1Idx = cellStart; comp1Idx < cellEnd; ++comp1Idx)
            {
                uint comp1 = componentID[comp1Idx];

                // Skip obstacles (already handled in first pass)
                if(comp1 < numObstacles)
                    continue;

                for(uint comp2Idx = targetCellStart; comp2Idx < targetCellEnd;
                    ++comp2Idx)
                {
                    uint comp2 = componentID[comp2Idx];

                    // Skip obstacles and ensure ordering
                    if(comp2 < numObstacles || comp1 >= comp2)
                        continue;

                    uint globalIndex      = atomicAdd(pairCount, 1);
                    pairList[globalIndex] = make_uint2(comp1, comp2);
                }
            }
        }
    }
}