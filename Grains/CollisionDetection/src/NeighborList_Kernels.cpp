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
__GLOBAL__ void updateNeighborList_LC_Device(const uint* particleID,
                                             const uint* particleHash,
                                             const uint* cellNeighborsList,
                                             const uint* cellStartID,
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

    // Initialize the pair count
    if(tID == 0)
        pairCount[0] = 0;

    const uint  i             = particleID[tID];
    const uint  cell          = particleHash[i];
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
        cellStart = cellStartID[c];

        // Skip empty cells
        if(cellStart == UINT_MAX)
            continue;

        // Get the end of the cell
        uint k = c;
        do
        {
            ++k;
            cellEnd = cellStartID[k];
        } while(cellEnd == UINT_MAX && k < numCells);
        // Last cell case
        if(k == numCells)
            cellEnd = numParticles;

        // Number of particles in the cell
        numParticlesInCell = cellEnd - cellStart;
        for(uint p = 0; p < numParticlesInCell; ++p)
        {
            j = particleID[cellStart + p];
            if(i >= j)
                continue; // Avoid duplicates and self-pairs
            // Use atomic operation to get unique index
            uint globalIndex      = atomicAdd(pairCount, 1);
            pairList[globalIndex] = make_uint2(i, j);
        }
    }
}