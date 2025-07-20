#include "Transform3.hh"

// -----------------------------------------------------------------------------
// Updates the neighbor list on host using an O(n^2) algorithm
__HOST__ void updateNeighborList_Nsq_Host(const uint nParticles,
                                          uint2*     pairList)
{
    for(uint i = 0; i < nParticles; ++i)
        for(uint j = i + 1; j < nParticles; ++j)
            pairList[i + j * (j - 1) / 2] = make_uint2(i, j);
}

// -----------------------------------------------------------------------------
// Updates the neighbor list on device using an O(n^2) algorithm
__GLOBAL__ void updateNeighborList_Nsq_Device(const uint nParticles,
                                              uint2*     pairList)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nParticles)
        return;

    for(uint j = tID + 1; j < nParticles; ++j)
        pairList[tID + j * (j - 1) / 2] = make_uint2(tID, j);
}

// -----------------------------------------------------------------------------
// Updates the neighbor list on host using a linked cell approach
__HOST__ void updateNeighborList_LC_Host(
    const std::vector<std::list<uint>>& cellParticles,
    const uint*                         cellNeighborsList,
    uint2*                              pairList,
    uint*                               pairCount)
{
    constexpr uint NUM_NEIGHBOR_CELLS = 27; // Number of neighboring cells
    uint           counter            = 0;
    // Iterate through all cells
    for(uint cellID = 0; cellID < cellParticles.size(); ++cellID)
    {
        const auto& currentCellParticles = cellParticles[cellID];
        // Skip empty cells
        if(currentCellParticles.empty())
            continue;
        // Check particles within the same cell
        for(auto it1 = currentCellParticles.begin();
            it1 != currentCellParticles.end();
            ++it1)
        {
            for(auto it2 = std::next(it1); it2 != currentCellParticles.end();
                ++it2)
            {
                uint particleID1    = *it1;
                uint particleID2    = *it2;
                pairList[counter++] = make_uint2(particleID1, particleID2);
            }
        }
        // Loop over all neighboring cells
        const uint* neighborCells
            = &cellNeighborsList[NUM_NEIGHBOR_CELLS * cellID];
        for(uint nCellID = 0; nCellID < NUM_NEIGHBOR_CELLS; ++nCellID)
        {
            // Get the neighboring cell hash
            uint c = neighborCells[nCellID];
            // Check if the neighboring cell is valid
            if(c == UINT_MAX || c == cellID || c < cellID)
                continue;
            const auto& neighborCellParticles = cellParticles[c];
            // Check all particle pairs between current cell and neighbor cell
            for(uint particleID1 : currentCellParticles)
                for(uint particleID2 : neighborCellParticles)
                    pairList[counter++] = make_uint2(particleID1, particleID2);
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