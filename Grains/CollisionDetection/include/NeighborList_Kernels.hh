#ifndef _NEIGHBORLIST_KERNELS_HH_
#define _NEIGHBORLIST_KERNELS_HH_

#include "Transform3.hh"

// =============================================================================
/** @brief The class NeighborList_Kernels.

    This header file contains the declarations of the various kernels used for
    updating the neighbor list in the simulation. These kernels support both
    host and device.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name NeighborList_Kernels: External Kernels */
//@{
/** @brief Updates the neighbor list on host using an O(n^2) algorithm
    @param nObstacles number of obstacles
    @param nParticles number of particles
    @param pairList array of pairs */
__HOST__ void updateNeighborList_Nsq_Host(const uint nObstacles,
                                          const uint nParticles,
                                          uint2*     pairList);

/** @brief Updates the neighbor list on device using an O(n^2) algorithm
    @param nObstacles number of obstacles
    @param nParticles number of particles
    @param pairList array of pairs */
__GLOBAL__ void updateNeighborList_Nsq_Device(const uint nObstacles,
                                              const uint nParticles,
                                              uint2*     pairList);

/** @brief Updates the neighbor list on host using a linked cell approach
    @param cellParticles vector of lists containing particle IDs for each cell
    @param cellNeighborsList array of neighboring cells for each cell
    @param pairList array of pairs
    @param pairCount number of pairs found */
__HOST__ void updateNeighborList_LC_Host(
    const uint*                         componentID,
    const uint*                         cellID,
    const std::vector<std::list<uint>>& cellComponents,
    const uint*                         cellNeighborsList,
    const uint                          maxObstacleID,
    const uint                          numObstacles,
    const uint                          numParticles,
    uint2*                              pairList,
    uint*                               pairCount);

/** @brief Updates the neighbor list on device using a linked cell approach
    @param particleID array of particle IDs
    @param particleHash array of particle hashes (cells they belong to)
    @param cellNeighborsList array of neighboring cells for each cell
    @param cellStartID array of start IDs for each cell
    @param numParticles number of particles
    @param numCells number of cells
    @param pairList array of pairs
    @param pairCount pointer to device memory for storing the total pair count */
__GLOBAL__ void updateNeighborList_LC_Device(const uint* particleID,
                                             const uint* particleHash,
                                             const uint* cellNeighborsList,
                                             const uint* cellStartID,
                                             const uint  numParticles,
                                             const uint  numCells,
                                             uint2*      pairList,
                                             uint*       pairCount);

#endif