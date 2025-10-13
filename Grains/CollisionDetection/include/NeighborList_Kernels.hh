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
    @param cellNeighborsList array of neighboring cells for each cell
    @param obstacleIDs array of obstacle IDs
    @param obstacleCellIDs array of obstacle cell IDs
    @param particleIDs array of particle IDs
    @param cellIDs array of cell IDs
    @param cellParticles vector of lists containing particle IDs for each cell
    @param maxCellsPerObstacle maximum number of cells per obstacle
    @param numObstacles number of obstacles
    @param numParticles number of particles
    @param pairList array of pairs
    @param pairCount pointer to host memory for storing the total pair count */
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
    uint*                               pairCount);

/** @brief Generate obstacle-particle pairs on device
    @param obstacleIDs array of obstacle IDs and cell counts
    @param obstacleCellIDs array of obstacle cell IDs
    @param cellStartIDs array of start IDs for each cell
    @param particleIDs array of particle IDs
    @param maxCellsPerObstacle maximum number of cells per obstacle
    @param numObstacles number of obstacles
    @param numParticles number of particles
    @param numCells number of cells
    @param pairList array of pairs
    @param pairCount pointer to device memory for storing the total pair count */
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
                                         uint*        pairCount);

/** @brief Updates the neighbor list on device using a linked cell approach
    @param cellNeighborsList array of neighboring cells for each cell
    @param particleIDs array of particle IDs
    @param cellIDs array of cell IDs
    @param cellStartIDs array of start IDs for each cell
    @param numObstacles number of obstacles
    @param numParticles number of particles
    @param numCells number of cells
    @param pairList array of pairs
    @param pairCount pointer to device memory for storing the total pair count */
__GLOBAL__ void updateNeighborList_LC_Device(const uint* cellNeighborsList,
                                             const uint* particleIDs,
                                             const uint* cellIDs,
                                             const uint* cellStartIDs,
                                             const uint  numObstacles,
                                             const uint  numParticles,
                                             const uint  numCells,
                                             uint2*      pairList,
                                             uint*       pairCount);

#endif