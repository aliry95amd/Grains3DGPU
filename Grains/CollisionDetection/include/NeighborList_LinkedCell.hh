#ifndef _NEIGHBORLIST_LINKEDCELL_HH_
#define _NEIGHBORLIST_LINKEDCELL_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "LinkedCell.hh"
#include "LinkedCellFactory.hh"
#include "LinkedCell_Host.hh"
#include "NeighborList.hh"
#include "NeighborList_Kernels.hh"

#if defined(__CUDACC__)
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#endif

// =================================================================================================
/** @brief The class NeighborList_LinkedCell.

    This is a derived class of NeighborList. It implements the neighbor list
    creation using an O(n) algorithm. This is useful for systems with large
    number of components.

    @author A.Yazdani - 2025 - Construction */
// =================================================================================================
template <typename T, MemType M>
class NeighborList_LinkedCell : public NeighborList<T, M>
{
    using NL = NeighborList<T, M>;
    using NL::m_needsUpdate;
    using NL::m_pairCount;
    using NL::m_pairList;

protected:
    /** @name Parameters */
    //@{
    /** \brief LinkedCell */
    LinkedCell<T, M>* m_LinkedCell;
    /** \brief Buffer of number of neighbors for each particle */
    GrainsMemBuffer<uint, M> m_numNeighbors;
    /** \brief Buffer of prefix sums for neighbor counts */
    GrainsMemBuffer<uint, M> m_numNeighborsPrefixSums;
    //@}

public:
    /** @name Constructors */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor */
    NeighborList_LinkedCell() = default;

    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param linkedCellParameters Linked cell parameters
        @param nObstacles number of obstacles
        @param nParticles number of particles */
    NeighborList_LinkedCell(const GrainsMemBuffer<RigidBody<T>*, M>* rb,
                            const GrainsMemBuffer<Vector3<T>, M>&    positions,
                            const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                            const LinkedCellParameters<T>&           linkedCellParameters,
                            const uint                               nObstacles,
                            const uint                               nParticles)
    {
        // Create the LinkedCell buffer
        LinkedCellFactory<T, M>::create(rb,
                                        positions,
                                        quaternions,
                                        linkedCellParameters,
                                        nObstacles,
                                        nParticles,
                                        m_LinkedCell);

        // TODO: Reduce init size
        m_pairList.initialize(nObstacles * nParticles + nParticles * (nParticles - 1) / 2);
        m_pairList.fill();

        *m_pairCount = 0;

        if constexpr(M == MemType::DEVICE)
        {
            // Initialize neighbor counting buffers
            m_numNeighbors.initialize(nParticles);
            m_numNeighbors.fill(0);

            m_numNeighborsPrefixSums.initialize(nParticles + 1);  // +1 for total
            m_numNeighborsPrefixSums.fill(0);
        }

        m_needsUpdate = true;  // Initially, we need to create the list
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor */
    ~NeighborList_LinkedCell() override = default;
    //@}

    /** @name Methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Updates the neighbor list
        @param positions array of positions
        @param nObstacles number of obstacles
        @param nParticles number of particles */
    void updateNeighborList(GrainsMemBuffer<Vector3<T>, M>& positions,
                            const uint                      nObstacles,
                            const uint                      nParticles) final
    {
        if(!m_needsUpdate)
            return;

        if constexpr(M == MemType::HOST)
        {
            auto* LC_host    = static_cast<LinkedCell_Host<T>*>(m_LinkedCell);
            bool  LC_updated = LC_host->updateLinkedCells();
            // Check if the linked cell structure was updated.
            // If not, we bypass the neighbor list update.
            if(LC_updated)
            {
                m_pairList.clear();
                updateNeighborList_LC_Host(LC_host->getCellNeighborsList(),
                                           LC_host->getObstacleIDs(),
                                           LC_host->getObstacleCellIDs(),
                                           LC_host->getParticleIDs(),
                                           LC_host->getCellIDs(),
                                           LC_host->getCellParticles(),
                                           LC_host->getMaxCellsPerObstacle(),
                                           nObstacles,
                                           nParticles,
                                           m_pairList.getData(),
                                           m_pairCount);
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            bool LC_updated = m_LinkedCell->updateLinkedCells();

            // Check if the linked cell structure was updated.
            // If not, we bypass the neighbor list update.
            if(LC_updated)
            {
                // Determine the type linked cell used
                using GP = GrainsParameters<T>;
                auto& CD = GP::m_collisionDetection;
                auto& LC = CD.linkedCellParameters;

                // Reset pair count
                *m_pairCount = 0;

                if(nObstacles > 0)
                {
                    if(LC.type == LinkedCellType::ATOMIC)
                    {
                        generateObstacleParticlePairs_AT_Device<<<nObstacles, 64>>>(
                            m_LinkedCell->getObstacleIDs(),
                            m_LinkedCell->getObstacleCellIDs(),
                            m_LinkedCell->getParticleIDArray(),
                            m_LinkedCell->getNumParticlesPerCell(),
                            m_LinkedCell->getNumParticlesPrefixSums(),
                            m_LinkedCell->getMaxCellsPerObstacle(),
                            nObstacles,
                            nParticles,
                            m_LinkedCell->getNumCells(),
                            m_pairList.getData(),
                            m_pairCount);
                    }
                    else if(LC.type == LinkedCellType::SORTBASED)
                    {
                        generateObstacleParticlePairs_SB_Device<<<nObstacles, 64>>>(
                            m_LinkedCell->getObstacleIDs(),
                            m_LinkedCell->getObstacleCellIDs(),
                            m_LinkedCell->getCellStartIDs(),
                            m_LinkedCell->getParticleIDs(),
                            m_LinkedCell->getMaxCellsPerObstacle(),
                            nObstacles,
                            nParticles,
                            m_LinkedCell->getNumCells(),
                            m_pairList.getData(),
                            m_pairCount);
                    }
                }

                // Two-phase atomic-free particle-particle neighbor generation
                uint numBlocks, numThreads;
                computeOptimalThreadsAndBlocks(nParticles,
                                               GrainsParameters<T>::m_GPU,
                                               numBlocks,
                                               numThreads);

                // Phase 1: Count neighbors per particle
                countNeighbors_Device<<<numBlocks, numThreads>>>(
                    m_LinkedCell->getCellNeighborsList(),
                    m_LinkedCell->getParticleIDs(),
                    m_LinkedCell->getCellIDs(),
                    m_LinkedCell->getNumParticlesPerCell(),
                    nParticles,
                    m_numNeighbors.getData());

                // Phase 2: Compute prefix sum (using Thrust)
                thrust::device_ptr<uint> numNeighbors_ptr(m_numNeighbors.getData());
                thrust::device_ptr<uint> prefixSums_ptr(m_numNeighborsPrefixSums.getData());
                thrust::exclusive_scan(numNeighbors_ptr,
                                       numNeighbors_ptr + nParticles,
                                       prefixSums_ptr);

                // Get total pair count using async copy from exclusive scan
                // result
                // Async copy last elements
                // prefix_sum[n-1] + neighbor_count[n-1] = total
                uint lastPrefixSum, lastNeighborCount;
                cudaMemcpyAsync(&lastPrefixSum,
                                &m_numNeighborsPrefixSums.getData()[nParticles - 1],
                                sizeof(uint),
                                cudaMemcpyDeviceToHost);
                cudaMemcpyAsync(&lastNeighborCount,
                                &m_numNeighbors.getData()[nParticles - 1],
                                sizeof(uint),
                                cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();

                uint totalPairs = lastPrefixSum + lastNeighborCount;
                // Add obstacle-particle pairs
                if(nObstacles > 0)
                    totalPairs += *m_pairCount;

                // Increase pair list size if needed
                if(totalPairs > m_pairList.getSize())
                {
                    m_pairList.free();
                    m_pairList.initialize(totalPairs);
                }

                // Phase 3: Write neighbor pairs using prefix sums
                if(LC.type == LinkedCellType::ATOMIC)
                {
                    updateNeighborList_LC_AT_Device<<<numBlocks, numThreads>>>(
                        m_LinkedCell->getCellNeighborsList(),
                        m_LinkedCell->getParticleIDs(),
                        m_LinkedCell->getCellIDs(),
                        m_LinkedCell->getParticleIDArray(),
                        m_LinkedCell->getNumParticlesPerCell(),
                        m_LinkedCell->getNumParticlesPrefixSums(),
                        m_numNeighborsPrefixSums.getData(),
                        nObstacles,
                        nParticles,
                        m_LinkedCell->getNumCells(),
                        m_pairList.getData(),
                        m_pairCount);  // Offset for obstacle pairs
                }
                else if(LC.type == LinkedCellType::SORTBASED)
                {
                    updateNeighborList_LC_SB_Device<<<numBlocks, numThreads>>>(
                        m_LinkedCell->getCellNeighborsList(),
                        m_LinkedCell->getParticleIDs(),
                        m_LinkedCell->getCellIDs(),
                        m_LinkedCell->getCellStartIDs(),
                        m_numNeighborsPrefixSums.getData(),
                        nObstacles,
                        nParticles,
                        m_LinkedCell->getNumCells(),
                        m_pairList.getData(),
                        m_pairCount);  // Offset for obstacle pairs
                }
                cudaDeviceSynchronize();
            }
        }

        m_needsUpdate = true;
    }
    //@}
};

#endif