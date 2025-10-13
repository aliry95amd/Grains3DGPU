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

// =============================================================================
/** @brief The class NeighborList_LinkedCell.

    This is a derived class of NeighborList. It implements the neighbor list
    creation using an O(n^2) algorithm. This is useful for systems with a small
    number of components since we bypass LinkedCell and Bounding Volume and use
    a brute force approach.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T, MemType M>
class NeighborList_LinkedCell : public NeighborList<T, M>
{
    using NL = NeighborList<T, M>;
    using NL::m_hPairCount;
    using NL::m_needsUpdate;
    using NL::m_pairCount;
    using NL::m_pairList;

protected:
    /** @name Parameters */
    //@{
    /** \brief LinkedCell */
    LinkedCell<T, M>* m_LinkedCell;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Constructor */
    NeighborList_LinkedCell() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param linkedCellFactor factor to multiply the minimum cell size
        @param nObstacles number of obstacles
        @param nParticles number of particles */
    NeighborList_LinkedCell(
        const GrainsMemBuffer<RigidBody<T>*, M>* rb,
        const GrainsMemBuffer<Vector3<T>, M>&    positions,
        const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
        const Vector3<T>&                        minCorner,
        const Vector3<T>&                        maxCorner,
        const T                                  linkedCellFactor,
        const uint                               nObstacles,
        const uint                               nParticles)
    {
        // Create the LinkedCell buffer
        LinkedCellFactory<T, M>::create(rb,
                                        positions,
                                        quaternions,
                                        minCorner,
                                        maxCorner,
                                        linkedCellFactor,
                                        nObstacles,
                                        nParticles,
                                        m_LinkedCell);

        // TODO: Reduce init size
        m_pairList.initialize(nObstacles * nParticles
                              + nParticles * (nParticles - 1) / 2);
        m_pairList.fill();

        m_pairCount.initialize(1);
        m_pairCount.fill(0);

        m_hPairCount.initialize(1);
        m_hPairCount.fill(0);

        m_needsUpdate = true; // Initially, we need to create the list
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~NeighborList_LinkedCell() override = default;
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
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
                                           m_pairCount.getData());
                // Update the actual size of the pair list
                m_pairList.setSize(m_pairCount[0]);
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            auto* LC_device
                = static_cast<LinkedCell_SortBased<T>*>(m_LinkedCell);
            bool LC_updated = LC_device->updateLinkedCells();
            // Check if the linked cell structure was updated.
            // If not, we bypass the neighbor list update.
            if(LC_updated)
            {
                uint numBlocks, numThreads;
                computeOptimalThreadsAndBlocks(positions.getSize(),
                                               GrainsParameters<T>::m_GPU,
                                               numBlocks,
                                               numThreads);
                updateNeighborList_LC_Device<<<numBlocks, numThreads>>>(
                    LC_device->getCellNeighborsList(),
                    LC_device->getParticleIDs(),
                    LC_device->getCellIDs(),
                    LC_device->getCellStartIDs(),
                    nObstacles,
                    nParticles,
                    LC_device->getNumCells(),
                    m_pairList.getData(),
                    m_pairCount.getData());
                cudaDeviceSynchronize();
                // Copy the actual pair count and update size
                m_pairCount.copyTo(m_hPairCount);
                m_pairList.setSize(m_hPairCount[0]);
            }
        }

        m_needsUpdate = true;
    }

    // -------------------------------------------------------------------------
    /** @brief Collect IDs from the candidate's cell and neighbors. 
        @param positions positions buffer
        @param candidate candidate world-space position to insert
        @param nObstacles number of obstacles
        @param nInserted number of particles already inserted
        @param out output buffer of indices (will be appended) */
    void collectPotentialNeighbors(
        const GrainsMemBuffer<Vector3<T>, M>& positions,
        const Vector3<T>&                     candidate,
        const uint                            nObstacles,
        const uint                            nInserted,
        std::vector<uint>&                    out) final
    {
        if constexpr(M == MemType::HOST)
        {
            auto*      LC_host = static_cast<LinkedCell_Host<T>*>(m_LinkedCell);
            const uint maxIndex = nObstacles + nInserted;
            LC_host->collectPotentialNeighbors(candidate, maxIndex, out);
        }
        else if constexpr(M == MemType::DEVICE)
        {
            GAbort("NeighborList_LinkedCell::collectPotentialNeighbors is not "
                   "implemented for DEVICE");
        }
    }
    //@}
};

#endif