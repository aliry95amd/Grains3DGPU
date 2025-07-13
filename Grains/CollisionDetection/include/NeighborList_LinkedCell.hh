#ifndef _NEIGHBORLIST_LINKEDCELL_HH_
#define _NEIGHBORLIST_LINKEDCELL_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "LinkedCell.hh"
#include "LinkedCell_Host.hh"
#include "LinkedCell_SortBased.hh"
#include "NeighborList.hh"
#include "NeighborList_Kernels.hh"
#include "Transform3.hh"

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
    static_assert(M == MemType::HOST || M == MemType::DEVICE,
                  "NeighborList_LinkedCell only supports MemType::HOST or "
                  "MemType::DEVICE");

    using NL = NeighborList<T, M>;
    using NL::m_needsUpdate;
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
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSize size of the cell 
        @param nParticles number of particles */
    NeighborList_LinkedCell(const Vector3<T>& minCorner,
                            const Vector3<T>& maxCorner,
                            const T           cellSize,
                            const uint        nParticles)
    {
        // Initialize the LinkedCell buffer
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            m_LinkedCell = new LinkedCell_Host<T>(minCorner,
                                                  maxCorner,
                                                  cellSize,
                                                  nParticles);
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            m_LinkedCell = new LinkedCell_SortBased<T>(minCorner,
                                                       maxCorner,
                                                       cellSize,
                                                       nParticles);
        }
        // TODO: reserve the max for now, but we can optimize this later
        m_pairList.reserve(nParticles * (nParticles - 1) / 2);
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
        @param transforms array of transformations */
    void updateNeighborList(GrainsMemBuffer<Transform3<T>, M>& transforms) final
    {
        if(!m_needsUpdate)
            return;

        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            auto* LC_host = static_cast<LinkedCell_Host<T>*>(m_LinkedCell);
            LC_host->updateLinkedCells(transforms);
            updateNeighborList_LC_Host(LC_host->getCellParticles(),
                                       LC_host->getCellNeighborsList(),
                                       m_pairList.getData());
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            auto* LC_device
                = static_cast<LinkedCell_SortBased<T>*>(m_LinkedCell);
            LC_device->updateLinkedCells(transforms);
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(m_pairList.getSize(),
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            updateNeighborList_LC_Device<<<numBlocks, numThreads>>>(
                LC_device->getParticleIDs(),
                LC_device->getParticleHashes(),
                LC_device->getCellNeighborsList(),
                LC_device->getCellStartIDs(),
                transforms.getSize(),
                m_pairList.getData());
        }

        m_needsUpdate = true;
    }
    //@}
};

#endif