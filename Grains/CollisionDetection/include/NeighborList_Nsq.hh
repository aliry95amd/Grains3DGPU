#ifndef _NEIGHBORLIST_NSQ_HH_
#define _NEIGHBORLIST_NSQ_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "NeighborList.hh"
#include "NeighborList_Kernels.hh"
#include "Transform3.hh"

// =============================================================================
/** @brief The class NeighborList_Nsq.

    This is a derived class of NeighborList. It implements the neighbor list
    creation using an O(n^2) algorithm. This is useful for systems with a small
    number of components since we bypass LinkedCell and Bounding Volume and use
    a brute force approach.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T, MemType M>
class NeighborList_Nsq : public NeighborList<T, M>
{
    using NL = NeighborList<T, M>;
    using NL::m_needsUpdate;
    using NL::m_pairList;

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Constructor */
    NeighborList_Nsq() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with number of particles
        @param nParticles number of particles */
    NeighborList_Nsq(const uint nParticles)
    {
        m_pairList.reserve(nParticles * (nParticles - 1) / 2);
        m_needsUpdate = true; // Initially, we need to create the list
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~NeighborList_Nsq() override = default;
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Updates the neighbor list 
    @param transforms memory buffer of transformations */
    void updateNeighborList(GrainsMemBuffer<Transform3<T>, M>& transforms) final
    {
        if(!m_needsUpdate)
            return;

        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            updateNeighborList_Nsq_Host(transforms.getSize(),
                                        m_pairList.getData());
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(m_pairList.getSize(),
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            updateNeighborList_Nsq_Device<<<numBlocks, numThreads>>>(
                transforms.getSize(),
                m_pairList.getData());
        }
        else
            GAbort("Unsupported memory type for "
                   "NeighborList_Nsq::updateNeighborList()");

        m_needsUpdate = false;
    }
    //@}
};

#endif