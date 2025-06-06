#ifndef _NEIGHBORLIST_NSQ_HH_
#define _NEIGHBORLIST_NSQ_HH_

#include "GrainsParameters.hh"
#include "NeighborList.hh"
#include "NeighborList_Nsq_Kernels.hh"
#include "Transform3.hh"
#include <cuda_runtime.h>
#include <iostream>

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
public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters 
    @param nPairs number of pairs */
    NeighborList_Nsq(const uint nPairs)
        : NeighborList<T, M>(nPairs)
    {
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~NeighborList_Nsq() override = default;
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Creates the neighbor list 
    @param transforms array of transformations
    @param nParticles number of particles */
    void createNeighborList(const Transform3<T>* transforms,
                            const uint           nParticles) override
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            createNeighborList_Host(transforms,
                                    nParticles,
                                    this->m_pairList.getData());
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            createNeighborList_Device<<<GP::m_numBlocksPerGrid,
                                        GP::m_numThreadsPerBlock>>>(
                transforms,
                nParticles,
                this->m_pairList.getData());
        }
        else
            GAbort("Unsupported memory type for createNeighborList()");
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the neighbor list 
    @param transforms array of transformations
    @param nParticles number of particles */
    void updateNeighborList(const Transform3<T>* transforms,
                            const uint           nParticles) override
    {
        // For O(N^2) algorithm, we don't need to update the list since it
        // remains the same. This is a dummy implementation, but it is needed to
        // satisfy the interface.
        return;
    }

    // -------------------------------------------------------------------------
    /** @brief Returns true if update is needed 
    @param transforms array of transformations
    @param nParticles number of particles */
    bool needsUpdate(const Transform3<T>* transforms,
                     const uint           nParticles) const override
    {
        // For O(N^2) algorithm, we don't need to update the list since it
        // remains the same. This is a dummy implementation, but it is needed to
        // satisfy the interface.
        return false;
    }
    //@}
};

// =============================================================================
/** @name NeighborList_Nsq: External kernels */
//@{
/** @brief Creates the neighbor list on host
@param transforms array of transformations
@param nParticles number of particles
@param pairList array of pairs */
template <typename T>
__HOST__ bool createNeighborList_Host(const Transform3<T>* transforms,
                                      const uint           nParticles,
                                      uint2*               pairList)
{
    for(uint i = 0; i < nParticles; ++i)
        for(uint j = i + 1; j < nParticles; ++j)
            pairList[i + j * (j + 1) / 2] = make_uint2(i, j);
    return true;
};

// -----------------------------------------------------------------------------
/** @brief Creates the neighbor list on device
@param transforms array of transformations
@param nParticles number of particles
@param pairList array of pairs */
template <typename T>
__GLOBAL__ bool createNeighborList_Device(const Transform3<T>* transforms,
                                          const uint           nParticles,
                                          uint2*               pairList)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= nParticles)
        return false;

    for(uint j = tid + 1; j < nParticles; ++j)
        pairList[tid + j * (j + 1) / 2] = make_uint2(tid, j);
    return true;
};

#endif