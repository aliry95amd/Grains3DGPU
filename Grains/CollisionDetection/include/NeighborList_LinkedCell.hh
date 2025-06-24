#ifndef _NEIGHBORLIST_NSQ_HH_
#define _NEIGHBORLIST_NSQ_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "LinkedCell.hh"
#include "NeighborList.hh"
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
    using NL = NeighborList<T, M>;
    using NL::m_pairList;

protected:
    /** @name Parameters */
    //@{
    /** \brief Buffer for LinkedCell. Possible to have multiple LinkedCell
    instances, but for now we use only one */
    GrainsMemBuffer<LinkedCell<T>*, M>* m_LinkedCell;
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
        uint numCells = 0;
        LinkedCellFactory<T>::create(GP::LinkedCellType,
                                     m_LinkedCell,
                                     numCells);
        m_pairList.reserve(nParticles * (nParticles - 1) / 2);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~NeighborList_LinkedCell() override = default;
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
        //     // First - finding the cell hash for each particle
        //     computeLinearLinkedCellHashGPU_kernel<<<numBlocks, numThreads>>>(
        //     LC,
        //     m_transform,
        //     m_nParticles,
        //     m_particleCellHash);

        //     // Second - sorting the particle ids according to the cell hash
        // thrust::sort_by_key(
        //     thrust::device_ptr<uint>(m_particleCellHash),
        //     thrust::device_ptr<uint>(m_particleCellHash + m_nParticles),
        //     thrust::device_ptr<uint>(m_particleId));

        //     sortComponentsAndFindCellStart_kernel<<<numBlocks, numThreads, sMemSize>>>(
        //     m_particleCellHash,
        //     m_nParticles,
        //     m_cellHashStart,
        //     m_cellHashEnd);

        //     for(int pId = 0; pId < m_nParticles; pId++)
        //     {
        //         // Parameters of the primary particle
        //         const uint             particleId = m_particleId[pId];
        //         const uint             cellHash   = m_particleCellHash[pId];
        //         const Transform3<T>&   trA   = m_transform[particleId];
        //         const uint*            neighborsList = (*LC)->getNeighbors(cellHash);
        //         // Loop over all neighboring particles
        //         for(int i = 0; i < 27; ++i)
        //         {
        //             // Get the neighboring cell hash
        //             uint neighborCellHash = neighborsList[i];
        //             // Check if the neighboring cell is valid
        //             if(neighborCellHash == UINT_MAX)
        //                 continue;
        //             // for(auto id : m_cell[neighborCellHash])
        //             // {
        //             //     const uint secondaryId = id;
        //             //     // To skip self-collision
        //             //     if(secondaryId == particleId)
        //             //         continue;
        //             // }
        //             int startId = cellHashStart[neighborCellHash];
        //             int endId   = cellHashEnd[neighborCellHash];
        //             for(int id = startId; id < endId; id++)
        //             {
        //                 const uint secondaryId = m_particleId[id];
        //                 // To skip self-collision
        //                 if(secondaryId == particle
        //         }
        //     }
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
__HOST__ void createNeighborList_Host(const Transform3<T>* transforms,
                                      const uint           nParticles,
                                      uint2*               pairList)
{
    for(uint i = 0; i < nParticles; ++i)
        for(uint j = i + 1; j < nParticles; ++j)
            pairList[i + j * (j - 1) / 2] = make_uint2(i, j);
};

// -----------------------------------------------------------------------------
/** @brief Creates the neighbor list on device
@param transforms array of transformations
@param nParticles number of particles
@param pairList array of pairs */
template <typename T>
__GLOBAL__ void createNeighborList_Device(const Transform3<T>* transforms,
                                          const uint           nParticles,
                                          uint2*               pairList)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nParticles)
        return;

    for(uint j = tID + 1; j < nParticles; ++j)
        pairList[tID + j * (j - 1) / 2] = make_uint2(tID, j);
};

#endif