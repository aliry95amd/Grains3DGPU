#ifndef _LINKEDCELL_SORTBASED_HH_
#define _LINKEDCELL_SORTBASED_HH_

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"

#include "GrainsMemBuffer.hh"
#include "LinkedCell.hh"
#include "LinkedCell_Kernels.hh"
#include "Misc_Kernels.hh"

// =============================================================================
/** @brief The class LinkedCell_SortBased.

    This class provides functionalities to manage linked cells for collision 
    detection in the simulation using a sort-based approach. It is a derived 
    class of LinkedCell and implements the update of linked cells based 
    on sorting the particle hashes. This is designed to work on the device (GPU).
    This has optimal space complexity, while the time complexity is O(n) for
    computing the particle hashes and O(nk) for sorting the particle IDs using
    a parallel radix sort algorithm, where n is the number of particles and k is
    the number of cells.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T>
class LinkedCell_SortBased : public LinkedCell<T, MemType::DEVICE>
{
    using LC = LinkedCell<T, MemType::DEVICE>;
    using LC::m_cells;
    using LC::m_neighborCells;
    using LC::m_numCells;
    using LC::m_particleHash;
    using LC::m_particleID;

protected:
    /** @name Parameters */
    //@{
    /** \brief Buffer to store start ID for each cell */
    GrainsMemBuffer<uint, MemType::DEVICE> m_cellStartID;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor */
    LinkedCell_SortBased() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSize size of the cell
        @param nParticles number of particles */
    LinkedCell_SortBased(const Vector3<T>& minCorner,
                         const Vector3<T>& maxCorner,
                         const T           cellSize,
                         const uint        nParticles)
        : LinkedCell<T, MemType::DEVICE>(
              minCorner, maxCorner, cellSize, nParticles)
    {
        m_cellStartID.reserve(m_numCells);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell_SortBased() = default;
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets cell start IDs */
    const uint* getCellStartIDs() const
    {
        return m_cellStartID.getData();
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells based on the transformations
    @param transforms buffer of transformations */
    void updateLinkedCells(
        GrainsMemBuffer<Transform3<T>, MemType::DEVICE>& transforms)
    {
        const uint numParticles = transforms.getSize();
        // Update the particle hashes
        this->updateParticlesHash(transforms);

        // Sorting the particle ids according to the cell hash
        thrust::sort_by_key(
            thrust::device_ptr<uint>(m_particleHash.getData()),
            thrust::device_ptr<uint>(m_particleHash.getData() + numParticles),
            thrust::device_ptr<uint>(m_particleID.getData()));

        // Finding the start of each cell
        uint numBlocks, numThreads;
        computeOptimalThreadsAndBlocks(m_numCells,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);
        m_cellStartID.fill();
        computeOptimalThreadsAndBlocks(numParticles,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);
        uint sMemSize = sizeof(uint) * (numThreads + 1);
        computeCellStart_Kernel<<<numBlocks, numThreads, sMemSize>>>(
            m_particleHash.getData(),
            numParticles,
            m_cellStartID.getData());
    }
};

#endif