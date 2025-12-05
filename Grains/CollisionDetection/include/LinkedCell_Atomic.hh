#ifndef _LINKEDCELL_ATOMIC_HH_
#define _LINKEDCELL_ATOMIC_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "LinkedCell.hh"
#include "LinkedCell_Kernels.hh"

// =============================================================================
/** @brief The class LinkedCell_Atomic.

    This class provides functionalities to manage linked cells for collision
    detection in the simulation using an atomic approach. It is a derived
    class of LinkedCell and implements the update of linked cells based
    on atomic operations. This is designed to work on the device (GPU).
    This should give better time complexity than the sort-based approach.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T>
class LinkedCell_Atomic : public LinkedCell<T, MemType::DEVICE>
{
    using LC = LinkedCell<T, MemType::DEVICE>;
    using LC::m_cellID;
    using LC::m_cells;
    using LC::m_neighborCells;
    using LC::m_numCells;
    using LC::m_numParticles;
    using LC::m_numParticlesPerCell;
    using LC::m_particleID;
    using LC::m_useAdaptiveSkin;

protected:
    /** @name Parameters */
    //@{
    /** \brief Buffer to store particle IDs for each cell */
    GrainsMemBuffer<uint, MemType::DEVICE> m_particleInCells;
    /** \brief Buffer to store the prefix sums of the number of particles 
        per cell */
    GrainsMemBuffer<uint, MemType::DEVICE> m_numParticlesPrefixSums;
    /** \brief Buffer for atomic counters during particle writing */
    GrainsMemBuffer<uint, MemType::DEVICE> m_cellCounters;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor */
    LinkedCell_Atomic() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param linkedCellParameters Linked cell parameters
        @param nObstacles number of obstacles
        @param nParticles number of particles */
    LinkedCell_Atomic(
        const GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* rb,
        const GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
        const GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
        const LinkedCellParameters<T>& linkedCellParameters,
        const uint                     nObstacles,
        const uint                     nParticles)
        : LinkedCell<T, MemType::DEVICE>(rb,
                                         positions,
                                         quaternions,
                                         linkedCellParameters,
                                         nObstacles,
                                         nParticles)
        , m_particleInCells(nParticles)
        , m_numParticlesPrefixSums(m_numCells)
        , m_cellCounters(m_numCells)
    {
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell_Atomic() = default;
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets cell start IDs (unsupported for atomic variant) */
    const uint* getCellStartIDs() const override
    {
        GAbort("LinkedCell_Atomic::getCellStartIDs is not supported in atomic "
               "variant");
        return nullptr;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particle IDs array */
    const uint* getParticleIDArray() const override
    {
        return m_particleInCells.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets number of particles prefix sums */
    const uint* getNumParticlesPrefixSums() const override
    {
        return m_numParticlesPrefixSums.getData();
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells based on the transformations */
    bool updateLinkedCells()
    {
        // Update the cells only if needed
        bool updated;
        if(m_useAdaptiveSkin)
            updated = this->updateCellAdaptive();
        else
            updated = this->updateCellFixed();

        if(!updated)
            return false;

        // Prefix sum to find the start index of each cell in the particleIDArray
        thrust::device_ptr<uint> numParticles_ptr(
            m_numParticlesPerCell.getData());
        thrust::device_ptr<uint> prefixSums_ptr(
            m_numParticlesPrefixSums.getData());
        thrust::exclusive_scan(numParticles_ptr,
                               numParticles_ptr + m_numCells,
                               prefixSums_ptr);

        // Write the particle IDs into the particleInCells
        m_cellCounters.fill(0);
        uint numBlocks, numThreads;
        computeOptimalThreadsAndBlocks(m_numParticles,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);
        writeParticleIDs_Kernel<<<numBlocks, numThreads>>>(
            m_particleID.getData(),
            m_cellID.getData(),
            m_numParticlesPrefixSums.getData(),
            m_numParticles,
            m_particleInCells.getData(),
            m_cellCounters.getData());

        return true;
    }
};

#endif