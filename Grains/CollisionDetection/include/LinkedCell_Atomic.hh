#ifndef _LINKEDCELL_ATOMIC_HH_
#define _LINKEDCELL_ATOMIC_HH_

#include "GrainsMemBuffer.hh"
#include "LinkedCell.hh"
#include "LinkedCell_Kernels.hh"

// =================================================================================================
/** @brief The class LinkedCell_Atomic.

    This class provides functionalities to manage linked cells for collision detection in the
    simulation using an atomic approach. It is a derived class of LinkedCell and implements the
    update of linked cells based on atomic operations. This is designed to work on the device (GPU).
    This should give better time complexity than the sort-based approach.

    @author A.Yazdani - 2025 - Construction */
// =================================================================================================
template <typename T>
class LinkedCell_Atomic : public LinkedCell<T, MemType::DEVICE>
{
    using LC = LinkedCell<T, MemType::DEVICE>;
    using LC::m_cells;
    using LC::m_cellSizeWithoutSkin;
    using LC::m_maxDisplacementSquared;
    using LC::m_neighborCells;
    using LC::m_numCells;
    using LC::m_numIterationsSinceLastUpdate;
    using LC::m_numObstacles;
    using LC::m_numParticles;
    using LC::m_oldPosition;
    using LC::m_positions;
    using LC::m_skinThickness;
    using LC::m_useAdaptiveSkin;

protected:
    /** @name Parameters */
    //@{
    /** \brief Packed uint64 buffer: upper 32 bits = cellID, lower 32 bits = particleID */
    GrainsMemBuffer<uint64_t, MemType::DEVICE> m_packedCellParticleIDs;
    /** \brief Buffer of number of particles per cell */
    GrainsMemBuffer<uint, MemType::DEVICE> m_numParticlesPerCell;
    /** \brief Buffer to store particle IDs for each cell */
    GrainsMemBuffer<uint, MemType::DEVICE> m_particleInCells;
    /** \brief Buffer to store the prefix sums of the number of particles per cell */
    GrainsMemBuffer<uint, MemType::DEVICE> m_numParticlesPrefixSums;
    /** \brief Buffer for atomic counters during particle writing */
    GrainsMemBuffer<uint, MemType::DEVICE> m_cellCounters;
    /** \brief CUDA streams for concurrent kernel execution */
    cudaEvent_t  m_resizeComplete;  // Event to track cell resize completion
    cudaStream_t m_stream0;         // Old position copy
    cudaStream_t m_stream1;         // Neighbor cell generation
    cudaStream_t m_stream2;         // Particle packing
    //@}

public:
    /** @name Constructors */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor */
    LinkedCell_Atomic() = default;

    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param linkedCellParameters Linked cell parameters
        @param nObstacles number of obstacles
        @param nParticles number of particles */
    LinkedCell_Atomic(const GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* rb,
                      const GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                      const GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                      const LinkedCellParameters<T>&                         linkedCellParameters,
                      const uint                                             nObstacles,
                      const uint                                             nParticles)
        : LinkedCell<T, MemType::DEVICE>(
              rb, positions, quaternions, linkedCellParameters, nObstacles, nParticles)
        , m_packedCellParticleIDs(nParticles)
        , m_numParticlesPerCell(m_numCells)
        , m_particleInCells(nParticles)
        , m_numParticlesPrefixSums(m_numCells)
        , m_cellCounters(m_numCells)
    {
        m_numParticlesPerCell.fill(0);
        cudaEventCreate(&m_resizeComplete);
        cudaStreamCreate(&m_stream0);
        cudaStreamCreate(&m_stream1);
        cudaStreamCreate(&m_stream2);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell_Atomic()
    {
        cudaEventDestroy(m_resizeComplete);
        cudaStreamDestroy(m_stream0);
        cudaStreamDestroy(m_stream1);
        cudaStreamDestroy(m_stream2);
    }
    //@}

    /** @name Get methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Gets number of particles per cell */
    const uint* getNumParticlesPerCell() const override
    {
        return m_numParticlesPerCell.getData();
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets particle IDs array */
    const uint* getParticleIDArray() const override
    {
        return m_particleInCells.getData();
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets number of particles prefix sums */
    const uint* getNumParticlesPrefixSums() const override
    {
        return m_numParticlesPrefixSums.getData();
    }
    //@}

    /** @name Methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Prepares linked cell update: handles cell resizing, neighbor recomputation

        Concurrent execution strategy using CUDA streams:

        Timeline (adaptive skin mode):
        ┌─────────────────────────────────────────────────────────────────────────┐
        │ Default Stream: resizeCells_Device → cudaMemcpyAsync (D2H) → [event]    │
        └──────────────────────────────────────┬──────────────────────────────────┘
                                               │ (resize complete event)
                                               │
        ┌──────────────────────────────────────▼──────────────────────────────────┐
        │ Stream 1: [wait event] → memset(neighbors) → generateNeighborCells      │
        └─────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────────┐
        │ Stream 0: cudaMemcpyAsync (old positions, D2D) [independent]            │
        └─────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────────┐
        │ Stream 2: memset(particleCounts) → packCellParticleIDs_Atomic           │
        └─────────────────────────────────────────────────────────────────────────┘

        Dependencies:
        - Stream 1 waits for resize completion (cudaStreamWaitEvent)
        - Stream 2's memset must complete before pack (atomic counting needs zero)
        - Streams 0, 1, 2 run concurrently
        - All streams synchronized at end before returning */
    void prepareLinkedCellUpdate()
    {
        // Compute thread configuration once
        uint numBlocks, numThreads;
        computeOptimalThreadsAndBlocks(m_numParticles,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);

        // Full update - adjust skin thickness and resize cells
        if(m_useAdaptiveSkin)
        {
            m_skinThickness                = this->computeSkinThickness();
            T cellSize                     = m_cellSizeWithoutSkin + m_skinThickness;
            m_maxDisplacementSquared       = T(0);
            m_numIterationsSinceLastUpdate = 0;

            // Resize cells on default stream (blocking but minimal impact)
            uint* d_numCells;
            cudaMalloc(&d_numCells, sizeof(uint));
            resizeCells_Device<<<1, 1>>>(m_cells.getData(), cellSize, d_numCells);
            cudaMemcpyAsync(&m_numCells, d_numCells, sizeof(uint), cudaMemcpyDeviceToHost, 0);
            cudaStreamSynchronize(0);
            cudaFree(d_numCells);
            cudaEventRecord(m_resizeComplete, 0);

            // Resize buffers
            m_neighborCells.reserve(m_numCells * 27);
            m_numParticlesPerCell.reserve(m_numCells);
            m_numParticlesPrefixSums.reserve(m_numCells);
            m_cellCounters.reserve(m_numCells);

            // Copy old positions on stream0 (independent)
            cudaMemcpyAsync(m_oldPosition.getData() + m_numObstacles,
                            m_positions->getData() + m_numObstacles,
                            m_numParticles * sizeof(Vector3<T>),
                            cudaMemcpyDeviceToDevice,
                            m_stream0);
        }

        /* Launch concurrent operations */

        // Stream 1: Generate neighbor cells (waits for resize if needed)
        if(m_useAdaptiveSkin)
            cudaStreamWaitEvent(m_stream1, m_resizeComplete, 0);

        uint numBlocksCells, numThreadsCells;
        computeOptimalThreadsAndBlocks(m_numCells,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocksCells,
                                       numThreadsCells);
        cudaMemsetAsync(m_neighborCells.getData(),
                        0xFF,
                        m_neighborCells.getSize() * sizeof(uint),
                        m_stream1);
        generateNeighborCells_Device<<<numBlocksCells, numThreadsCells, 0, m_stream1>>>(
            m_cells.getData(),
            m_numCells,
            m_neighborCells.getData());

        // Stream 2: Reset particle counts + pack IDs (sequential within stream)
        cudaMemsetAsync(m_numParticlesPerCell.getData(),
                        0,
                        m_numParticlesPerCell.getSize() * sizeof(uint),
                        m_stream2);
        packCellParticleIDs_Atomic_Device<<<numBlocks, numThreads, 0, m_stream2>>>(
            m_cells.getData(),
            m_positions->getData() + m_numObstacles,
            m_numParticles,
            m_numObstacles,
            m_packedCellParticleIDs.getData(),
            m_numParticlesPerCell.getData());

        // Synchronize all streams before returning
        cudaStreamSynchronize(m_stream0);  // Old position copy
        cudaStreamSynchronize(m_stream1);  // Neighbor generation
        cudaStreamSynchronize(m_stream2);  // Particle packing
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Updates the linked cells based on the transformations */
    bool updateLinkedCells() override
    {
        auto& SS = GrainsParameters<T>::m_simulationState;

        if(this->needsUpdate())
        {
            // Complete preparation: resize, pack IDs, generate neighbors, link obstacles
            this->prepareLinkedCellUpdate();
            // Relink obstacles
            this->linkObstacles();
        }
        else
        {
            // Only relink obstacles if they moved
            if(SS.obstaclesMoved)
                this->linkObstacles();
            return false;
        }

        // Prefix sum to find the start index of each cell in the particleIDArray
        thrust::device_ptr<uint> numParticles_ptr(m_numParticlesPerCell.getData());
        thrust::device_ptr<uint> prefixSums_ptr(m_numParticlesPrefixSums.getData());
        thrust::exclusive_scan(numParticles_ptr, numParticles_ptr + m_numCells, prefixSums_ptr);

        // Write the particle IDs into the particleInCells using packed uint64 data
        uint numBlocks, numThreads;
        computeOptimalThreadsAndBlocks(m_numParticles,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);
        writeParticleIDs_Kernel<<<numBlocks, numThreads>>>(m_packedCellParticleIDs.getData(),
                                                           m_numParticlesPrefixSums.getData(),
                                                           m_numParticles,
                                                           m_particleInCells.getData(),
                                                           m_cellCounters.getData());

        return true;
    }
    //@}
};

#endif