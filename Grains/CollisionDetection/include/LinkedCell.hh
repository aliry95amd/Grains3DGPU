#ifndef _LINKEDCELL_HH_
#define _LINKEDCELL_HH_

#include "Cells.hh"
#include "CellsFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "Transform3.hh"

// =============================================================================
/** @brief The class LinkedCell.

    This class provides functionalities to to manage linked cells for
    collision detection in the simulation. It is essentially a wrapper around
    the LinkedCell class, providing methods to create and update the neighbor
    list based on the linked cells. This wrapper is designed to work only on 
    host.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T, MemType M>
class LinkedCell
{
    static_assert(M == MemType::HOST || M == MemType::DEVICE,
                  "LinkedCell only supports MemType::HOST or MemType::DEVICE");

protected:
    /** @name Parameters */
    //@{
    /** \brief Cells object. We allocate a buffer for later if we want to work 
        with multiple cells objects */
    GrainsMemBuffer<Cells<T>*, M> m_cells;
    /** \brief Buffer to store particle IDs */
    GrainsMemBuffer<uint, M> m_particleID;
    /** \brief Buffer to store particle hashes (cells they belong to) */
    GrainsMemBuffer<uint, M> m_particleHash;
    /** \brief Buffer to store neighbor cell IDs */
    GrainsMemBuffer<uint, M> m_neighborCells;
    /** \brief Number of cells in the grid */
    uint m_numCells;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor */
    LinkedCell() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSize size of the cell
        @param nParticles number of particles */
    LinkedCell(const Vector3<T>& minCorner,
               const Vector3<T>& maxCorner,
               const T           cellSize,
               const uint        nParticles)
    {
        // Initialize the LinkedCell buffer
        m_cells.reserve(1);
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            CellsFactory<T>::create(m_cells, &m_numCells);
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            GrainsMemBuffer<Cells<T>*, MemType::HOST> h_cells;
            CellsFactory<T>::create(h_cells, &m_numCells);
            CellsFactory<T>::copyHostToDevice(h_cells, m_cells);
            // Free the host buffer
            delete h_cells[0];
        }
        m_particleID.reserve(nParticles);
        m_particleHash.reserve(nParticles);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell()
    {
        // Clean up the Cells object
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            if(m_cells.getSize() > 0 && m_cells[0] != nullptr)
                delete m_cells[0];
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            if(m_cells.getSize() > 0 && m_cells.getData()[0] != nullptr)
                cudaFree(m_cells.getData()[0]);
        }
    }
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets linked cell list */
    const Cells<T>** getLinkedCell() const
    {
        return m_cells.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets neighbor cells */
    uint* getCellNeighborsList()
    {
        // First, ask the Cells object to specify the size required for neighbor
        // cells.
        uint numNeighborCells = 0;
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            numNeighborCells = m_cells[0]->getSizeOfNeighborCells();
            m_neighborCells.reserve(numNeighborCells);
            m_cells[0]->generateNeighborCells(m_neighborCells.getData());
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            m_neighborCells.reserve(14 * m_numCells);
            getCellNeighborsList_Device<<<1, 1>>>(m_cells.getData(),
                                                  m_neighborCells.getData());
        }
        return m_neighborCells.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particle IDs */
    const uint* getParticleIDs() const
    {
        return m_particleID.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particle hashes */
    const uint* getParticleHashes() const
    {
        return m_particleHash.getData();
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Updates the particle hashes
    @param transforms buffer of transformations */
    void updateParticlesHash(GrainsMemBuffer<Transform3<T>, M>& transforms)
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            computeHash_Host(m_cells.getData(),
                             transforms.getData(),
                             transforms.getSize(),
                             m_particleHash.getData());
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(transforms.getSize(),
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            computeHash_Device<<<numBlocks, numThreads>>>(
                m_cells.getData(),
                transforms.getData(),
                transforms.getSize(),
                m_particleHash.getData());
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells
    @param transforms buffer of transformations */
    virtual void
        updateLinkedCells(GrainsMemBuffer<Transform3<T>, M>& transforms)
        = 0;
    //@}
};

// =============================================================================
/** @name LinkedCell: External kernels */
//@{
/** @brief Gets the neighbor cells array
    @param cells pointer to the Cells object
    @param tr transformations */
template <typename T>
__GLOBAL__ void getCellNeighborsList_Device(const Cells<T>* const* cells,
                                            uint* neighborCells)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= 1)
        return;
    cells[0]->generateNeighborCells(neighborCells);
}

// -----------------------------------------------------------------------------
/** @brief Computes the cell hash for a given point
    @param cells pointer to the Cells object
    @param tr transformations
    @param numParticles number of particles
    @param particleHash particle hash */
template <typename T>
void computeHash_Host(const Cells<T>* const* cells,
                      const Transform3<T>*   tr,
                      uint                   numParticles,
                      uint*                  particleHash)
{
    for(uint i = 0; i < numParticles; ++i)
        particleHash[i] = cells[0]->computeCellHash(tr[i].getOrigin());
}

// -----------------------------------------------------------------------------
/** @brief Computes the cell hash for a given point
    @param cells pointer to the Cells object
    @param tr transformations
    @param numParticles number of particles
    @param particleHash particle hash */
template <typename T>
__GLOBAL__ void computeHash_Device(const Cells<T>* const* cells,
                                   const Transform3<T>*   tr,
                                   uint                   numParticles,
                                   uint*                  particleHash)
{
    // TODO: Load cells to shared memory if needed
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numParticles)
        return;

    particleHash[tID] = cells[0]->computeCellHash(tr[tID].getOrigin());
}

#endif