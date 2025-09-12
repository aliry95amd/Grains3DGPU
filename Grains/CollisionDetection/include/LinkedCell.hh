#ifndef _LINKEDCELL_HH_
#define _LINKEDCELL_HH_

#include "thrust/device_ptr.h"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform_reduce.h>
#include <thrust/tuple.h>

#include "Cells.hh"
#include "CellsFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "LinkedCell_Kernels.hh"
#include "VectorMath.hh"

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
    /** \brief Particles position in the last update */
    GrainsMemBuffer<Vector3<T>, M> m_oldPosition;
    /** \brief Cells object. We allocate a buffer for later if we want to work 
        with multiple cells objects */
    GrainsMemBuffer<Cells<T>*, M> m_cells;
    /** \brief Buffer to store particle IDs */
    GrainsMemBuffer<uint, M> m_particleID;
    /** \brief Buffer to store particle hashes (cells they belong to) */
    GrainsMemBuffer<uint, M> m_particleHash;
    /** \brief Buffer to store neighbor cell IDs */
    GrainsMemBuffer<uint, M> m_neighborCells;
    /** \brief Cell size. This is the minimum possible size for the cells. skin
        thickness will be added to this value. */
    T m_cellSizeWithoutSkin;
    /** \brief Skin thickness */
    T m_skinThickness;
    /** \brief Maximum displacement of particles since the last update (squared)*/
    T m_maxDisplacementSquared;
    /** \brief Number of iterations since the last update */
    uint m_numIterationsSinceLastUpdate;
    /** \brief Number of cells in the grid */
    uint m_numCells;
    /** \brief Is update needed */
    bool m_needsUpdate;
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
        : m_oldPosition(nParticles)
        , m_particleHash(nParticles, UINT_MAX)
        , m_cellSizeWithoutSkin(cellSize)
        , m_skinThickness(T(0.1) * cellSize)
        , m_maxDisplacementSquared(0)
        , m_numIterationsSinceLastUpdate(0)
        , m_needsUpdate(true)
    {
        // Note that we start with smallest size possible (largest number of cells)
        // so the buffers are allocated with the largest size possible.
        T size = m_cellSizeWithoutSkin;
        // Initialize the LinkedCell buffer
        m_cells.reserve(1);
        if constexpr(M == MemType::HOST)
        {
            CellsFactory<T>::create(minCorner,
                                    maxCorner,
                                    size,
                                    m_cells,
                                    &m_numCells);
        }
        else if constexpr(M == MemType::DEVICE)
        {
            GrainsMemBuffer<Cells<T>*, MemType::HOST> h_cells;
            CellsFactory<T>::create(minCorner,
                                    maxCorner,
                                    size,
                                    h_cells,
                                    &m_numCells);
            CellsFactory<T>::copyHostToDevice(h_cells, m_cells);
            // Free the host buffer
            delete h_cells[0];
        }
        // m_particleID is initialized to 0, 1, 2, 3, ...
        m_particleID.reserve(nParticles);
        m_particleID.sequence();
        // Neighbor cells are initialized to UINT_MAX
        m_neighborCells.reserve(m_numCells * 27); // 26 neighbors + self
        m_neighborCells.fill(UINT_MAX);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell()
    {
        // Clean up the Cells object
        if constexpr(M == MemType::HOST)
        {
            if(m_cells.getSize() > 0 && m_cells[0] != nullptr)
                delete m_cells[0];
        }
        else if constexpr(M == MemType::DEVICE)
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
    Cells<T>* const* getLinkedCell() const
    {
        return m_cells.getData();
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

    // -------------------------------------------------------------------------
    /** @brief Gets neighbor cells */
    const uint* getCellNeighborsList() const
    {
        return m_neighborCells.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets cell size without skin thickness */
    T getCellSizeWithoutSkin() const
    {
        return m_cellSizeWithoutSkin;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets skin thickness */
    T getSkinThickness() const
    {
        return m_skinThickness;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets maximum displacement */
    T getMaxDisplacement() const
    {
        return sqrt(m_maxDisplacementSquared);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets number of iterations since last update */
    uint getNumIterationsSinceLastUpdate() const
    {
        return m_numIterationsSinceLastUpdate;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets number of cells */
    uint getNumCells() const
    {
        return m_numCells;
    }

    // -------------------------------------------------------------------------
    /** @brief Checks if an update is needed */
    bool needsUpdate() const
    {
        return m_needsUpdate;
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Generates neighbor cells */
    void generateNeighborCells()
    {
        if constexpr(M == MemType::HOST)
        {
            m_cells[0]->generateNeighborCells(m_neighborCells.getData());
        }
        else if constexpr(M == MemType::DEVICE)
        {
            // Assign one thread to each cell
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(m_numCells,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            generateNeighborCells_Device<<<numBlocks, numThreads>>>(
                m_cells.getData(),
                m_numCells,
                m_neighborCells.getData());
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Computes the maximum displacement
        @param positions new positions of the particles */
    void computeMaxDisplacement(const GrainsMemBuffer<Vector3<T>, M>& positions)
    {
        ++m_numIterationsSinceLastUpdate;
        if constexpr(M == MemType::HOST)
        {
            m_maxDisplacementSquared = 0;
            for(uint i = 0; i < m_oldPosition.getSize(); ++i)
            {
                T dispSquared = norm2(positions[i] - m_oldPosition[i]);
                if(dispSquared > m_maxDisplacementSquared)
                    m_maxDisplacementSquared = dispSquared;
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            // Convert raw pointers to thrust device pointers
            thrust::device_ptr<const Vector3<T>> old_begin
                = thrust::device_pointer_cast(m_oldPosition.getData());
            thrust::device_ptr<const Vector3<T>> old_end
                = old_begin + m_oldPosition.getSize();
            thrust::device_ptr<const Vector3<T>> pos_begin
                = thrust::device_pointer_cast(positions.getData());
            thrust::device_ptr<const Vector3<T>> pos_end
                = pos_begin + positions.getSize();

            m_maxDisplacementSquared = thrust::transform_reduce(
                thrust::make_zip_iterator(
                    thrust::make_tuple(old_begin, pos_begin)),
                thrust::make_zip_iterator(thrust::make_tuple(old_end, pos_end)),
                [] __device__(
                    thrust::tuple<const Vector3<T>, const Vector3<T>> tup)
                    -> T {
                    return norm2(thrust::get<1>(tup) - thrust::get<0>(tup));
                },
                T(0),
                thrust::maximum<T>());
        }

        // Condition to check if an update is needed:
        // d_max > skinThickness / 2
        // But we are using d_max^2, so we need to square the skin thickness
        m_needsUpdate = (T(4) * m_maxDisplacementSquared
                         > m_skinThickness * m_skinThickness);
        if(m_needsUpdate)
        {
            adjustSkinThickness();
            m_oldPosition.copyFrom(positions); // Update old positions
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Adjusts the skin thickness */
    void adjustSkinThickness()
    {
        // Desired number of iterations before next update
        constexpr T desiredNumIterationsToUpdate = 100;
        const T     newThickness = T(2) * sqrt(m_maxDisplacementSquared)
                               * desiredNumIterationsToUpdate
                               / m_numIterationsSinceLastUpdate;
        constexpr T mu  = T(0.4); // Smoothing factor
        m_skinThickness = mu * newThickness + (1 - mu) * m_skinThickness;
        // Cap the skin thickness at 20% of the cell size
        if(m_skinThickness > T(0.2) * m_cellSizeWithoutSkin)
            m_skinThickness = T(0.2) * m_cellSizeWithoutSkin;
    }

    // -------------------------------------------------------------------------
    /** @brief Updates cells */
    void updateCells()
    {
        T cellSize = m_cellSizeWithoutSkin + m_skinThickness;
        if constexpr(M == MemType::HOST)
        {
            m_cells[0]->resize(cellSize);
            m_numCells = m_cells[0]->getNumCells();
        }
        else if constexpr(M == MemType::DEVICE)
        {
            uint* d_numCells;
            cudaMalloc(&d_numCells, sizeof(uint));
            resizeCells_Device<<<1, 1>>>(m_cells.getData(),
                                         cellSize,
                                         d_numCells);
            cudaMemcpy(&m_numCells,
                       d_numCells,
                       sizeof(uint),
                       cudaMemcpyDeviceToHost);
            cudaFree(d_numCells);
        }
        m_neighborCells.fill(UINT_MAX);
        generateNeighborCells();

        // reset
        m_maxDisplacementSquared       = T(0);
        m_numIterationsSinceLastUpdate = 0;
        m_needsUpdate                  = false;
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the particle hashes
    @param positions buffer of positions */
    void updateParticlesHash(GrainsMemBuffer<Vector3<T>, M>& positions)
    {
        if constexpr(M == MemType::HOST)
        {
            Vector3<T>* p = positions.getData();
            for(uint i = 0; i < positions.getSize(); ++i)
                m_particleHash[i] = m_cells[0]->computeCellHash(p[i]);
        }
        else if constexpr(M == MemType::DEVICE)
        {
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(positions.getSize(),
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            computeHash_Device<<<numBlocks, numThreads>>>(
                m_cells.getData(),
                positions.getData(),
                positions.getSize(),
                m_particleHash.getData());
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells and returns if the LinkedCell has been
        updated
    @param positions buffer of positions */
    virtual bool updateLinkedCells(GrainsMemBuffer<Vector3<T>, M>& positions)
        = 0;
    //@}
};

#endif