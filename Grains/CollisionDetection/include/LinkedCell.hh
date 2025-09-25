#ifndef _LINKEDCELL_HH_
#define _LINKEDCELL_HH_

#include "thrust/device_ptr.h"
#include <thrust/find.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
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
    Note that number of obstacles is fixed after construction. This is a hard
    constraint since we allocate buffers based on this number.

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
    /** \brief Non-owning pointer to the rigid bodies buffer (stable address) 
        We assume that this buffer remains valid during the lifetime of this 
        object. */
    const GrainsMemBuffer<RigidBody<T>*, M>* m_rb = nullptr;
    /** \brief Non-owning pointer to positions buffer (obstacles + particles) */
    const GrainsMemBuffer<Vector3<T>, M>* m_positions = nullptr;
    /** \brief Non-owning pointer to quaternions buffer (obstacles + particles) */
    const GrainsMemBuffer<Quaternion<T>, M>* m_quaternions = nullptr;
    /** \brief Particles position in the last update */
    GrainsMemBuffer<Vector3<T>, M> m_oldPosition;
    /** \brief Cells object. We allocate a buffer for later if we want to work 
        with multiple cells objects */
    GrainsMemBuffer<Cells<T>*, M> m_cells;
    /** \brief Buffer to store neighbor cell IDs */
    GrainsMemBuffer<uint, M> m_neighborCells;
    /** \brief Buffer of component IDs */
    GrainsMemBuffer<uint, M> m_componentID;
    /** \brief Buffer of cells that components belong to */
    GrainsMemBuffer<uint, M> m_cellID;
    /** \brief Cell size. This is the minimum possible size for the cells. skin
        thickness will be added to this value. */
    T m_cellSizeWithoutSkin;
    /** \brief Skin thickness */
    T m_skinThickness;
    /** \brief Maximum displacement of particles since the last update (squared) */
    T m_maxDisplacementSquared;
    /** \brief Number of iterations since the last update */
    uint m_numIterationsSinceLastUpdate;
    /** \brief Size of the obstacles buffer */
    uint m_obstaclesBufferSize;
    /** \brief Number of obstacles */
    uint m_numObstacles;
    /** \brief Maximum number of cells an obstacle can occupy */
    uint m_maxCellsPerObstacle;
    /** \brief Number of particles */
    uint m_numParticles;
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
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSizeFactor factor to multiply the minimum cell size
        @param nObstacles number of obstacles
        @param nParticles number of particles
        @param nCellsForEachObstacle number of cells for each obstacle */
    LinkedCell(const GrainsMemBuffer<RigidBody<T>*, M>* rb,
               const GrainsMemBuffer<Vector3<T>, M>&    positions,
               const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
               const Vector3<T>&                        minCorner,
               const Vector3<T>&                        maxCorner,
               const T                                  cellSizeFactor,
               const uint                               nObstacles,
               const uint                               nParticles)
        : m_maxDisplacementSquared(0)
        , m_numIterationsSinceLastUpdate(0)
        , m_numObstacles(nObstacles)
        , m_numParticles(nParticles)
    {
        // Store non-owning pointer to rigid body buffer (must remain valid)
        m_rb          = rb;
        m_positions   = &positions;
        m_quaternions = &quaternions;
        GAssert(positions.getSize() == nObstacles + nParticles
                    && quaternions.getSize() == nObstacles + nParticles,
                "LinkedCell: positions or quaternions size does not match "
                "nObstacles + nParticles");

        // Find the maximum circumscribed radius of particles and obstacles
        T maxRadiusParticles = computeMaxRadius(nObstacles, m_rb->getSize());

        // Adjust the cell based on the obstacles and particles
        m_cellSizeWithoutSkin = T(2) * maxRadiusParticles * cellSizeFactor;

        // Initialize the LinkedCell buffer
        // Note that we start with smallest size possible (largest number of
        // cells) so the buffers are allocated with the largest size possible.
        m_cells.reserve(1);
        if constexpr(M == MemType::HOST)
        {
            CellsFactory<T>::create(minCorner,
                                    maxCorner,
                                    m_cellSizeWithoutSkin,
                                    m_cells,
                                    &m_numCells);
        }
        else if constexpr(M == MemType::DEVICE)
        {
            GrainsMemBuffer<Cells<T>*, MemType::HOST> h_cells;
            CellsFactory<T>::create(minCorner,
                                    maxCorner,
                                    m_cellSizeWithoutSkin,
                                    h_cells,
                                    &m_numCells);
            CellsFactory<T>::copyHostToDevice(h_cells, m_cells);
            // Free the host buffer
            delete h_cells[0];
        }

        // Initialize neighbor cells buffer
        m_neighborCells.reserve(m_numCells * 27); // 26 neighbors + self
        m_neighborCells.fill(UINT_MAX);
        generateNeighborCells();

        // Compute initial skin thickness
        m_skinThickness = 0.1 * m_cellSizeWithoutSkin;

        // Adjust the maximum number of cells per obstacle
        T maxRadiusObstacles        = computeMaxRadius(0, nObstacles);
        T maxCellsPerObstaclePerDim = static_cast<uint>(
            ceil(T(2) * maxRadiusObstacles / m_cellSizeWithoutSkin));
        m_maxCellsPerObstacle = maxCellsPerObstaclePerDim
                                * maxCellsPerObstaclePerDim
                                * maxCellsPerObstaclePerDim;
        m_maxCellsPerObstacle = std::max(m_maxCellsPerObstacle, m_numCells);

        // Force update at the first step
        bool updated = updateCellFixed();
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

        // Set non-owning pointers to nullptr
        m_rb          = nullptr;
        m_positions   = nullptr;
        m_quaternions = nullptr;
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
    /** @brief Gets component IDs */
    const uint* getComponentIDs() const
    {
        return m_componentID.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets cell IDs */
    const uint* getCellIDs() const
    {
        return m_cellID.getData();
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
    //@}

    /** @name Set methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Sets the component ID */
    void setComponentID()
    {
        // Component ID should be as the following:
        // First few elements are for obstacles. linkObstacles method should
        // set them.
        // Elements after m_obstaclesBufferSize are for particles:
        if constexpr(M == MemType::HOST)
        {
            for(uint i = 0; i < m_numParticles; ++i)
                m_componentID[m_obstaclesBufferSize + i] = m_numObstacles + i;
        }
        else if constexpr(M == MemType::DEVICE)
        {
            uint numThreads, numBlocks;
            computeOptimalThreadsAndBlocks(m_numParticles,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            fillComponentID_Device<<<numBlocks, numThreads>>>(
                m_componentID.getData(),
                m_obstaclesBufferSize,
                m_numObstacles,
                m_numParticles);
            cudaDeviceSynchronize();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the cell ID */
    void setCellID()
    {
        // Cell ID should be as the following:
        // First few elements are for obstacles. linkObstacles method should
        // set them.
        // Elements after m_obstaclesBufferSize are for particles:
        if constexpr(M == MemType::HOST)
        {
            for(uint i = 0; i < m_numParticles; ++i)
                m_cellID[m_obstaclesBufferSize + i] = UINT_MAX;
        }
        else if constexpr(M == MemType::DEVICE)
        {
            uint numThreads, numBlocks;
            computeOptimalThreadsAndBlocks(m_numParticles,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            fillCellID_Device<<<numBlocks, numThreads>>>(m_cellID.getData(),
                                                         m_obstaclesBufferSize,
                                                         m_numParticles);
            cudaDeviceSynchronize();
        }
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Computes the maximum radius of rigid bodies given an interval 
        @param startID start ID of the interval
        @param endID end ID of the interval */
    T computeMaxRadius(const uint startID, const uint endID) const
    {
        T maxRadius = T(0), radius = T(0);
        if constexpr(M == MemType::HOST)
        {
            for(uint i = startID; i < endID; ++i)
            {
                radius = m_rb->at(i)->getCircumscribedRadius();
                if(radius > maxRadius)
                    maxRadius = radius;
            }
        }
        if constexpr(M == MemType::DEVICE)
        {
            using Tup = thrust::tuple<T, T>;

            struct max_radius
            {
                __device__ Tup
                    operator()(const RigidBody<T>* const& rb_ptr) const
                {
                    T radius = rb_ptr->getCircumscribedRadius();
                    return thrust::make_tuple(radius, T(1));
                }
            };

            struct tuple_max_reducer
            {
                __device__ Tup operator()(const Tup& a, const Tup& b) const
                {
                    return thrust::make_tuple(
                        thrust::max(thrust::get<0>(a), thrust::get<0>(b)),
                        thrust::get<1>(a) + thrust::get<1>(b));
                }
            };

            auto begin = thrust::device_pointer_cast(m_rb->getData() + startID);
            auto end   = begin + (endID - startID);
            auto result
                = thrust::transform_reduce(begin,
                                           end,
                                           max_radius(),
                                           thrust::make_tuple(T(0), T(0)),
                                           tuple_max_reducer());
            maxRadius = thrust::get<0>(result);
        }

        return (maxRadius);
    }

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
            cudaDeviceSynchronize();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Determines if obstacles have moved
        @param positions new positions of the obstacles */
    bool haveObstaclesMoved() const
    {
        if(m_numObstacles == 0)
            return false;

        if constexpr(M == MemType::HOST)
        {
            for(uint i = 0; i < m_numObstacles; ++i)
            {
                if(m_positions->at(i) != m_oldPosition[i])
                    return true;
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            using Tup = thrust::tuple<uint, Vector3<T>, Vector3<T>>;
            struct has_moved
            {
                __device__ bool operator()(const Tup& t) const
                {
                    return thrust::get<1>(t) != thrust::get<2>(t);
                }
            } moved_pred;

            auto ids_begin = thrust::make_counting_iterator<uint>(0);
            auto pos_begin
                = thrust::device_pointer_cast(m_positions->getData());
            auto old_begin
                = thrust::device_pointer_cast(m_oldPosition.getData());
            auto zip_begin = thrust::make_zip_iterator(
                thrust::make_tuple(ids_begin, pos_begin, old_begin));
            auto zip_end = zip_begin + m_numObstacles;

            auto it = thrust::find_if(zip_begin, zip_end, moved_pred);
            return it != zip_end;
        }

        return false;
    }

    // -------------------------------------------------------------------------
    /** @brief Links obstacles to cells and determines enough buffer size for 
        obstacles */
    void linkObstacles()
    {
        // If there is no obstacle, return 0
        if(m_numObstacles == 0)
        {
            m_obstaclesBufferSize = 0;
            return;
        }

        // Workspace to store the IDs of the obstacles and their cell hashes
        // We allocate the maximum possible size
        uint maxBufferSize = m_maxCellsPerObstacle * m_numObstacles;
        GrainsMemBuffer<uint, M> componentID(maxBufferSize, UINT_MAX);
        GrainsMemBuffer<uint, M> cellID(maxBufferSize, UINT_MAX);

        if constexpr(M == MemType::HOST)
        {
            uint             index        = 0; // insertion index
            const T          cellSize     = m_cells[0]->getCellSize();
            const T          halfCellSize = T(0.5) * cellSize;
            const Vector3<T> cellBBox(halfCellSize, halfCellSize, halfCellSize);
            const Vector3<T>& minCorner = m_cells[0]->getMinCornerLinkedCell();
            for(uint r = 0; r < m_numObstacles; ++r)
            {
                const Vector3<T> BBox
                    = (*m_rb)[r]->getConvex()->computeBoundingBox();
                Vector3<T> cellCenter = minCorner;
                for(int i = 0; i < m_numCells; ++i)
                {
                    const uint3 hash = m_cells[0]->computeCellID(i);
                    cellCenter[0] = minCorner[0] + (hash.x + T(0.5)) * cellSize;
                    cellCenter[1] = minCorner[1] + (hash.y + T(0.5)) * cellSize;
                    cellCenter[2] = minCorner[2] + (hash.z + T(0.5)) * cellSize;
                    // Check intersection
                    // Note that the relative position and quaternion of
                    // the obstacle is used here
                    if(intersectOrientedBoundingBox(cellBBox,
                                                    BBox,
                                                    m_positions->at(r)
                                                        - cellCenter,
                                                    m_quaternions->at(r)))
                    {
                        componentID[index] = r;
                        cellID[index]      = m_cells[0]->computeCellHash(hash);
                        index++;
                    }
                }
            }

            // Set the size of the obstacles buffer
            m_obstaclesBufferSize = index;
        }
        else if constexpr(M == MemType::DEVICE)
        {
            // Launch one block per obstacle
            uint numBlocks = m_numObstacles;
            // single thread in each block will work on multiple cells with
            // 1024 strides
            uint numThreads = 1024;

            linkObstacles_Device<T>
                <<<numBlocks, numThreads>>>(m_rb->getData(),
                                            m_positions->getData(),
                                            m_quaternions->getData(),
                                            m_cells.getData(),
                                            m_numObstacles,
                                            m_numCells,
                                            m_maxCellsPerObstacle,
                                            componentID.getData(),
                                            cellID.getData());
            cudaDeviceSynchronize();

            // Compact in-place: remove entries with ID == UINT_MAX
            using Tup = thrust::tuple<uint, uint>;
            struct is_invalid
            {
                __device__ bool operator()(const Tup& t) const
                {
                    return thrust::get<0>(t) == UINT_MAX;
                }
            } invalid_pred;

            const uint total = m_maxCellsPerObstacle * m_numObstacles;
            auto ids_begin = thrust::device_pointer_cast(componentID.getData());
            auto ids_end   = ids_begin + total;
            auto hash_begin = thrust::device_pointer_cast(cellID.getData());
            auto zip_begin  = thrust::make_zip_iterator(
                thrust::make_tuple(ids_begin, hash_begin));
            auto zip_end = zip_begin + total;
            auto new_end = thrust::remove_if(zip_begin, zip_end, invalid_pred);

            // Set the size of the obstacles buffer
            m_obstaclesBufferSize = static_cast<uint>(new_end - zip_begin);
        }

        // Copy the info to member buffers
        // Resize the buffers (for copying)
        componentID.resize(m_obstaclesBufferSize);
        cellID.resize(m_obstaclesBufferSize);
        // Copy the info to buffers
        m_componentID.copyFrom(componentID);
        m_cellID.copyFrom(cellID);
    }

    // -------------------------------------------------------------------------
    /** @brief Computes the maximum displacement */
    T computeMaxDisplacement() const
    {
        T maxDisplacementSquared = 0;

        if constexpr(M == MemType::HOST)
        {
            for(uint i = 0; i < m_oldPosition.getSize(); ++i)
            {
                T dispSquared = norm2(m_positions->at(i) - m_oldPosition[i]);
                if(dispSquared > maxDisplacementSquared)
                    maxDisplacementSquared = dispSquared;
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
                = thrust::device_pointer_cast(m_positions->getData());
            thrust::device_ptr<const Vector3<T>> pos_end
                = pos_begin + m_positions->getSize();

            maxDisplacementSquared = thrust::transform_reduce(
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

        return maxDisplacementSquared;
    }

    // -------------------------------------------------------------------------
    /** @brief Computes the skin thickness based on the maximum displacement */
    T computeSkinThickness() const
    {
        // Desired number of iterations before next update
        constexpr T desiredNumIterationsToUpdate = 100;
        // Smoothing factor
        constexpr T mu = T(0.4);
        // Max Cap the skin thickness at 20% of the cell size
        constexpr T maxSkinThickness = T(0.2);

        const T newThickness = T(2) * sqrt(m_maxDisplacementSquared)
                               * desiredNumIterationsToUpdate
                               / m_numIterationsSinceLastUpdate;
        T skinThickness = mu * newThickness + (1 - mu) * m_skinThickness;
        if(skinThickness > maxSkinThickness * m_cellSizeWithoutSkin)
            skinThickness = maxSkinThickness * m_cellSizeWithoutSkin;

        return skinThickness;
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the which cell particles belong to */
    void updateCellIDs()
    {
        if constexpr(M == MemType::HOST)
        {
            const Vector3<T>* p = m_positions->getData() + m_numObstacles;
            // No need to update obstacles as they should be
            // Particles
            const uint offset = m_obstaclesBufferSize;
            for(uint i = 0; i < m_numParticles; ++i)
                m_cellID[offset + i] = m_cells[0]->computeCellHash(p[i]);
        }
        else if constexpr(M == MemType::DEVICE)
        {
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(m_numParticles,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            computeHash_Device<<<numBlocks, numThreads>>>(m_cells.getData(),
                                                          m_positions->getData()
                                                              + m_numObstacles,
                                                          m_obstaclesBufferSize,
                                                          m_numParticles,
                                                          m_cellID.getData());
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Determines if an update is required and updates cells if so */
    bool updateCellAdaptive()
    {
        T cellSize = T(0);

        // Increment the number of iterations since last update
        ++m_numIterationsSinceLastUpdate;

        // Compute the maximum displacement
        m_maxDisplacementSquared = computeMaxDisplacement();

        // Condition to check if an update is needed:
        // d_max > skinThickness / 2
        // But we are using d_max^2, so we need to square the skin thickness
        bool needsUpdate = (T(4) * m_maxDisplacementSquared
                            > m_skinThickness * m_skinThickness);

        // If no update is needed, return false
        if(!needsUpdate)
            return false;

        // If we reach here, an update is needed
        // Adjust skin thickness
        m_skinThickness = computeSkinThickness();
        cellSize        = m_cellSizeWithoutSkin + m_skinThickness;

        // Update old positions
        m_oldPosition.copyFrom(*m_positions);

        // Resize the cells
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

        // Since Cell size may have changed, we need to recompute the neighbor
        // cells
        // Note: reserve does not change the size if the capacity is enough
        m_neighborCells.reserve(m_numCells * 27); // 26 neighbors + self
        m_neighborCells.fill(UINT_MAX);
        generateNeighborCells();

        // link obstacles
        linkObstacles();

        // Set the component IDs
        // Note: reserve does not change the size if capacity is enough
        m_componentID.reserve(m_obstaclesBufferSize + m_numParticles);
        setComponentID();

        // Set the cell IDs
        // Note: reserve does not change the size if capacity is enough
        m_cellID.reserve(m_obstaclesBufferSize + m_numParticles);
        setCellID();

        // Reset parameters
        m_maxDisplacementSquared       = T(0);
        m_numIterationsSinceLastUpdate = 0;

        // Finally, update the cell IDs of the particles
        updateCellIDs();

        return true;
    }

    // -------------------------------------------------------------------------
    /** @brief Updates links on the current fixed grid (no resizing/skin) */
    bool updateCellFixed()
    {
        if(haveObstaclesMoved() == true)
        {
            // Relink obstacles on the existing grid
            linkObstacles();

            // Refresh component and cell IDs on current buffers
            m_componentID.reserve(m_obstaclesBufferSize + m_numParticles);
            setComponentID();

            m_cellID.reserve(m_obstaclesBufferSize + m_numParticles);
            setCellID();
        }

        // Update particle cell IDs
        updateCellIDs();

        // We performed an update
        return true;
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells and returns if the LinkedCell has been
        updated */
    virtual bool updateLinkedCells() = 0;
    //@}
};

#endif