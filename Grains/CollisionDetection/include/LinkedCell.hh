#ifndef _LINKEDCELL_HH_
#define _LINKEDCELL_HH_

#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/extrema.h"
#include "thrust/find.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/transform_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/remove.h"
#include "thrust/transform_reduce.h"
#include "thrust/tuple.h"

#include "Cells.hh"
#include "CellsFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "LinkedCell_Kernels.hh"
#include "QuaternionMath.hh"
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
    /** \brief Non-owning pointer to positions buffer */
    const GrainsMemBuffer<Vector3<T>, M>* m_positions = nullptr;
    /** \brief Non-owning pointer to quaternions buffer */
    const GrainsMemBuffer<Quaternion<T>, M>* m_quaternions = nullptr;
    /** \brief Particles position in the last update */
    GrainsMemBuffer<Vector3<T>, M> m_oldPosition;
    /** \brief Cells object. We allocate a buffer for later if we want to work 
        with multiple cells objects */
    GrainsMemBuffer<Cells<T>*, M> m_cells;
    /** \brief Buffer to store neighbor cell IDs */
    GrainsMemBuffer<uint, M> m_neighborCells;
    /** \brief Buffer of particle IDs */
    GrainsMemBuffer<uint, M> m_particleID;
    /** \brief Buffer of cells that particles belong to. This is a one-to-one 
        mapping from particle IDs to cell IDs, i.e., for index i,
        m_particleID[i] is the ID of particle i (p_i), and m_cellID[i] is the ID
        of the cell that particle p_i belongs to. */
    GrainsMemBuffer<uint, M> m_cellID;
    /** \brief Buffer of obstacle IDs and the number of cells that have to be 
        checked for a possible contact with a particle. This is essentially the
        number of cells each obstacle occupies + 1-ring.*/
    GrainsMemBuffer<uint2, M> m_obstacleID;
    /** \brief Buffer of the cell IDs that that have to be checked for a
        possible contact with an obstacle. */
    GrainsMemBuffer<uint, M> m_obstacleCellID;
    /** \brief Cell size. This is the minimum possible size for the cells. skin
        thickness will be added to this value. */
    T m_cellSizeWithoutSkin;
    /** \brief Skin thickness */
    T m_skinThickness;
    /** \brief Maximum displacement of particles since the last update. Note 
        that we store the squared value */
    T m_maxDisplacementSquared;
    /** \brief Number of iterations since the last update */
    uint m_numIterationsSinceLastUpdate;
    /** \brief Number of obstacles */
    uint m_numObstacles;
    /** \brief Maximum number of cells an obstacle can occupy + 1-ring */
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
        : m_oldPosition(nObstacles + nParticles)
        , m_particleID(nParticles)
        , m_cellID(nParticles)
        , m_obstacleID(nObstacles)
        , m_maxDisplacementSquared(0)
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

        // Find the maximum circumscribed radius of particles
        T maxRadiusParticles = computeMaxRadius(nObstacles, m_rb->getSize());

        // The minimum cell size should be at least twice the maximum radius
        // of particles. We multiply this value by a factor (>=1) to get the
        // final cell size.
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
            GrainsMemBuffer<Cells<T>*, MemType::HOST> h_cells(1);
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
        m_neighborCells.initialize(m_numCells * 27); // 26 neighbors + self
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
        m_maxCellsPerObstacle = std::min(m_maxCellsPerObstacle, m_numCells);
        m_obstacleCellID.initialize(m_maxCellsPerObstacle * nObstacles);

        // Force update at the first step
        bool updated = updateCellFixed<true>();
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
    /** @brief Gets neighbor cells */
    const uint* getCellNeighborsList() const
    {
        return m_neighborCells.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particle IDs */
    const uint* getParticleIDs() const
    {
        return m_particleID.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets cell IDs */
    const uint* getCellIDs() const
    {
        return m_cellID.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacle IDs */
    const uint2* getObstacleIDs() const
    {
        return m_obstacleID.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacle cell IDs */
    const uint* getObstacleCellIDs() const
    {
        return m_obstacleCellID.getData();
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
    /** @brief Gets maximum number of cells an obstacle can occupy */
    uint getMaxCellsPerObstacle() const
    {
        return m_maxCellsPerObstacle;
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
    /** @brief Sets the particle ID */
    void setParticleID()
    {
        // Initialize with sequence starting from m_numObstacles
        m_particleID.sequence(m_numObstacles);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the cell ID */
    void setCellID()
    {
        m_cellID.fill(UINT_MAX);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the obstacle ID */
    void setObstacleID()
    {
        if constexpr(M == MemType::HOST)
        {
            for(uint i = 0; i < m_numObstacles; ++i)
            {
                m_obstacleID[i] = make_uint2(i, 0);
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(m_numObstacles,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            fillObstacleID_Device<<<numBlocks, numThreads>>>(
                m_obstacleID.getData(),
                m_numObstacles);
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
        GAssert(endID >= startID,
                "LinkedCell::computeMaxRadius: endID must be >= startID");

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
            GrainsMemBuffer<T, MemType::DEVICE> radii(endID - startID, T(0));

            // Launch kernel to extract radii
            uint numBlocks, numThreads;
            computeOptimalThreadsAndBlocks(endID - startID,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);

            computeMaxRadius_Device<<<numBlocks, numThreads>>>(m_rb->getData(),
                                                               startID,
                                                               endID,
                                                               radii.getData());
            cudaDeviceSynchronize();

            // Use thrust to find maximum
            auto max_it = thrust::max_element(
                thrust::device_pointer_cast(radii.getData()),
                thrust::device_pointer_cast(radii.getData() + radii.getSize()));
            maxRadius = *max_it;
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
    /** @brief Determines if obstacles have moved */
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
            // Define the functor
            struct obstacle_has_moved
            {
                __device__ bool operator()(
                    const thrust::tuple<uint, Vector3<T>, Vector3<T>>& t) const
                {
                    return thrust::get<1>(t) != thrust::get<2>(t);
                }
            };

            auto pos_begin
                = thrust::device_pointer_cast(m_positions->getData());
            auto old_begin
                = thrust::device_pointer_cast(m_oldPosition.getData());
            auto ids_begin = thrust::make_counting_iterator<uint>(0);

            auto zip_begin = thrust::make_zip_iterator(
                thrust::make_tuple(ids_begin, pos_begin, old_begin));
            auto zip_end = zip_begin + m_numObstacles;

            obstacle_has_moved moved_pred;
            auto it = thrust::find_if(zip_begin, zip_end, moved_pred);
            return it != zip_end;
        }

        return false;
    }

    // -------------------------------------------------------------------------
    /** @brief Links obstacles to cells */
    void linkObstacles()
    {
        // If there is no obstacle
        if(m_numObstacles == 0)
            return;

        if constexpr(M == MemType::HOST)
        {
            // Lambda to extract support point from rigid body in given world
            // direction
            auto support
                = [this](uint              obstacleIndex,
                         const Vector3<T>& worldDirection) -> Vector3<T> {
                // Transform world direction to local coordinates using inverse
                // rotation
                const Quaternion<T>& q = m_quaternions->at(obstacleIndex);
                const Vector3<T>     localDirection = q << worldDirection;
                Vector3<T> supPt = (*m_rb)[obstacleIndex]->getConvex()->support(
                    localDirection);
                transform(q, m_positions->at(obstacleIndex), supPt);
                return supPt;
            };

            // Cell info
            const uint4 numCells = m_cells[0]->getNumCellsPerDirection();

            for(uint i = 0; i < m_numObstacles; ++i)
            {
                // offset in the obstacleCellID buffer
                const uint offset = i * m_maxCellsPerObstacle;

                // AABB by querying support in all 6 axis directions
                const Vector3<T> minExt(support(i, Vector3<T>(-1, 0, 0))[X],
                                        support(i, Vector3<T>(0, -1, 0))[Y],
                                        support(i, Vector3<T>(0, 0, -1))[Z]);
                const Vector3<T> maxExt(support(i, Vector3<T>(1, 0, 0))[X],
                                        support(i, Vector3<T>(0, 1, 0))[Y],
                                        support(i, Vector3<T>(0, 0, 1))[Z]);

                // Convert world coordinates to cell coordinates
                const uint3 minCell = m_cells[0]->computeCellID(minExt, false);
                const uint3 maxCell = m_cells[0]->computeCellID(maxExt, false);

                uint cellCount = 0;
                int  minX      = std::max((int)minCell.x - 1, 0);
                int  maxX = std::min((int)maxCell.x + 1, (int)numCells.x - 1);
                int  minY = std::max((int)minCell.y - 1, 0);
                int  maxY = std::min((int)maxCell.y + 1, (int)numCells.y - 1);
                int  minZ = std::max((int)minCell.z - 1, 0);
                int  maxZ = std::min((int)maxCell.z + 1, (int)numCells.z - 1);

                // Nested loops with 1-ring expansion
                for(int x = minX; x <= maxX; ++x)
                {
                    for(int y = minY; y <= maxY; ++y)
                    {
                        for(int z = minZ; z <= maxZ; ++z)
                        {
                            uint cellHash = m_cells[0]->computeCellHash(
                                make_uint3((uint)x, (uint)y, (uint)z));
                            m_obstacleCellID[offset + cellCount] = cellHash;
                            ++cellCount;
                        }
                    }
                }

                // Update the count in obstacleID buffer
                m_obstacleID[i].y = cellCount;
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            // Launch one block per obstacle with one thread per block
            const uint numBlocks  = m_numObstacles;
            const uint numThreads = 1;
            linkObstacles_Device<T>
                <<<numBlocks, numThreads>>>(m_rb->getData(),
                                            m_positions->getData(),
                                            m_quaternions->getData(),
                                            m_cells.getData(),
                                            m_numObstacles,
                                            m_maxCellsPerObstacle,
                                            m_obstacleID.getData(),
                                            m_obstacleCellID.getData());
            cudaDeviceSynchronize();
        }
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
                thrust::device,
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

            for(uint i = 0; i < m_numParticles; ++i)
                m_cellID[i] = m_cells[0]->computeCellHash(p[i]);
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

        // Set the particle IDs
        setParticleID();

        // Set the cell IDs
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
    template <bool forceUpdate = false>
    bool updateCellFixed()
    {
        if constexpr(forceUpdate)
        {
            // Relink obstacles on the existing grid
            linkObstacles();
            setParticleID();
            setCellID();
        }
        else
        {
            if(haveObstaclesMoved() == true)
            {
                // Relink obstacles on the existing grid
                linkObstacles();
                setParticleID();
                setCellID();
            }
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