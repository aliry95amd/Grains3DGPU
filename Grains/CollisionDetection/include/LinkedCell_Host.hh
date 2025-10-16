#ifndef _LINKEDCELL_HOST_HH_
#define _LINKEDCELL_HOST_HH_

#include <cstring>
#include <list>
#include <unordered_map>
#include <vector>

#include "Cells.hh"
#include "CellsFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "LinkedCell.hh"
#include "Transform3.hh"

// =============================================================================
/** @brief The class LinkedCell_Host.

    This class provides functionalities to manage linked cells for
    collision detection in the simulation on the host. It uses std::vector
    of std::list for each cell to efficiently manage particle assignments.
    When updating, it checks if particle cell IDs have changed and moves
    particles between cells accordingly.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T>
class LinkedCell_Host : public LinkedCell<T, MemType::HOST>
{
    using LC = LinkedCell<T, MemType::HOST>;
    using LC::m_cellID;
    using LC::m_cells;
    using LC::m_neighborCells;
    using LC::m_numCells;
    using LC::m_numObstacles;
    using LC::m_numParticles;
    using LC::m_particleID;

private:
    /** @name Host-specific storage */
    //@{
    /** \brief Vector of lists, one list per cell containing particle IDs */
    std::vector<std::list<uint>> m_cellParticles;
    /** \brief Map to store iterators to particle positions in cell lists for
        O(1) removal */
    std::unordered_map<uint, std::list<uint>::iterator> m_particleIteratorMap;
    /** \brief Temporary buffer to store old particle hashes during updates */
    GrainsMemBuffer<uint, MemType::HOST> m_oldCellID;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSizeFactor factor to multiply the minimum cell size
        @param nObstacles number of obstacles
        @param nParticles number of particles */
    LinkedCell_Host(
        const GrainsMemBuffer<RigidBody<T>*, MemType::HOST>* rb,
        const GrainsMemBuffer<Vector3<T>, MemType::HOST>&    positions,
        const GrainsMemBuffer<Quaternion<T>, MemType::HOST>& quaternions,
        const Vector3<T>&                                    minCorner,
        const Vector3<T>&                                    maxCorner,
        const T                                              cellSizeFactor,
        const uint                                           nObstacles,
        const uint                                           nParticles)
        : LinkedCell<T, MemType::HOST>(rb,
                                       positions,
                                       quaternions,
                                       minCorner,
                                       maxCorner,
                                       cellSizeFactor,
                                       nObstacles,
                                       nParticles)
    {
        // Initialize vector of lists for each cell
        m_cellParticles.resize(m_numCells);
        // Reserve space in iterator map for efficiency
        m_particleIteratorMap.reserve(nParticles);
        // Initialize old cell IDs buffer with maximum size to accommodate all
        // possible particles
        m_oldCellID.initialize(m_numParticles);
        m_oldCellID.fill(UINT_MAX);

        // Populate initial cell assignments
        populateInitialCells();
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~LinkedCell_Host() = default;
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets all cell particle lists */
    const std::vector<std::list<uint>>& getCellParticles() const
    {
        return m_cellParticles;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles in a specific cell
        @param cellID the cell ID */
    const std::list<uint>& getParticlesInCell(uint cellID) const
    {
        return m_cellParticles[cellID];
    }

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Populates initial cell assignments for all components */
    void populateInitialCells()
    {
        for(uint i = 0; i < m_numParticles; ++i)
        {
            uint particleID = m_particleID[i];
            uint cellID     = m_cellID[i];
            addParticleToCell(particleID, cellID);
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Adds a particle to a specific cell
        @param particleID the particle ID
        @param cellID the cell ID */
    void addParticleToCell(uint particleID, uint cellID)
    {
        if(cellID == UINT_MAX)
            return;

        // Add particle to the cell's list
        m_cellParticles[cellID].push_front(particleID);

        // Store the iterator for O(1) removal later
        m_particleIteratorMap[particleID] = m_cellParticles[cellID].begin();
    }

    // -------------------------------------------------------------------------
    /** @brief Removes a particle from its current cell
        @param particleID the particle ID
        @param cellID the cell ID */
    void removeParticleFromCurrentCell(uint particleID, uint cellID)
    {
        if(cellID == UINT_MAX)
            return;

        auto iterIt = m_particleIteratorMap.find(particleID);

        if(iterIt != m_particleIteratorMap.end())
        {
            // Remove from the cell's list using the stored iterator
            m_cellParticles[cellID].erase(iterIt->second);

            // Clean up iterator map
            m_particleIteratorMap.erase(iterIt);
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Moves a particle from one cell to another
        @param particleID the particle ID
        @param oldCellID the old cell ID
        @param newCellID the new cell ID */
    void moveParticleToCell(uint particleID, uint oldCellID, uint newCellID)
    {
        removeParticleFromCurrentCell(particleID, oldCellID);
        addParticleToCell(particleID, newCellID);
    }

    // -------------------------------------------------------------------------
    /** @brief Handles cell grid resize by updating particle lists */
    void handleCellResize()
    {
        if(m_cellParticles.size() != m_numCells)
        {
            // Clear all existing assignments
            m_cellParticles.clear();
            m_particleIteratorMap.clear();

            // Resize to new cell count
            m_cellParticles.resize(m_numCells);

            // Repopulate all particles
            populateInitialCells();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells based on particle transformations */
    bool updateLinkedCells() override
    {
        // Store old cell IDs before updating
        // Manually copy to preserve m_oldCellID's maximum size (copyFrom would resize it)
        m_oldCellID.copyFrom(m_cellID);

        // Update particle hashes with new positions
        bool isUpdated = this->updateCellFixed();

        // Handle potential cell grid resize
        handleCellResize();

        // Process particles that have changed cells
        for(uint i = 0; i < m_numParticles; ++i)
        {
            uint particleID = m_particleID[i];
            uint newCellID  = m_cellID[i];
            uint oldCellID  = m_oldCellID[i];
            if(oldCellID != newCellID)
                moveParticleToCell(particleID, oldCellID, newCellID);
        }

        return isUpdated;
    }

    // -------------------------------------------------------------------------
    /** @brief Collects particle IDs in the candidate's cell and its neighbors.
        Appends IDs less than maxIndex into out.
        @param candidate candidate world-space position
        @param maxIndex only IDs < maxIndex are considered existing
        @param out output buffer to append IDs */
    void collectPotentialNeighbors(const Vector3<T>&  candidate,
                                   uint               maxIndex,
                                   std::vector<uint>& out) const
    {
        constexpr uint  NUM_NEIGHBOR_CELLS = 27; // Number of neighboring cells
        const Cells<T>* cells              = this->getLinkedCell()[0];
        const uint      candidateCellID    = cells->computeCellHash(candidate);

        // Collect obstacles that might interact with this candidate position
        const uint2* obstacleIDs         = this->getObstacleIDs();
        const uint*  obstacleCellIDs     = this->getObstacleCellIDs();
        const uint   maxCellsPerObstacle = this->getMaxCellsPerObstacle();

        for(uint i = 0; i < m_numObstacles; ++i)
        {
            const uint obstacleIndex      = obstacleIDs[i].x;
            const uint numCellsToTraverse = obstacleIDs[i].y;
            const uint offset             = i * maxCellsPerObstacle;

            // Check if candidate cell intersects with any of obstacle's cells
            for(uint c = 0; c < numCellsToTraverse; ++c)
            {
                const uint obstacleCell = obstacleCellIDs[offset + c];
                if(obstacleCell == candidateCellID)
                {
                    out.push_back(obstacleIndex);
                    break; // Found intersection, no need to check other cells for this obstacle
                }
            }
        }

        // Same-cell particles
        const auto& currentCellParticles = m_cellParticles[candidateCellID];
        for(const uint p : currentCellParticles)
            if(p < maxIndex)
                out.push_back(p);

        // Neighbor cells for particles
        const uint* allNeighbors = this->getCellNeighborsList();
        const uint* neighborCells
            = &allNeighbors[NUM_NEIGHBOR_CELLS * candidateCellID];
        for(uint n = 0; n < NUM_NEIGHBOR_CELLS; ++n)
        {
            const uint c = neighborCells[n];
            if(c == UINT_MAX || c == candidateCellID)
                continue;
            const auto& neighborCellParticles = m_cellParticles[c];
            for(const uint p : neighborCellParticles)
                if(p < maxIndex)
                    out.push_back(p);
        }
    }
    //@}
};

#endif