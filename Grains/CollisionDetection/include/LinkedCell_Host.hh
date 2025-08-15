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
    using LC::m_cells;
    using LC::m_neighborCells;
    using LC::m_numCells;
    using LC::m_particleHash;
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
    GrainsMemBuffer<uint, MemType::HOST> m_oldParticleHashes;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSize size of the cell
        @param nParticles number of particles */
    LinkedCell_Host(const Vector3<T>& minCorner,
                    const Vector3<T>& maxCorner,
                    const T           cellSize,
                    const uint        nParticles)
        : LinkedCell<T, MemType::HOST>(
              minCorner, maxCorner, cellSize, nParticles)
    {
        // Initialize vector of lists for each cell
        m_cellParticles.resize(m_numCells);
        // Reserve space in iterator map for efficiency
        m_particleIteratorMap.reserve(nParticles);
        // Initialize old particle hashes buffer
        m_oldParticleHashes.reserve(nParticles);
        // Initialize old particle hashes to UINT_MAX, maybe faster than fill
        std::fill(m_oldParticleHashes.getData(),
                  m_oldParticleHashes.getData() + nParticles,
                  UINT_MAX);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell_Host() = default;
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
    /** @brief Adds a particle to a specific cell
        @param particleID the particle ID
        @param cellID the cell ID */
    void addParticleToCell(uint particleID, uint cellID)
    {
        // Add particle to the cell's list
        m_cellParticles[cellID].push_front(particleID);

        // Store the iterator for O(1) removal later
        m_particleIteratorMap[particleID] = m_cellParticles[cellID].begin();

        // Update particle hash in base class array directly
        m_particleHash[particleID] = cellID;
    }

    // -------------------------------------------------------------------------
    /** @brief Removes a particle from its current cell
        @param particleID the particle ID */
    void removeParticleFromCurrentCell(uint particleID)
    {
        auto iterIt = m_particleIteratorMap.find(particleID);

        if(iterIt != m_particleIteratorMap.end())
        {
            // Get current cell from base class array directly
            uint cellID = m_particleHash[particleID];

            // Remove from the cell's list using the stored iterator
            m_cellParticles[cellID].erase(iterIt->second);

            // Clean up iterator map
            m_particleIteratorMap.erase(iterIt);

            // Mark particle as not assigned to any cell
            m_particleHash.getData()[particleID] = UINT_MAX;
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Moves a particle from one cell to another
        @param particleID the particle ID
        @param newCellID the new cell ID */
    void moveParticleToCell(uint particleID, uint newCellID)
    {
        removeParticleFromCurrentCell(particleID);
        addParticleToCell(particleID, newCellID);
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells based on particle transformations
        @param positions buffer of positions */
    bool updateLinkedCells(
        GrainsMemBuffer<Vector3<T>, MemType::HOST>& positions) override
    {
        uint numParticles = positions.getSize();

        // Store old particle hashes before updating
        std::memcpy(m_oldParticleHashes.getData(),
                    m_particleHash.getData(),
                    numParticles * sizeof(uint));

        // Update particle hashes with new positions
        this->updateParticlesHash(positions);

        // Process each particle
        for(uint i = 0; i < numParticles; ++i)
        {
            uint newCellID = m_particleHash[i];
            uint oldCellID = m_oldParticleHashes[i];
            // Particle has moved to a different cell
            if(oldCellID != newCellID)
                moveParticleToCell(i, newCellID);
        }

        return true;
    }
    //@}
};

#endif