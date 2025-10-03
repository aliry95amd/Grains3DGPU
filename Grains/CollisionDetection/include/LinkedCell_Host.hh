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
    using LC::m_cellID;
    using LC::m_cells;
    using LC::m_componentID;
    using LC::m_neighborCells;
    using LC::m_numCells;
    using LC::m_numObstacles;
    using LC::m_numParticles;
    using LC::m_obstaclesBufferSize;

private:
    /** @name Host-specific storage */
    //@{
    /** \brief Vector of lists, one list per cell containing component IDs */
    std::vector<std::list<uint>> m_cellComponents;
    /** \brief Map to store iterators to component positions in cell lists for
        O(1) removal */
    std::unordered_map<uint, std::list<uint>::iterator> m_componentIteratorMap;
    /** \brief Temporary buffer to store old component hashes during updates */
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
        @param nParticles number of particles
        @param nCellsForEachObstacle number of cells for each obstacle */
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
        m_cellComponents.resize(m_numCells);
        // Reserve space in iterator map for efficiency
        m_componentIteratorMap.reserve(nParticles);
        // Initialize old cell IDs buffer
        m_oldCellID.initialize(m_cellID.getSize());
        m_oldCellID.fill(UINT_MAX);

        // Populate initial cell assignments
        populateInitialCells();
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~LinkedCell_Host() = default;
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets all cell component lists */
    const std::vector<std::list<uint>>& getCellComponents() const
    {
        return m_cellComponents;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components in a specific cell
        @param cellID the cell ID */
    const std::list<uint>& getComponentsInCell(uint cellID) const
    {
        return m_cellComponents[cellID];
    }

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Populates initial cell assignments for all components */
    void populateInitialCells()
    {
        for(uint i = 0; i < m_obstaclesBufferSize + m_numParticles; ++i)
        {
            uint componentID = m_componentID[i];
            uint cellID      = m_cellID[i];
            addComponentToCell(componentID, cellID);
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Adds a component to a specific cell
        @param componentID the component ID
        @param cellID the cell ID */
    void addComponentToCell(uint componentID, uint cellID)
    {
        if(cellID == UINT_MAX)
            return;

        // Add component to the cell's list
        m_cellComponents[cellID].push_front(componentID);

        // Store the iterator for O(1) removal later
        m_componentIteratorMap[componentID] = m_cellComponents[cellID].begin();
    }

    // -------------------------------------------------------------------------
    /** @brief Removes a component from its current cell
        @param componentID the component ID
        @param cellID the cell ID */
    void removeComponentFromCurrentCell(uint componentID, uint cellID)
    {
        if(cellID == UINT_MAX)
            return;

        auto iterIt = m_componentIteratorMap.find(componentID);

        if(iterIt != m_componentIteratorMap.end())
        {
            // Remove from the cell's list using the stored iterator
            m_cellComponents[cellID].erase(iterIt->second);

            // Clean up iterator map
            m_componentIteratorMap.erase(iterIt);
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Moves a component from one cell to another
        @param componentID the component ID
        @param oldCellID the old cell ID
        @param newCellID the new cell ID */
    void moveComponentToCell(uint componentID, uint oldCellID, uint newCellID)
    {
        removeComponentFromCurrentCell(componentID, oldCellID);
        addComponentToCell(componentID, newCellID);
    }

    // -------------------------------------------------------------------------
    /** @brief Handles cell grid resize by updating component lists */
    void handleCellResize()
    {
        if(m_cellComponents.size() != m_numCells)
        {
            // Clear all existing assignments
            m_cellComponents.clear();
            m_componentIteratorMap.clear();

            // Resize to new cell count
            m_cellComponents.resize(m_numCells);

            // Repopulate all components
            populateInitialCells();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Updates the linked cells based on component transformations */
    bool updateLinkedCells() override
    {
        // Ensure old buffer has correct size before copying
        m_oldCellID.resize(m_cellID.getSize());

        // Store old cell IDs before updating
        std::memcpy(m_oldCellID.getData(),
                    m_cellID.getData(),
                    m_cellID.getBytes());

        // Update component hashes with new positions
        bool isUpdated = this->updateCellFixed();

        // Handle potential cell grid resize
        handleCellResize();

        // Process components
        for(uint i = 0; i < m_obstaclesBufferSize + m_numParticles; ++i)
        {
            uint comp      = m_componentID[i];
            uint newCellID = m_cellID[i];
            uint oldCellID = m_oldCellID[i];
            if(oldCellID != newCellID)
                moveComponentToCell(comp, oldCellID, newCellID);
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
        const uint      cellID             = cells->computeCellHash(candidate);

        // Same-cell particles
        const auto& currentCellParticles = m_cellComponents[cellID];
        for(const uint p : currentCellParticles)
            if(p < maxIndex)
                out.push_back(p);

        // Neighbor cells
        const uint* allNeighbors  = this->getCellNeighborsList();
        const uint* neighborCells = &allNeighbors[NUM_NEIGHBOR_CELLS * cellID];
        for(uint n = 0; n < NUM_NEIGHBOR_CELLS; ++n)
        {
            const uint c = neighborCells[n];
            if(c == UINT_MAX || c == cellID)
                continue;
            const auto& neighborCellParticles = m_cellComponents[c];
            for(const uint p : neighborCellParticles)
                if(p < maxIndex)
                    out.push_back(p);
        }
    }
    //@}
};

#endif