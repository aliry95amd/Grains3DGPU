#ifndef _NEIGHBORLIST_HH_
#define _NEIGHBORLIST_HH_

#include "GrainsMemBuffer.hh"
#include "Transform3.hh"

// =============================================================================
/** @brief The class NeighborList.

    This class provides functionalities to create a neighbor list for components
    in the simulation. It is used to limit the collision detection to only
    neighboring components. It is one of the main differences between the Grains3D 
    and its GPU version as it is useful to avoid thread divergence.
    This is the base class and derived classes should implement the methods.
    This design gives the flexibility to use different types of neighbor lists.
    For instance, the neighbor list can be created using an O(n^2) algorithm for
    systems with a small number of components or using a more sophisticated
    algorithm for larger systems.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T, MemType M>
class NeighborList
{
protected:
    /** @name Parameters */
    //@{
    /** \brief Pair list */
    GrainsMemBuffer<uint2, M> m_pairList;
    /** \brief Pair count */
    GrainsMemBuffer<uint, M> m_pairCount;
    /** \brief If neighbor list needs update */
    bool m_needsUpdate;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor (forbidden except in derived classes) */
    NeighborList() = default;

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~NeighborList() = default;
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets pair list */
    const GrainsMemBuffer<uint2, M>& getBuffer() const
    {
        return m_pairList;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets pair list data */
    uint2* getData()
    {
        return m_pairList.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets size of pair list */
    uint getSize() const
    {
        return m_pairList.getSize();
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Updates the neighbor list 
    @param transforms array of transformations */
    virtual void
        updateNeighborList(GrainsMemBuffer<Transform3<T>, M>& transforms)
        = 0;

    // -------------------------------------------------------------------------
    /** @brief Returns true if update is needed 
    @param transforms array of transformations */
    bool needsUpdate() const
    {
        return m_needsUpdate;
    }
    //@}
};

#endif