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
    /** \brief Number of pairs */
    uint m_nPairs;

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor (forbidden except in derived classes) */
    __HOSTDEVICE__
    NeighborList()
        : m_nPairs(0)
    {
    }

    // -------------------------------------------------------------------------
    /** @brief Constructor with number of pairs */
    __HOSTDEVICE__
    NeighborList(const uint nPairs)
        : m_nPairs(nPairs)
    {
        m_pairList.reserve(nPairs);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    __HOSTDEVICE__
    virtual ~NeighborList()
    {
        m_pairList.free();
    }
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets pair list */
    uint2* getList() const
    {
        return m_pairList.getData();
    }

    // -------------------------------------------------------------------------
    /** @brief Gets number of pairs in the list */
    uint getNumberOfPairs() const
    {
        return m_nPairs;
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Creates the neighbor list 
    @param transforms array of transformations
    @param nParticles number of particles */
    virtual void createNeighborList(const Transform3<T>* transforms,
                                    const uint           nParticles)
        = 0;

    // -------------------------------------------------------------------------
    /** @brief Updates the neighbor list 
    @param transforms array of transformations
    @param nParticles number of particles */
    virtual void updateNeighborList(const Transform3<T>* transforms,
                                    const uint           nParticles)
        = 0;
    //@}
};

#endif