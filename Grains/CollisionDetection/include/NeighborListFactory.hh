#ifndef _NEIGHBORLISTFACTORY_HH_
#define _NEIGHBORLISTFACTORY_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "NeighborList.hh"
#include "NeighborList_LinkedCell.hh"
#include "NeighborList_Nsq.hh"

// =============================================================================
/** @brief The class NeighborListFactory.

	Creates the neighbor list for the simulation.

    @author A.YAZDANI - 2025 - Construction */
// =============================================================================
template <typename T, MemType M>
class NeighborListFactory
{
    static_assert(
        M == MemType::HOST || M == MemType::DEVICE,
        "NeighborListFactory only supports MemType::HOST or MemType::DEVICE");

private:
    /**@name Contructors & Destructor */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor (forbidden) */
    NeighborListFactory() = default;

    // -------------------------------------------------------------------------
    /** @brief Destructor (forbidden) */
    ~NeighborListFactory() = default;
    //@}

public:
    /**@name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Creates and returns a buffer of NeighborList objects
        @param NL Memory buffer for storing the neighbor list object
        @param numParticles Total number of particles in the simulation */
    static void create(NeighborList<T, M>*& NL)
    {
        using GP = GrainsParameters<T>;

        if(GP::m_neighborListType == 0)
        {
            // brute-force neighbor list
            NL = new NeighborList_Nsq<T, M>(GP::m_numParticles);
        }
        else if(GP::m_neighborListType == 1)
        {
            T cellSize = T(2) * GP::m_maxRadius * GP::m_linkedCellSizeFactor;
            // Linked cell neighbor list
            NL = new NeighborList_LinkedCell<T, M>(GP::m_origin,
                                                   GP::m_maxCoordinate,
                                                   cellSize,
                                                   GP::m_numParticles);
        }
        else
            GAbort("Unknown neighbor list type! Aborting Grains!");
    }
    //@}
};

#endif
