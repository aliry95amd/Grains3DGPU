#ifndef _NEIGHBORLISTFACTORY_HH_
#define _NEIGHBORLISTFACTORY_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "NeighborList.hh"
#include "NeighborList_LinkedCell.hh"
#include "NeighborList_Nsq.hh"

// =================================================================================================
/** @brief The class NeighborListFactory.

    Creates the neighbor list for the simulation.

    @author A.YAZDANI - 2025 - Construction */
// =================================================================================================
template <typename T, MemType M>
class NeighborListFactory
{
    static_assert(M == MemType::HOST || M == MemType::DEVICE,
                  "NeighborListFactory only supports MemType::HOST or MemType::DEVICE");

private:
    /**@name Contructors & Destructor */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor (forbidden) */
    NeighborListFactory() = default;

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor (forbidden) */
    ~NeighborListFactory() = default;
    //@}

public:
    /**@name Methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Creates and returns a buffer of NeighborList objects
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param CD Collision detection parameters
        @param nObstacles number of obstacles
        @param nParticles number of particles
        @param NL Memory buffer for storing the neighbor list object */
    static void create(const GrainsMemBuffer<RigidBody<T>*, M>* rb,
                       const GrainsMemBuffer<Vector3<T>, M>&    positions,
                       const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                       const CollisionDetectionParameters<T>&   CD,
                       const uint                               nObstacles,
                       const uint                               nParticles,
                       NeighborList<T, M>*&                     NL)
    {
        // Assertions
        GAssert(rb->getSize() == nObstacles + nParticles, "Rigid body size mismatch");
        GAssert(positions.getSize() == nObstacles + nParticles, "Positions size mismatch");
        GAssert(quaternions.getSize() == nObstacles + nParticles, "Quaternions size mismatch");

        // Global parameters
        NeighborListType type = CD.neighborListType;

        if(type == NeighborListType::NSQ)
        {
            NL = new NeighborList_Nsq<T, M>(nObstacles, nParticles);
        }
        else if(type == NeighborListType::LINKEDCELL)
        {
            NL = new NeighborList_LinkedCell<T, M>(rb,
                                                   positions,
                                                   quaternions,
                                                   CD.linkedCellParameters,
                                                   nObstacles,
                                                   nParticles);
        }

        // Sanity check to ensure neighbor list was created
        GAssert(NL != nullptr, "Neighbor list creation failed.");
    }
    //@}
};

#endif
