#ifndef _LINKEDCELLFACTORY_HH_
#define _LINKEDCELLFACTORY_HH_

#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "LinkedCell.hh"
#include "LinkedCell_Atomic.hh"
#include "LinkedCell_AtomicFixed.hh"
#include "LinkedCell_Host.hh"
#include "LinkedCell_SortBased.hh"

// =================================================================================================
/** @brief The class LinkedCellFactory.

    Creates the linked cell structure for the simulation.

    @author A.YAZDANI - 2025 - Construction */
// =================================================================================================
template <typename T, MemType M>
class LinkedCellFactory
{
    static_assert(M == MemType::HOST || M == MemType::DEVICE,
                  "LinkedCellFactory only supports MemType::HOST or MemType::DEVICE");

private:
    /** @name Constructors & Destructor */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor (forbidden) */
    LinkedCellFactory() = default;

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor (forbidden) */
    ~LinkedCellFactory() = default;
    //@}

public:
    /**@name Methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Creates and returns a buffer of LinkedCell objects
        @param rb Rigid body buffer
        @param positions Positions buffer
        @param quaternions Quaternions buffer
        @param linkedCellParameters Linked cell parameters
        @param nObstacles number of obstacles
        @param nParticles number of particles
        @param LC Memory buffer for storing the linked cell object */
    static void create(const GrainsMemBuffer<RigidBody<T>*, M>* rb,
                       const GrainsMemBuffer<Vector3<T>, M>&    positions,
                       const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                       const LinkedCellParameters<T>&           linkedCellParameters,
                       const uint                               nObstacles,
                       const uint                               nParticles,
                       LinkedCell<T, M>*&                       LC)
    {
        auto type = linkedCellParameters.type;
        // Create the linked cell object
        if constexpr(M == MemType::HOST)
        {
            LC = new LinkedCell_Host<T>(rb,
                                        positions,
                                        quaternions,
                                        linkedCellParameters,
                                        nObstacles,
                                        nParticles);
        }
        else if constexpr(M == MemType::DEVICE)
        {
            if(type == LinkedCellType::SORTBASED)
            {
                LC = new LinkedCell_SortBased<T>(rb,
                                                 positions,
                                                 quaternions,
                                                 linkedCellParameters,
                                                 nObstacles,
                                                 nParticles);
            }
            else if(type == LinkedCellType::ATOMIC)
            {
                LC = new LinkedCell_Atomic<T>(rb,
                                              positions,
                                              quaternions,
                                              linkedCellParameters,
                                              nObstacles,
                                              nParticles);
            }
            else if(type == LinkedCellType::ATOMICFIXED)
            {
                // Use maxNumCellsPerObstacle as maxParticlesPerCell proxy for now
                // TODO: Add separate parameter in LinkedCellParameters if needed
                uint maxPerCell = linkedCellParameters.maxNumCellsPerObstacle > 0
                                      ? linkedCellParameters.maxNumCellsPerObstacle
                                      : 64;  // Default fallback
                LC              = new LinkedCell_AtomicFixed<T>(rb,
                                                   positions,
                                                   quaternions,
                                                   linkedCellParameters,
                                                   nObstacles,
                                                   nParticles,
                                                   maxPerCell);
            }
            else
                GAbort("LinkedCell type not supported on device. Aborting "
                       "Grains!");
        }

        // Sanity check to ensure LinkedCell was created
        GAssert(LC != nullptr, "LinkedCell creation failed.");
    }
    //@}
};

#endif
