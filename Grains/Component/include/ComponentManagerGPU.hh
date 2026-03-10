#ifndef _COMPONENTMANAGERGPU_HH_
#define _COMPONENTMANAGERGPU_HH_

#include "ComponentManager.hh"

// =================================================================================================
/** @brief The class ComponentManagerGPU.

    Components in the simulation running on GPU.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
template <typename T>
class ComponentManagerGPU : public ComponentManager<T, MemType::DEVICE>
{
    using CM = ComponentManager<T, MemType::DEVICE>;
    using CM::m_numObstacles;
    using CM::m_numParticles;

    using CM::m_componentId;
    using CM::m_position;
    using CM::m_quaternion;
    using CM::m_rigidBody;
    using CM::m_rigidBodyId;
    using CM::m_torce;
    using CM::m_velocity;

    using CM::m_collisionDetectionModule;

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor */
    ComponentManagerGPU();

    /** @brief Constructor with the number of particles, and obstacles.
        @param rigidBody Pointer to the components rigid body buffer
        @param nObstacles Number of obstacles
        @param nParticles Number of particles */
    ComponentManagerGPU(GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* rigidBody,
                        uint                                             nObstacles,
                        uint                                             nParticles);

    /** @brief Destructor */
    ~ComponentManagerGPU();
    //@}

    /** @name Get methods */
    //@{
    //@}

    /** @name Set methods */
    //@{
    //@}

    /** @name Manager methods */
    //@{
    //@}

    /** @name Methods */
    //@{
    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    void moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI) final;

    /** @brief Performs the second velocity half-kick (KDK Step 3; no-op for single-pass schemes)
        @param TI time integration scheme */
    void advanceVelocity(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI) final;
    //@}
};

#endif