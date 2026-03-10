#ifndef _COMPONENTMANAGERCPU_HH_
#define _COMPONENTMANAGERCPU_HH_

#include "ComponentManager.hh"

// =================================================================================================
/** @brief The class ComponentManagerCPU.

    Components in the simulation running on CPU.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
template <typename T>
class ComponentManagerCPU : public ComponentManager<T, MemType::HOST>
{
    using CM = ComponentManager<T, MemType::HOST>;
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
    ComponentManagerCPU();

    /** @brief Constructor with the number of particles, and obstacles.
        @param rigidBody Pointer to the components rigid body buffer
        @param nObstacles Number of obstacles
        @param nParticles Number of particles */
    ComponentManagerCPU(GrainsMemBuffer<RigidBody<T>*, MemType::HOST>* rigidBody,
                        uint                                           nObstacles,
                        uint                                           nParticles);

    /** @brief Destructor */
    ~ComponentManagerCPU();
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
        @param CF array of all contact force models */
    void moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI) final;

    /** @brief Performs the second velocity half-kick (KDK Step 3; no-op for single-pass schemes)
        @param TI time integration scheme */
    void advanceVelocity(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI) final;
    //@}
};

#endif