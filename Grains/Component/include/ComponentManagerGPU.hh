#ifndef _COMPONENTMANAGERGPU_HH_
#define _COMPONENTMANAGERGPU_HH_

#include "ComponentManager.hh"

// =============================================================================
/** @brief The class ComponentManagerGPU.

    Components in the simulation running on GPU.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
template <typename T>
class ComponentManagerGPU : public ComponentManager<T, MemType::DEVICE>
{
    using CM = ComponentManager<T, MemType::DEVICE>;
    using CM::m_nObstacles;
    using CM::m_nParticles;

    using CM::m_obstaclePosition;
    using CM::m_obstacleQuaternion;
    using CM::m_obstacleRB;
    using CM::m_obstacleRigidBodyId;
    using CM::m_obstacleVelocity;

    using CM::m_particleId;
    using CM::m_particleRB;
    using CM::m_position;
    using CM::m_quaternion;
    using CM::m_rigidBodyId;
    using CM::m_torce;
    using CM::m_velocity;

    using CM::m_contactInfo;
    using CM::m_neighborList;
    using CM::m_relTransform;

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor */
    ComponentManagerGPU();

    /** @brief Constructor with the number of particles, and obstacles 
        @param particleRB Pointer to the particles rigid body buffer
        @param obstacleRB Pointer to the obstacles rigid body buffer
        @param nParticles Number of particles
        @param nObstacles Number of obstacles */
    ComponentManagerGPU(
        GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* particleRB,
        GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* obstacleRB,
        uint                                             nParticles,
        uint                                             nObstacles);

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
    /** @brief Allocates memory for the component manager */
    void allocate();

    /** @brief Initializes data members to default values */
    void initialize();
    //@}

    /** @name Methods */
    //@{
    /** @brief Updates neighbor list */
    void updateNeighborList() final;

    /** @brief Computes the relative transformations */
    void computeRelativeTransformations() final;

    /** @brief Detects collisions between particles and obstacles */
    void detectCollisionsObstacles() final;

    /** @brief Detects collisions between particles and particles */
    void detectCollisionsParticles() final;

    /** @brief Detects collision between particles and particles and */
    void detectCollisions() final;

    /** @brief Computes contact forces between different components
        @param CF array of all contact force models */
    void computeContactForces(const GrainsMemBuffer<ContactForceModel<T>*,
                                                    MemType::DEVICE>& CF) final;

    /** @brief Adds external forces such as gravity */
    void addExternalForces() final;

    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    void moveParticles(
        const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI) final;
    //@}
};

#endif