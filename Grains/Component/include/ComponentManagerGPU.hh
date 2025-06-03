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
    using CM::m_nCells;
    using CM::m_nObstacles;
    using CM::m_nParticles;
    using CM::m_obstacleRB;
    using CM::m_obstacleRigidBodyId;
    using CM::m_obstacleTransform;
    using CM::m_particleId;
    using CM::m_particleRB;
    using CM::m_rigidBodyId;
    using CM::m_torce;
    using CM::m_transform;
    using CM::m_velocity;

protected:
    /** @name Parameters */
    //@{
    /** \brief Particles cell hash */
    GrainsMemBuffer<uint, MemType::DEVICE> m_particleCellHash;
    /** \brief cells hash start */
    GrainsMemBuffer<uint, MemType::DEVICE> m_cellHashStart;
    /** \brief cells hash end */
    GrainsMemBuffer<uint, MemType::DEVICE> m_cellHashEnd;
    //@}

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor */
    ComponentManagerGPU();

    /** @brief Constructor with the number of particles, obstacles, and cells. 
        @param particleRB Pointer to the particles rigid body buffer
        @param obstacleRB Pointer to the obstacles rigid body buffer
        @param nParticles Number of particles
        @param nObstacles Number of obstacles
        @param nCells Number of cells */
    ComponentManagerGPU(
        GrainsMemBuffer<RigidBody<T, T>*, MemType::DEVICE>* particleRB,
        GrainsMemBuffer<RigidBody<T, T>*, MemType::DEVICE>* obstacleRB,
        uint                                                nParticles,
        uint                                                nObstacles,
        uint                                                nCells);

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
    void allocate() final;

    /** @brief Initializes data members to default values */
    void initialize() final;
    //@}

    /** @name Methods */
    //@{
    /** @name Methods */
    //@{
    /** @brief Updates links between particles and linked cell
        @param LC linked cell */
    void updateLinks(
        const GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>& LC) final;

    /** @brief Detects collision between particles and obstacles and 
    computes forces
        @param CF array of all contact force models */
    void detectCollisionAndComputeContactForcesObstacles(
        const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
        final;

    /** @brief Detects collision between particles and particles and
    computes forces
        @param LC linked cell
        @param CF array of all contact force models */
    void detectCollisionAndComputeContactForcesParticles(
        const GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>&        LC,
        const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
        final;

    /** @brief Detects collision between components and computes forces
        @param LC linked cell
        @param CF array of all contact force models */
    void detectCollisionAndComputeContactForces(
        const GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>&        LC,
        const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
        final;

    /** @brief Adds external forces such as gravity */
    void addExternalForces() final;

    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    void moveParticles(
        const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI) final;
    //@}
};

#endif