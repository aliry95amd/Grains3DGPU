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

    using CM::m_contactInfo;
    using CM::m_contactInfoWorld;
    using CM::m_neighborList;
    using CM::m_relPosition;
    using CM::m_relQuaternion;

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
    /** @brief Updates neighbor list */
    void updateNeighborList() final;

    /** @brief Computes the relative transformations */
    void computeRelativeTransformations() final;

    /** @brief Detects collisions between components */
    // template <GJKType GJKVARIANT = GJKType::JOHNSON, bool GJKACC = false>
    void detectCollisionsComponents() final;

    /** @brief Transforms contact info to world frame and flags active pairs */
    void transformContactInfoToWorld() final;

    /** @brief Detects collision */
    void detectCollisions() final;

    /** @brief Computes contact forces between different components
        @param CF array of all contact force models */
    void
        computeContactForces(const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF) final;

    /** @brief Adds external forces such as gravity */
    void addExternalForces() final;

    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    void moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI) final;

    /** @brief Performs the second velocity half-kick (KDK Step 3; no-op for single-pass schemes)
        @param TI time integration scheme */
    void advanceVelocity(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI) final;
    //@}
};

#endif