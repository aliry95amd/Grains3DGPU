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

    using CM::m_activePairs;
    using CM::m_contactInfo;
    using CM::m_contactInfoWorld;
    using CM::m_neighborList;
    using CM::m_relPosition;
    using CM::m_relQuaternion;

private:
    // Persistent buffers to avoid per-call allocations for compaction
    GrainsMemBuffer<uint, MemType::DEVICE> m_prefixScan;
    GrainsMemBuffer<uint, MemType::DEVICE> m_activeIndex;

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
    /** @brief Initializes buffers for pair-dependent data */
    void initialize();

    /** @brief Resizes pair-dependent buffers based on current neighbor list size.
        @param size New size for the pair-dependent buffers */
    void resizePairBuffers(const uint size);
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
    void computeContactForces(
        const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF) final;

    /** @brief Adds external forces such as gravity */
    void addExternalForces() final;

    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    void moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI) final;
    //@}
};

#endif