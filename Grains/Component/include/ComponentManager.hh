#ifndef _COMPONENTMANAGER_HH_
#define _COMPONENTMANAGER_HH_

#include "CollisionDetectionModule.hh"
#include "ContactForceModel.hh"
#include "ForceModule.hh"
#include "ForceModuleFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "Insertion.hh"
#include "Kinematics.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"
#include "Torce.hh"
#include "Transform3.hh"

#include "ContactInfo.hh"
#include "NeighborList.hh"

// =================================================================================================
/** @brief The class ComponentManager.

    This is just an abstract class to make sure all derived classess follow the same set of methods.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
template <typename T, MemType M = MemType::HOST>
class ComponentManager
{
protected:
    /** @name Parameters */
    //@{
    /** \brief Collision detection module */
    std::unique_ptr<CollisionDetectionModule<T, M>> m_collisionDetectionModule;

    // TODO: What to do with pointers? Better design? unique_ptr?
    /** \brief Pointer to buffer of components rigid bodies */
    const GrainsMemBuffer<RigidBody<T>*, M>* m_rigidBody;

    /** \brief Components rigid body Id */
    GrainsMemBuffer<uint, M> m_rigidBodyId;
    /** \brief Components position */
    GrainsMemBuffer<Vector3<T>, M> m_position;
    /** \brief Components quaternion */
    GrainsMemBuffer<Quaternion<T>, M> m_quaternion;
    /** \brief Components velocities */
    GrainsMemBuffer<Kinematics<T>, M> m_velocity;
    /** \brief Components torce */
    GrainsMemBuffer<Torce<T>, M> m_torce;
    /** \brief Components Id */
    GrainsMemBuffer<uint, M> m_componentId;

    /** \brief Per-pair contact information in world frame */
    GrainsMemBuffer<ContactInfo<T>, M> m_contactInfo;
    /** \brief Pair list (indices of interacting particle pairs, populated by CDModule) */
    GrainsMemBuffer<uint2, M> m_pairList;

    /** \brief Force computation module (owns contact table + GPU intermediate buffers) */
    std::unique_ptr<ForceModule<T, M>> m_forceModule;

    /** \brief Number of obstacles in manager */
    uint m_numObstacles;
    /** \brief Number of particles in manager */
    uint m_numParticles;
    /** \brief Number of active pairs in manager */
    uint m_numPairs;
    //@}

public:
    /** @name Constructors */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor (forbidden except in derived classes) */
    ComponentManager() = default;

    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor with the number of particles, and obstacles
        @param rigidBody Pointer to the components rigid body buffer
        @param nObstacles Number of obstacles
        @param nParticles Number of particles */
    ComponentManager(GrainsMemBuffer<RigidBody<T>*, M>* rigidBody, uint nObstacles, uint nParticles)
        : m_rigidBody(rigidBody)
        , m_rigidBodyId(nParticles + nObstacles)
        , m_position(nParticles + nObstacles)
        , m_quaternion(nParticles + nObstacles)
        , m_velocity(nParticles + nObstacles)
        , m_torce(nParticles + nObstacles)
        , m_componentId(nParticles + nObstacles)
        , m_numObstacles(nObstacles)
        , m_numParticles(nParticles)
        , m_numPairs(0)
    {
        GAssert(m_rigidBody->getSize() == m_numParticles + m_numObstacles,
                "Rigid body size mismatch");
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~ComponentManager() = default;
    //@}

    /** @name Get methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components rigid body Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getRigidBodyId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_rigidBodyId.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components positions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getPosition(GrainsMemBuffer<Vector3<T>, destM>& buffer) const
    {
        m_position.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components quaternions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getQuaternion(GrainsMemBuffer<Quaternion<T>, destM>& buffer) const
    {
        m_quaternion.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components velocities
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getVelocity(GrainsMemBuffer<Kinematics<T>, destM>& buffer) const
    {
        m_velocity.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components torces
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getTorce(GrainsMemBuffer<Torce<T>, destM>& buffer) const
    {
        m_torce.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getComponentId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_componentId.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets contact information in world frame
        @param buffer destination buffer to copy data into */
    template <MemType destM>
    void getContactInfoWorld(GrainsMemBuffer<ContactInfo<T>, destM>& buffer) const
    {
        m_contactInfo.copyTo(buffer);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components rigid body Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getRigidBodyId() const
    {
        static_assert(M == MemType::HOST, "getRigidBodyId() only available for HOST memory");
        return m_rigidBodyId;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components positions */
    const GrainsMemBuffer<Vector3<T>, MemType::HOST>& getPosition() const
    {
        static_assert(M == MemType::HOST, "getPosition() only available for HOST memory");
        return m_position;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components quaternions */
    const GrainsMemBuffer<Quaternion<T>, MemType::HOST>& getQuaternion() const
    {
        static_assert(M == MemType::HOST, "getQuaternion() only available for HOST memory");
        return m_quaternion;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components velocities */
    const GrainsMemBuffer<Kinematics<T>, MemType::HOST>& getVelocity() const
    {
        static_assert(M == MemType::HOST, "getVelocity() only available for HOST memory");
        return m_velocity;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components torces */
    const GrainsMemBuffer<Torce<T>, MemType::HOST>& getTorce() const
    {
        static_assert(M == MemType::HOST, "getTorce() only available for HOST memory");
        return m_torce;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets components Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getComponentId() const
    {
        static_assert(M == MemType::HOST, "getComponentId() only available for HOST memory");
        return m_componentId;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets neighbor list (delegated to CollisionDetectionModule) */
    const NeighborList<T, M>* getNeighborList() const
    {
        return m_collisionDetectionModule ? m_collisionDetectionModule->getNeighborList() : nullptr;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets the collision detection module */
    const CollisionDetectionModule<T, M>* getCollisionDetectionModule() const
    {
        return m_collisionDetectionModule.get();
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets the number of particles in manager */
    uint getNumberOfParticles() const
    {
        return m_numParticles;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Gets the number of obstacles in manager */
    uint getNumberOfObstacles() const
    {
        return m_numObstacles;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Sets the number of particles (used by multi-GPU to include/exclude ghosts). */
    void setNumberOfParticles(uint n)
    {
        m_numParticles = n;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Direct access to the position buffer (for ghost exchange / migration). */
    GrainsMemBuffer<Vector3<T>, M>& getPositionBuffer() { return m_position; }
    /** @brief Direct access to the quaternion buffer. */
    GrainsMemBuffer<Quaternion<T>, M>& getQuaternionBuffer() { return m_quaternion; }
    /** @brief Direct access to the velocity buffer. */
    GrainsMemBuffer<Kinematics<T>, M>& getVelocityBuffer() { return m_velocity; }
    /** @brief Direct access to the torce buffer. */
    GrainsMemBuffer<Torce<T>, M>& getTorceBuffer() { return m_torce; }
    /** @brief Direct access to the rigid body id buffer. */
    GrainsMemBuffer<uint, M>& getRigidBodyIdBuffer() { return m_rigidBodyId; }
    /** @brief Direct access to the component id buffer. */
    GrainsMemBuffer<uint, M>& getComponentIdBuffer() { return m_componentId; }
    //@}

    /** @name Set methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Sets components rigid body Ids
        @param id host buffer containing the rigid body Ids */
    template <MemType srcM>
    void setRigidBodyId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_rigidBodyId.copyFrom(id);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Sets components positions
        @param p host buffer containing the positions */
    template <MemType srcM>
    void setPosition(const GrainsMemBuffer<Vector3<T>, srcM>& p)
    {
        m_position.copyFrom(p);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Sets components quaternions
        @param q host buffer containing the quaternions */
    template <MemType srcM>
    void setQuaternion(const GrainsMemBuffer<Quaternion<T>, srcM>& q)
    {
        m_quaternion.copyFrom(q);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Sets components velocities
        @param v host buffer containing the velocities */
    template <MemType srcM>
    void setVelocity(const GrainsMemBuffer<Kinematics<T>, srcM>& v)
    {
        m_velocity.copyFrom(v);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Sets components torces
        @param t host buffer containing the torces */
    template <MemType srcM>
    void setTorce(const GrainsMemBuffer<Torce<T>, srcM>& t)
    {
        m_torce.copyFrom(t);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Sets the array of components Ids
        @param id host buffer containing the components Ids */
    template <MemType srcM>
    void setComponentId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_componentId.copyFrom(id);
    }
    //@}

    /** @name Manager methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Initializes the CollisionDetectionModule and the contact hash table */
    void initialize()
    {
        m_collisionDetectionModule = std::make_unique<CollisionDetectionModule<T, M>>(
            m_rigidBody,
            m_position,
            m_quaternion,
            GrainsParameters<T>::m_collisionDetection,
            m_numObstacles,
            m_numParticles);

        // Size m_contactInfo and m_pairList to match the module's initial pair buffer capacity
        size_t pairCapacity = m_collisionDetectionModule->getPairBufferSize();
        m_contactInfo.initialize(pairCapacity);
        m_pairList.initialize(pairCapacity);

        // Create ForceModule (owns contact table + GPU intermediate buffers)
        m_forceModule = ForceModuleFactory<T, M>::create(pairCapacity);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Copies particle state from this manager to another.
        @param other destination component manager */
    template <MemType srcM>
    void copyTo(const std::unique_ptr<ComponentManager<T, srcM>>& other)
    {
        other->setRigidBodyId(m_rigidBodyId);
        other->setPosition(m_position);
        other->setQuaternion(m_quaternion);
        other->setVelocity(m_velocity);
        other->setTorce(m_torce);
        other->setComponentId(m_componentId);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Copies data to ComponentManagerCPU object for post-processing.
        @param other other component manager */
    void copyTo_PostProcessing(const std::unique_ptr<ComponentManager<T, MemType::HOST>>& other)
    {
        // RigidBodyId
        other->setRigidBodyId(m_rigidBodyId);

        // Position
        other->setPosition(m_position);

        // Quaternion
        other->setQuaternion(m_quaternion);

        // Velocity
        other->setVelocity(m_velocity);
    }
    //@}

    /** @name Methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Initializes transformations for components in the simulation
        @param initPosition initial position of components
        @param initOrientation initial orientation of components */
    template <MemType srcM>
    void initializeComponents(const GrainsMemBuffer<Vector3<T>, srcM>&    initPosition,
                              const GrainsMemBuffer<Quaternion<T>, srcM>& initOrientation)
    {
        // We can only initialize on host
        static_assert(M == MemType::HOST,
                      "Cannot initialize components directly on the device. Try "
                      "initializing on host first, and copy to device. Aborting Grains!");
        // Making sure that we have data for all components and the number of
        // initial TR matches the number of RBs
        uint nComponents = m_numParticles + m_numObstacles;
        assert(initPosition.getSize() == nComponents && initOrientation.getSize() == nComponents);

        // Assigning
        for(uint i = 0; i < nComponents; ++i)
        {
            m_position[i]   = initPosition[i];
            m_quaternion[i] = initOrientation[i];
        }
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Inserts particles according to a given insertion policy
        @param ins insertion policy */
    void insertParticles(const std::unique_ptr<Insertion<T>>& insertionPolicy)
    {
        // We can only insert on host
        static_assert(M == MemType::HOST,
                      "Cannot insert particles directly on the device. Try inserting on "
                      "host first, and copy to device. Aborting Grains!");

        // This adds all particles to the system all at once in the beginning
        insertionPolicy->insert(m_rigidBody,
                                m_position,
                                m_quaternion,
                                m_velocity,
                                GrainsParameters<T>::m_collisionDetection.linkedCellParameters,
                                m_numObstacles,
                                m_numParticles);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Runs the full collision detection pipeline via CollisionDetectionModule.
        Delegates sorting, neighbor list update, bounding volume, and narrow phase steps, as well as
        contact info transformation to world frame */
    void detectCollisions()
    {
        m_collisionDetectionModule->run(m_rigidBody->getData(),
                                        m_position,
                                        m_quaternion,
                                        m_velocity,
                                        m_torce,
                                        m_rigidBodyId,
                                        m_componentId,
                                        m_contactInfo,
                                        m_pairList,
                                        m_numPairs,
                                        m_numObstacles,
                                        m_numParticles);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Computes contact forces and external forces via ForceModule.
        Delegates the complete force pipeline (contact forces + gravity) to m_forceModule->run().
        @param CF array of all contact force models */
    void computeContactForces(const GrainsMemBuffer<ContactForceModel<T>*, M>& CF)
    {
        m_forceModule->run(CF,
                           m_rigidBody,
                           m_position,
                           m_velocity,
                           m_pairList,
                           m_contactInfo,
                           m_numPairs,
                           m_torce,
                           m_numObstacles,
                           m_numParticles);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief No-op: external forces (gravity) are applied inside computeContactForces via
        ForceModule::run().  Kept for API compatibility with the simulation loop. */
    void addExternalForces() {}

    // ---------------------------------------------------------------------------------------------
    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    virtual void moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, M>& TI) = 0;

    // ---------------------------------------------------------------------------------------------
    /** @brief Performs the second velocity half-kick for split-step schemes (e.g. Leapfrog).
        For single-pass schemes (e.g. FirstOrderExplicit) this is a no-op because
        TimeIntegrator::AdvanceVelocity defaults to an empty body.
        @param TI time integration scheme */
    virtual void advanceVelocity(const GrainsMemBuffer<TimeIntegrator<T>*, M>& TI) = 0;
    //@}
};

#endif