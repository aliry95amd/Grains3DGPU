#ifndef _COMPONENTMANAGER_HH_
#define _COMPONENTMANAGER_HH_

#include "ComponentManagerCommon.hh"
#include "ContactForceModel.hh"
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
#include "NeighborListFactory.hh"

// =============================================================================
/** @brief The class ComponentManager.

    This is just an abstract class to make sure all derived classess follow the
    same set of methods.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
template <typename T, MemType M = MemType::HOST>
class ComponentManager
{
protected:
    /** @name Parameters */
    //@{
    // TODO: What to do with pointers? Better design? unique_ptr?
    /** \brief Pointer to buffer of particles rigid bodies. */
    const GrainsMemBuffer<RigidBody<T>*, M>* m_particleRB;
    /** \brief Pointer to buffer of obstacles rigid bodies. */
    const GrainsMemBuffer<RigidBody<T>*, M>* m_obstacleRB;

    /** \brief Particles rigid body Id */
    GrainsMemBuffer<uint, M> m_rigidBodyId;
    /** \brief Particles transformation */
    GrainsMemBuffer<Transform3<T>, M> m_transform;
    /** \brief Particles velocities */
    GrainsMemBuffer<Kinematics<T>, M> m_velocity;
    /** \brief Particles torce */
    GrainsMemBuffer<Torce<T>, M> m_torce;
    /** \brief Particles quaternion */
    GrainsMemBuffer<Quaternion<T>, M> m_quaternion;
    /** \brief Particles Id */
    GrainsMemBuffer<uint, M> m_particleId;

    /** \brief Obstacles rigid body Id */
    GrainsMemBuffer<uint, M> m_obstacleRigidBodyId;
    /** \brief Obstacles transformation */
    GrainsMemBuffer<Transform3<T>, M> m_obstacleTransform;
    /** \brief Obstacles velocities */
    GrainsMemBuffer<Kinematics<T>, M> m_obstacleVelocity;

    /** \brief Number of particles in manager */
    uint m_nParticles;
    /** \brief Number of obstacles in manager */
    uint m_nObstacles;
    /** \brief Number of pairs in manager */
    uint m_nPairs;

    /** \brief Neighbor list object */
    NeighborList<T, M>* m_neighborList;
    /** \brief Relative transformation */
    GrainsMemBuffer<Transform3<T>, M> m_relTransform;
    /** \brief Contact information */
    GrainsMemBuffer<ContactInfo<T>, M> m_contactInfo;
    // /** \brief Rigid bodies bounding volume */
    // GrainsMemBuffer<BoundingVolume<T>, M> m_boundingVolume;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor (forbidden except in derived classes) */
    ComponentManager() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with the number of particles, and obstacles 
        @param particleRB Pointer to the particles rigid body buffer
        @param obstacleRB Pointer to the obstacles rigid body buffer
        @param nParticles Number of particles
        @param nObstacles Number of obstacles */
    ComponentManager(GrainsMemBuffer<RigidBody<T>*, M>* particleRB,
                     GrainsMemBuffer<RigidBody<T>*, M>* obstacleRB,
                     uint                               nParticles,
                     uint                               nObstacles)
        : m_particleRB(particleRB)
        , m_obstacleRB(obstacleRB)
        , m_nParticles(nParticles)
        , m_nObstacles(nObstacles)
    {
        NeighborListFactory<T, M>::create(m_neighborList);
        m_nPairs = m_neighborList->getSize();
        initializeToDefault();
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~ComponentManager()
    {
        if(m_neighborList)
            delete m_neighborList;
    }
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Gets particles rigid body Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getRigidBodyId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_rigidBodyId.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles transformations
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getTransform(GrainsMemBuffer<Transform3<T>, destM>& buffer) const
    {
        m_transform.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles velocities
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getVelocity(GrainsMemBuffer<Kinematics<T>, destM>& buffer) const
    {
        m_velocity.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles torces
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getTorce(GrainsMemBuffer<Torce<T>, destM>& buffer) const
    {
        m_torce.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets the array of particles Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getParticleId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_particleId.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles quaternions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getQuaternion(GrainsMemBuffer<Quaternion<T>, destM>& buffer) const
    {
        m_quaternion.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles rigid body Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getObstaclesRigidBodyId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_obstacleRigidBodyId.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles transformation
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getObstaclesTransform(
        GrainsMemBuffer<Transform3<T>, destM>& buffer) const
    {
        m_obstacleTransform.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles velocities
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getObstaclesVelocity(
        GrainsMemBuffer<Kinematics<T>, destM>& buffer) const
    {
        m_obstacleVelocity.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets relative transformations
     @param buffer host buffer to copy data to */
    template <MemType destM>
    void getRelativeTransform(
        GrainsMemBuffer<Transform3<T>, destM>& buffer) const
    {
        m_relTransform.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets contact information
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getContactInfo(GrainsMemBuffer<ContactInfo<T>, destM>& buffer) const
    {
        m_contactInfo.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles rigid body Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getRigidBodyId() const
    {
        static_assert(M == MemType::HOST,
                      "getRigidBodyId() only available for HOST memory");
        return m_rigidBodyId;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles transformations */
    const GrainsMemBuffer<Transform3<T>, MemType::HOST>& getTransform() const
    {
        static_assert(M == MemType::HOST,
                      "getTransform() only available for HOST memory");
        return m_transform;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles velocities */
    const GrainsMemBuffer<Kinematics<T>, MemType::HOST>& getVelocity() const
    {
        static_assert(M == MemType::HOST,
                      "getVelocity() only available for HOST memory");
        return m_velocity;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles torces */
    const GrainsMemBuffer<Torce<T>, MemType::HOST>& getTorce() const
    {
        static_assert(M == MemType::HOST,
                      "getTorce() only available for HOST memory");
        return m_torce;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets particles quaternions */
    const GrainsMemBuffer<Quaternion<T>, MemType::HOST>& getQuaternion() const
    {
        static_assert(M == MemType::HOST,
                      "getQuaternion() only available for HOST memory");
        return m_quaternion;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets the array of particles Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getParticleId() const
    {
        static_assert(M == MemType::HOST,
                      "getParticleId() only available for HOST memory");
        return m_particleId;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles rigid body Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getObstaclesRigidBodyId() const
    {
        static_assert(
            M == MemType::HOST,
            "getObstaclesRigidBodyId() only available for HOST memory");
        return m_obstacleRigidBodyId;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles transformation */
    const GrainsMemBuffer<Transform3<T>, MemType::HOST>&
        getObstaclesTransform() const
    {
        static_assert(M == MemType::HOST,
                      "getObstaclesTransform() only available for HOST memory");
        return m_obstacleTransform;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles velocities */
    const GrainsMemBuffer<Kinematics<T>, MemType::HOST>&
        getObstaclesVelocity() const
    {
        static_assert(M == MemType::HOST,
                      "getObstacleVelocity() only available for HOST memory");
        return m_obstacleVelocity;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets the number of particles in manager */
    uint getNumberOfParticles() const
    {
        return m_nParticles;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets the number of obstacles in manager */
    uint getNumberOfObstacles() const
    {
        return m_nObstacles;
    }
    //@}

    /** @name Set methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Sets the array of particles rigid body Ids
        @param id host buffer containing the rigid body Ids */
    template <MemType srcM>
    void setRigidBodyId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_rigidBodyId.copyFrom(id);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets particles transformations
        @param t host buffer containing the transformations */
    template <MemType srcM>
    void setTransform(const GrainsMemBuffer<Transform3<T>, srcM>& t)
    {
        m_transform.copyFrom(t);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets particles velocities
        @param v host buffer containing the velocities */
    template <MemType srcM>
    void setVelocity(const GrainsMemBuffer<Kinematics<T>, srcM>& v)
    {
        m_velocity.copyFrom(v);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets particles torces
        @param t host buffer containing the torces */
    template <MemType srcM>
    void setTorce(const GrainsMemBuffer<Torce<T>, srcM>& t)
    {
        m_torce.copyFrom(t);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets particles torces
        @param t host buffer containing the torces */
    template <MemType srcM>
    void setQuaternion(const GrainsMemBuffer<Quaternion<T>, srcM>& t)
    {
        m_quaternion.copyFrom(t);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the array of particles Ids
        @param id host buffer containing the particles Ids */
    template <MemType srcM>
    void setParticleId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_particleId.copyFrom(id);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the array of obstacles rigid body Ids
        @param id host buffer containing the rigid body Ids */
    template <MemType srcM>
    void setObstaclesRigidBodyId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_obstacleRigidBodyId.copyFrom(id);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets obstacles transformations
        @param t host buffer containing the transformations */
    template <MemType srcM>
    void setObstaclesTransform(const GrainsMemBuffer<Transform3<T>, srcM>& t)
    {
        m_obstacleTransform.copyFrom(t);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets obstacles velocities
        @param v host buffer containing the velocities */
    template <MemType srcM>
    void setObstaclesVelocity(const GrainsMemBuffer<Kinematics<T>, srcM>& v)
    {
        m_obstacleVelocity.copyFrom(v);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the relative transformations
        @param relTransform host buffer containing the rel transformations */
    template <MemType srcM>
    void setRelativeTransform(
        const GrainsMemBuffer<Transform3<T>, srcM>& relTransform)
    {
        m_relTransform.copyFrom(relTransform);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the contact information
        @param contactInfo host buffer containing the contact information */
    template <MemType srcM>
    void
        setContactInfo(const GrainsMemBuffer<ContactInfo<T>, srcM>& contactInfo)
    {
        m_contactInfo.copyFrom(contactInfo);
    }
    //@}

    /** @name Manager methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Initializes (and reserves memory) the members to default */
    void initializeToDefault()
    {
        initDefault(m_rigidBodyId, m_nParticles);
        initDefault(m_transform, m_nParticles);
        initDefault(m_velocity, m_nParticles);
        initDefault(m_torce, m_nParticles);
        initDefault(m_quaternion, m_nParticles);
        initDefault(m_particleId, m_nParticles);

        initDefault(m_obstacleRigidBodyId, m_nObstacles);
        initDefault(m_obstacleTransform, m_nObstacles);
        initDefault(m_obstacleVelocity, m_nObstacles);

        initDefault(m_relTransform, m_nPairs);
        initDefault(m_contactInfo, m_nPairs);
    }

    // -------------------------------------------------------------------------
    /** @brief Copies data from another ComponentManager object.
        @param other other component manager */
    template <MemType srcM>
    void copyTo(const std::unique_ptr<ComponentManager<T, srcM>>& other)
    {
        // Particles
        other->setRigidBodyId(m_rigidBodyId);
        other->setTransform(m_transform);
        other->setVelocity(m_velocity);
        other->setTorce(m_torce);
        other->setQuaternion(m_quaternion);
        other->setParticleId(m_particleId);
        // Obstacles
        other->setObstaclesRigidBodyId(m_obstacleRigidBodyId);
        other->setObstaclesTransform(m_obstacleTransform);
        other->setObstaclesVelocity(m_obstacleVelocity);
        // Neighbor list
        other->setRelativeTransform(m_relTransform);
        other->setContactInfo(m_contactInfo);
    }

    // -------------------------------------------------------------------------
    /** @brief Copies data to ComponentManagerCPU object for post-processing.
        @param other other component manager */
    void copyTo_PostProcessing(
        const std::unique_ptr<ComponentManager<T, MemType::HOST>>& other)
    {
        // RigidBodyId
        other->setRigidBodyId(m_rigidBodyId);

        // Transform
        other->setTransform(m_transform);

        // Velocity
        other->setVelocity(m_velocity);

        // Obstacle Transform
        other->setObstaclesTransform(m_obstacleTransform);

        // Obstacle Velocity
        other->setObstaclesVelocity(m_obstacleVelocity);
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Initializes transformations for particles in the simulation
        @param initTr initial transformation of particles */
    template <MemType srcM>
    void initializeParticles(const GrainsMemBuffer<Transform3<T>, srcM>& initTr)
    {
        // We can only initialize on host
        static_assert(
            M == MemType::HOST,
            "Cannot initialize particles directly on the device. Try "
            "initializing on host first, and copy to device. Aborting Grains!");
        // Making sure that we have data for all particles and the number of
        // initial TR matches the number of RBs
        assert(initTr.getSize() == m_nParticles);

        // Assigning
        for(uint i = 0; i < m_nParticles; ++i)
        {
            m_transform[i]  = initTr[i];
            m_quaternion[i] = Quaternion<T>(initTr[i].getBasis());
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Initializes transformations for obstacles in the simulation
        @param initTr initial transformation of obstacles */
    template <MemType srcM>
    void initializeObstacles(const GrainsMemBuffer<Transform3<T>, srcM>& initTr)
    {
        // We can only initialize on host
        static_assert(
            M == MemType::HOST,
            "Cannot initialize obstacles directly on the device. Try "
            "initializing on host first, and copy to device. Aborting Grains!");
        // Making sure that we have data for all obstacles and the number of
        // initial TR matches the number of RBs
        assert(initTr.getSize() == m_nObstacles);

        // Assigning
        for(uint i = 0; i < m_nObstacles; ++i)
        {
            m_obstacleTransform[i] = initTr[i];
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Inserts particles according to a given insertion policy
        @param ins insertion policy */
    void insertParticles(const std::unique_ptr<Insertion<T>>& ins)
    {
        // We can only insert on host
        static_assert(
            M == MemType::HOST,
            "Cannot insert particles directly on the device. Try inserting on "
            "host first, and copy to device. Aborting Grains!");

        std::pair<Transform3<T>, Kinematics<T>> insData;
        // Inserting particles
        for(uint i = 0; i < m_nParticles; ++i)
        {
            // Fetching insertion data from ins
            insData = ins->fetchInsertionData();

            // m_transform
            m_transform[i].composeLeftByRotation(insData.first);
            m_transform[i].setOrigin(insData.first.getOrigin());

            // m_velocity
            m_velocity[i] = insData.second;
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Updates neighbor list */
    virtual void updateNeighborList() = 0;

    // -------------------------------------------------------------------------
    /** @brief Computes the relative transformations */
    virtual void computeRelativeTransformations() = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collisions between particles and obstacles */
    virtual void detectCollisionsObstacles() = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collisions between particles and particles */
    virtual void detectCollisionsParticles() = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collision between particles and particles and */
    virtual void detectCollisions() = 0;

    // -------------------------------------------------------------------------
    /** @brief Computes contact forces between different components
        @param CF array of all contact force models */
    virtual void computeContactForces(
        const GrainsMemBuffer<ContactForceModel<T>*, M>& CF)
        = 0;

    // -------------------------------------------------------------------------
    /** @brief Adds external forces such as gravity */
    virtual void addExternalForces() = 0;

    // -------------------------------------------------------------------------
    /** @brief Updates the position and velocities of particles
        @param TI time integration scheme */
    virtual void moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, M>& TI)
        = 0;
    //@}
};

#endif