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
    /** \brief Particles position */
    GrainsMemBuffer<Vector3<T>, M> m_position;
    /** \brief Particles quaternion */
    GrainsMemBuffer<Quaternion<T>, M> m_quaternion;
    /** \brief Particles velocities */
    GrainsMemBuffer<Kinematics<T>, M> m_velocity;
    /** \brief Particles torce */
    GrainsMemBuffer<Torce<T>, M> m_torce;
    /** \brief Particles Id */
    GrainsMemBuffer<uint, M> m_particleId;

    /** \brief Obstacles rigid body Id */
    GrainsMemBuffer<uint, M> m_obstacleRigidBodyId;
    /** \brief Obstacles position */
    GrainsMemBuffer<Vector3<T>, M> m_obstaclePosition;
    /** \brief Particles quaternion */
    GrainsMemBuffer<Quaternion<T>, M> m_obstacleQuaternion;
    /** \brief Obstacles velocities */
    GrainsMemBuffer<Kinematics<T>, M> m_obstacleVelocity;

    /** \brief Number of particles in manager */
    uint m_nParticles;
    /** \brief Number of obstacles in manager */
    uint m_nObstacles;

    /** \brief Neighbor list object */
    NeighborList<T, M>* m_neighborList;
    /** \brief Relative position */
    GrainsMemBuffer<Vector3<T>, M> m_relPosition;
    /** \brief Relative quaternion */
    GrainsMemBuffer<Quaternion<T>, M> m_relQuaternion;
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
        , m_rigidBodyId(nParticles)
        , m_position(nParticles)
        , m_quaternion(nParticles)
        , m_velocity(nParticles)
        , m_torce(nParticles)
        , m_particleId(nParticles)
        , m_obstacleRigidBodyId(nObstacles)
        , m_obstaclePosition(nObstacles)
        , m_obstacleQuaternion(nObstacles)
        , m_obstacleVelocity(nObstacles)
        , m_nParticles(nParticles)
        , m_nObstacles(nObstacles)
    {
        NeighborListFactory<T, M>::create(m_neighborList);

        // Initialize with maximum possible pairs for dynamic sizing
        // TODO: Make this dynamic
        uint maxPairs = m_nParticles * (m_nParticles - 1) / 2;
        m_relPosition.allocate(maxPairs);
        m_relPosition.fill();
        m_relQuaternion.allocate(maxPairs);
        m_relQuaternion.fill();
        m_contactInfo.allocate(maxPairs);
        m_contactInfo.fill();
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
    /** @brief Gets particles positions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getPosition(GrainsMemBuffer<Vector3<T>, destM>& buffer) const
    {
        m_position.copyTo(buffer);
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
    /** @brief Gets obstacles rigid body Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getObstaclesRigidBodyId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_obstacleRigidBodyId.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles positions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getObstaclesPosition(GrainsMemBuffer<Vector3<T>, destM>& buffer) const
    {
        m_obstaclePosition.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles quaternions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getObstaclesQuaternion(
        GrainsMemBuffer<Quaternion<T>, destM>& buffer) const
    {
        m_obstacleQuaternion.copyTo(buffer);
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
    /** @brief Gets relative position
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getRelativePosition(GrainsMemBuffer<Vector3<T>, destM>& buffer) const
    {
        m_relPosition.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets relative quaternion
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getRelativeQuaternion(
        GrainsMemBuffer<Quaternion<T>, destM>& buffer) const
    {
        m_relQuaternion.copyTo(buffer);
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
    /** @brief Gets particles positions */
    const GrainsMemBuffer<Vector3<T>, MemType::HOST>& getPosition() const
    {
        static_assert(M == MemType::HOST,
                      "getPosition() only available for HOST memory");
        return m_position;
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
    /** @brief Gets obstacles positions */
    const GrainsMemBuffer<Vector3<T>, MemType::HOST>&
        getObstaclesPosition() const
    {
        static_assert(M == MemType::HOST,
                      "getObstaclesPosition() only available for HOST memory");
        return m_obstaclePosition;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets obstacles quaternions */
    const GrainsMemBuffer<Quaternion<T>, MemType::HOST>&
        getObstaclesQuaternion() const
    {
        static_assert(
            M == MemType::HOST,
            "getObstaclesQuaternion() only available for HOST memory");
        return m_obstacleQuaternion;
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
    /** @brief Sets particles positions
        @param p host buffer containing the positions */
    template <MemType srcM>
    void setPosition(const GrainsMemBuffer<Vector3<T>, srcM>& p)
    {
        m_position.copyFrom(p);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets particles quaternions
        @param q host buffer containing the quaternions */
    template <MemType srcM>
    void setQuaternion(const GrainsMemBuffer<Quaternion<T>, srcM>& q)
    {
        m_quaternion.copyFrom(q);
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
    /** @brief Sets obstacles positions
        @param p host buffer containing the positions */
    template <MemType srcM>
    void setObstaclesPosition(const GrainsMemBuffer<Vector3<T>, srcM>& p)
    {
        m_obstaclePosition.copyFrom(p);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets obstacles quaternions
        @param q host buffer containing the quaternions */
    template <MemType srcM>
    void setObstaclesQuaternion(const GrainsMemBuffer<Quaternion<T>, srcM>& q)
    {
        m_obstacleQuaternion.copyFrom(q);
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
    /** @brief Sets the relative position
        @param relPosition host buffer containing the relative positions */
    template <MemType srcM>
    void setRelativePosition(
        const GrainsMemBuffer<Vector3<T>, srcM>& relPosition)
    {
        m_relPosition.copyFrom(relPosition);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the relative quaternion
        @param relQuaternion host buffer containing the relative quaternions */
    template <MemType srcM>
    void setRelativeQuaternion(
        const GrainsMemBuffer<Quaternion<T>, srcM>& relQuaternion)
    {
        m_relQuaternion.copyFrom(relQuaternion);
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
    /** @brief Resizes pair-dependent buffers based on current neighbor list size */
    void resizePairBuffers()
    {
        uint pairCount = m_neighborList->getSize();
        m_relPosition.setSize(pairCount);
        m_relQuaternion.setSize(pairCount);
        m_contactInfo.setSize(pairCount);
    }

    // -------------------------------------------------------------------------
    /** @brief Copies data from another ComponentManager object.
        @param other other component manager */
    template <MemType srcM>
    void copyTo(const std::unique_ptr<ComponentManager<T, srcM>>& other)
    {
        // Particles
        other->setRigidBodyId(m_rigidBodyId);
        other->setPosition(m_position);
        other->setQuaternion(m_quaternion);
        other->setVelocity(m_velocity);
        other->setTorce(m_torce);
        other->setParticleId(m_particleId);
        // Obstacles
        other->setObstaclesRigidBodyId(m_obstacleRigidBodyId);
        other->setObstaclesPosition(m_obstaclePosition);
        other->setObstaclesQuaternion(m_obstacleQuaternion);
        other->setObstaclesVelocity(m_obstacleVelocity);
        // Neighbor list
        other->setRelativePosition(m_relPosition);
        other->setRelativeQuaternion(m_relQuaternion);
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

        // Position
        other->setPosition(m_position);

        // Quaternion
        other->setQuaternion(m_quaternion);

        // Velocity
        other->setVelocity(m_velocity);

        // Obstacle Position
        other->setObstaclesPosition(m_obstaclePosition);

        // Obstacle Quaternion
        other->setObstaclesQuaternion(m_obstacleQuaternion);

        // Obstacle Velocity
        other->setObstaclesVelocity(m_obstacleVelocity);
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Initializes transformations for particles in the simulation
        @param initPosition initial position of particles
        @param initOrientation initial orientation of particles */
    template <MemType srcM>
    void initializeParticles(
        const GrainsMemBuffer<Vector3<T>, srcM>&    initPosition,
        const GrainsMemBuffer<Quaternion<T>, srcM>& initOrientation)
    {
        // We can only initialize on host
        static_assert(
            M == MemType::HOST,
            "Cannot initialize particles directly on the device. Try "
            "initializing on host first, and copy to device. Aborting Grains!");
        // Making sure that we have data for all particles and the number of
        // initial TR matches the number of RBs
        assert(initPosition.getSize() == m_nParticles
               && initOrientation.getSize() == m_nParticles);

        // Assigning
        for(uint i = 0; i < m_nParticles; ++i)
        {
            m_position[i]   = initPosition[i];
            m_quaternion[i] = initOrientation[i];
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Initializes transformations for obstacles in the simulation
        @param initPosition initial positions of obstacles
        @param initOrientation initial orientations of obstacles */
    template <MemType srcM>
    void initializeObstacles(
        const GrainsMemBuffer<Vector3<T>, srcM>&    initPosition,
        const GrainsMemBuffer<Quaternion<T>, srcM>& initOrientation)
    {
        // We can only initialize on host
        static_assert(
            M == MemType::HOST,
            "Cannot initialize obstacles directly on the device. Try "
            "initializing on host first, and copy to device. Aborting Grains!");
        // Making sure that we have data for all obstacles and the number of
        // initial TR matches the number of RBs
        assert(initPosition.getSize() == m_nObstacles
               && initOrientation.getSize() == m_nObstacles);

        // Assigning
        for(uint i = 0; i < m_nObstacles; ++i)
        {
            m_obstaclePosition[i]   = initPosition[i];
            m_obstacleQuaternion[i] = initOrientation[i];
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
            insData         = ins->fetchInsertionData();
            m_position[i]   = insData.first.getOrigin();
            m_quaternion[i] = insData.first.getRotation() * m_quaternion[i];
            m_velocity[i]   = insData.second;
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