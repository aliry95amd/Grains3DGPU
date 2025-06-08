#ifndef _COMPONENTMANAGER_HH_
#define _COMPONENTMANAGER_HH_

#include "CollisionDetection.hh"
#include "ContactForceModel.hh"
#include "ContactForceModelFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "Insertion.hh"
#include "Kinematics.hh"
#include "LinkedCell.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"
#include "Torce.hh"
#include "Transform3.hh"

#include "NeighborList.hh"
#include "NeighborList_Nsq.hh"

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
    const GrainsMemBuffer<RigidBody<T, T>*, M>* m_particleRB;
    /** \brief Pointer to buffer of obstacles rigid bodies. */
    const GrainsMemBuffer<RigidBody<T, T>*, M>* m_obstacleRB;
    /** \brief Particles rigid body Id */
    GrainsMemBuffer<uint, M> m_rigidBodyId;
    /** \brief Particles transformation */
    GrainsMemBuffer<Transform3<T>, M> m_transform;
    /** \brief Particles velocities */
    GrainsMemBuffer<Kinematics<T>, M> m_velocity;
    /** \brief Particles torce */
    GrainsMemBuffer<Torce<T>, M> m_torce;
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
    /** \brief Number of cells in manager */
    uint m_nCells;

    /** \brief Neighbor list object */
    NeighborList<T, M>* m_neighborList;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor (forbidden except in derived classes) */
    ComponentManager() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with the number of particles, obstacles, and cells. 
        @param particleRB Pointer to the particles rigid body buffer
        @param obstacleRB Pointer to the obstacles rigid body buffer
        @param nParticles Number of particles
        @param nObstacles Number of obstacles
        @param nCells Number of cells */
    ComponentManager(GrainsMemBuffer<RigidBody<T, T>*, M>* particleRB,
                     GrainsMemBuffer<RigidBody<T, T>*, M>* obstacleRB,
                     uint                                  nParticles,
                     uint                                  nObstacles,
                     uint                                  nCells)
        : m_particleRB(particleRB)
        , m_obstacleRB(obstacleRB)
        , m_nParticles(nParticles)
        , m_nObstacles(nObstacles)
        , m_nCells(nCells)
    {
        allocate();
        m_neighborList = new NeighborList_Nsq<T, M>(nParticles);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    virtual ~ComponentManager() = default;
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

    // -------------------------------------------------------------------------
    /** @brief Gets the number of cells in manager */
    uint getNumberOfCells() const
    {
        return m_nCells;
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
    //@}

    /** @name Manager methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Allocates memory for the component manager */
    virtual void allocate()
    {
        m_rigidBodyId.reserve(m_nParticles);
        m_transform.reserve(m_nParticles);
        m_velocity.reserve(m_nParticles);
        m_torce.reserve(m_nParticles);
        m_particleId.reserve(m_nParticles);

        m_obstacleRigidBodyId.reserve(m_nObstacles);
        m_obstacleTransform.reserve(m_nObstacles);
        m_obstacleVelocity.reserve(m_nObstacles);
    }

    // -------------------------------------------------------------------------
    /** @brief Copies data from another ComponentManager object.
        @param other other component manager */
    template <MemType srcM>
    void copyFrom(const std::unique_ptr<ComponentManager<T, srcM>>& other)
    {
        // RigidBodyId
        GrainsMemBuffer<uint, srcM> tmpRigidBodyId(m_nParticles);
        other->getRigidBodyId(tmpRigidBodyId);
        setRigidBodyId(tmpRigidBodyId);

        // Transform
        GrainsMemBuffer<Transform3<T>, srcM> tmpTransform(m_nParticles);
        other->getTransform(tmpTransform);
        setTransform(tmpTransform);

        // Velocity
        GrainsMemBuffer<Kinematics<T>, srcM> tmpVelocity(m_nParticles);
        other->getVelocity(tmpVelocity);
        setVelocity(tmpVelocity);

        // Torce
        GrainsMemBuffer<Torce<T>, srcM> tmpTorce(m_nParticles);
        other->getTorce(tmpTorce);
        setTorce(tmpTorce);

        // ParticleId
        GrainsMemBuffer<uint, srcM> tmpParticleId(m_nParticles);
        other->getParticleId(tmpParticleId);
        setParticleId(tmpParticleId);

        // Obstacle RigidBodyId
        GrainsMemBuffer<uint, srcM> tmpRigidBodyIdObstacles(m_nObstacles);
        other->getObstaclesRigidBodyId(tmpRigidBodyIdObstacles);
        setObstaclesRigidBodyId(tmpRigidBodyIdObstacles);

        // Obstacle Transform
        GrainsMemBuffer<Transform3<T>, srcM> tmpTransformObstacles(
            m_nObstacles);
        other->getObstaclesTransform(tmpTransformObstacles);
        setObstaclesTransform(tmpTransformObstacles);

        // Obstacle Velocity
        GrainsMemBuffer<Kinematics<T>, srcM> tmpVelocityObstacles(m_nObstacles);
        other->getObstaclesVelocity(tmpVelocityObstacles);
        setObstaclesVelocity(tmpVelocityObstacles);
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
            // m_rigidBodyId
            m_rigidBodyId[i] = i;

            // m_transform
            m_transform[i] = initTr[i];
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
            // m_rigidBodyId
            m_obstacleRigidBodyId[i] = i;

            // m_transform
            m_obstacleTransform[i] = initTr[i];

            // m_velocity
            m_obstacleVelocity[i] = Kinematics<T>();
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
    /** @brief Updates links between particles and linked cell
        @param LC linked cell */
    virtual void updateLinks(const GrainsMemBuffer<LinkedCell<T>*, M>& LC) = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collision between particles and obstacles and computes 
    forces
        @param CF array of all contact force models */
    virtual void detectCollisionAndComputeContactForcesObstacles(
        const GrainsMemBuffer<ContactForceModel<T>*, M>& CF)
        = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collision between particles and particles and computes 
    forces
        @param LC linked cell
        @param CF array of all contact force models */
    virtual void detectCollisionAndComputeContactForcesParticles(
        const GrainsMemBuffer<LinkedCell<T>*, M>&        LC,
        const GrainsMemBuffer<ContactForceModel<T>*, M>& CF)
        = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collision between components and computes forces
        @param LC linked cell
        @param CF array of all contact force models */
    virtual void detectCollisionAndComputeContactForces(
        const GrainsMemBuffer<LinkedCell<T>*, M>&        LC,
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