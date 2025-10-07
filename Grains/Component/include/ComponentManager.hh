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
    /** \brief Pointer to buffer of components rigid bodies */
    const GrainsMemBuffer<RigidBody<T>*, M>* m_rigidBody;
    /** \brief Pointer to buffer of reference rigid bodies for collision detection */
    const GrainsMemBuffer<RigidBody<T>*, M>* m_referenceRigidBodies;

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
    /** \brief Contact information in world frame */
    GrainsMemBuffer<ContactInfo<T>, M> m_contactInfoWorld;
    /** \brief Active contact pairs */
    GrainsMemBuffer<uint, M> m_activePairs;
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
        @param rigidBody Pointer to the components rigid body buffer
        @param referenceRigidBodies Pointer to the reference rigid bodies buffer
        @param nObstacles Number of obstacles
        @param nParticles Number of particles */
    ComponentManager(GrainsMemBuffer<RigidBody<T>*, M>* rigidBody,
                     GrainsMemBuffer<RigidBody<T>*, M>* referenceRigidBodies,
                     uint                               nObstacles,
                     uint                               nParticles)
        : m_rigidBody(rigidBody)
        , m_referenceRigidBodies(referenceRigidBodies)
        , m_rigidBodyId(nParticles + nObstacles)
        , m_position(nParticles + nObstacles)
        , m_quaternion(nParticles + nObstacles)
        , m_velocity(nParticles + nObstacles)
        , m_torce(nParticles + nObstacles)
        , m_componentId(nParticles + nObstacles)
        , m_nParticles(nParticles)
        , m_nObstacles(nObstacles)
    {
        GAssert(m_rigidBody->getSize() == m_nParticles + m_nObstacles,
                "Rigid body size mismatch");
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
    /** @brief Gets components rigid body Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getRigidBodyId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_rigidBodyId.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components positions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getPosition(GrainsMemBuffer<Vector3<T>, destM>& buffer) const
    {
        m_position.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components quaternions
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getQuaternion(GrainsMemBuffer<Quaternion<T>, destM>& buffer) const
    {
        m_quaternion.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components velocities
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getVelocity(GrainsMemBuffer<Kinematics<T>, destM>& buffer) const
    {
        m_velocity.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components torces
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getTorce(GrainsMemBuffer<Torce<T>, destM>& buffer) const
    {
        m_torce.copyTo(buffer);
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components Ids
        @param buffer host buffer to copy data to */
    template <MemType destM>
    void getComponentId(GrainsMemBuffer<uint, destM>& buffer) const
    {
        m_componentId.copyTo(buffer);
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
    /** @brief Gets components rigid body Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getRigidBodyId() const
    {
        static_assert(M == MemType::HOST,
                      "getRigidBodyId() only available for HOST memory");
        return m_rigidBodyId;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components positions */
    const GrainsMemBuffer<Vector3<T>, MemType::HOST>& getPosition() const
    {
        static_assert(M == MemType::HOST,
                      "getPosition() only available for HOST memory");
        return m_position;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components quaternions */
    const GrainsMemBuffer<Quaternion<T>, MemType::HOST>& getQuaternion() const
    {
        static_assert(M == MemType::HOST,
                      "getQuaternion() only available for HOST memory");
        return m_quaternion;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components velocities */
    const GrainsMemBuffer<Kinematics<T>, MemType::HOST>& getVelocity() const
    {
        static_assert(M == MemType::HOST,
                      "getVelocity() only available for HOST memory");
        return m_velocity;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components torces */
    const GrainsMemBuffer<Torce<T>, MemType::HOST>& getTorce() const
    {
        static_assert(M == MemType::HOST,
                      "getTorce() only available for HOST memory");
        return m_torce;
    }

    // -------------------------------------------------------------------------
    /** @brief Gets components Ids */
    const GrainsMemBuffer<uint, MemType::HOST>& getComponentId() const
    {
        static_assert(M == MemType::HOST,
                      "getComponentId() only available for HOST memory");
        return m_componentId;
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
    /** @brief Sets components rigid body Ids
        @param id host buffer containing the rigid body Ids */
    template <MemType srcM>
    void setRigidBodyId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_rigidBodyId.copyFrom(id);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets components positions
        @param p host buffer containing the positions */
    template <MemType srcM>
    void setPosition(const GrainsMemBuffer<Vector3<T>, srcM>& p)
    {
        m_position.copyFrom(p);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets components quaternions
        @param q host buffer containing the quaternions */
    template <MemType srcM>
    void setQuaternion(const GrainsMemBuffer<Quaternion<T>, srcM>& q)
    {
        m_quaternion.copyFrom(q);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets components velocities
        @param v host buffer containing the velocities */
    template <MemType srcM>
    void setVelocity(const GrainsMemBuffer<Kinematics<T>, srcM>& v)
    {
        m_velocity.copyFrom(v);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets components torces
        @param t host buffer containing the torces */
    template <MemType srcM>
    void setTorce(const GrainsMemBuffer<Torce<T>, srcM>& t)
    {
        m_torce.copyFrom(t);
    }

    // -------------------------------------------------------------------------
    /** @brief Sets the array of components Ids
        @param id host buffer containing the components Ids */
    template <MemType srcM>
    void setComponentId(const GrainsMemBuffer<uint, srcM>& id)
    {
        m_componentId.copyFrom(id);
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
    /** @brief Initializes buffers for pair-dependent data */
    void initialize()
    {
        NeighborListFactory<T, M>::create(m_rigidBody,
                                          m_referenceRigidBodies,
                                          m_position,
                                          m_quaternion,
                                          m_nObstacles,
                                          m_nParticles,
                                          m_neighborList);

        // Initialize with maximum possible pairs for dynamic sizing
        // TODO: Make this dynamic
        uint maxPairs = m_nObstacles * m_nParticles
                        + m_nParticles * (m_nParticles - 1) / 2;
        resizePairBuffers(maxPairs);
    }

    // -------------------------------------------------------------------------
    /** @brief Resizes pair-dependent buffers based on current neighbor list 
        size
        @param size new size for the pair buffers */
    virtual void resizePairBuffers(const uint size)
    {
        m_relPosition.setSize(size);
        m_relQuaternion.setSize(size);
        m_contactInfo.setSize(size);
        m_contactInfoWorld.setSize(size);
        m_activePairs.setSize(size);
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
        other->setComponentId(m_componentId);
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
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Initializes transformations for components in the simulation
        @param initPosition initial position of components
        @param initOrientation initial orientation of components */
    template <MemType srcM>
    void initializeComponents(
        const GrainsMemBuffer<Vector3<T>, srcM>&    initPosition,
        const GrainsMemBuffer<Quaternion<T>, srcM>& initOrientation)
    {
        // We can only initialize on host
        static_assert(
            M == MemType::HOST,
            "Cannot initialize components directly on the device. Try "
            "initializing on host first, and copy to device. Aborting Grains!");
        // Making sure that we have data for all components and the number of
        // initial TR matches the number of RBs
        uint nComponents = m_nParticles + m_nObstacles;
        assert(initPosition.getSize() == nComponents
               && initOrientation.getSize() == nComponents);

        // Assigning
        for(uint i = 0; i < nComponents; ++i)
        {
            m_position[i]   = initPosition[i];
            m_quaternion[i] = initOrientation[i];
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Inserts particles according to a given insertion policy
        @param ins insertion policy */
    void insertParticles(const std::unique_ptr<Insertion<T>>& insertionPolicy)
    {
        // We can only insert on host
        static_assert(
            M == MemType::HOST,
            "Cannot insert particles directly on the device. Try inserting on "
            "host first, and copy to device. Aborting Grains!");

        // This adds all particles to the system all at once in the beginning
        insertionPolicy->insert(m_rigidBody,
                                m_position,
                                m_quaternion,
                                m_velocity,
                                m_nObstacles,
                                m_nParticles);
    }

    // -------------------------------------------------------------------------
    /** @brief Updates neighbor list */
    virtual void updateNeighborList() = 0;

    // -------------------------------------------------------------------------
    /** @brief Computes the relative transformations */
    virtual void computeRelativeTransformations() = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collisions between components */
    virtual void detectCollisionsComponents() = 0;

    // -------------------------------------------------------------------------
    /** @brief Transforms contact info to world frame and flags active pairs */
    virtual void transformContactInfoToWorld() = 0;

    // -------------------------------------------------------------------------
    /** @brief Detects collision */
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