#include "ComponentManager.hh"
#include "BodyTag.hh"
#include "ComponentManagerCommon.hh"
#include "ComponentManagerGPU_Kernels.hh"
#include "ForceModuleFactory.hh"
#include "QuaternionMath.hh"
#include "VectorMath.hh"

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
ComponentManager<T, M>::ComponentManager(GrainsMemBuffer<RigidBody<T>*, M>* rigidBody,
                                         uint                               nObstacles,
                                         uint                               nParticles,
                                         uint                               nComposites,
                                         uint                               nSubBodies)
    : m_rigidBody(rigidBody)
    , m_position(nParticles + nObstacles)
    , m_quaternion(nParticles + nObstacles)
    , m_velocity(nParticles + nObstacles)
    , m_torce(nParticles + nObstacles)
    , m_bodyTag(nParticles + nObstacles)
    , m_localPos(nParticles + nObstacles)
    , m_localQuat(nParticles + nObstacles)
    , m_masterSlot(nComposites + 1)
    , m_counts{nObstacles, nParticles, 0u, nComposites, nSubBodies}
{
    GAssert(m_rigidBody->getSize() == m_counts.numParticles + m_counts.numObstacles,
            "Rigid body size mismatch");
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<Vector3<T>, M>& ComponentManager<T, M>::getLocalPos() const
{
    return m_localPos;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<Quaternion<T>, M>& ComponentManager<T, M>::getLocalQuat() const
{
    return m_localQuat;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<uint, M>& ComponentManager<T, M>::getBodyTag() const
{
    return m_bodyTag;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<Vector3<T>, M>& ComponentManager<T, M>::getPosition() const
{
    return m_position;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<Quaternion<T>, M>& ComponentManager<T, M>::getQuaternion() const
{
    return m_quaternion;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<Kinematics<T>, M>& ComponentManager<T, M>::getVelocity() const
{
    return m_velocity;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const GrainsMemBuffer<Torce<T>, M>& ComponentManager<T, M>::getTorce() const
{
    return m_torce;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const NeighborList<T, M>* ComponentManager<T, M>::getNeighborList() const
{
    return m_collisionDetectionModule ? m_collisionDetectionModule->getNeighborList() : nullptr;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
const CollisionDetectionModule<T, M>* ComponentManager<T, M>::getCollisionDetectionModule() const
{
    return m_collisionDetectionModule.get();
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
uint ComponentManager<T, M>::getNumberOfParticles() const
{
    return m_counts.numParticles;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
uint ComponentManager<T, M>::getNumberOfObstacles() const
{
    return m_counts.numObstacles;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
uint ComponentManager<T, M>::getNumberOfComposites() const
{
    return m_counts.numComposites;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
uint ComponentManager<T, M>::getNumberOfSubBodies() const
{
    return m_counts.numSubBodies;
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
void ComponentManager<T, M>::initialize()
{
    m_collisionDetectionModule = std::make_unique<CollisionDetectionModule<T, M>>(
        m_rigidBody,
        m_position,
        m_quaternion,
        GrainsParameters<T>::m_collisionDetection,
        m_counts.numObstacles,
        m_counts.numParticles);

    // Size m_contactInfo and m_pairList to match the module's initial pair buffer capacity
    size_t pairCapacity = m_collisionDetectionModule->getPairBufferSize();
    m_contactInfo.initialize(pairCapacity);
    m_pairList.initialize(pairCapacity);

    // Create ForceModule (owns contact table + GPU intermediate buffers)
    m_forceModule = ForceModuleFactory<T, M>::create(pairCapacity);
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
void ComponentManager<T, M>::copyTo_PostProcessing(
    const std::unique_ptr<ComponentManager<T, MemType::HOST>>& other)
{
    other->setPosition(m_position);
    other->setQuaternion(m_quaternion);
    other->setVelocity(m_velocity);
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
void ComponentManager<T, M>::initializeComponents(
    const GrainsMemBuffer<Vector3<T>, MemType::HOST>&    initPosition,
    const GrainsMemBuffer<Quaternion<T>, MemType::HOST>& initOrientation,
    const GrainsMemBuffer<uint, MemType::HOST>&          initBodyTags,
    const GrainsMemBuffer<Vector3<T>, MemType::HOST>&    initLocalPos,
    const GrainsMemBuffer<Quaternion<T>, MemType::HOST>& initLocalQuat)
{
    if constexpr(M == MemType::HOST)
    {
        uint nComponents = m_counts.numParticles + m_counts.numObstacles;
        assert(initPosition.getSize() == nComponents && initOrientation.getSize() == nComponents
               && initBodyTags.getSize() == nComponents && initLocalPos.getSize() == nComponents
               && initLocalQuat.getSize() == nComponents);

        for(uint i = 0; i < nComponents; ++i)
        {
            m_position[i]   = initPosition[i];
            m_quaternion[i] = initOrientation[i];
            m_bodyTag[i]    = initBodyTags[i];
            m_localPos[i]   = initLocalPos[i];
            m_localQuat[i]  = initLocalQuat[i];
        }

        // Populate masterSlot: sub-body with localIdx=0 is the master of each composite
        for(uint i = 0; i < nComponents; ++i)
        {
            uint tag = m_bodyTag[i];
            if(isSubBody(tag) && getSubBodyLocalIdx(tag) == 0u)
                m_masterSlot[getCompositeIdx(tag)] = i;
        }
    }
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
void ComponentManager<T, M>::insertParticles(const std::unique_ptr<Insertion<T>>& insertionPolicy)
{
    // Insertion is a HOST-only operation. The DEVICE branch is discarded by if constexpr so
    // that type-incompatible calls to insertionPolicy->insert() are never compiled for DEVICE.
    if constexpr(M == MemType::HOST)
    {
        insertionPolicy->insert(m_rigidBody,
                                m_position,
                                m_quaternion,
                                m_velocity,
                                GrainsParameters<T>::m_collisionDetection.linkedCellParameters,
                                m_counts.numObstacles,
                                m_counts.numParticles);
    }
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
void ComponentManager<T, M>::detectCollisions()
{
    m_collisionDetectionModule->run(m_rigidBody->getData(),
                                    m_position,
                                    m_quaternion,
                                    m_velocity,
                                    m_torce,
                                    m_bodyTag,
                                    m_localPos,
                                    m_localQuat,
                                    m_masterSlot,
                                    m_contactInfo,
                                    m_pairList,
                                    m_counts);
}

// -------------------------------------------------------------------------------------------------
template <typename T, MemType M>
void ComponentManager<T, M>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, M>& CF)
{
    m_forceModule->run(CF,
                       m_rigidBody,
                       m_position,
                       m_velocity,
                       m_pairList,
                       m_contactInfo,
                       m_torce,
                       m_bodyTag,
                       m_masterSlot,
                       m_counts);
}

// -------------------------------------------------------------------------------------------------
// Updates the position and quaternion of master particles (non-masters are updated by
// updateSubBodyPositions after this call).
template <typename T, MemType M>
void ComponentManager<T, M>::moveParticles(const GrainsMemBuffer<TimeIntegrator<T>*, M>& TI)
{
    if constexpr(M == MemType::HOST)
    {
        for(uint pID = m_counts.numObstacles; pID < m_counts.numObstacles + m_counts.numParticles;
            ++pID)
        {
            if(isSubBody(m_bodyTag[pID]) && getSubBodyLocalIdx(m_bodyTag[pID]) != 0u)
                continue;
            moveParticles_common(TI.getData(),
                                 m_rigidBody->getData(),
                                 m_position.getData(),
                                 m_quaternion.getData(),
                                 m_velocity.getData(),
                                 m_torce.getData(),
                                 pID);
        }
    }
    else
    {
        uint numThreads, numBlocks;
        computeOptimalThreadsAndBlocks(m_counts.numParticles,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);
        moveParticles_Kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                        m_rigidBody->getData(),
                                                        m_position.getData(),
                                                        m_quaternion.getData(),
                                                        m_velocity.getData(),
                                                        m_torce.getData(),
                                                        m_bodyTag.getData(),
                                                        m_counts.numObstacles,
                                                        m_counts.numParticles);
    }
    updateSubBodyPositions();
}

// -------------------------------------------------------------------------------------------------
// Performs the second velocity half-kick (KDK Step 3; no-op for single-pass schemes).
template <typename T, MemType M>
void ComponentManager<T, M>::advanceVelocity(const GrainsMemBuffer<TimeIntegrator<T>*, M>& TI)
{
    if constexpr(M == MemType::HOST)
    {
        for(uint pID = m_counts.numObstacles; pID < m_counts.numObstacles + m_counts.numParticles;
            ++pID)
        {
            if(isSubBody(m_bodyTag[pID]) && getSubBodyLocalIdx(m_bodyTag[pID]) != 0u)
                continue;
            advanceVelocity_common(TI.getData(),
                                   m_rigidBody->getData(),
                                   m_quaternion.getData(),
                                   m_velocity.getData(),
                                   m_torce.getData(),
                                   pID);
        }
    }
    else
    {
        uint numThreads, numBlocks;
        computeOptimalThreadsAndBlocks(m_counts.numParticles,
                                       GrainsParameters<T>::m_GPU,
                                       numBlocks,
                                       numThreads);
        advanceVelocity_Kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                          m_rigidBody->getData(),
                                                          m_quaternion.getData(),
                                                          m_velocity.getData(),
                                                          m_torce.getData(),
                                                          m_bodyTag.getData(),
                                                          m_counts.numObstacles,
                                                          m_counts.numParticles);
    }
}

// -------------------------------------------------------------------------------------------------
// Slaves non-master sub-body world transforms to their composite master.
template <typename T, MemType M>
void ComponentManager<T, M>::updateSubBodyPositions()
{
    if(m_counts.numSubBodies == 0)
        return;

    if constexpr(M == MemType::HOST)
    {
        const uint nTotal = m_counts.numObstacles + m_counts.numParticles;
        for(uint cID = m_counts.numObstacles; cID < nTotal; ++cID)
        {
            const uint tag = m_bodyTag[cID];
            if(!isSubBody(tag) || getSubBodyLocalIdx(tag) == 0u)
                continue;
            const uint mSlot  = m_masterSlot[getCompositeIdx(tag)];
            m_position[cID]   = m_position[mSlot] + (m_quaternion[mSlot] >> m_localPos[cID]);
            m_quaternion[cID] = m_quaternion[mSlot] * m_localQuat[cID];
            T qn              = norm(m_quaternion[cID]);
            if(qn > T(1e-12))
                m_quaternion[cID] *= (T(1) / qn);
        }
    }
    else
    {
        const uint nTotal = m_counts.numObstacles + m_counts.numParticles;
        uint       numThreads, numBlocks;
        computeOptimalThreadsAndBlocks(nTotal, GrainsParameters<T>::m_GPU, numBlocks, numThreads);
        updateSubBodyPositions_Kernel<<<numBlocks, numThreads>>>(m_position.getData(),
                                                                 m_quaternion.getData(),
                                                                 m_localPos.getData(),
                                                                 m_localQuat.getData(),
                                                                 m_masterSlot.getData(),
                                                                 m_bodyTag.getData(),
                                                                 nTotal);
    }
}

// --------------------------------------------------------------------------------------------------
// Explicit instantiations
template class ComponentManager<float, MemType::HOST>;
template class ComponentManager<double, MemType::HOST>;
template class ComponentManager<float, MemType::DEVICE>;
template class ComponentManager<double, MemType::DEVICE>;
