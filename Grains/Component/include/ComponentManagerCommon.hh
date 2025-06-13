#ifndef _COMPONENTMANAGERCOMMON_HH_
#define _COMPONENTMANAGERCOMMON_HH_

#include "CollisionDetection.hh"
#include "ContactForceModel.hh"
#include "ContactForceModelFactory.hh"
#include "GrainsParameters.hh"
#include "Kinematics.hh"
#include "LinkedCell.hh"
#include "Quaternion.hh"
#include "QuaternionMath.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"
#include "Torce.hh"
#include "Transform3.hh"
#include "Vector3.hh"

// =============================================================================
/** @brief ComponentManager common functions between host and device.

    This is a header-only file that contains common functions between CPU/GPU
    for the ComponentManager class. The functions are templated to allow
    for flexibility in usage. The functions are marked as inline to allow for 
    better optimization by the compiler.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name ComponentManager common functions between host and device */
//@{
/** @brief Initializes a buffer to default 
    @param buffer the buffer to be initialized
    @param size the size of the buffer */
template <typename T>
__GLOBAL__ void initDefault_Kernel(T* buffer, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= size)
        return;
    buffer[idx] = T();
}

// -----------------------------------------------------------------------------
/** @brief Initializes a buffer to default
    @param buffer the buffer to be initialized
    @param size the size of the buffer */
template <typename T, MemType M>
void initDefault(GrainsMemBuffer<T, M>& buffer, size_t size)
{
    buffer.reserve(size);
    if constexpr(M == MemType::HOST || M == MemType::PINNED)
    {
        for(size_t i = 0; i < size; ++i)
            buffer[i] = T();
    }
    else if constexpr(M == MemType::DEVICE)
    {
        initDefault_Kernel<<<(size + 255) / 256, 256>>>(buffer.getData(), size);
    }
}

// -----------------------------------------------------------------------------
/** @brief Computes relative transformations per pair
    @param pairList list of rigid bodies pairs
    @param transform transformation of the rigid bodies
    @param relativeTransform relative transformation of the rigid bodies
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    computeRelativeTransformations_common(const uint2*         pairList,
                                          const Transform3<T>* transform,
                                          Transform3<T>* relativeTransform,
                                          const uint     pairID)
{
    const uint2 pair          = pairList[pairID];
    const uint  idA           = pair.x;
    const uint  idB           = pair.y;
    relativeTransform[pairID] = transform[idB];
    relativeTransform[pairID].relativeToTransform(transform[idA]);
    // TODO: apply crust thickness
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between particles and obstacles
    @param pairList list of rigid bodies pairs
    @param particleRB rigid body of particles
    @param obstacleRB rigid body of obstacles
    @param transform transformation of the particles
    @param obstacleTransform transformation of the obstacles
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    detectCollisionsObstacles_common(const uint2*               pairList,
                                     const RigidBody<T>* const* particleRB,
                                     const RigidBody<T>* const* obstacleRB,
                                     const Transform3<T>*       transform,
                                     const Transform3<T>* obstacleTransform,
                                     ContactInfo<T>*      contactInfo,
                                     const uint           pairID)
{
    const uint2          pair = pairList[pairID];
    const uint           idA  = pair.x;
    const uint           idB  = pair.y;
    const RigidBody<T>&  rbA  = *(particleRB[idA]);
    const RigidBody<T>&  rbB  = *(obstacleRB[idB]);
    const Transform3<T>& trA  = transform[idA];
    const Transform3<T>& trB  = obstacleTransform[idB];
    closestPointsRigidBodies(rbA, rbB, trA, trB, contactInfo[pairID]);
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between particles and particles
    @param pairList list of rigid bodies pairs
    @param particleRB rigid body of particles
    @param transform transformation of the particles
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    detectCollisionsParticles_common(const uint2*               pairList,
                                     const RigidBody<T>* const* particleRB,
                                     const Transform3<T>*       transform,
                                     ContactInfo<T>*            contactInfo,
                                     const uint                 pairID)
{
    const uint2          pair = pairList[pairID];
    const uint           idA  = pair.x;
    const uint           idB  = pair.y;
    const RigidBody<T>&  rbA  = *(particleRB[idA]);
    const RigidBody<T>&  rbB  = *(particleRB[idB]);
    const Transform3<T>& tr   = transform[pairID];
    closestPointsRigidBodies(rbA, rbB, tr, contactInfo[pairID]);
}

// -----------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param pairList list of rigid bodies pairs
    @param particleRB rigid body of particles
    @param transform transformation of the particles
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    computeContactForces_common(ContactForceModel<T>**     CF,
                                const uint2*               pairList,
                                const ContactInfo<T>*      contactInfo,
                                const RigidBody<T>* const* particleRB,
                                const Kinematics<T>*       velocity,
                                Torce<T>*                  torce,
                                const Transform3<T>*       transform,
                                const uint                 pairID)
{
    const ContactInfo<T>& ci = contactInfo[pairID];
    // Compute the forces
    if(ci.getOverlapDistance() < T(0))
    {
        const uint2         pair      = pairList[pairID];
        const uint          idA       = pair.x;
        const uint          idB       = pair.y;
        const RigidBody<T>* rbA       = particleRB[idA];
        const uint          materialA = rbA->getMaterial();
        const T             massA     = rbA->getMass();
        const RigidBody<T>* rbB       = particleRB[idB];
        const uint          materialB = rbB->getMaterial();
        const T             massB     = rbB->getMass();
        // CF ID given materialIDs
        uint contactForceID
            = ContactForceModelFactory<T>::computeHash(materialA, materialB);
        // velocities of the particles
        const Kinematics<T>& vA(velocity[idA]);
        const Kinematics<T>& vB(velocity[idB]);
        // geometric point of contact
        const Vector3<T>& contactPt(ci.getContactPoint());
        // relative velocity at contact point
        const Vector3<T>& relVel(vA.kinematicsAtPoint(contactPt)
                                 - vB.kinematicsAtPoint(contactPt));
        // relative angular velocity
        const Vector3<T>& relAngVel(vA.getAngularComponent()
                                    - vB.getAngularComponent());
        CF[contactForceID]->computeForces(ci,
                                          relVel,
                                          relAngVel,
                                          massA,
                                          massB,
                                          transform[idA].getOrigin(),
                                          torce[idA],
                                          torce[idB]);
    }
}

// -----------------------------------------------------------------------------
/** @brief Adds gravity to the particle
    @param particleRB the rigid body of the particle
    @param rigidBodyId the rigid body ID of the particle
    @param g the gravitational acceleration vector
    @param torce the torce acting on the particle */
template <typename T>
__HOSTDEVICE__ static INLINE void
    addExternalForces_common(const Vector3<T>&          g,
                             const RigidBody<T>* const* particleRB,
                             Torce<T>*                  torce,
                             const uint                 pID)
{
    const RigidBody<T>* rb   = particleRB[pID];
    const T             mass = rb->getMass();
    // Adding the gravitational force to the torce
    torce[pID].addForce(mass * g);
}

// -----------------------------------------------------------------------------
/** @brief Moves a particle using the given time integration method
    @param TI the time integrator
    @param particleRB the rigid body of the particle
    @param transform the transformation of the particle
    @param kinematics the kinematics of the particle
    @param torce the torce acting on the particle
    @param rigidBodyId the rigid body ID of the particle
    @param pId the ID of the particle */
template <typename T>
__HOSTDEVICE__ static INLINE void
    moveParticles_common(const TimeIntegrator<T>* const* TI,
                         const RigidBody<T>* const*      particleRB,
                         Transform3<T>*                  transform,
                         Kinematics<T>*                  kinematics,
                         Torce<T>*                       torce,
                         const uint*                     rigidBodyId,
                         const uint                      pId)
{
    // // Rigid body
    // const RigidBody<T>* rb = particleRB[rigidBodyId[pId]];
    // // First, we compute quaternion of orientation
    // Quaternion<T> qRot(transform.getBasis());
    // // Computing momentums in the space-fixed coordinate
    // const Kinematics<T>& momentum
    //     = rb->computeMomentum(kinematics.getAngularComponent(), torce, qRot);
    // // Reset torces
    // torce.reset();
    // // Finally, we move particles using the given time integration
    // Vector3<T>    transMotion;
    // Quaternion<T> rotMotion;
    // TI[0]->Move(momentum, kinematics, transMotion, rotMotion);

    // // Quaternion and rotation quaternion conjugate
    // Vector3<T>    om = kinematics.getAngularComponent();
    // Quaternion<T> qRotCon(qRot.conjugate());
    // // Write torque in body-fixed coordinates system
    // Vector3<T> angAcc(qRot.multToVector3(om * qRotCon));
    // // and update the transformation of the component
    // transform.updateTransform(transMotion, rotMotion);
    // // TODO
    // // qRot = qRotChange * qRot;
    // // qRotChange = T( 0.5 ) * ( m_velocity[ pId ].getAngularComponent() * qRot );
}

// /** @brief Resets the particle configurations
// @param transform the transformation of the particle
// */
// template <typename T>
// __HOSTDEVICE__ static INLINE void resetParticle(
//     const Transform3<T>& transform,
//     Torce<T>& torce
// )
// {
//     torce.reset();
// }
#endif