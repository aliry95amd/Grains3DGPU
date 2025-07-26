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
/** @brief Computes relative transformations per pair
    @param pairList list of rigid bodies pairs
    @param position position of the rigid bodies
    @param quaternion quaternion of the rigid bodies
    @param relativePosition output relative position of the rigid bodies
    @param relativeQuaternion output relative quaternion of the rigid bodies
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    computeRelativeTransformations_common(const uint2*         pairList,
                                          const Vector3<T>*    position,
                                          const Quaternion<T>* quaternion,
                                          Vector3<T>*          relativePosition,
                                          Quaternion<T>* relativeQuaternion,
                                          const uint     pairID)
{
    const uint2 pair           = pairList[pairID];
    const uint  idA            = pair.x;
    const uint  idB            = pair.y;
    relativePosition[pairID]   = position[idB] - position[idA];
    relativeQuaternion[pairID] = quaternion[idB] * inverse(quaternion[idA]);
    // TODO: apply crust thickness
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between particles and obstacles
    @param pairList list of rigid bodies pairs
    @param particleRB rigid body of particles
    @param obstacleRB rigid body of obstacles
    @param position position of the particles
    @param obstaclePosition position of the obstacles
    @param quaternion quaternion of the particles
    @param obstacleQuaternion quaternion of the obstacles
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    detectCollisionsObstacles_common(const uint2*               pairList,
                                     const RigidBody<T>* const* particleRB,
                                     const RigidBody<T>* const* obstacleRB,
                                     const Vector3<T>*          position,
                                     const Vector3<T>*    obstaclePosition,
                                     const Quaternion<T>* quaternion,
                                     const Quaternion<T>* obstacleQuaternion,
                                     ContactInfo<T>*      contactInfo,
                                     const uint           pairID)
{
    const uint2          pair  = pairList[pairID];
    const uint           idA   = pair.x;
    const uint           idB   = pair.y;
    const RigidBody<T>&  rbA   = *(particleRB[idA]);
    const RigidBody<T>&  rbB   = *(obstacleRB[idB]);
    const Vector3<T>&    posA  = position[idA];
    const Vector3<T>&    posB  = obstaclePosition[idB];
    const Quaternion<T>& quatA = quaternion[idA];
    const Quaternion<T>& quatB = obstacleQuaternion[idB];
    closestPointsRigidBodies(rbA,
                             rbB,
                             posA,
                             posB,
                             quatA,
                             quatB,
                             contactInfo[pairID]);
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between particles and particles
    @param pairList list of rigid bodies pairs
    @param particleRB rigid body of particles
    @param position relative position of the particles
    @param quaternion relative quaternion of the particles
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    detectCollisionsParticles_common(const uint2*               pairList,
                                     const RigidBody<T>* const* particleRB,
                                     const Vector3<T>*          position,
                                     const Quaternion<T>*       quaternion,
                                     ContactInfo<T>*            contactInfo,
                                     const uint                 pairID)
{
    const uint2          pair  = pairList[pairID];
    const uint           idA   = pair.x;
    const uint           idB   = pair.y;
    const RigidBody<T>&  rbA   = *(particleRB[idA]);
    const RigidBody<T>&  rbB   = *(particleRB[idB]);
    const Vector3<T>&    v_b2a = position[pairID];
    const Quaternion<T>& q_b2a = quaternion[pairID];
    closestPointsRigidBodies(rbA, rbB, v_b2a, q_b2a, contactInfo[pairID]);
}

// -----------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of rigid bodies pairs
    @param contactInfo contact information
    @param particleRB rigid body of particles
    @param velocity kinematics of the particles
    @param torce torce acting on the particles
    @param relPosition relative position of the particles
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    computeContactForces_common(const ContactForceModel<T>* const* CF,
                                const uint2*                       pairList,
                                const ContactInfo<T>*              contactInfo,
                                const RigidBody<T>* const*         particleRB,
                                const Kinematics<T>*               velocity,
                                Torce<T>*                          torce,
                                const Vector3<T>*                  relPosition,
                                const uint                         pairID)
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
                                          relPosition[pairID],
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
    @param position the position of the particle
    @param quaternion the quaternion of the particle
    @param kinematics the kinematics of the particle
    @param torce the torce acting on the particle
    @param rigidBodyId the rigid body ID of the particle
    @param pID the ID of the particle */
template <typename T>
__HOSTDEVICE__ static INLINE void
    moveParticles_common(const TimeIntegrator<T>* const* TI,
                         const RigidBody<T>* const*      particleRB,
                         Vector3<T>*                     position,
                         Quaternion<T>*                  quaternion,
                         Kinematics<T>*                  kinematics,
                         Torce<T>*                       torce,
                         const uint*                     rigidBodyId,
                         const uint                      pID)
{
    // Rigid body
    const RigidBody<T>* rb = particleRB[pID];
    // Computing momentums in the space-fixed coordinate
    const Kinematics<T>& momentum
        = rb->computeMomentum(kinematics[pID].getAngularComponent(),
                              torce[pID],
                              quaternion[pID]);
    // Reset torces
    torce[pID].reset();
    // Finally, we move particles using the given time integration
    Vector3<T>    transMotion;
    Quaternion<T> rotMotion;
    TI[0]->Move(momentum, kinematics[pID], transMotion, rotMotion);

    position[pID] += transMotion;
    quaternion[pID] *= rotMotion;
}

#endif