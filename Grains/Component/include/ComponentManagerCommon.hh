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
    @param position position of the components
    @param quaternion quaternion of the components
    @param relativePosition output relative position of the components
    @param relativeQuaternion output relative quaternion of the components
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
    const uint2 pair         = pairList[pairID];
    const uint  idA          = pair.x;
    const uint  idB          = pair.y;
    relativePosition[pairID] = quaternion[idA]
                               << (position[idB] - position[idA]);
    relativeQuaternion[pairID] = inverse(quaternion[idA]) * quaternion[idB];
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between components
    @param pairList list of contact pairs
    @param rigidBody rigid body
    @param relPosition relative position of the components
    @param relQuaternion relative quaternion of the components
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    detectCollisionsComponents_common(const uint2*               pairList,
                                      const RigidBody<T>* const* rigidBody,
                                      const Vector3<T>*          relPosition,
                                      const Quaternion<T>*       relQuaternion,
                                      ContactInfo<T>*            contactInfo,
                                      const uint                 pairID)
{
    const uint2          pair  = pairList[pairID];
    const uint           idA   = pair.x;
    const uint           idB   = pair.y;
    const RigidBody<T>&  rbA   = *(rigidBody[idA]);
    const RigidBody<T>&  rbB   = *(rigidBody[idB]);
    const Vector3<T>&    v_b2a = relPosition[pairID];
    const Quaternion<T>& q_b2a = relQuaternion[pairID];
    closestPointsRigidBodies(rbA, rbB, v_b2a, q_b2a, contactInfo[pairID]);
}

// -----------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of pairs
    @param contactInfo contact information
    @param rigidBody rigid body of components
    @param velocity kinematics of the components
    @param torce torce acting on the components
    @param relPosition relative position of the components
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    computeContactForces_common(const ContactForceModel<T>* const* CF,
                                const uint2*                       pairList,
                                const ContactInfo<T>*              contactInfo,
                                const RigidBody<T>* const*         rigidBody,
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
        const RigidBody<T>* rbA       = rigidBody[idA];
        const uint          materialA = rbA->getMaterial();
        const T             massA     = rbA->getMass();
        const RigidBody<T>* rbB       = rigidBody[idB];
        const uint          materialB = rbB->getMaterial();
        const T             massB     = rbB->getMass();
        // CF ID given materialIDs
        uint contactForceID
            = ContactForceModelFactory<T>::computeHash(materialA, materialB);
        // velocities of the components
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
        // note that we will add torce to obstacles as well.
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
/** @brief Adds gravity to the component
    @param g the gravitational acceleration vector
    @param rigidBody the rigid body of the component
    @param torce the torce acting on the component
    @param cID the ID of the component */
template <typename T>
__HOSTDEVICE__ static INLINE void
    addExternalForces_common(const Vector3<T>&          g,
                             const RigidBody<T>* const* rigidBody,
                             Torce<T>*                  torce,
                             const uint                 cID)
{
    const RigidBody<T>* rb   = rigidBody[cID];
    const T             mass = rb->getMass();
    // Adding the gravitational force to the torce
    torce[cID].addForce(mass * g);
}

// -----------------------------------------------------------------------------
/** @brief Moves a component using the given time integration method
    @param TI the time integrator
    @param rigidBody the rigid body of the components
    @param position the position of the component
    @param quaternion the quaternion of the component
    @param kinematics the kinematics of the component
    @param torce the torce acting on the component
    @param cID the ID of the component */
template <typename T>
__HOSTDEVICE__ static INLINE void
    moveParticles_common(const TimeIntegrator<T>* const* TI,
                         const RigidBody<T>* const*      rigidBody,
                         Vector3<T>*                     position,
                         Quaternion<T>*                  quaternion,
                         Kinematics<T>*                  kinematics,
                         Torce<T>*                       torce,
                         const uint                      cID)
{
    // Rigid body
    const RigidBody<T>* rb = rigidBody[cID];
    // Computing momentums in the space-fixed coordinate
    const Kinematics<T>& momentum
        = rb->computeMomentum(kinematics[cID].getAngularComponent(),
                              torce[cID],
                              quaternion[cID]);
    // Reset torces
    torce[cID].reset();
    // Finally, we move particles using the given time integration
    Vector3<T>    transMotion;
    Quaternion<T> rotMotion;
    TI[0]->Move(momentum, kinematics[cID], transMotion, rotMotion);

    position[cID] += transMotion;
    quaternion[cID] *= rotMotion;
}

#endif