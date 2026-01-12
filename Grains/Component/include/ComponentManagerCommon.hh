#ifndef _COMPONENTMANAGERCOMMON_HH_
#define _COMPONENTMANAGERCOMMON_HH_

#include "CollisionDetection.hh"
#include "ContactForceModel.hh"
#include "ContactForceModelFactory.hh"
#include "GrainsParameters.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "QuaternionMath.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"
#include "Torce.hh"
#include "Transform3.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief ComponentManager common functions between host and device.

    This is a header-only file that contains common functions between CPU/GPU
    for the ComponentManager class. The functions are templated to allow
    for flexibility in usage. The functions are marked as inline to allow for
    better optimization by the compiler.

    @author A.Yazdani - 2025 - Construction */
// =================================================================================================
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
                                          Quaternion<T>*       relativeQuaternion,
                                          const uint           pairID)
{
    const uint2 pair           = pairList[pairID];
    const uint  idA            = pair.x;
    const uint  idB            = pair.y;
    relativePosition[pairID]   = quaternion[idA] << (position[idB] - position[idA]);
    relativeQuaternion[pairID] = inverse(quaternion[idA]) * quaternion[idB];
}

// -------------------------------------------------------------------------------------------------
/** @brief Detects collisions between components
    @param pairList list of contact pairs
    @param rigidBody rigid body
    @param relPosition relative position of the components
    @param relQuaternion relative quaternion of the components
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T, GJKType GJKVARIANT = GJKType::JOHNSON, bool GJKACC = false>
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
    closestPointsRigidBodies<T, GJKVARIANT, GJKACC>(rbA, rbB, v_b2a, q_b2a, contactInfo[pairID]);
}

// -------------------------------------------------------------------------------------------------
/** @brief Detects collisions between components using global coordinates
    @param pairList list of contact pairs
    @param rigidBody rigid body
    @param position global position of the components
    @param quaternion global quaternion of the components
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T, GJKType GJKVARIANT = GJKType::JOHNSON, bool GJKACC = false>
__HOSTDEVICE__ static INLINE void
    detectCollisionsComponentsGlobal_common(const uint2*               pairList,
                                            const RigidBody<T>* const* rigidBody,
                                            const Vector3<T>*          position,
                                            const Quaternion<T>*       quaternion,
                                            ContactInfo<T>*            contactInfo,
                                            const uint                 pairID)
{
    const uint2          pair  = pairList[pairID];
    const uint           idA   = pair.x;
    const uint           idB   = pair.y;
    const RigidBody<T>&  rbA   = *(rigidBody[idA]);
    const RigidBody<T>&  rbB   = *(rigidBody[idB]);
    const Vector3<T>&    v_a2w = position[idA];
    const Vector3<T>&    v_b2w = position[idB];
    const Quaternion<T>& q_a2w = quaternion[idA];
    const Quaternion<T>& q_b2w = quaternion[idB];
    closestPointsRigidBodies<T, GJKVARIANT, GJKACC>(rbA,
                                                    rbB,
                                                    v_a2w,
                                                    v_b2w,
                                                    q_a2w,
                                                    q_b2w,
                                                    contactInfo[pairID]);
}

// -------------------------------------------------------------------------------------------------
/** @brief Computes relative transformations per pair using Transform3
    @param pairList list of rigid bodies pairs
    @param transform transforms of the components
    @param relativeTransform output relative transforms of the components
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void
    computeRelativeTransformations_common(const uint2*         pairList,
                                          const Transform3<T>* transform,
                                          Transform3<T>*       relativeTransform,
                                          const uint           pairID)
{
    const uint2   pair = pairList[pairID];
    const uint    idA  = pair.x;
    const uint    idB  = pair.y;
    Transform3<T> invA;
    invA.setToInverseTransform(transform[idA]);
    relativeTransform[pairID].setToTransformsComposition(invA, transform[idB]);
}

// -------------------------------------------------------------------------------------------------
/** @brief Detects collisions between components using Transform3 (relative)
    @param pairList list of contact pairs
    @param rigidBody rigid body
    @param relTransform relative transforms of the components
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T, GJKType GJKVARIANT = GJKType::JOHNSON, bool GJKACC = false>
__HOSTDEVICE__ static INLINE void
    detectCollisionsComponents_common(const uint2*               pairList,
                                      const RigidBody<T>* const* rigidBody,
                                      const Transform3<T>*       relTransform,
                                      ContactInfo<T>*            contactInfo,
                                      const uint                 pairID)
{
    const uint2          pair = pairList[pairID];
    const uint           idA  = pair.x;
    const uint           idB  = pair.y;
    const RigidBody<T>&  rbA  = *(rigidBody[idA]);
    const RigidBody<T>&  rbB  = *(rigidBody[idB]);
    const Transform3<T>& b2a  = relTransform[pairID];
    closestPointsRigidBodies<T, GJKVARIANT, GJKACC>(rbA, rbB, b2a, contactInfo[pairID]);
}

// -------------------------------------------------------------------------------------------------
/** @brief Detects collisions between components using Transform3 (global coordinates)
    @param pairList list of contact pairs
    @param rigidBody rigid body
    @param transform global transforms of the components
    @param contactInfo contact information
    @param pairID ID of the pair */
template <typename T, GJKType GJKVARIANT = GJKType::JOHNSON, bool GJKACC = false>
__HOSTDEVICE__ static INLINE void
    detectCollisionsComponentsGlobal_common(const uint2*               pairList,
                                            const RigidBody<T>* const* rigidBody,
                                            const Transform3<T>*       transform,
                                            ContactInfo<T>*            contactInfo,
                                            const uint                 pairID)
{
    const uint2          pair = pairList[pairID];
    const uint           idA  = pair.x;
    const uint           idB  = pair.y;
    const RigidBody<T>&  rbA  = *(rigidBody[idA]);
    const RigidBody<T>&  rbB  = *(rigidBody[idB]);
    const Transform3<T>& a2w  = transform[idA];
    const Transform3<T>& b2w  = transform[idB];
    closestPointsRigidBodies<T, GJKVARIANT, GJKACC>(rbA, rbB, a2w, b2w, contactInfo[pairID]);
}

// -------------------------------------------------------------------------------------------------
/** @brief Flags active contacts and transforms CI from A-local to world.
    @param pairList list of contact pairs
    @param position world positions of components
    @param quaternion world orientations of components
    @param contactInfoLocal CI computed in A-local frame (input)
    @param contactInfoWorld CI written in world frame (output, only for actives)
    @param active flag buffer (1 if active/contact, else 0)
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void transformContactInfo_common(const uint2*         pairList,
                                                              const Vector3<T>*    position,
                                                              const Quaternion<T>* quaternion,
                                                              ContactInfo<T>*      contactInfoLocal,
                                                              ContactInfo<T>*      contactInfoWorld,
                                                              uint*                active,
                                                              const uint           pairID)
{
    ContactInfo<T>& ciL = contactInfoLocal[pairID];
    active[pairID]      = (ciL.getOverlapDistance() < T(0)) ? 1 : 0;

    // Transform point and vector from A-local to world using A's pose
    const uint           idA = pairList[pairID].x;
    ContactInfo<T>&      ciW = contactInfoWorld[pairID];
    const Quaternion<T>& qA  = quaternion[idA];
    ciW.setContactPoint((qA >> ciL.getContactPoint()) + position[idA]);
    ciW.setContactVector((qA >> ciL.getContactVector()));
    ciW.setOverlapDistance(ciL.getOverlapDistance());

    // reset the distance so we don't compute the torce twice
    ciL.setOverlapDistance(T(0));
}

// -------------------------------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of pairs
    @param contactInfo contact information in the world frame
    @param rigidBody rigid body of components
    @param position position of the components
    @param velocity kinematics of the components
    @param torce torce acting on the components
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void computeContactForces_common(const ContactForceModel<T>* const* CF,
                                                              const uint2*          pairList,
                                                              const ContactInfo<T>* contactInfo,
                                                              const RigidBody<T>* const* rigidBody,
                                                              const Vector3<T>*          position,
                                                              const Kinematics<T>*       velocity,
                                                              Torce<T>*                  torce,
                                                              const uint                 pairID)
{
    ContactInfo<T>& ci = const_cast<ContactInfo<T>&>(contactInfo[pairID]);
    // Compute the forces
    // On device path, this is redundant.
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
        uint contactForceID = ContactForceModelFactory<T>::computeHash(materialA, materialB);
        // velocities of the components
        const Kinematics<T>& vA(velocity[idA]);
        const Kinematics<T>& vB(velocity[idB]);
        // geometric point of contact
        const Vector3<T>& contactPt(ci.getContactPoint());
        // relative velocity at contact point
        const Vector3<T>& relVel(vA.kinematicsAtPoint(contactPt) - vB.kinematicsAtPoint(contactPt));
        // relative angular velocity
        const Vector3<T>& relAngVel(vA.getAngularComponent() - vB.getAngularComponent());
        // note that we will add torce to obstacles as well.
        CF[contactForceID]->computeForces(ci,
                                          relVel,
                                          relAngVel,
                                          position[idA],
                                          position[idB],
                                          massA,
                                          massB,
                                          torce[idA],
                                          torce[idB]);
    }
    // reset the distance so we don't compute the torce twice
    ci.setOverlapDistance(T(0));
}

// -------------------------------------------------------------------------------------------------
/** @brief Adds gravity to the component
    @param g the gravitational acceleration vector
    @param rigidBody the rigid body of the component
    @param torce the torce acting on the component
    @param cID the ID of the component */
template <typename T>
__HOSTDEVICE__ static INLINE void addExternalForces_common(const Vector3<T>&          g,
                                                           const RigidBody<T>* const* rigidBody,
                                                           Torce<T>*                  torce,
                                                           const uint                 cID)
{
    const RigidBody<T>* rb   = rigidBody[cID];
    const T             mass = rb->getMass();
    // Adding the gravitational force to the torce
    torce[cID].addForce(mass * g);
}

// -------------------------------------------------------------------------------------------------
/** @brief Moves a component using the given time integration method
    @param TI the time integrator
    @param rigidBody the rigid body of the components
    @param position the position of the component
    @param quaternion the quaternion of the component
    @param kinematics the kinematics of the component
    @param torce the torce acting on the component
    @param cID the ID of the component */
template <typename T>
__HOSTDEVICE__ static INLINE void moveParticles_common(const TimeIntegrator<T>* const* TI,
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
        = rb->computeMomentum(kinematics[cID].getAngularComponent(), torce[cID], quaternion[cID]);
    // Reset torces
    torce[cID].reset();
    // Finally, we move particles using the given time integration
    Vector3<T>    transMotion;
    Quaternion<T> rotMotion;
    TI[0]->Move(momentum, kinematics[cID], transMotion, rotMotion);

    position[cID] += transMotion;
    quaternion[cID] *= rotMotion;

    const T* rotMotionBuffer = rotMotion.getBuffer();
}

#endif