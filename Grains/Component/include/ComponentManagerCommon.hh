#ifndef _COMPONENTMANAGERCOMMON_HH_
#define _COMPONENTMANAGERCOMMON_HH_

#include "CollisionDetection.hh"
#include "ContactForceModel.hh"
#include "ContactForceModelFactory.hh"
#include "ContactTable.hh"
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
    // We want B expressed in A-local coordinates: b2a = inv(A2W) ∘ B2W.
    // setToTransformsComposition(t1, t2) computes: this = t2 ∘ t1.
    invA.setToInverseTransform(transform[idA]);
    relativeTransform[pairID].setToTransformsComposition(transform[idB], invA);
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
/** @brief Transforms contact info from A-local to world.
    @param pairList list of contact pairs
    @param position world positions of components
    @param quaternion world orientations of components
    @param contactInfoLocal CI computed in A-local frame (input)
    @param contactInfoWorld CI written in world frame (output, only for actives)
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void transformContactInfo_common(const uint2*         pairList,
                                                              const Vector3<T>*    position,
                                                              const Quaternion<T>* quaternion,
                                                              ContactInfo<T>*      contactInfoLocal,
                                                              ContactInfo<T>*      contactInfoWorld,
                                                              const uint           pairID)
{
    // Transform point and vector from A-local to world using A's pose
    uint                              idA      = pairList[pairID].x;
    const Quaternion<T>&              qA       = quaternion[idA];
    const ContactInfo<T>              ciL      = contactInfoLocal[pairID];
    typename ContactInfo<T>::Snapshot snapshot = ciL.getSnapshot();
    snapshot.contactPoint                      = qA >> snapshot.contactPoint + position[idA];
    snapshot.contactVector                     = qA >> snapshot.contactVector;
    contactInfoWorld[pairID].setSnapshot(snapshot);
}

// -------------------------------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of pairs
    @param contactInfo contact information in the world frame
    @param position position of the components
    @param velocity kinematics of the components
    @param torce torce acting on the components
    @param contactMemory view of contact memory (hash table + history data)
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void computeContactForces_common(const ContactForceModel<T>* const* CF,
                                                              const uint2*          pairList,
                                                              const ContactInfo<T>* contactInfo,
                                                              const Vector3<T>*     position,
                                                              const Kinematics<T>*  velocity,
                                                              Torce<T>*             torce,
                                                              ContactMemoryView<T>  contactMemory,
                                                              const uint            pairID)
{
    // Contact Details
    ContactInfo<T> ci = contactInfo[pairID];
    // one load of metadata
    typename ContactInfo<T>::Snapshot snapshot = ci.getSnapshot();
    bool isContact = snapshot.overlapDistance < T(0);  // is in contact/negative distance

    // Compute the forces
    if(isContact)  // On device path, this check is redundant.
    {
        const uint2 pair = pairList[pairID];
        const uint  idA  = pair.x;
        const uint  idB  = pair.y;

        // velocities of the components
        const Kinematics<T>& vA(velocity[idA]);
        const Kinematics<T>& vB(velocity[idB]);
        // geometric point of contact
        const Vector3<T>& contactPt(ci.getContactPoint());
        // relative velocity at contact point
        const Vector3<T>& relVel(vA.kinematicsAtPoint(contactPt - position[idA])
                                 - vB.kinematicsAtPoint(contactPt - position[idB]));
        // relative angular velocity
        const Vector3<T>& relAngVel(vA.getAngularComponent() - vB.getAngularComponent());

        // Look up or create contact history entry
        ContactHistory<T>* historyPtr = nullptr;
        if(contactMemory.m_historyData != nullptr)
        {
            uint historyIndex;
            contactMemory.findOrInsert(pair, historyIndex);
            historyPtr = &(contactMemory.m_historyData[historyIndex]);
        }

        // note that we will add torce to obstacles as well.
        uint contactForceID = snapshot.contactHash;
        CF[contactForceID]->computeForces(ci,
                                          relVel,
                                          relAngVel,
                                          position[idA],
                                          position[idB],
                                          historyPtr,
                                          torce[idA],
                                          torce[idB]);

        // printf("Contact Point: (%f, %f, %f), Normal: (%f, %f, %f), Overlap Distance: %f\n",
        //        snapshot.contactPoint[X],
        //        snapshot.contactPoint[Y],
        //        snapshot.contactPoint[Z],
        //        snapshot.contactVector[X],
        //        snapshot.contactVector[Y],
        //        snapshot.contactVector[Z],
        //        snapshot.overlapDistance);
        // printf("Pair %u: relVel = (%f, %f, %f), relAngVel = (%f, %f, %f), delTorceA = (%f, %f,
        // %f, "
        //        "%f, %f, %f), delTorceB = (%f, %f, %f, %f, %f, %f)\n",
        //        pairID,
        //        relVel[X],
        //        relVel[Y],
        //        relVel[Z],
        //        relAngVel[X],
        //        relAngVel[Y],
        //        relAngVel[Z],
        //        torce[idA].getForce()[X],
        //        torce[idA].getForce()[Y],
        //        torce[idA].getForce()[Z],
        //        torce[idA].getTorque()[X],
        //        torce[idA].getTorque()[Y],
        //        torce[idA].getTorque()[Z],
        //        torce[idB].getForce()[X],
        //        torce[idB].getForce()[Y],
        //        torce[idB].getForce()[Z],
        //        torce[idB].getTorque()[X],
        //        torce[idB].getTorque()[Y],
        //        torce[idB].getTorque()[Z]);
    }
    // reset the distance so we don't compute the torce twice
    ci.setOverlapDistance(T(0));
}

// -------------------------------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of pairs
    @param contactInfo contact information in the world frame
    @param position position of the components
    @param velocity kinematics of the components
    @param torceA torce acting on the first component
    @param torceB torce acting on the second component
    @param contactMemory view of contact memory (hash table + history data)
    @param pairID ID of the pair */
template <typename T>
__HOSTDEVICE__ static INLINE void computeContactForces_common(const ContactForceModel<T>* const* CF,
                                                              const uint2*          pairList,
                                                              const ContactInfo<T>* contactInfo,
                                                              const Vector3<T>*     position,
                                                              const Kinematics<T>*  velocity,
                                                              Torce<T>*             torceA,
                                                              Torce<T>*             torceB,
                                                              ContactMemoryView<T>  contactMemory,
                                                              const uint            pairID)
{
    // Contact Details
    ContactInfo<T> ci = contactInfo[pairID];
    // one load of metadata
    typename ContactInfo<T>::Snapshot snapshot = ci.getSnapshot();
    bool isContact = snapshot.overlapDistance < T(0);  // is in contact/negative distance

    // Compute the forces
    if(isContact)  // On device path, this check is redundant.
    {
        const uint2 pair = pairList[pairID];
        const uint  idA  = pair.x;
        const uint  idB  = pair.y;

        // velocities of the components
        const Kinematics<T>& vA(velocity[idA]);
        const Kinematics<T>& vB(velocity[idB]);
        // geometric point of contact
        const Vector3<T>& contactPt(ci.getContactPoint());
        // relative velocity at contact point
        const Vector3<T>& relVel(vA.kinematicsAtPoint(contactPt - position[idA])
                                 - vB.kinematicsAtPoint(contactPt - position[idB]));
        // relative angular velocity
        const Vector3<T>& relAngVel(vA.getAngularComponent() - vB.getAngularComponent());

        // Look up or create contact history entry
        ContactHistory<T>* historyPtr = nullptr;
        if(contactMemory.m_historyData != nullptr)
        {
            uint historyIndex;
            contactMemory.findOrInsert(pair, historyIndex);
            historyPtr = &(contactMemory.m_historyData[historyIndex]);
        }

        // note that we will add torce to obstacles as well.
        uint contactForceID = snapshot.contactHash;
        CF[contactForceID]->computeForces(ci,
                                          relVel,
                                          relAngVel,
                                          position[idA],
                                          position[idB],
                                          historyPtr,
                                          torceA[pairID],
                                          torceB[pairID]);

        // printf("Contact Point: (%f, %f, %f), Normal: (%f, %f, %f), Overlap Distance: %f\n",
        //        snapshot.contactPoint[X],
        //        snapshot.contactPoint[Y],
        //        snapshot.contactPoint[Z],
        //        snapshot.contactVector[X],
        //        snapshot.contactVector[Y],
        //        snapshot.contactVector[Z],
        //        snapshot.overlapDistance);
        // printf("Pair %u: relVel = (%f, %f, %f), relAngVel = (%f, %f, %f), delTorceA = (%f, %f,
        // %f, "
        //        "%f, %f, %f), delTorceB = (%f, %f, %f, %f, %f, %f)\n",
        //        pairID,
        //        relVel[X],
        //        relVel[Y],
        //        relVel[Z],
        //        relAngVel[X],
        //        relAngVel[Y],
        //        relAngVel[Z],
        //        torceA[pairID].getForce()[X],
        //        torceA[pairID].getForce()[Y],
        //        torceA[pairID].getForce()[Z],
        //        torceA[pairID].getTorque()[X],
        //        torceA[pairID].getTorque()[Y],
        //        torceA[pairID].getTorque()[Z],
        //        torceB[pairID].getForce()[X],
        //        torceB[pairID].getForce()[Y],
        //        torceB[pairID].getForce()[Z],
        //        torceB[pairID].getTorque()[X],
        //        torceB[pairID].getTorque()[Y],
        //        torceB[pairID].getTorque()[Z]);
    }
    // reset the distance so we don't compute the torce twice
    ci.setOverlapDistance(T(0));
}

// -------------------------------------------------------------------------------------------------
/** @brief Reduces per-pair intermediate torces to per-particle torces (DEVICE kernel helper)
    @param pairList list of pairs
    @param intermediateTorceA intermediate torce storage for particle A in each pair
    @param intermediateTorceB intermediate torce storage for particle B in each pair
    @param torce final per-particle torce array (accumulated atomically)
    @param pairID ID of the pair */
template <typename T>
__device__ static INLINE void reduceTorces_common(const uint2*    pairList,
                                                  const Torce<T>* intermediateTorceA,
                                                  const Torce<T>* intermediateTorceB,
                                                  Torce<T>*       torce,
                                                  const uint      pairID)
{
    const uint2     pair = pairList[pairID];
    const uint      idA  = pair.x;
    const uint      idB  = pair.y;
    const Torce<T>& tA   = intermediateTorceA[pairID];
    const Torce<T>& tB   = intermediateTorceB[pairID];

    // Atomically accumulate all components (6 atomics per particle)
    torce[idA].addTorceAtomic(tA);
    torce[idB].addTorceAtomic(tB);
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
    T qn = norm(quaternion[cID]);
    if(qn > EPS<T>)
        quaternion[cID] *= (T(1) / qn);
}

// -------------------------------------------------------------------------------------------------
/** @brief Performs the second velocity half-kick for split-step schemes (KDK leapfrog Step 3).
    Computes the acceleration from the current torce (forces at x_{n+1}) and delegates to
    TI[0]->AdvanceVelocity. The torce is intentionally NOT reset here so that it remains
    available as a_n for the next call to moveParticles_common.
    @param TI the time integrator
    @param rigidBody the rigid body of the components
    @param quaternion the quaternion of the component
    @param kinematics the kinematics of the component (velocity updated in-place)
    @param torce the accumulated torce at x_{n+1} (read but not reset)
    @param cID the ID of the component */
template <typename T>
__HOSTDEVICE__ static INLINE void advanceVelocity_common(const TimeIntegrator<T>* const* TI,
                                                         const RigidBody<T>* const*      rigidBody,
                                                         const Quaternion<T>*            quaternion,
                                                         Kinematics<T>*                  kinematics,
                                                         const Torce<T>*                 torce,
                                                         const uint                      cID)
{
    const RigidBody<T>* rb = rigidBody[cID];
    // Compute acceleration from forces at x_{n+1} — torce is NOT reset
    const Kinematics<T> acceleration
        = rb->computeMomentum(kinematics[cID].getAngularComponent(), torce[cID], quaternion[cID]);
    TI[0]->AdvanceVelocity(acceleration, kinematics[cID]);
}

#endif