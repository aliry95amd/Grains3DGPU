#ifndef _COLLISIONDETECTION_HH_
#define _COLLISIONDETECTION_HH_

#include "ContactInfo.hh"
#include "RigidBody.hh"
#include "Transform3.hh"

// =================================================================================================
/** @brief The header for Rigid bodies collision detections.

    Functions for collision detection between two rigid bodies.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
/** @name CollisionDetection : External methods */
//@{
/** @brief Returns whether 2 rigid bodies intersect - relative transformation.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_b2a position describing convex B in the A's reference frame
    @param q_b2a rotation describing convex B in the A's reference frame */
template <typename T>
__HOSTDEVICE__ bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                         const RigidBody<T>&  rbB,
                                         const Vector3<T>&    v_b2a,
                                         const Quaternion<T>& q_b2a);

/** @brief Returns whether 2 rigid bodies intersect.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_a2w position describing convex A in the world reference frame
    @param v_b2w position describing convex B in the world reference frame
    @param q_a2w rotation describing convex A in the world reference frame
    @param q_b2w rotation describing convex B in the world reference frame */
template <typename T>
__HOSTDEVICE__ bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                         const RigidBody<T>&  rbB,
                                         const Vector3<T>&    v_a2w,
                                         const Vector3<T>&    v_b2w,
                                         const Quaternion<T>& q_a2w,
                                         const Quaternion<T>& q_b2w);

/** @brief Returns the contact information (if any) for 2 rigid bodies - relative transformation.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_b2a position describing convex B in the A's reference frame
    @param q_b2a rotation describing convex B in the A's reference frame
    @param contactInfo output contact information */
template <typename T>
__HOSTDEVICE__ void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                             const RigidBody<T>&  rbB,
                                             const Vector3<T>&    v_b2a,
                                             const Quaternion<T>& q_b2a,
                                             ContactInfo<T>&      contactInfo);

/** @brief Returns the contact information (if any) for 2 rigid bodies.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_a2w position describing convex A in the world reference frame
    @param v_b2w position describing convex B in the world reference frame
    @param q_a2w rotation describing convex A in the world reference frame
    @param q_b2w rotation describing convex B in the world reference frame
    @param contactInfo output contact information */
template <typename T>
__HOSTDEVICE__ void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                             const RigidBody<T>&  rbB,
                                             const Vector3<T>&    v_a2w,
                                             const Vector3<T>&    v_b2w,
                                             const Quaternion<T>& q_a2w,
                                             const Quaternion<T>& q_b2w,
                                             ContactInfo<T>&      contactInfo);

/** @brief Returns the di (if any) for 2 rigid bodies.
    @param rbA first rigid body
    @param rbB second rigid body
    @param a2w geometric tramsformation describing convex A in the world reference frame
    @param b2w geometric tramsformation describing convex B in the world reference frame */
template <typename T>
__HOSTDEVICE__ T distanceRigidBodies(const RigidBody<T>&  rbA,
                                     const RigidBody<T>&  rbB,
                                     const Transform3<T>& a2w,
                                     const Transform3<T>& b2w,
                                     const uint           method);
//@}

#endif
