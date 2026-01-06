#ifndef _COLLISIONDETECTION_HH_
#define _COLLISIONDETECTION_HH_

#include "ContactInfo.hh"
#include "GJK.hh"
#include "GrainsUtils.hh"
#include "MatrixMath.hh"
#include "MiscMath.hh"
#include "OBB.hh"
#include "QuaternionMath.hh"
#include "Rectangle.hh"
#include "RigidBody.hh"
#include "Transform3.hh"

// =================================================================================================
/** @brief The header-only file for Rigid bodies collision detections.

    Functions for collision detection between two rigid bodies.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
/** @name CollisionDetection Low-Level Methods */
//@{
// Returns whether two rigid bodies os spherical shape intersect
template <typename T>
__HOSTDEVICE__ static INLINE bool
    intersectSpheres(const RigidBody<T>& rbA, const RigidBody<T>& rbB, const Vector3<T>& v_b2a)
{
    T radiiSum = rbA.getCircumscribedRadius() + rbB.getCircumscribedRadius();
    T dist2    = norm2(v_b2a);
    return (dist2 < radiiSum * radiiSum);
}

// -------------------------------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies of spherical shape
template <typename T>
__HOSTDEVICE__ static INLINE void closestPointsSpheres(const RigidBody<T>& rbA,
                                                       const RigidBody<T>& rbB,
                                                       const Vector3<T>&   v_b2a,
                                                       ContactInfo<T>&     contactInfo)
{
    T          rA    = rbA.getCircumscribedRadius();
    T          rB    = rbB.getCircumscribedRadius();
    Vector3<T> vecBA = v_b2a;
    // We calculate the overlap, and then normalize the distance vector.
    T overlap = norm(vecBA) - rA - rB;
    contactInfo.setOverlapDistance(overlap);
    if(overlap < T(0))
    {
        contactInfo.setContactPoint((rA + T(.5) * overlap) * vecBA);
        contactInfo.setContactVector(overlap * vecBA);
    }
    return;
}

// -------------------------------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies of spherical shape
template <typename T>
__HOSTDEVICE__ static INLINE void closestPointsSpheres(const RigidBody<T>& rbA,
                                                       const RigidBody<T>& rbB,
                                                       const Vector3<T>&   v_a2w,
                                                       const Vector3<T>&   v_b2w,
                                                       ContactInfo<T>&     contactInfo)
{
    T          rA    = rbA.getCircumscribedRadius();
    T          rB    = rbB.getCircumscribedRadius();
    Vector3<T> cenA  = v_a2w;
    Vector3<T> vecBA = v_b2w - cenA;
    // We calculate the overlap, and then normalize the distance vector.
    T overlap = norm(vecBA) - rA - rB;
    contactInfo.setOverlapDistance(overlap);
    if(overlap < T(0))
    {
        contactInfo.setContactPoint(cenA + (rA + T(.5) * overlap) * vecBA);
        contactInfo.setContactVector(overlap * vecBA);
    }
    return;
}

// -------------------------------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies if the first one is a rectangle
template <typename T>
__HOSTDEVICE__ static INLINE void closestPointsRectangle(const RigidBody<T>&  rbA,
                                                         const RigidBody<T>&  rbB,
                                                         const Transform3<T>& b2a,
                                                         ContactInfo<T>&      contactInfo)
{
    const Convex<T>* rect    = rbA.getConvex();
    const Convex<T>* convexB = rbB.getConvex();

    const Vector3<T> r = b2a.getOrigin()[Z] > 0 ? Vector3<T>(0, 0, -1) : Vector3<T>(0, 0, 1);
    // Contact point on the particle
    const Vector3<T> ptA = (b2a)(convexB->support(r * b2a.getBasis()));
    if(ptA[Z] < T(0))
    {
        // The projection point on the rectangle plane
        const Vector3<T> ptB(ptA[X], ptA[Y], T(0));
        // The projection point lies on the rectangle?
        if(rect->isInside(ptB))
        {
            contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
            contactInfo.setContactVector(ptA - ptB);
            contactInfo.setOverlapDistance(-norm(ptA - ptB));
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies if the first one is a rectangle
template <typename T>
__HOSTDEVICE__ static INLINE void closestPointsRectangle(const RigidBody<T>&  rbA,
                                                         const RigidBody<T>&  rbB,
                                                         const Transform3<T>& a2w,
                                                         const Transform3<T>& b2w,
                                                         ContactInfo<T>&      contactInfo)
{
    const Convex<T>* rect    = rbA.getConvex();
    const Convex<T>* convexB = rbB.getConvex();

    // rectangle center
    const Vector3<T>& c(a2w.getOrigin());
    const Matrix3<T>& m(a2w.getBasis());
    // rectangle normal is a2w.getBasis() * [0, 0, 1] which is the last column of the transform
    Vector3<T> r(m(XZ), m(YZ), m(ZZ));
    r.normalized();
    r *= copysign(T(1), r * (b2w.getOrigin() - c));
    // Contact point on the particle
    const Vector3<T> ptA = (b2w)(convexB->support((-r) * b2w.getBasis()));
    if(r * (ptA - c) < T(0))
    {
        // The projection point on the rectangle plane
        const Vector3<T> ptB = ((c - ptA) * r) * r + ptA;
        // The projection point lies on the rectangle?
        if(rect->isInside(inverse(m) * (ptB - c)))
        {
            contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
            contactInfo.setContactVector(ptA - ptB);
            contactInfo.setOverlapDistance(-norm(ptA - ptB));
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies if the first one is a rectangle
template <typename T>
__HOSTDEVICE__ static INLINE void closestPointsRectangle(const RigidBody<T>&  rbA,
                                                         const RigidBody<T>&  rbB,
                                                         const Vector3<T>&    v_b2a,
                                                         const Quaternion<T>& q_b2a,
                                                         ContactInfo<T>&      contactInfo)
{
    const Convex<T>* rect    = rbA.getConvex();
    const Convex<T>* convexB = rbB.getConvex();

    const Vector3<T> r = v_b2a[Z] > 0 ? Vector3<T>(0, 0, -1) : Vector3<T>(0, 0, 1);
    // Contact point on the particle
    Vector3<T> ptA = convexB->support(q_b2a << r);
    transform(q_b2a, v_b2a, ptA);
    if(ptA[Z] < T(0))
    {
        // The projection point on the rectangle plane
        const Vector3<T> ptB(ptA[X], ptA[Y], T(0));
        // The projection point lies on the rectangle?
        // if(rect->isInside(ptB))
        // {
        contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
        contactInfo.setContactVector(ptA - ptB);
        contactInfo.setOverlapDistance(-norm(ptA - ptB));
        // }
    }
}

// -------------------------------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies if the first one is a rectangle
template <typename T>
__HOSTDEVICE__ static INLINE void closestPointsRectangle(const RigidBody<T>&  rbA,
                                                         const RigidBody<T>&  rbB,
                                                         const Vector3<T>&    v_a2w,
                                                         const Vector3<T>&    v_b2w,
                                                         const Quaternion<T>& q_a2w,
                                                         const Quaternion<T>& q_b2w,
                                                         ContactInfo<T>&      contactInfo)
{
    const Convex<T>* rect    = rbA.getConvex();
    const Convex<T>* convexB = rbB.getConvex();

    // rectangle normal is a2w.getBasis() * [0, 0, 1] which is the last column of the transform
    Vector3<T> r = q_a2w >> Vector3<T>(0, 0, 1);
    r.normalized();
    r *= copysign(T(1), r * (v_b2w - v_a2w));
    // Contact point on the particle
    const Vector3<T> ptA = q_b2w >> convexB->support(q_b2w << r) + v_b2w;
    if(r * (ptA - v_a2w) < T(0))
    {
        // The projection point on the rectangle plane
        const Vector3<T> ptB = ((v_a2w - ptA) * r) * r + ptA;
        // The projection point lies on the rectangle?
        if(rect->isInside(q_a2w << (ptB - v_a2w)))
        {
            contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
            contactInfo.setContactVector(ptA - ptB);
            contactInfo.setOverlapDistance(-norm(ptA - ptB));
        }
    }
}
//@}

/* ============================================================================================== */
/** @name CollisionDetection High-Level Methods */
//@{
// -------------------------------------------------------------------------------------------------
/** @brief Returns whether 2 rigid bodies intersect - relative transformation.
    @param rbA first rigid body
    @param rbB second rigid body
    @param b2a geometric transformation describing convex B in the A's reference frame */
template <typename T>
__HOSTDEVICE__ inline bool
    intersectRigidBodies(const RigidBody<T>& rbA, const RigidBody<T>& rbB, const Transform3<T>& b2a)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    return (intersectGJK(convexA, convexB, b2a));
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns whether 2 rigid bodies intersect.
    @param rbA first rigid body
    @param rbB second rigid body
    @param a2w geometric transformation describing convex A in the world reference frame
    @param b2w geometric transformation describing convex B in the world reference frame */
template <typename T>
__HOSTDEVICE__ inline bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                                const RigidBody<T>&  rbB,
                                                const Transform3<T>& a2w,
                                                const Transform3<T>& b2w)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    return (intersectGJK(convexA, convexB, a2w, b2w));
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns whether 2 rigid bodies intersect - relative transformation.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_b2a position describing convex B in the A's reference frame
    @param q_b2a rotation describing convex B in the A's reference frame */
template <typename T>
__HOSTDEVICE__ inline bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                                const RigidBody<T>&  rbB,
                                                const Vector3<T>&    v_b2a,
                                                const Quaternion<T>& q_b2a)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    return (intersectGJK(convexA, convexB, v_b2a, q_b2a));
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns whether 2 rigid bodies intersect.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_a2w position describing convex A in the world reference frame
    @param v_b2w position describing convex B in the world reference frame
    @param q_a2w rotation describing convex A in the world reference frame
    @param q_b2w rotation describing convex B in the world reference frame */
template <typename T>
__HOSTDEVICE__ inline bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                                const RigidBody<T>&  rbB,
                                                const Vector3<T>&    v_a2w,
                                                const Vector3<T>&    v_b2w,
                                                const Quaternion<T>& q_a2w,
                                                const Quaternion<T>& q_b2w)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    return (intersectGJK(convexA, convexB, v_a2w, v_b2w, q_a2w, q_b2w));
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns the contact information (if any) for 2 rigid bodies - relative transformation.
    @param rbA first rigid body
    @param rbB second rigid body
    @param b2a geometric transformation describing convex B in the A's reference frame
    @param contactInfo output contact information */
template <typename T, GJKType GJKVARIANT, bool GJKACC>
__HOSTDEVICE__ inline void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                                    const RigidBody<T>&  rbB,
                                                    const Transform3<T>& b2a,
                                                    ContactInfo<T>&      contactInfo)
{
    /* -------------------------------------------------------------------------
    Comments on the contactInfo. It applies to all variant of this function:
    1. If actual overlap distance (GJK dist - crustA - crustB < 0), there is
    contact otherwise no contact. Although we can enforce an early exit, but
    other threads are most likely still running, so we continue to have
    consistent code path.

    2. ptA and ptB are in their respective local coordinate systems and
    represent points on the actual rigid bodies, not the shrunken versions.
    Contact point definition as the mid point between ptA and ptB

    3. If contact, overlap is negative and overlap_vector is from B to A
    If no contact, overlap is positive and we do not care about the direction
    of overlap_vector. Assuming A and B are the centers of the 2 convex
    bodies overlap_vector = overlap * Vector3(A to B)
    ------------------------------------------------------------------------- */
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());

    // If both convexes are spheres, we use a specific method
    if(convexA.getConvexType() == ConvexType::SPHERE
       && convexB.getConvexType() == ConvexType::SPHERE)
    {
        closestPointsSpheres(rbA, rbB, b2a.getOrigin(), contactInfo);
        return;
    }
    else if(convexA.getConvexType() == ConvexType::RECTANGLE)
    {
        closestPointsRectangle(rbA, rbB, b2a, contactInfo);
        return;
    }

    /* General Case --------------------------------------------------------- */
    T crustA = rbA.getCrustThickness();
    T crustB = rbB.getCrustThickness();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;

    auto computeDistance = [&]() {
        return computeClosestPoints_GJK<T, GJKVARIANT, GJKACC>(convexA,
                                                               convexB,
                                                               b2a,
                                                               crustA,
                                                               crustB,
                                                               ptA,
                                                               ptB,
                                                               nbIterGJK);
    };

    T distance = computeDistance();

    // If bodies are too close
    while(fabs(distance) < HIGHEPS<T>)
    {
        Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
        crustA *= 10;
        crustB *= 10;
        distance = computeDistance();
    }

    // Computation of the actual overlap
    distance -= crustA + crustB;
    contactInfo.setOverlapDistance(distance);
    // ptA = (a2a)(ptA);
    ptB = (b2a)(ptB);
    contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
    Vector3<T> contactVec(ptA - ptB);
    contactVec.normalize();
    round(contactVec);
    contactVec *= -distance;
    contactInfo.setContactVector(contactVec);
    return;
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns the contact information (if any) for 2 rigid bodies.
    @param rbA first rigid body
    @param rbB second rigid body
    @param a2w geometric transformation describing convex A in the world reference frame
    @param b2w geometric transformation describing convex B in the world reference frame
    @param contactInfo output contact information */
template <typename T, GJKType GJKVARIANT, bool GJKACC>
__HOSTDEVICE__ inline void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                                    const RigidBody<T>&  rbB,
                                                    const Transform3<T>& a2w,
                                                    const Transform3<T>& b2w,
                                                    ContactInfo<T>&      contactInfo)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());

    // If both convexes are spheres, we use a specific method
    if(convexA.getConvexType() == ConvexType::SPHERE
       && convexB.getConvexType() == ConvexType::SPHERE)
    {
        closestPointsSpheres(rbA, rbB, a2w.getOrigin(), b2w.getOrigin(), contactInfo);
        return;
    }
    else if(convexA.getConvexType() == ConvexType::RECTANGLE)
    {
        closestPointsRectangle(rbA, rbB, a2w, b2w, contactInfo);
        return;
    }

    /* General Case --------------------------------------------------------- */
    T crustA = rbA.getCrustThickness();
    T crustB = rbB.getCrustThickness();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;

    auto computeDistance = [&]() {
        return computeClosestPoints_GJK<T, GJKVARIANT, GJKACC>(convexA,
                                                               convexB,
                                                               a2w,
                                                               b2w,
                                                               crustA,
                                                               crustB,
                                                               ptA,
                                                               ptB,
                                                               nbIterGJK);
    };

    T distance = computeDistance();

    // If bodies are too close
    while(fabs(distance) < HIGHEPS<T>)
    {
        Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
        crustA *= 10;
        crustB *= 10;
        distance = computeDistance();
    }

    // Computation of the actual overlap
    distance -= crustA + crustB;
    contactInfo.setOverlapDistance(distance);
    ptA = (a2w)(ptA);
    ptB = (b2w)(ptB);
    contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
    Vector3<T> contactVec = ptA - ptB;
    contactVec.normalize();
    round(contactVec);
    contactVec *= -distance;
    contactInfo.setContactVector(contactVec);
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns the contact information (if any) for 2 rigid bodies - relative transformation.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_b2a position describing convex B in the A's reference frame
    @param q_b2a rotation describing convex B in the A's reference frame
    @param contactInfo output contact information */
template <typename T, GJKType GJKVARIANT, bool GJKACC>
__HOSTDEVICE__ inline void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                                    const RigidBody<T>&  rbB,
                                                    const Vector3<T>&    v_b2a,
                                                    const Quaternion<T>& q_b2a,
                                                    ContactInfo<T>&      contactInfo)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());

    // If both convexes are spheres, we use a specific method
    if(convexA.getConvexType() == ConvexType::SPHERE
       && convexB.getConvexType() == ConvexType::SPHERE)
    {
        closestPointsSpheres(rbA, rbB, v_b2a, contactInfo);
        return;
    }
    else if(convexA.getConvexType() == ConvexType::RECTANGLE)
    {
        closestPointsRectangle(rbA, rbB, v_b2a, q_b2a, contactInfo);
        return;
    }

    /* General Case --------------------------------------------------------- */
    T crustA = rbA.getCrustThickness();
    T crustB = rbB.getCrustThickness();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;

    auto computeDistance = [&]() {
        return computeClosestPoints_GJK<T, GJKVARIANT, GJKACC>(convexA,
                                                               convexB,
                                                               v_b2a,
                                                               q_b2a,
                                                               crustA,
                                                               crustB,
                                                               ptA,
                                                               ptB,
                                                               nbIterGJK);
    };

    T distance = computeDistance();

    // If bodies are too close
    while(fabs(distance) < HIGHEPS<T>)
    {
        Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
        crustA *= 10;
        crustB *= 10;
        distance = computeDistance();
    }

    // Computation of the actual overlap
    distance -= crustA + crustB;
    contactInfo.setOverlapDistance(distance);
    // transform(q_a2a, v_a2a, ptA);
    transform(q_b2a, v_b2a, ptB);
    contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
    Vector3<T> contactVec(ptA - ptB);
    contactVec.normalize();
    round(contactVec);
    contactVec *= -distance;
    contactInfo.setContactVector(contactVec);
    return;
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns the contact information (if any) for 2 rigid bodies.
    @param rbA first rigid body
    @param rbB second rigid body
    @param v_a2w position describing convex A in the world reference frame
    @param v_b2w position describing convex B in the world reference frame
    @param q_a2w rotation describing convex A in the world reference frame
    @param q_b2w rotation describing convex B in the world reference frame
    @param contactInfo output contact information */
template <typename T, GJKType GJKVARIANT, bool GJKACC>
__HOSTDEVICE__ inline void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                                    const RigidBody<T>&  rbB,
                                                    const Vector3<T>&    v_a2w,
                                                    const Vector3<T>&    v_b2w,
                                                    const Quaternion<T>& q_a2w,
                                                    const Quaternion<T>& q_b2w,
                                                    ContactInfo<T>&      contactInfo)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());

    // If both convexes are spheres, we use a specific method
    if(convexA.getConvexType() == ConvexType::SPHERE
       && convexB.getConvexType() == ConvexType::SPHERE)
    {
        closestPointsSpheres(rbA, rbB, v_a2w, v_b2w, contactInfo);
        return;
    }
    else if(convexA.getConvexType() == ConvexType::RECTANGLE)
    {
        closestPointsRectangle(rbA, rbB, v_a2w, v_b2w, q_a2w, q_b2w, contactInfo);
        return;
    }

    /* General Case --------------------------------------------------------- */
    T crustA = rbA.getCrustThickness();
    T crustB = rbB.getCrustThickness();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;

    auto computeDistance = [&]() {
        return computeClosestPoints_GJK<T, GJKVARIANT, GJKACC>(convexA,
                                                               convexB,
                                                               v_a2w,
                                                               v_b2w,
                                                               q_a2w,
                                                               q_b2w,
                                                               crustA,
                                                               crustB,
                                                               ptA,
                                                               ptB,
                                                               nbIterGJK);
    };

    T distance = computeDistance();

    // If bodies are too close
    while(fabs(distance) < HIGHEPS<T>)
    {
        Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
        crustA *= 10;
        crustB *= 10;
        distance = computeDistance();
    }

    // Computation of the actual overlap
    distance -= crustA + crustB;
    contactInfo.setOverlapDistance(distance);
    transform(q_a2w, v_a2w, ptA);
    transform(q_b2w, v_b2w, ptB);
    contactInfo.setContactPoint(T(0.5) * (ptA + ptB));
    Vector3<T> contactVec = ptA - ptB;
    contactVec.normalize();
    round(contactVec);
    contactVec *= -distance;
    contactInfo.setContactVector(contactVec);
}

// -------------------------------------------------------------------------------------------------
/** @brief Returns the distance between 2 rigid bodies.
    @param rbA first rigid body
    @param rbB second rigid body
    @param a2w geometric transformation describing convex A in the world reference frame
    @param b2w geometric transformation describing convex B in the world reference frame
    @param method method identifier (currently unused) */
template <typename T, GJKType GJKVARIANT, bool GJKACC>
__HOSTDEVICE__ inline T distanceRigidBodies(const RigidBody<T>&  rbA,
                                            const RigidBody<T>&  rbB,
                                            const Transform3<T>& a2w,
                                            const Transform3<T>& b2w,
                                            const uint           method)
{
    Convex<T> const* convexA = rbA.getConvex();
    Convex<T> const* convexB = rbB.getConvex();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;
    T          distance  = 0;
    distance             = computeClosestPoints_GJK<T, GJKVARIANT, GJKACC>(*convexA,
                                                               *convexB,
                                                               a2w,
                                                               b2w,
                                                               rbA.getCrustThickness(),
                                                               rbB.getCrustThickness(),
                                                               ptA,
                                                               ptB,
                                                               nbIterGJK);
    return (distance);
}
//@}

#endif
