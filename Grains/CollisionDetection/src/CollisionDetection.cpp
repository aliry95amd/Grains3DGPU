#include "CollisionDetection.hh"
#include "GJK_AY.hh"
#include "GJK_JH.hh"
#include "GJK_SV.hh"
#include "MatrixMath.hh"
#include "MiscMath.hh"
#include "OBB.hh"

/* ========================================================================== */
/*                             Low-Level Methods                              */
/* ========================================================================== */
// Returns whether two rigid bodies os spherical shape intersect
template <typename T>
__HOSTDEVICE__ static INLINE bool intersectSpheres(const RigidBody<T>& rbA,
                                                   const RigidBody<T>& rbB,
                                                   const Vector3<T>&   b2a)
{
    T radiiSum = rbA.getCircumscribedRadius() + rbB.getCircumscribedRadius();
    T dist2    = norm2(b2a);
    return (dist2 < radiiSum * radiiSum);
}

// -----------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies of spherical
// shape
template <typename T>
__HOSTDEVICE__ static INLINE void
    closestPointsSpheres(const RigidBody<T>&  rbA,
                         const RigidBody<T>&  rbB,
                         const Transform3<T>& b2a,
                         ContactInfo<T>&      contactInfo)
{
    T          rA    = rbA.getCircumscribedRadius();
    T          rB    = rbB.getCircumscribedRadius();
    Vector3<T> vecBA = b2a.getOrigin();
    // We calculate the overlap, and then normalize the distance vector.
    T overlap = vecBA.norm() - rA - rB;
    contactInfo.setOverlapDistance(overlap);
    if(overlap < T(0))
    {
        contactInfo.setContactPoint((rA + T(.5) * overlap) * vecBA);
        contactInfo.setContactVector(overlap * vecBA);
    }
    return;
}

// -----------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies of spherical
// shape
template <typename T>
__HOSTDEVICE__ static INLINE void
    closestPointsSpheres(const RigidBody<T>&  rbA,
                         const RigidBody<T>&  rbB,
                         const Transform3<T>& a2w,
                         const Transform3<T>& b2w,
                         ContactInfo<T>&      contactInfo)
{
    T          rA    = rbA.getCircumscribedRadius();
    T          rB    = rbB.getCircumscribedRadius();
    Vector3<T> cenA  = a2w.getOrigin();
    Vector3<T> vecBA = b2w.getOrigin() - cenA;
    // We calculate the overlap, and then normalize the distance vector.
    T overlap = vecBA.norm() - rA - rB;
    contactInfo.setOverlapDistance(overlap);
    if(overlap < T(0))
    {
        contactInfo.setContactPoint(cenA + (rA + T(.5) * overlap) * vecBA);
        contactInfo.setContactVector(overlap * vecBA);
    }
    return;
}

// -----------------------------------------------------------------------------
// // Returns the contact information (if any) for 2 rigid bodies if the SECOND ONE
// // is a rectangle
// template <typename T>
// __HOSTDEVICE__ static INLINE ContactInfo<T>
//                              closestPointsRectangle(const RigidBody<T>& rbA,
//                                                     const RigidBody<T>& rbB,
//                                                     const Transform3<T>&   a2w,
//                                                     const Transform3<T>&   b2w)
// {
//     const Convex<T>& convexA = *(rbA.getConvex());
//     const Convex<T>& convexB = *(rbB.getConvex());

//     // rectangle center
//     const Vector3<T>& rPt = b2w.getOrigin();
//     // rectangle normal is b2w.getBasis() * [0, 0, 1] which is the last column
//     // of the transformation matrix
//     Vector3<T> rNorm(b2w.getBasis()[XZ],
//                      b2w.getBasis()[YZ],
//                      b2w.getBasis()[ZZ]);
//     rNorm.normalized();
//     rNorm = copysign(T(1), rNorm * (a2w.getOrigin() - rPt)) * rNorm;
//     // Contact point on the particle
//     Vector3<T> pointA = (a2w)(convexA->support((-rNorm) * a2w.getBasis()));
//     if(rNorm * (pointA - rPt) < T(0))
//     {
//         // The projection point on the rectangle plane
//         Vector3<T> pointB = ((rPt - pointA) * rNorm) * rNorm + pointA;
//         // The projection point lies on the rectangle?
//         // TODO:
//         // if ( ( pointB - rPt ).isInBox(  ) )
//         // {
//         Vector3<T> contactPt  = T(0.5) * (pointA + pointB);
//         Vector3<T> contactVec = pointB - pointA;
//         T          overlap    = -norm(contactVec);
//         return (ContactInfo<T>(contactPt, contactVec, overlap));
//         // }
//     }
//     else
//         return (noContact);
// }

/* ========================================================================== */
/*                             High-Level Methods                             */
/* ========================================================================== */
// Returns whether 2 rigid bodies intersect
template <typename T>
__HOSTDEVICE__ bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                         const RigidBody<T>&  rbB,
                                         const Transform3<T>& a2w,
                                         const Transform3<T>& b2w)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    return (intersectGJK(convexA, convexB, a2w, b2w));
}

// -----------------------------------------------------------------------------
// Returns whether 2 rigid bodies intersect using the GJK algorithm - relative
// transformation
template <typename T>
__HOSTDEVICE__ bool intersectRigidBodies(const RigidBody<T>&  rbA,
                                         const RigidBody<T>&  rbB,
                                         const Transform3<T>& b2a)
{
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    return (intersectGJK(convexA, convexB, b2a));
}

// -----------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies - relative
// transformation
template <typename T>
__HOSTDEVICE__ void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                             const RigidBody<T>&  rbB,
                                             const Transform3<T>& b2a,
                                             ContactInfo<T>&      contactInfo)
{
    // Comment on the direction of the overlap vector
    // Assuming A and B are the centers of the 2 convex bodies
    // overlap_vector = overlap * Vector3(A to B)
    // If contact, overlap is negative and overlap_vector is from B to A
    // If no contact, overlap is positive and we do not care about the direction
    // of overlap_vector

    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());

    // If both convexes are spheres, we use a specific method
    if(convexA.getConvexType() == ConvexType::SPHERE
       && convexB.getConvexType() == ConvexType::SPHERE)
    {
        closestPointsSpheres(rbA, rbB, b2a, contactInfo);
        return;
    }

    // General case for convexes
    // Sum of crust thicknesses
    T ctSum = rbA.getCrustThickness() + rbB.getCrustThickness();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;
    T          distance  = computeClosestPoints_GJK_JH(convexA,
                                             convexB,
                                             b2a,
                                             ptA,
                                             ptB,
                                             nbIterGJK);

    printf("GJK_JH: nbIterGJK = %d, ctSum = %f, distance = %f\n",
           nbIterGJK,
           ctSum,
           distance);
    // Computation of the actual overlap
    // distance = distance - crustA - crustB
    // If actual overlap distance < 0 => contact otherwise no contact
    distance -= ctSum;
    // TODO: What if too much overlap?
    contactInfo.setOverlapDistance(distance);
    if(distance > T(0))
        return;

    // Points A and B are in their respective local coordinate systems
    // We transform ptB into the the local coordinate system of A
    // ptA = ptA;
    ptB = (b2a)(ptB);

    // Contact point definition as the mid point between ptA and ptB
    contactInfo.setContactPoint(T(0.5) * (ptA + ptB));

    // Computation of the actual overlap vector
    // If contact, crustA + crustB - distance > 0, the overlap vector is
    // directed from B to A
    // If no contact, crustA + crustB - distance < 0 and we do not care
    // about the direction of the overlap vector
    Vector3<T> contactVec(ptA - ptB);
    contactVec.normalize();
    contactVec.round();
    contactVec *= -distance;
    contactInfo.setContactVector(contactVec);
    return;
}

// -----------------------------------------------------------------------------
// Returns the contact information (if any) for 2 rigid bodies
template <typename T>
__HOSTDEVICE__ void closestPointsRigidBodies(const RigidBody<T>&  rbA,
                                             const RigidBody<T>&  rbB,
                                             const Transform3<T>& a2w,
                                             const Transform3<T>& b2w,
                                             ContactInfo<T>&      contactInfo)
{
    // Comment on the direction of the overlap vector
    // Assuming A and B are the centers of the 2 convex bodies
    // overlap_vector = overlap * Vector3(A to B)
    // If contact, overlap is negative and overlap_vector is from B to A
    // If no contact, overlap is positive and we do not care about the direction
    // of overlap_vector

    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());

    // If both convexes are spheres, we use a specific method
    if(convexA.getConvexType() == ConvexType::SPHERE
       && convexB.getConvexType() == ConvexType::SPHERE)
    {
        closestPointsSpheres(rbA, rbB, a2w, b2w, contactInfo);
        return;
    }

    // General case for convexes
    // Sum of crust thicknesses
    T ctSum = rbA.getCrustThickness() + rbB.getCrustThickness();

    Vector3<T> ptA, ptB;
    uint       nbIterGJK = 0;
    T          distance  = computeClosestPoints_GJK_JH(convexA,
                                             convexB,
                                             a2w,
                                             b2w,
                                             ptA,
                                             ptB,
                                             nbIterGJK);

    // Computation of the actual overlap
    // distance = distance - crustA - crustB
    // If actual overlap distance < 0 => contact otherwise no contact
    distance -= ctSum;
    // TODO: What if too much overlap?
    contactInfo.setOverlapDistance(distance);
    if(distance > T(0))
        return;

    // Points A and B are in their respective local coordinate systems
    // Thus we transform them into the world coordinate system
    ptA = (a2w)(ptA);
    ptB = (b2w)(ptB);

    // Contact point definition as the mid point between ptA and ptB
    contactInfo.setContactPoint(T(0.5) * (ptA + ptB));

    // Computation of the actual overlap vector
    // If contact, crustA + crustB - distance > 0, the overlap vector is
    // directed from B to A
    // If no contact, crustA + crustB - distance < 0 and we do not care
    // about the direction of the overlap vector
    Vector3<T> contactVec = ptA - ptB;
    contactVec.normalize();
    contactVec.round();
    contactVec *= -distance;
    contactInfo.setContactVector(contactVec);
}

// -----------------------------------------------------------------------------
// Returns the distance between 2 rigid bodies
template <typename T>
__HOSTDEVICE__ T distanceRigidBodies(const RigidBody<T>&  rbA,
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
    // if ( method == 1 )
    // {
    distance = computeClosestPoints_GJK_JH(*convexA,
                                           *convexB,
                                           a2w,
                                           b2w,
                                           ptA,
                                           ptB,
                                           nbIterGJK);
    // }
    // else if ( method == 2 )
    // {
    // distance = computeClosestPoints_GJK_SV( *convexA,
    //                                         *convexB,
    //                                         a2w,
    //                                         b2w,
    //                                         ptA,
    //                                         ptB,
    //                                         nbIterGJK );
    // }
    // else if ( method == 3 )
    // {
    // distance = computeClosestPoints_GJK_AY( *convexA,
    //                                         *convexB,
    //                                         a2w,
    //                                         b2w,
    //                                         ptA,
    //                                         ptB,
    //                                         nbIterGJK );
    // }
    return (distance);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
#define X(T)                                                                \
    template __HOSTDEVICE__ bool intersectRigidBodies(                      \
        const RigidBody<T>&  rbA,                                           \
        const RigidBody<T>&  rbB,                                           \
        const Transform3<T>& b2a);                                          \
    template __HOSTDEVICE__ bool intersectRigidBodies(                      \
        const RigidBody<T>&  rbA,                                           \
        const RigidBody<T>&  rbB,                                           \
        const Transform3<T>& a2w,                                           \
        const Transform3<T>& b2w);                                          \
    template __HOSTDEVICE__ void closestPointsRigidBodies(                  \
        const RigidBody<T>&  rbA,                                           \
        const RigidBody<T>&  rbB,                                           \
        const Transform3<T>& b2a,                                           \
        ContactInfo<T>&      contactInfo);                                       \
    template __HOSTDEVICE__ void closestPointsRigidBodies(                  \
        const RigidBody<T>&  rbA,                                           \
        const RigidBody<T>&  rbB,                                           \
        const Transform3<T>& a2w,                                           \
        const Transform3<T>& b2w,                                           \
        ContactInfo<T>&      contactInfo);                                       \
    template __HOSTDEVICE__ T distanceRigidBodies(const RigidBody<T>&  rbA, \
                                                  const RigidBody<T>&  rbB, \
                                                  const Transform3<T>& a2w, \
                                                  const Transform3<T>& b2w, \
                                                  const uint           method);
X(float)
X(double)
#undef X