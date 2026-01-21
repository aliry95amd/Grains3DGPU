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
/** @brief Template alias for ContactMetaDataPacker */
template <typename T>
using ContactMetaDataPacker = BitPacker<uint32_t,
                                        ContactInfo<T>::B_OVERLAP_SIGN,
                                        ContactInfo<T>::B_CONTACT_HASH,
                                        ContactInfo<T>::B_AVG_MASS>;

/** @brief Helper function to extract rigid body properties and build contact metadata. This
    function extracts material, mass, and crust properties from two rigid bodies and constructs the
    contact metadata including material hash and average mass.
    @param rbA first rigid body
    @param rbB second rigid body
    @param crustA output crust thickness for body A
    @param crustB output crust thickness for body B
    @param contactMetaData output contact metadata */
template <typename T>
__HOSTDEVICE__ static INLINE void buildContactMetaData(const RigidBody<T>&       rbA,
                                                       const RigidBody<T>&       rbB,
                                                       T&                        crustA,
                                                       T&                        crustB,
                                                       ContactMetaDataPacker<T>& contactMetaData)
{
    // Load all properties at once using BitPacker
    using RBPacker
        = BitPacker<uint64_t, RigidBody<T>::B_MAT, RigidBody<T>::B_MASS, RigidBody<T>::B_CRUST>;

    // Component A
    const RBPacker& propertiesA = rbA.getProperties();
    crustA          = propertiesA.template getFixed<2, T>(RigidBody<T>::DEFAULT_CRUST_MIN,
                                                 RigidBody<T>::DEFAULT_CRUST_MAX);
    float massA     = propertiesA.template getFixed<1, T>(RigidBody<T>::DEFAULT_MASS_MIN,
                                                      RigidBody<T>::DEFAULT_MASS_MAX);
    uint  materialA = propertiesA.template get<0>();

    // Component B
    const RBPacker& propertiesB = rbB.getProperties();
    crustB          = propertiesB.template getFixed<2, T>(RigidBody<T>::DEFAULT_CRUST_MIN,
                                                 RigidBody<T>::DEFAULT_CRUST_MAX);
    float massB     = propertiesB.template getFixed<1, T>(RigidBody<T>::DEFAULT_MASS_MIN,
                                                      RigidBody<T>::DEFAULT_MASS_MAX);
    uint  materialB = propertiesB.template get<0>();

    // Compute symmetric material hash (order-independent)
    uint matHash = triangularHash(materialA, materialB);
    // Compute average mass (with protection against infinity mass (obstacle))
    massA = (massA == 0.f) ? 0.f : 1.f / massA;
    massB = (massB == 0.f) ? 0.f : 1.f / massB;

    // Set average mass with quantization
    bool saturated = false;
    contactMetaData.template set<1>(matHash);
    contactMetaData.template setFixed<2, T>((1.f / (massA + massB)),
                                            ContactInfo<T>::DEFAULT_AVG_MASS_MIN,
                                            ContactInfo<T>::DEFAULT_AVG_MASS_MAX,
                                            saturated);
}

// -------------------------------------------------------------------------------------------------
// Returns whether two rigid bodies os spherical shape intersect
template <typename T>
__HOSTDEVICE__ static INLINE bool
    intersectSpheres(const RigidBody<T>& rbA, const RigidBody<T>& rbB, const Vector3<T>& v_b2a)
{
    T radiiSum = rbA.getCircumscribedRadius() + rbB.getCircumscribedRadius();
    T dist2    = norm2(v_b2a);
    return (dist2 < radiiSum * radiiSum);
}
//@}

/* ============================================================================================== */
/* High-Level Methods                                                                             */
/* ============================================================================================== */
/** @name CollisionDetection High-Level Methods */
//@{
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
    /* ---------------------------------------------------------------------------------------------
    Comments on the contactInfo. It applies to all variants of this function:
    1. If actual overlap distance (GJK dist - crustA - crustB < 0), there is contact otherwise no
    contact. Although we can enforce an early exit, but other threads are most likely still
    running, so we continue to have consistent code path.

    2. ptA and ptB are in their respective local coordinate systems and represent points on the
    actual rigid bodies, not the shrunken versions. Contact point definition as the mid point
    between ptA and ptB.

    3. If contact, overlap is negative and overlap_vector is from B to A If no contact, overlap is
    positive and we do not care about the direction of overlap_vector. Assuming A and B are the
    centers of the 2 convex bodies overlap_vector = overlap * Vector3(A to B)
    --------------------------------------------------------------------------------------------- */
    const Convex<T>& convexA = *(rbA.getConvex());
    const Convex<T>& convexB = *(rbB.getConvex());
    const ConvexType typeA   = convexA.getConvexType();
    const ConvexType typeB   = convexB.getConvexType();

    // Extract properties and build contact metadata using helper function
    T                        crustA, crustB;
    ContactMetaDataPacker<T> contactMetaData;
    buildContactMetaData(rbA, rbB, crustA, crustB, contactMetaData);

    T          distance = std::numeric_limits<T>::max();
    Vector3<T> ptA, ptB;

    // Sphere-Sphere Case
    if(typeA == ConvexType::SPHERE && typeB == ConvexType::SPHERE)
    {
        T                 rA    = rbA.getCircumscribedRadius();
        T                 rB    = rbB.getCircumscribedRadius();
        const Vector3<T>& vecBA = b2a.getOrigin();
        distance                = norm(vecBA) - rA - rB;
        ptA                     = (rA + T(.5) * distance) * vecBA;
        ptB                     = ptA + distance * vecBA;
    }
    // Rectangle-Particle Case
    else if(typeA == ConvexType::RECTANGLE)
    {
        const Vector3<T> r = b2a.getOrigin()[Z] > 0 ? Vector3<T>(0, 0, -1) : Vector3<T>(0, 0, 1);
        ptA                = (b2a)(convexB.support(r * b2a.getBasis()));
        if(ptA[Z] < T(0))
        {
            ptB = Vector3<T>(ptA[X], ptA[Y], T(0));
            if(convexA.isInside(ptB))
            {
                distance = -norm(ptA - ptB);
            }
        }
    }
    // General Case
    else
    {
        uint nbIterGJK       = 0;
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

        // If bodies are too close, increase crust thickness and recompute
        constexpr uint maxCrustIterations = 4;
        uint           crustIteration     = 0;
        do
        {
            distance = computeDistance();
            if(fabs(distance) >= HIGHEPS<T>)
                break;
            Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
            crustA *= 10;
            crustB *= 10;
            crustIteration++;
        } while(crustIteration < maxCrustIterations);

        // Actual overlap
        distance -= crustA + crustB;
        // ptA = (a2a)(ptA);
        ptB = (b2a)(ptB);
    }

    // Set contact information and metadata
    contactMetaData.template set<0>(distance < T(0) ? 1 : 0);
    contactInfo.setContactInfo(T(0.5) * (ptA + ptB),
                               (ptA - ptB).normalized(),
                               distance,
                               contactMetaData.getValue());
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
    const ConvexType typeA   = convexA.getConvexType();
    const ConvexType typeB   = convexB.getConvexType();

    // Extract properties and build contact metadata using helper function
    T                        crustA, crustB;
    ContactMetaDataPacker<T> contactMetaData;
    buildContactMetaData(rbA, rbB, crustA, crustB, contactMetaData);

    T          distance = std::numeric_limits<T>::max();
    Vector3<T> ptA, ptB;

    // Sphere-Sphere Case
    if(typeA == ConvexType::SPHERE && typeB == ConvexType::SPHERE)
    {
        T                 rA    = rbA.getCircumscribedRadius();
        T                 rB    = rbB.getCircumscribedRadius();
        const Vector3<T>& cenA  = a2w.getOrigin();
        const Vector3<T>& vecBA = b2w.getOrigin() - cenA;
        distance                = norm(vecBA) - rA - rB;
        ptA                     = cenA + (rA + T(.5) * distance) * vecBA;
        ptB                     = ptA + distance * vecBA;
    }
    // Rectangle-Particle Case
    else if(typeA == ConvexType::RECTANGLE)
    {
        const Vector3<T>& c = a2w.getOrigin();
        const Matrix3<T>& m = a2w.getBasis();
        Vector3<T>        r(m(XZ), m(YZ), m(ZZ));
        r.normalized();
        r *= copysign(T(1), r * (b2w.getOrigin() - c));
        ptA = (b2w)(convexB.support((-r) * b2w.getBasis()));
        if(r * (ptA - c) < T(0))
        {
            ptB = ((c - ptA) * r) * r + ptA;
            if(convexA.isInside(inverse(m) * (ptB - c)))
            {
                distance = -norm(ptA - ptB);
            }
        }
    }
    // General Case
    else
    {
        uint nbIterGJK       = 0;
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

        // If bodies are too close, increase crust thickness and recompute
        constexpr uint maxCrustIterations = 4;
        uint           crustIteration     = 0;
        do
        {
            distance = computeDistance();
            if(fabs(distance) >= HIGHEPS<T>)
                break;
            Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
            crustA *= 10;
            crustB *= 10;
            crustIteration++;
        } while(crustIteration < maxCrustIterations);

        // Computation of the actual overlap
        distance -= crustA + crustB;
        ptA = (a2w)(ptA);
        ptB = (b2w)(ptB);
    }

    // Set contact information and metadata
    contactMetaData.template set<0>(distance < T(0) ? 1 : 0);
    contactInfo.setContactInfo(T(0.5) * (ptA + ptB),
                               (ptA - ptB).normalized(),
                               distance,
                               contactMetaData.getValue());
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
    const ConvexType typeA   = convexA.getConvexType();
    const ConvexType typeB   = convexB.getConvexType();

    // Extract properties and build contact metadata using helper function
    T                        crustA, crustB;
    ContactMetaDataPacker<T> contactMetaData;
    buildContactMetaData(rbA, rbB, crustA, crustB, contactMetaData);

    T          distance = std::numeric_limits<T>::max();
    Vector3<T> ptA, ptB;

    // Sphere-Sphere Case
    if(typeA == ConvexType::SPHERE && typeB == ConvexType::SPHERE)
    {
        T          rA    = rbA.getCircumscribedRadius();
        T          rB    = rbB.getCircumscribedRadius();
        Vector3<T> vecBA = v_b2a;
        distance         = norm(vecBA) - rA - rB;
        ptA              = (rA + T(.5) * distance) * vecBA;
        ptB              = ptA + distance * vecBA;
    }
    // Rectangle-Particle Case
    else if(typeA == ConvexType::RECTANGLE)
    {
        const Vector3<T> r = v_b2a[Z] > 0 ? Vector3<T>(0, 0, -1) : Vector3<T>(0, 0, 1);
        ptA                = convexB.support(q_b2a << r);
        transform(q_b2a, v_b2a, ptA);
        if(ptA[Z] < T(0))
        {
            ptB = Vector3<T>(ptA[X], ptA[Y], T(0));
            if(convexA.isInside(ptB))
            {
                distance = -norm(ptA - ptB);
            }
        }
    }
    // General Case
    else
    {
        uint nbIterGJK       = 0;
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

        // If bodies are too close, increase crust thickness and recompute
        constexpr uint maxCrustIterations = 4;
        uint           crustIteration     = 0;
        do
        {
            distance = computeDistance();
            if(fabs(distance) >= HIGHEPS<T>)
                break;
            Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
            crustA *= 10;
            crustB *= 10;
            crustIteration++;
        } while(crustIteration < maxCrustIterations);

        // Computation of the actual overlap
        distance -= crustA + crustB;
        // transform(q_a2a, v_a2a, ptA);
        transform(q_b2a, v_b2a, ptB);
    }

    // Set contact information and metadata
    contactMetaData.template set<0>(distance < T(0) ? 1 : 0);
    contactInfo.setContactInfo(T(0.5) * (ptA + ptB),
                               (ptA - ptB).normalized(),
                               distance,
                               contactMetaData.getValue());
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
    const ConvexType typeA   = convexA.getConvexType();
    const ConvexType typeB   = convexB.getConvexType();

    // Extract properties and build contact metadata using helper function
    T                        crustA, crustB;
    ContactMetaDataPacker<T> contactMetaData;
    buildContactMetaData(rbA, rbB, crustA, crustB, contactMetaData);

    T          distance = std::numeric_limits<T>::max();
    Vector3<T> ptA, ptB;

    // Sphere-Sphere Case
    if(typeA == ConvexType::SPHERE && typeB == ConvexType::SPHERE)
    {
        T          rA    = rbA.getCircumscribedRadius();
        T          rB    = rbB.getCircumscribedRadius();
        Vector3<T> cenA  = v_a2w;
        Vector3<T> vecBA = v_b2w - cenA;
        distance         = norm(vecBA) - rA - rB;
        ptA              = cenA + (rA + T(.5) * distance) * vecBA;
        ptB              = ptA + distance * vecBA;
    }
    // Rectangle-Particle Case
    else if(typeA == ConvexType::RECTANGLE)
    {
        Vector3<T> r = q_a2w >> Vector3<T>(0, 0, 1);
        r.normalized();
        r *= copysign(T(1), r * (v_b2w - v_a2w));
        ptA = q_b2w >> convexB.support(q_b2w << r) + v_b2w;
        if(r * (ptA - v_a2w) < T(0))
        {
            ptB = ((v_a2w - ptA) * r) * r + ptA;
            if(convexA.isInside(q_a2w << (ptB - v_a2w)))
            {
                distance = -norm(ptA - ptB);
            }
        }
    }
    // General Case
    else
    {
        uint nbIterGJK       = 0;
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

        // If bodies are too close, increase crust thickness and recompute
        constexpr uint maxCrustIterations = 4;
        uint           crustIteration     = 0;
        do
        {
            distance = computeDistance();
            if(fabs(distance) >= HIGHEPS<T>)
                break;
            Gout("Warning: GJK too close bodies, increasing crust thicknesses ...");
            crustA *= 10;
            crustB *= 10;
            crustIteration++;
        } while(crustIteration < maxCrustIterations);

        // Computation of the actual overlap
        distance -= crustA + crustB;
        transform(q_a2w, v_a2w, ptA);
        transform(q_b2w, v_b2w, ptB);
    }

    // Set contact information and metadata
    contactMetaData.template set<0>(distance < T(0) ? 1 : 0);
    contactInfo.setContactInfo(T(0.5) * (ptA + ptB),
                               (ptA - ptB).normalized(),
                               distance,
                               contactMetaData.getValue());
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
