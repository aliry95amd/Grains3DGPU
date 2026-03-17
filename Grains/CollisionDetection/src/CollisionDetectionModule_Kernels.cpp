#include "CollisionDetectionModule_Kernels.hh"
#include "CollisionDetectionCommon.hh"
#include <cstdint>

// -------------------------------------------------------------------------------------------------
// Computes per-pair relative position / quaternion (B in A-local frame)
template <typename T>
__GLOBAL__ void computeRelativeTransformations_Kernel(const Vector3<T>*    position,
                                                      const Quaternion<T>* quaternion,
                                                      const uint2*         pairList,
                                                      Vector3<T>*          relPosition,
                                                      Quaternion<T>*       relQuaternion,
                                                      const uint           nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nPairs)
        return;

    computeRelativeTransformations_common(pairList,
                                          position,
                                          quaternion,
                                          relPosition,
                                          relQuaternion,
                                          tID);
}

// -------------------------------------------------------------------------------------------------
// BV pre-filter kernel: writes per-pair pass/fail flag (1 = pass, 0 = reject) and writes
// a no-contact sentinel into contactInfo for every rejected pair.
template <typename T, BoundingVolumeType BVType>
__GLOBAL__ void filterPairsBV_Kernel(const RigidBody<T>* const* rigidBodies,
                                     const uint2*               pairList,
                                     const Vector3<T>*          relPosition,
                                     const Quaternion<T>*       relQuaternion,
                                     ContactInfo<T>*            contactInfo,
                                     uint8_t*                   bvPassFlags,
                                     const uint                 nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nPairs)
        return;

    const uint2         pair = pairList[tID];
    const RigidBody<T>& rbA  = *(rigidBodies[pair.x]);
    const RigidBody<T>& rbB  = *(rigidBodies[pair.y]);

    if(filterPairBV_common<T, BVType>(rbA, rbB, relPosition[tID], relQuaternion[tID]))
    {
        bvPassFlags[tID] = 1;
    }
    else
    {
        bvPassFlags[tID] = 0;
        contactInfo[tID].setOverlapDistance(T(1));
    }
}

// -------------------------------------------------------------------------------------------------
// Narrow-phase GJK detection (relative vec/quat).
// When activePairIndices is non-null each thread resolves its pair index through the indirection
// table (BV-compacted path). When null the thread ID is used directly (BV-off path, all pairs).
template <typename T, GJKType GJKVARIANT, bool GJKACC>
__GLOBAL__ void detectCollisionsComponents_Kernel(const RigidBody<T>* const* rigidBody,
                                                  const uint2*               pairList,
                                                  const uint*                activePairIndices,
                                                  const Vector3<T>*          relPosition,
                                                  const Quaternion<T>*       relQuaternion,
                                                  ContactInfo<T>*            contactInfo,
                                                  const uint                 nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nPairs)
        return;

    const uint pairIdx = (activePairIndices != nullptr) ? activePairIndices[tID] : tID;
    detectCollisionsComponents_common<T, GJKVARIANT, GJKACC, BoundingVolumeType::OFF>(pairList,
                                                                                      rigidBody,
                                                                                      relPosition,
                                                                                      relQuaternion,
                                                                                      contactInfo,
                                                                                      pairIdx);
}

// -------------------------------------------------------------------------------------------------
// Narrow-phase GJK detection using absolute world-frame positions and quaternions.
// When activePairIndices is non-null each thread resolves its pair index through the indirection
// table (BV-compacted path). When null the thread ID is used directly (BV-off path, all pairs).
template <typename T, GJKType GJKVARIANT, bool GJKACC, BoundingVolumeType BVType>
__GLOBAL__ void detectCollisionsComponentsGlobal_Kernel(const RigidBody<T>* const* rigidBody,
                                                        const uint2*               pairList,
                                                        const uint*          activePairIndices,
                                                        const Vector3<T>*    position,
                                                        const Quaternion<T>* quaternion,
                                                        ContactInfo<T>*      contactInfo,
                                                        const uint           nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nPairs)
        return;

    const uint pairIdx = (activePairIndices != nullptr) ? activePairIndices[tID] : tID;
    detectCollisionsComponentsGlobal_common<T, GJKVARIANT, GJKACC, BVType>(pairList,
                                                                           rigidBody,
                                                                           position,
                                                                           quaternion,
                                                                           contactInfo,
                                                                           pairIdx);
}

// -------------------------------------------------------------------------------------------------
// Transforms contact info from A-local frame to world frame
template <typename T>
__GLOBAL__ void transformContactInfo_Kernel(const Vector3<T>*    position,
                                            const Quaternion<T>* quaternion,
                                            const uint2*         pairList,
                                            ContactInfo<T>*      contactInfoLocal,
                                            ContactInfo<T>*      contactInfoWorld,
                                            const uint           nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nPairs)
        return;

    transformContactInfo_common(pairList,
                                position,
                                quaternion,
                                contactInfoLocal,
                                contactInfoWorld,
                                tID);
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiations
#define X(T)                                                                                \
    template __GLOBAL__ void computeRelativeTransformations_Kernel<T>(const Vector3<T>*,    \
                                                                      const Quaternion<T>*, \
                                                                      const uint2*,         \
                                                                      Vector3<T>*,          \
                                                                      Quaternion<T>*,       \
                                                                      const uint);          \
    template __GLOBAL__ void transformContactInfo_Kernel<T>(const Vector3<T>*,              \
                                                            const Quaternion<T>*,           \
                                                            const uint2*,                   \
                                                            ContactInfo<T>*,                \
                                                            ContactInfo<T>*,                \
                                                            const uint);
X(float)
X(double)
#undef X

#define X(T, GJK, ACC)                                                       \
    template __GLOBAL__ void detectCollisionsComponents_Kernel<T, GJK, ACC>( \
        const RigidBody<T>* const*,                                          \
        const uint2*,                                                        \
        const uint*,                                                         \
        const Vector3<T>*,                                                   \
        const Quaternion<T>*,                                                \
        ContactInfo<T>*,                                                     \
        const uint);
X(float, GJKType::JOHNSON, false)
X(double, GJKType::JOHNSON, false)
X(float, GJKType::JOHNSON, true)
X(double, GJKType::JOHNSON, true)
X(float, GJKType::SIGNEDVOLUME, false)
X(double, GJKType::SIGNEDVOLUME, false)
X(float, GJKType::SIGNEDVOLUME, true)
X(double, GJKType::SIGNEDVOLUME, true)
#undef X

#define X(T, GJK, ACC, BV)                                                             \
    template __GLOBAL__ void detectCollisionsComponentsGlobal_Kernel<T, GJK, ACC, BV>( \
        const RigidBody<T>* const*,                                                    \
        const uint2*,                                                                  \
        const uint*,                                                                   \
        const Vector3<T>*,                                                             \
        const Quaternion<T>*,                                                          \
        ContactInfo<T>*,                                                               \
        const uint);
X(float, GJKType::JOHNSON, false, BoundingVolumeType::OFF)
X(double, GJKType::JOHNSON, false, BoundingVolumeType::OFF)
X(float, GJKType::JOHNSON, true, BoundingVolumeType::OFF)
X(double, GJKType::JOHNSON, true, BoundingVolumeType::OFF)
X(float, GJKType::SIGNEDVOLUME, false, BoundingVolumeType::OFF)
X(double, GJKType::SIGNEDVOLUME, false, BoundingVolumeType::OFF)
X(float, GJKType::SIGNEDVOLUME, true, BoundingVolumeType::OFF)
X(double, GJKType::SIGNEDVOLUME, true, BoundingVolumeType::OFF)
X(float, GJKType::JOHNSON, false, BoundingVolumeType::OBB)
X(double, GJKType::JOHNSON, false, BoundingVolumeType::OBB)
X(float, GJKType::JOHNSON, true, BoundingVolumeType::OBB)
X(double, GJKType::JOHNSON, true, BoundingVolumeType::OBB)
X(float, GJKType::SIGNEDVOLUME, false, BoundingVolumeType::OBB)
X(double, GJKType::SIGNEDVOLUME, false, BoundingVolumeType::OBB)
X(float, GJKType::SIGNEDVOLUME, true, BoundingVolumeType::OBB)
X(double, GJKType::SIGNEDVOLUME, true, BoundingVolumeType::OBB)
X(float, GJKType::JOHNSON, false, BoundingVolumeType::OBC)
X(double, GJKType::JOHNSON, false, BoundingVolumeType::OBC)
X(float, GJKType::JOHNSON, true, BoundingVolumeType::OBC)
X(double, GJKType::JOHNSON, true, BoundingVolumeType::OBC)
X(float, GJKType::SIGNEDVOLUME, false, BoundingVolumeType::OBC)
X(double, GJKType::SIGNEDVOLUME, false, BoundingVolumeType::OBC)
X(float, GJKType::SIGNEDVOLUME, true, BoundingVolumeType::OBC)
X(double, GJKType::SIGNEDVOLUME, true, BoundingVolumeType::OBC)
#undef X

#define X(T, BV)                                                                     \
    template __GLOBAL__ void filterPairsBV_Kernel<T, BV>(const RigidBody<T>* const*, \
                                                         const uint2*,               \
                                                         const Vector3<T>*,          \
                                                         const Quaternion<T>*,       \
                                                         ContactInfo<T>*,            \
                                                         uint8_t*,                   \
                                                         const uint);
X(float, BoundingVolumeType::OBB)
X(double, BoundingVolumeType::OBB)
X(float, BoundingVolumeType::OBC)
X(double, BoundingVolumeType::OBC)
#undef X
