#ifndef _COLLISIONDETECTIONMODULE_KERNELS_HH_
#define _COLLISIONDETECTIONMODULE_KERNELS_HH_

#include <cstdint>
#include <cuda_runtime.h>

#include "ContactInfo.hh"
#include "GrainsParameters.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "Transform3.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief CUDA kernel declarations for CollisionDetectionModule.

    Definitions live in CollisionDetectionModule_Kernels.cpp (compiled by nvcc).

    @author A.Yazdani - 2026 - Construction */
// =================================================================================================
/** @name CollisionDetectionModule_Kernels */
//@{

/** @brief Computes relative transformations (vec/quat) for all pairs in parallel.
    @param pairList list of pairs
    @param position world positions
    @param quaternion world quaternions
    @param relPosition output relative positions (B in A-local)
    @param relQuaternion output relative quaternions (B in A-local)
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void computeRelativeTransformations_Kernel(const Vector3<T>*    position,
                                                      const Quaternion<T>* quaternion,
                                                      const uint2*         pairList,
                                                      Vector3<T>*          relPosition,
                                                      Quaternion<T>*       relQuaternion,
                                                      const uint           nPairs);

/** @brief Bounding volume pre-filter kernel (DEVICE path only).
    Executes sphere and OBB SAT checks for each pair, writes a pass/fail flag, and writes
    a no-contact sentinel into @p contactInfo for every rejected pair so that the downstream
    transformContactInfo pass reads a valid value even for pairs that skip GJK.
    @param pairList      List of pairs
    @param rigidBodies   Rigid body array
    @param relPosition   Per-pair relative positions (B in A-local)
    @param relQuaternion Per-pair relative quaternions (B in A-local)
    @param contactInfo   Per-pair contact info buffer; sentinel written for rejected pairs
    @param bvPassFlags   Output pass/fail flags (0 = reject, 1 = pass)
    @param nPairs        Number of pairs */
template <typename T, BoundingVolumeType BVType = BoundingVolumeType::OBB>
__GLOBAL__ void filterPairsBV_Kernel(const RigidBody<T>* const* rigidBodies,
                                     const uint2*               pairList,
                                     const Vector3<T>*          relPosition,
                                     const Quaternion<T>*       relQuaternion,
                                     ContactInfo<T>*            contactInfo,
                                     uint8_t*                   bvPassFlags,
                                     const uint                 nPairs);

/** @brief Narrow-phase GJK detection (relative vec/quat).
    When @p activePairIndices is non-null each thread resolves its original pair index through
    the indirection table (BV-compacted path, zero divergence); when null the thread ID is used
    directly (BV-off path, all pairs).
    @param rigidBody         Rigid body array
    @param pairList          Full pair list
    @param activePairIndices Compacted index table from CUB, or nullptr to process all pairs
    @param relPosition       Per-pair relative positions (B in A-local)
    @param relQuaternion     Per-pair relative quaternions (B in A-local)
    @param contactInfo       Output contact information (A-local frame)
    @param nPairs            Number of pairs to process (active or total) */
template <typename T, GJKType GJKVARIANT = GJKType::JOHNSON, bool GJKACC = false>
__GLOBAL__ void detectCollisionsComponents_Kernel(const RigidBody<T>* const* rigidBody,
                                                  const uint2*               pairList,
                                                  const uint*                activePairIndices,
                                                  const Vector3<T>*          relPosition,
                                                  const Quaternion<T>*       relQuaternion,
                                                  ContactInfo<T>*            contactInfo,
                                                  const uint                 nPairs);

/** @brief Transforms contact info from A-local frame to world frame for all pairs in parallel.
    @param pairList list of pairs
    @param position world positions
    @param quaternion world quaternions
    @param contactInfoLocal input contact info in A-local frame
    @param contactInfoWorld output contact info in world frame
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void transformContactInfo_Kernel(const Vector3<T>*    position,
                                            const Quaternion<T>* quaternion,
                                            const uint2*         pairList,
                                            ContactInfo<T>*      contactInfoLocal,
                                            ContactInfo<T>*      contactInfoWorld,
                                            const uint           nPairs);
//@}

#endif
