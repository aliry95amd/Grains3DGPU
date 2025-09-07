// TODO: CHANGE THE FORMAT FROM HH TO CUH LATER.
#ifndef _COMPONENTMANAGERGPU_KERNLES_CUH_
#define _COMPONENTMANAGERGPU_KERNLES_CUH_

#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/for_each.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/scan.h"
#include "thrust/sort.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include "CollisionDetection.hh"
#include "ComponentManagerCommon.hh"
#include "ContactForceModel.hh"
#include "ContactForceModelFactory.hh"
#include "GrainsParameters.hh"
#include "NeighborList.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"
#include "Transform3.hh"
#include "Vector3.hh"
#include "VectorMath.hh"

// =============================================================================
/** @brief The header for GPU kernels used in the ComponentManagerGPU class.

    Various GPU kernels used in the ComponentManagerGPU class.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
/** @name ComponentManagerGPU_Kernels : External methods */
//@{
/** @brief Build a compact list of active pair indices using exclusive scan + 
    scatter.
    @param flagsDev is a device array of uint flags (0/1) of length nPairs.
    @param prefixDev is a device array of length nPairs to store scan results.
    @param activeIdxDev is a device array with capacity >= nPairs to receive 
    indices. */
INLINE uint buildCompactActiveIndex(const uint* flagsDev,
                                    const uint  nPairs,
                                    uint*       prefixDev,
                                    uint*       activeIdxDev)
{
    auto flagsPtr  = thrust::device_pointer_cast(const_cast<uint*>(flagsDev));
    auto prefixPtr = thrust::device_pointer_cast(prefixDev);

    // Exclusive scan to compute output positions for active entries
    thrust::exclusive_scan(thrust::device,
                           flagsPtr,
                           flagsPtr + nPairs,
                           prefixPtr);

    // Compute total active as last prefix + last flag
    uint lastPrefix = 0u, lastFlag = 0u;
    cudaMemcpy(&lastPrefix,
               prefixDev + (nPairs - 1),
               sizeof(uint),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastFlag,
               flagsDev + (nPairs - 1),
               sizeof(uint),
               cudaMemcpyDeviceToHost);
    const uint nActive = lastPrefix + lastFlag;

    // Scatter indices for active entries into compact array
    thrust::for_each_n(thrust::device,
                       thrust::make_counting_iterator<uint>(0u),
                       nPairs,
                       [activeIdxDev, flagsDev, prefixDev] __device__(uint i) {
                           if(flagsDev[i])
                               activeIdxDev[prefixDev[i]] = i;
                       });

    return nActive;
}

// -----------------------------------------------------------------------------
/** @brief Computes the relative transformations between pairs of components
    @param pairList list of pairs of components
    @param position array of positions for components
    @param quaternion array of quaternions for components
    @param relPosition array of relative positions for particles
    @param relQuaternion array of relative quaternions for particles
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    computeRelativeTransformations_Kernel(const uint2*         pairList,
                                          const Vector3<T>*    position,
                                          const Quaternion<T>* quaternion,
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

// -----------------------------------------------------------------------------
/** @brief Detects collisions between components
    @param pairList list of pairs of components
    @param rigidBody array of rigid bodies for components
    @param relPosition array of relative positions for components
    @param relQuaternion array of relative quaternions for components
    @param contactInfo array to store contact information
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    detectCollisionsComponents_Kernel(const uint2*               pairList,
                                      const RigidBody<T>* const* rigidBody,
                                      const Vector3<T>*          relPosition,
                                      const Quaternion<T>*       relQuaternion,
                                      ContactInfo<T>*            contactInfo,
                                      const uint                 nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    detectCollisionsComponents_common(pairList,
                                      rigidBody,
                                      relPosition,
                                      relQuaternion,
                                      contactInfo,
                                      tID);
}

// -----------------------------------------------------------------------------
/** @brief Transforms contact info to world and flags actives. */
template <typename T>
__GLOBAL__ void transformContactInfo_Kernel(const uint2*         pairList,
                                            const Vector3<T>*    position,
                                            const Quaternion<T>* quaternion,
                                            ContactInfo<T>* contactInfoLocal,
                                            ContactInfo<T>* contactInfoWorld,
                                            uint*           activePairs,
                                            const uint      nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    transformContactInfo_common(pairList,
                                position,
                                quaternion,
                                contactInfoLocal,
                                contactInfoWorld,
                                activePairs,
                                tID);
}

// -----------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of rigid bodies pairs
    @param contactInfo contact information
    @param rigidBody rigid body of components
    @param position position of the components
    @param velocity kinematics of the components
    @param torce torce acting on the components
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    computeContactForces_Kernel(const ContactForceModel<T>* const* CF,
                                const uint2*                       pairList,
                                const ContactInfo<T>*              contactInfo,
                                const RigidBody<T>* const*         rigidBody,
                                const Vector3<T>*                  position,
                                const Kinematics<T>*               velocity,
                                Torce<T>*                          torce,
                                const uint                         nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    computeContactForces_common(CF,
                                pairList,
                                contactInfo,
                                rigidBody,
                                position,
                                velocity,
                                torce,
                                tID);
}

// -----------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of rigid bodies pairs
    @param contactInfo contact information
    @param activeIdx list of active pair indices
    @param rigidBody rigid body of components
    @param position position of the components
    @param velocity kinematics of the components
    @param torce torce acting on the components
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    computeContactForcesCompact_Kernel(const ContactForceModel<T>* const* CF,
                                       const uint2*               pairList,
                                       const ContactInfo<T>*      contactInfo,
                                       const uint*                activeIdx,
                                       const RigidBody<T>* const* rigidBody,
                                       const Vector3<T>*          position,
                                       const Kinematics<T>*       velocity,
                                       Torce<T>*                  torce,
                                       const uint                 nActive)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nActive)
        return;

    const uint i = activeIdx[tID];
    computeContactForces_common(CF,
                                pairList,
                                contactInfo,
                                rigidBody,
                                position,
                                velocity,
                                torce,
                                i);
}

// -----------------------------------------------------------------------------
/** @brief Adds external forces such as gravity
    @param gx the gravity field - the x component
    @param gy the gravity field - the y component
    @param gz the gravity field - the z component
    @param rigidBody array of rigid bodies for components
    @param torce array of components torces
    @param nObstacles number of obstacles
    @param nParticles number of particles */
template <typename T>
__GLOBAL__ void addExternalForces_Kernel(const T                    gX,
                                         const T                    gY,
                                         const T                    gZ,
                                         const RigidBody<T>* const* rigidBody,
                                         Torce<T>*                  torce,
                                         const uint                 nObstacles,
                                         const uint                 nParticles)
{
    uint pID = blockIdx.x * blockDim.x + threadIdx.x;

    if(pID >= nParticles)
        return;

    addExternalForces_common(Vector3<T>(gX, gY, gZ),
                             rigidBody,
                             torce,
                             nObstacles + pID);
}

// -----------------------------------------------------------------------------
/** @brief Updates the position and velocities of particles
    @param TI time integrator scheme
    @param rigidBody array of rigid bodies for components
    @param position the position of the component
    @param quaternion array of components quaternions
    @param velocity array of components velocities
    @param torce array of components torces
    @param nObstacles number of obstacles
    @param nParticles number of particles */
template <typename T>
__GLOBAL__ void moveParticles_Kernel(const TimeIntegrator<T>* const* TI,
                                     const RigidBody<T>* const*      rigidBody,
                                     Vector3<T>*                     position,
                                     Quaternion<T>*                  quaternion,
                                     Kinematics<T>*                  velocity,
                                     Torce<T>*                       torce,
                                     const uint                      nObstacles,
                                     const uint                      nParticles)
{
    uint pID = blockIdx.x * blockDim.x + threadIdx.x;

    if(pID >= nParticles)
        return;

    moveParticles_common(TI,
                         rigidBody,
                         position,
                         quaternion,
                         velocity,
                         torce,
                         nObstacles + pID);
}
//@}

#endif