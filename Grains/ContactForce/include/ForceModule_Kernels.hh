// TODO: CHANGE THE FORMAT FROM HH TO CUH LATER.
#ifndef _FORCEMODULE_KERNELS_CUH_
#define _FORCEMODULE_KERNELS_CUH_

#include "thrust/device_ptr.h"
#include "thrust/execution_policy.h"
#include "thrust/for_each.h"
#include "thrust/iterator/counting_iterator.h"
#include "thrust/scan.h"
#include <cuda_runtime.h>

#include "ContactForceModel.hh"
#include "ForceModuleCommon.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "RigidBody.hh"
#include "Vector3.hh"
#include "VectorMath.hh"

// =================================================================================================
/** @brief GPU kernels for the ForceModule class.

    Contains the contact-force and external-force kernels migrated from
    ComponentManagerGPU_Kernels.hh, plus the active-pair compaction helper
    buildCompactActiveIndex.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
/** @name ForceModule GPU kernels */
//@{

// -------------------------------------------------------------------------------------------------
/** @brief Build a compact list of active pair indices using exclusive scan + scatter.
    @param flagsDev device array of uint flags (0/1) of length nPairs
    @param nPairs   total number of pairs
    @param prefixDev device array of length nPairs to store scan results
    @param activeIdxDev device array with capacity >= nPairs to receive active indices
    @return number of active pairs */
INLINE uint buildCompactActiveIndex(const uint* flagsDev,
                                    const uint  nPairs,
                                    uint*       prefixDev,
                                    uint*       activeIdxDev)
{
    cudaErrCheck(cudaGetLastError());
    auto flagsPtr  = thrust::device_pointer_cast(const_cast<uint*>(flagsDev));
    auto prefixPtr = thrust::device_pointer_cast(prefixDev);

    // Exclusive scan to compute output positions for active entries
    thrust::exclusive_scan(flagsPtr, flagsPtr + nPairs, prefixPtr);

    // Compute total active as last prefix + last flag
    uint lastPrefix = 0u, lastFlag = 0u;
    cudaMemcpy(&lastPrefix, prefixDev + (nPairs - 1), sizeof(uint), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastFlag, flagsDev + (nPairs - 1), sizeof(uint), cudaMemcpyDeviceToHost);
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

// -------------------------------------------------------------------------------------------------
/** @brief Computes the contact forces (writes to intermediate per-pair storage).
    @param CF contact force models
    @param pairList list of rigid bodies pairs
    @param contactInfo contact information
    @param activeIdx list of active pair indices (nullptr = process all pairs)
    @param position position of the components
    @param velocity kinematics of the components
    @param intermediateTorceA intermediate torce storage for particle A in each pair
    @param intermediateTorceB intermediate torce storage for particle B in each pair
    @param contactMemory view of contact memory (hash table + history data)
    @param nActive number of active pairs */
template <typename T>
__GLOBAL__ void computeContactForces_Kernel(const ContactForceModel<T>* const* CF,
                                            const uint2*                       pairList,
                                            const ContactInfo<T>*              contactInfo,
                                            const uint*                        activeIdx,
                                            const Vector3<T>*                  position,
                                            const Kinematics<T>*               velocity,
                                            Torce<T>*                          intermediateTorceA,
                                            Torce<T>*                          intermediateTorceB,
                                            ContactMemoryView<T>               contactMemory,
                                            const uint                         nActive)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nActive)
        return;

    if(activeIdx == nullptr)
    {
        computeContactForces_common(CF,
                                    pairList,
                                    contactInfo,
                                    position,
                                    velocity,
                                    intermediateTorceA,
                                    intermediateTorceB,
                                    contactMemory,
                                    tID);
    }
    else
    {
        const uint i = activeIdx[tID];
        computeContactForces_common(CF,
                                    pairList,
                                    contactInfo,
                                    position,
                                    velocity,
                                    intermediateTorceA,
                                    intermediateTorceB,
                                    contactMemory,
                                    i);
    }
}

// -------------------------------------------------------------------------------------------------
/** @brief Reduces per-pair intermediate torces to per-particle torces using atomics.
    @param pairList list of rigid bodies pairs
    @param activeIdx list of active pair indices (nullptr = process all pairs)
    @param intermediateTorceA intermediate torce storage for particle A in each pair
    @param intermediateTorceB intermediate torce storage for particle B in each pair
    @param torce final per-particle torce array (accumulated atomically)
    @param nActive number of active pairs */
template <typename T>
__GLOBAL__ void reduceTorces_Kernel(const uint2*    pairList,
                                    const uint*     activeIdx,
                                    const Torce<T>* intermediateTorceA,
                                    const Torce<T>* intermediateTorceB,
                                    Torce<T>*       torce,
                                    const uint      nActive)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nActive)
        return;

    if(activeIdx == nullptr)
    {
        reduceTorces_common(pairList, intermediateTorceA, intermediateTorceB, torce, tID);
    }
    else
    {
        const uint i = activeIdx[tID];
        reduceTorces_common(pairList, intermediateTorceA, intermediateTorceB, torce, i);
    }
}

// -------------------------------------------------------------------------------------------------
/** @brief Adds external forces such as gravity.
    @param gX x-component of the gravity vector
    @param gY y-component of the gravity vector
    @param gZ z-component of the gravity vector
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

    addExternalForces_common(Vector3<T>(gX, gY, gZ), rigidBody, torce, nObstacles + pID);
}
//@}

#endif
