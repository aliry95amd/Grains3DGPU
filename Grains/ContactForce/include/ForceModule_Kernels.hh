// TODO: CHANGE THE FORMAT FROM HH TO CUH LATER.
#ifndef _FORCEMODULE_KERNELS_CUH_
#define _FORCEMODULE_KERNELS_CUH_

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "BodyTag.hh"
#include "ContactForceModel.hh"
#include "ForceModuleCommon.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "RigidBody.hh"
#include "Torce.hh"
#include "Vector3.hh"
#include "VectorMath.hh"

// =================================================================================================
/** @brief GPU kernels for the ForceModule class.

    Contains the contact-force and external-force kernels.

    @author A.Yazdani - 2026 - Construction */
// =================================================================================================
/** @name ForceModule GPU kernels */
//@{
/** @brief Flags pairs that are in contact (overlap distance < 0).
    @param contactInfo array of contact information
    @param flags       output flag array (1 = in contact, 0 = not in contact)
    @param nPairs      total number of pairs */
template <typename T>
__GLOBAL__ void
    flagActivePairs_Kernel(const ContactInfo<T>* contactInfo, uint* flags, const uint nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= nPairs)
        return;
    flags[tID] = (contactInfo[tID].getSnapshot().overlapDistance < T(0)) ? 1u : 0u;
}

// -------------------------------------------------------------------------------------------------
/** @brief Queries the temporary storage size required by CUB DeviceSelect::Flagged.
    @param nPairs       total number of pairs
    @param activeIdxDev device array receiving compact indices (used only for type deduction)
    @param numSelectedDev device pointer receiving the count of selected items
    @return required temporary storage size in bytes */
INLINE size_t queryCubSelectTempStorageBytes(uint nPairs, uint* activeIdxDev, uint* numSelectedDev)
{
    cub::CountingInputIterator<uint> countIter(0u);
    size_t                           bytes = 0;
    cudaErrCheck(cub::DeviceSelect::Flagged(nullptr,
                                            bytes,
                                            countIter,
                                            (const uint*)nullptr,
                                            activeIdxDev,
                                            numSelectedDev,
                                            static_cast<int>(nPairs)));
    return bytes;
}

// -------------------------------------------------------------------------------------------------
/** @brief Build a compact list of active pair indices using CUB DeviceSelect::Flagged.
    @param flagsDev        device array of uint flags (0/1) of length nPairs
    @param nPairs          total number of pairs
    @param activeIdxDev    device array with capacity >= nPairs to receive active indices
    @param numSelectedDev  device pointer to receive the count of selected pairs
    @param tempStorage     preallocated CUB temporary storage
    @param tempStorageBytes size of tempStorage in bytes
    @return number of active pairs (copied from device) */
INLINE uint buildCompactActiveIndex(const uint* flagsDev,
                                    uint        nPairs,
                                    uint*       activeIdxDev,
                                    uint*       numSelectedDev,
                                    void*       tempStorage,
                                    size_t      tempStorageBytes)
{
    cub::CountingInputIterator<uint> countIter(0u);
    cudaErrCheck(cub::DeviceSelect::Flagged(tempStorage,
                                            tempStorageBytes,
                                            countIter,
                                            flagsDev,
                                            activeIdxDev,
                                            numSelectedDev,
                                            static_cast<int>(nPairs)));
    uint nActive = 0u;
    cudaErrCheck(cudaMemcpy(&nActive, numSelectedDev, sizeof(uint), cudaMemcpyDeviceToHost));
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

// -------------------------------------------------------------------------------------------------
/** @brief Accumulates forces and torques from non-master sub-bodies into their composite master,
    then resets the sub-body torces. One thread per component.
    @param torce      per-component torce (sub-bodies read, master accumulated atomically, reset)
    @param position   world-frame positions (needed to compute moment arm r = sub - master)
    @param masterSlot lookup: masterSlot[compositeIdx] = current array slot of composite master
    @param bodyTag    per-component body tag (encodes isSubBody / compositeIdx / localIdx)
    @param nComponents total number of components (obstacles + particles) */
template <typename T>
__GLOBAL__ void assembleCompositeTorces_Kernel(Torce<T>*         torce,
                                               const Vector3<T>* position,
                                               const uint*       masterSlot,
                                               const uint*       bodyTag,
                                               const uint        nComponents)
{
    uint cID = blockIdx.x * blockDim.x + threadIdx.x;
    if(cID >= nComponents)
        return;

    const uint tag = bodyTag[cID];
    if(!isSubBody(tag) || getSubBodyLocalIdx(tag) == 0u)
        return;

    const uint       mSlot = masterSlot[getCompositeIdx(tag)];
    const Vector3<T> r     = position[cID] - position[mSlot];
    const Vector3<T> f     = torce[cID].getForce();
    const Vector3<T> tau   = torce[cID].getTorque() + (r ^ f);
    torce[mSlot].addForceAtomic(f);
    torce[mSlot].addTorqueAtomic(tau);
    torce[cID].reset();
}
//@}

#endif
