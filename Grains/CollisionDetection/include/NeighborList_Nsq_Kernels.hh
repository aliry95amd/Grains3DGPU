#ifndef _NEIGHBORLIST_NSQ_KERNELS_HH_
#define _NEIGHBORLIST_NSQ_KERNELS_HH_

#include "NeighborList_Nsq.hh"
#include <cuda_runtime.h>

// =============================================================================
/** @brief The header for the NeighborList_Nsq kernels.

    Axis-aligned Bounding Boxes (AABB) and Oriented Bounding Boxes (OBB)
    routines to find whether bounding boxes are in contact or not.
    AABB is deprecated, so try to use OBB.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name NeighborList_Nsq: External kernels */
//@{
/** @brief Creates the neighbor list on host
@param transforms array of transformations
@param nParticles number of particles
@param pairList array of pairs */
template <typename T>
__HOST__ bool createNeighborList_Host(const Transform3<T>* transforms,
                                      const uint           nParticles,
                                      uint2*               pairList)
{
    for(uint i = 0; i < nParticles; ++i)
        for(uint j = i + 1; j < nParticles; ++j)
            pairList[i + j * (j + 1) / 2] = make_uint2(i, j);
    return true;
};

/** @brief Creates the neighbor list on device
@param transforms array of transformations
@param nParticles number of particles
@param pairList array of pairs */
template <typename T>
__GLOBAL__ bool createNeighborList_Device(const Transform3<T>* transforms,
                                          const uint           nParticles,
                                          uint2*               pairList)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= nParticles)
        return false;

    for(uint j = tid + 1; j < nParticles; ++j)
        pairList[tid + j * (j + 1) / 2] = make_uint2(tid, j);
    return true;
};

#endif