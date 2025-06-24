// TODO: CHANGE THE FORMAT FROM HH TO CUH LATER.
#ifndef _COMPONENTMANAGERGPU_KERNLES_CUH_
#define _COMPONENTMANAGERGPU_KERNLES_CUH_

#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <cooperative_groups.h>

#include "CollisionDetection.hh"
#include "ComponentManagerCommon.hh"
#include "ContactForceModel.hh"
#include "ContactForceModelFactory.hh"
#include "GrainsParameters.hh"
#include "LinkedCell.hh"
#include "LinkedCellGPUWrapper.hh"
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
/** @brief Zeros out the array
    @param array array to be zero-ed out
    @param numElements number of elements in the array */
__GLOBAL__
void zeroOutArray_Kernel(uint* array, uint numElements)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= numElements)
        return;

    array[tID] = 0;
}

// -----------------------------------------------------------------------------
/** @brief Returns the start Id for each hash value in cellStart
    @param componentCellHash sorted array of cell hash values
    @param numComponents number of components
    @param cellStartAndEnd start and end indices as s1, e1, s2, e2, ... */
__GLOBAL__
void sortComponentsAndFindCellStart_Kernel(const uint* componentCellHash,
                                           uint        numComponents,
                                           uint*       cellStart,
                                           uint*       cellEnd)
{
    // Handle to thread block group
    cooperative_groups::thread_block cta
        = cooperative_groups::this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint                   tID = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;
    if(tID < numComponents)
    {
        hash = componentCellHash[tID];
        // Load hash data into shared memory so that we can look at neighboring
        // component's hash value without loading two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;
        // first thread in block must load neighboring component hash as well
        if(tID > 0 && threadIdx.x == 0)
            sharedHash[0] = componentCellHash[tID - 1];
    }
    cooperative_groups::sync(cta);

    if(tID < numComponents)
    {
        // If this component has a different cell hash value to the previous
        // component then it must be the first component in the cell.
        // As it isn't the first component, it must also be the end of the
        // previous component's cell.

        if(tID == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = tID;
            if(tID > 0)
                cellEnd[sharedHash[threadIdx.x]] = tID; // excluding
        }
        if(tID == numComponents - 1)
            cellEnd[hash] = tID + 1;
    }
    // // Now use the sorted index to reorder the pos and vel data
    // uint sortedIndex = gridParticleIndex[index];
    // float4 pos = oldPos[sortedIndex];
    // float4 vel = oldVel[sortedIndex];

    // sortedPos[index] = pos;
    // sortedVel[index] = vel;
}

// -----------------------------------------------------------------------------
/** @brief Computes the relative transformations between pairs of components
    @param pairList list of pairs of components
    @param transform array of transformations for components
    @param relativeTransform array to store the relative transformations
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    computeRelativeTransformations_Kernel(const uint2*         pairList,
                                          const Transform3<T>* transform,
                                          Transform3<T>* relativeTransform,
                                          const uint     nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    computeRelativeTransformations_common(pairList,
                                          transform,
                                          relativeTransform,
                                          tID);
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between particles and obstacles
    @param pairList list of pairs of components
    @param particleRB array of rigid bodies for particles
    @param obstacleRB array of rigid bodies for obstacles
    @param transform array of transformations for particles
    @param obstacleTransform array of transformations for obstacles
    @param contactInfo array to store contact information
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    detectCollisionsObstacles_Kernel(const uint2*               pairList,
                                     const RigidBody<T>* const* particleRB,
                                     const RigidBody<T>* const* obstacleRB,
                                     const Transform3<T>*       transform,
                                     const Transform3<T>* obstacleTransform,
                                     ContactInfo<T>*      contactInfo,
                                     const uint           nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    detectCollisionsObstacles_common(pairList,
                                     particleRB,
                                     obstacleRB,
                                     transform,
                                     obstacleTransform,
                                     contactInfo,
                                     tID);
}

// -----------------------------------------------------------------------------
/** @brief Detects collisions between particles and particles
    @param pairList list of pairs of components
    @param particleRB array of rigid bodies for particles
    @param relTransform array of relative transformations for particles
    @param contactInfo array to store contact information
    @param nPairs number of pairs */
template <typename T>
__GLOBAL__ void
    detectCollisionsParticles_Kernel(const uint2*               pairList,
                                     const RigidBody<T>* const* particleRB,
                                     const Transform3<T>*       relTransform,
                                     ContactInfo<T>*            contactInfo,
                                     const uint                 nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    detectCollisionsParticles_common(pairList,
                                     particleRB,
                                     relTransform,
                                     contactInfo,
                                     tID);
}

// -----------------------------------------------------------------------------
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of rigid bodies pairs
    @param contactInfo contact information
    @param particleRB rigid body of particles
    @param velocity kinematics of the particles
    @param torce torce acting on the particles
    @param relTransform transformation of the particles */
template <typename T>
__GLOBAL__ void
    computeContactForces_Kernel(const ContactForceModel<T>* const* CF,
                                const uint2*                       pairList,
                                const ContactInfo<T>*              contactInfo,
                                const RigidBody<T>* const*         particleRB,
                                const Kinematics<T>*               velocity,
                                Torce<T>*                          torce,
                                const Transform3<T>*               relTransform,
                                const uint                         nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    computeContactForces_common(CF,
                                pairList,
                                contactInfo,
                                particleRB,
                                velocity,
                                torce,
                                relTransform,
                                tID);
}

// -----------------------------------------------------------------------------
/** @brief Adds external forces such as gravity
    @param gx the gravity field - the x component
    @param gy the gravity field - the y component
    @param gz the gravity field - the z component
    @param particleRB array of rigid bodies for particles
    @param torce array of particles torces
    @param nParticles number of particles */
template <typename T>
__GLOBAL__ void addExternalForces_Kernel(const T                    gX,
                                         const T                    gY,
                                         const T                    gZ,
                                         const RigidBody<T>* const* particleRB,
                                         Torce<T>*                  torce,
                                         const uint                 nParticles)
{
    uint pID = blockIdx.x * blockDim.x + threadIdx.x;

    if(pID >= nParticles)
        return;

    addExternalForces_common(Vector3<T>(gX, gY, gZ), particleRB, torce, pID);
}

// -----------------------------------------------------------------------------
/** @brief Updates the position and velocities of particles
    @param TI time integrator scheme
    @param particleRB array of rigid bodies for particles
    @param transform array of particles transformations
    @param quaternion array of particles quaternions
    @param velocity array of particles velocities
    @param torce array of particles torces
    @param rigidBodyId array of rigid body IDs for particles
    @param nParticles number of particles */
template <typename T>
__GLOBAL__ void moveParticles_Kernel(const TimeIntegrator<T>* const* TI,
                                     const RigidBody<T>* const*      particleRB,
                                     Transform3<T>*                  transform,
                                     Quaternion<T>*                  quaternion,
                                     Kinematics<T>*                  velocity,
                                     Torce<T>*                       torce,
                                     const uint* rigidBodyId,
                                     int         nParticles)
{
    uint pID = blockIdx.x * blockDim.x + threadIdx.x;

    if(pID >= nParticles)
        return;

    moveParticles_common(TI,
                         particleRB,
                         transform,
                         quaternion,
                         velocity,
                         torce,
                         rigidBodyId,
                         pID);
}
//@}

#endif