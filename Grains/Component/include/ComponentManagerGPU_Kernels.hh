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
/** @brief Computes the contact forces
    @param CF contact force models
    @param pairList list of rigid bodies pairs
    @param contactInfo contact information
    @param rigidBody rigid body of components
    @param velocity kinematics of the components
    @param torce torce acting on the components
    @param relPosition relative position of the components */
template <typename T>
__GLOBAL__ void
    computeContactForces_Kernel(const ContactForceModel<T>* const* CF,
                                const uint2*                       pairList,
                                const ContactInfo<T>*              contactInfo,
                                const RigidBody<T>* const*         rigidBody,
                                const Kinematics<T>*               velocity,
                                Torce<T>*                          torce,
                                const Vector3<T>*                  relPosition,
                                const uint                         nPairs)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;

    if(tID >= nPairs)
        return;

    computeContactForces_common(CF,
                                pairList,
                                contactInfo,
                                rigidBody,
                                velocity,
                                torce,
                                relPosition,
                                tID);
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