// TODO: CHANGE THE FORMAT FROM HH TO CUH LATER.
#ifndef _COMPONENTMANAGERGPU_KERNLES_CUH_
#define _COMPONENTMANAGERGPU_KERNLES_CUH_

#include <cuda_runtime.h>

#include "ComponentManagerCommon.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"
#include "Torce.hh"
#include "Vector3.hh"

// Force computation kernels (buildCompactActiveIndex, computeContactForces_Kernel,
// reduceTorces_Kernel, addExternalForces_Kernel) have been moved to ForceModule_Kernels.hh.

// =================================================================================================
/** @brief GPU kernels for the ComponentManagerGPU class (particle motion).

    Contact-force kernels have been moved to ForceModule_Kernels.hh.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
/** @name ComponentManagerGPU_Kernels : External methods */
//@{

// -------------------------------------------------------------------------------------------------
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

    moveParticles_common(TI, rigidBody, position, quaternion, velocity, torce, nObstacles + pID);
}

// -------------------------------------------------------------------------------------------------
/** @brief Performs the second velocity half-kick for split-step schemes (KDK Leapfrog Step 3).
    For single-pass schemes the underlying AdvanceVelocity is a no-op.
    @param TI time integrator scheme
    @param rigidBody array of rigid bodies for components
    @param quaternion array of components quaternions
    @param velocity array of components velocities
    @param torce array of components torces (read but NOT reset)
    @param nObstacles number of obstacles
    @param nParticles number of particles */
template <typename T>
__GLOBAL__ void advanceVelocity_Kernel(const TimeIntegrator<T>* const* TI,
                                       const RigidBody<T>* const*      rigidBody,
                                       const Quaternion<T>*            quaternion,
                                       Kinematics<T>*                  velocity,
                                       const Torce<T>*                 torce,
                                       const uint                      nObstacles,
                                       const uint                      nParticles)
{
    uint pID = blockIdx.x * blockDim.x + threadIdx.x;

    if(pID >= nParticles)
        return;

    advanceVelocity_common(TI, rigidBody, quaternion, velocity, torce, nObstacles + pID);
}
//@}

#endif
