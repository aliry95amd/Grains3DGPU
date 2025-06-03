// TODO: CHANGE THE FORMAT FROM CPP TO CU LATER.
#include <cooperative_groups.h>

#include "CollisionDetection.hh"
#include "ComponentManagerCommon.hh"
#include "ComponentManagerGPU_Kernels.hh"
#include "ContactForceModelFactory.hh"
#include "GrainsParameters.hh"
#include "LinkedCell.hh"
#include "LinkedCellGPUWrapper.hh"
#include "NeighborList.hh"
#include "VectorMath.hh"

// -----------------------------------------------------------------------------
// Zeros out the array
__GLOBAL__
void zeroOutArray_kernel(uint* array, uint numElements)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= numElements)
        return;

    array[tid] = 0;
}

// -----------------------------------------------------------------------------
// Returns the start Id for each hash value in cellStart
__GLOBAL__
void sortComponentsAndFindCellStart_kernel(const uint* componentCellHash,
                                           uint        numComponents,
                                           uint*       cellStart,
                                           uint*       cellEnd)
{
    // Handle to thread block group
    cooperative_groups::thread_block cta
        = cooperative_groups::this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint                   tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;
    if(tid < numComponents)
    {
        hash = componentCellHash[tid];
        // Load hash data into shared memory so that we can look at neighboring
        // component's hash value without loading two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;
        // first thread in block must load neighboring component hash as well
        if(tid > 0 && threadIdx.x == 0)
            sharedHash[0] = componentCellHash[tid - 1];
    }
    cooperative_groups::sync(cta);

    if(tid < numComponents)
    {
        // If this component has a different cell hash value to the previous
        // component then it must be the first component in the cell.
        // As it isn't the first component, it must also be the end of the
        // previous component's cell.

        if(tid == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = tid;
            if(tid > 0)
                cellEnd[sharedHash[threadIdx.x]] = tid; // excluding
        }
        if(tid == numComponents - 1)
            cellEnd[hash] = tid + 1;
    }
    // // Now use the sorted index to reorder the pos and vel data
    // uint sortedIndex = gridParticleIndex[index];
    // float4 pos = oldPos[sortedIndex];
    // float4 vel = oldVel[sortedIndex];

    // sortedPos[index] = pos;
    // sortedVel[index] = vel;
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between particles and components
template <typename T, typename U>
__GLOBAL__ void detectCollisionAndComputeContactForcesObstacles_kernel(
    RigidBody<T, U> const* const*      particleRB,
    RigidBody<T, U> const* const*      obstacleRB,
    ContactForceModel<T> const* const* CF,
    uint*                              rigidBodyId,
    const Transform3<T>*               transform,
    const Kinematics<T>*               velocity,
    Torce<T>*                          torce,
    uint*                              obstacleRigidBodyId,
    const Transform3<T>*               obstacleTransform,
    int                                nParticles,
    int                                nObstacles)
{
    uint pId = blockIdx.x * blockDim.x + threadIdx.x;

    if(pId >= nParticles)
        return;

    RigidBody<T, U> const& rbA   = *(particleRB[rigidBodyId[pId]]);
    const Transform3<T>&   trA   = transform[pId];
    T                      massA = rbA.getMass();
    uint                   matA  = rbA.getMaterial();

    for(int oId = 0; oId < nObstacles; oId++)
    {
        RigidBody<T, U> const& rbB = *(obstacleRB[obstacleRigidBodyId[oId]]);
        const Transform3<T>&   trB = obstacleTransform[oId];
        ContactInfo<T> ci = closestPointsRigidBodies(rbA, rbB, trA, trB);
        if(ci.getOverlapDistance() < T(0))
        {
            // CF ID given materialIDs
            uint contactForceID
                = ContactForceModelFactory<T>::computeHash(matA,
                                                           rbB.getMaterial());
            // velocities of the particles
            Kinematics<T> v1(velocity[pId]);
            // Kinematics<T> v2( m_velocity[ oId ] );
            Kinematics<T> v2;
            // geometric point of contact
            Vector3<T> contactPt(ci.getContactPoint());
            // relative velocity at contact point
            Vector3<T> relVel(v1.kinematicsAtPoint(contactPt)
                              - v2.kinematicsAtPoint(contactPt));
            // relative angular velocity
            Vector3<T> relAngVel(v1.getAngularComponent()
                                 - v2.getAngularComponent());
            CF[contactForceID]->computeForces(ci,
                                              relVel,
                                              relAngVel,
                                              massA,
                                              rbB.getMass(),
                                              trA.getOrigin(),
                                              torce[pId]);
        }
    }
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between particles and particles
template <typename T, typename U>
__GLOBAL__ void detectCollisionAndComputeContactForcesParticles_kernel(
    RigidBody<T, U> const* const*      particleRB,
    LinkedCell<T> const* const*        LC,
    ContactForceModel<T> const* const* CF,
    uint*                              rigidBodyId,
    const Transform3<T>*               transform,
    const Kinematics<T>*               velocity,
    Torce<T>*                          torce,
    uint*                              particleId,
    uint*                              particleCellHash,
    uint*                              cellHashStart,
    uint*                              cellHashEnd,
    int                                nParticles)
{
    uint pId = blockIdx.x * blockDim.x + threadIdx.x;

    if(pId >= nParticles)
        return;

    uint const             primaryId = particleId[pId];
    uint const             cellHash  = particleCellHash[pId];
    RigidBody<T, U> const& rbA       = *(particleRB[rigidBodyId[primaryId]]);
    const Transform3<T>&   trA       = transform[primaryId];
    T                      massA     = rbA.getMass();
    uint                   matA      = rbA.getMaterial();
    const uint*            neighborsList = (*LC)->getNeighbors(cellHash);

    for(int i = 0; i < 27; ++i)
    {
        // Get the neighboring cell hash
        uint neighborCellHash = neighborsList[i];
        // Check if the neighboring cell is valid
        if(neighborCellHash == UINT_MAX)
            continue;
        int startId = cellHashStart[neighborCellHash];
        int endId   = cellHashEnd[neighborCellHash];
        for(int id = startId; id < endId; id++)
        {
            int secondaryId = particleId[id];
            // To skip the self-collision
            if(secondaryId == primaryId)
                continue;
            RigidBody<T, U> const& rbB
                = *(particleRB[rigidBodyId[secondaryId]]);
            const Transform3<T>& trB = transform[secondaryId];
            ContactInfo<T> ci = closestPointsRigidBodies(rbA, rbB, trA, trB);
            if(ci.getOverlapDistance() < T(0))
            {
                // CF ID given materialIDs
                uint contactForceID = ContactForceModelFactory<T>::computeHash(
                    matA,
                    rbB.getMaterial());
                // velocities of the particles
                Kinematics<T> v1(velocity[primaryId]);
                Kinematics<T> v2(velocity[secondaryId]);
                // relative velocity at contact point
                Vector3<T> relVel(v1.kinematicsAtPoint(ci.getContactPoint())
                                  - v2.kinematicsAtPoint(ci.getContactPoint()));
                // relative angular velocity
                Vector3<T> relAngVel(v1.getAngularComponent()
                                     - v2.getAngularComponent());
                CF[contactForceID]->computeForces(ci,
                                                  relVel,
                                                  relAngVel,
                                                  massA,
                                                  rbB.getMass(),
                                                  trA.getOrigin(),
                                                  torce[primaryId]);
            }
        }
    }
}

// // -----------------------------------------------------------------------------
// // Detects collision and computes forces between particles and particles
// template <typename T, typename U>
// __GLOBAL__ void
//     detectCollisionsParticles_kernel(RigidBody<T, U> const* const* particleRB,
//                                      const uint2*                  neighborList,
//                                      const Transform3<T>*          transform,
//                                      const int                     nPairs)
// {
//     uint pId = blockIdx.x * blockDim.x + threadIdx.x;

//     if(pId >= nPairs)
//         return;

//     const uint2 pair = neighborList[pId];
//     const uint  p1   = pair.x;
//     const uint  p2   = pair.y;
//     // Get the rigid body IDs of the particles
//     const uint             primaryId   = particleId[p1];
//     const uint             secondaryId = particleId[p2];
//     const RigidBody<T, U>& rbA         = *(particleRB[rigidBodyId[primaryId]]);
//     const RigidBody<T, U>& rbB = *(particleRB[rigidBodyId[secondaryId]]);
//     const Transform3<T>&   trA = transform[primaryId];
//     const Transform3<T>&   trB = transform[secondaryId];
//     // Compute the contact information
//     ContactInfo<T> ci = closestPointsRigidBodies(rbA, rbB, trA, trB);
// }

// // -----------------------------------------------------------------------------
// // Detects collision and computes forces between particles and particles
// template <typename T, typename U>
// __GLOBAL__ void
//     computeContactForces_Kernel(RigidBody<T, U> const* const*      particleRB,
//                                 ContactForceModel<T> const* const* CF,
//                                 const uint*                        rigidBodyId,
//                                 const Kinematics<T>*               velocity,
//                                 Torce<T>*                          torce,
//                                 const uint                         nPairs)
// {
//     uint pId = blockIdx.x * blockDim.x + threadIdx.x;

//     if(pId >= nPairs)
//         return;

//     const ContactInfo<T> ci = CI[pId];
//     if(ci.getOverlapDistance() >= T(0))
//         return;

//     const uint2            pair        = neighborList[pId];
//     const uint             p1          = pair.x;
//     const uint             p2          = pair.y;
//     const uint             primaryId   = particleId[p1];
//     const uint             secondaryId = particleId[p2];
//     const RigidBody<T, U>& rbA         = *(particleRB[rigidBodyId[primaryId]]);
//     const T                massA       = rbA.getMass();
//     const uint             matA        = rbA.getMaterial();
//     const RigidBody<T, U>& rbB   = *(particleRB[rigidBodyId[secondaryId]]);
//     const T                massB = rbB.getMass();
//     const uint             matB  = rbB.getMaterial();

//     // CF ID given materialIDs
//     const uint contactForceID
//         = ContactForceModelFactory<T>::computeHash(matA, matB);
//     // velocities of the particles
//     const Kinematics<T> v1(velocity[primaryId]);
//     const Kinematics<T> v2(velocity[secondaryId]);
//     // relative velocity at contact point
//     const Vector3<T> relVel(v1.kinematicsAtPoint(ci.getContactPoint())
//                             - v2.kinematicsAtPoint(ci.getContactPoint()));
//     // relative angular velocity
//     const Vector3<T> relAngVel(v1.getAngularComponent()
//                                - v2.getAngularComponent());
//     // CF[contactForceID]->computeForces(ci,
//     //                                     relVel,
//     //                                     relAngVel,
//     //                                     massA,
//     //                                     massB,
//     //                                     trA.getOrigin(),
//     //                                     torce[primaryId]);
// }

// -----------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T, typename U>
__GLOBAL__ void
    addExternalForces_kernel(RigidBody<T, U> const* const* particleRB,
                             const uint*                   rigidBodyId,
                             const T                       gX,
                             const T                       gY,
                             const T                       gZ,
                             Torce<T>*                     torce,
                             const uint                    nParticles)
{
    uint pId = blockIdx.x * blockDim.x + threadIdx.x;

    if(pId >= nParticles)
        return;

    addGravity(particleRB,
               rigidBodyId[pId],
               Vector3<T>(gX, gY, gZ),
               torce[pId]);
}

// -----------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T, typename U>
__GLOBAL__ void moveParticles_kernel(const RigidBody<T, U>* const*   RB,
                                     const TimeIntegrator<T>* const* TI,
                                     uint*          rigidBodyId,
                                     Transform3<T>* transform,
                                     Kinematics<T>* velocity,
                                     Torce<T>*      torce,
                                     int            nParticles)
{
    uint pId = blockIdx.x * blockDim.x + threadIdx.x;

    if(pId >= nParticles)
        return;

    moveParticle(RB,
                 TI,
                 transform[pId],
                 velocity[pId],
                 torce[pId],
                 rigidBodyId[pId],
                 pId);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
#define X(T, U)                                                     \
    template __GLOBAL__ void                                        \
        detectCollisionAndComputeContactForcesObstacles_kernel(     \
            RigidBody<T, U> const* const*      particleRB,          \
            RigidBody<T, U> const* const*      obstacleRB,          \
            ContactForceModel<T> const* const* CF,                  \
            uint*                              rigidBodyId,         \
            const Transform3<T>*               transform,           \
            const Kinematics<T>*               velocity,            \
            Torce<T>*                          torce,               \
            uint*                              obstacleRigidBodyId, \
            const Transform3<T>*               obstacleTransform,   \
            int                                nParticles,          \
            int                                nObstacles);                                        \
                                                                    \
    template __GLOBAL__ void                                        \
        detectCollisionAndComputeContactForcesParticles_kernel(     \
            RigidBody<T, U> const* const*      particleRB,          \
            LinkedCell<T> const* const*        LC,                  \
            ContactForceModel<T> const* const* CF,                  \
            uint*                              rigidBodyId,         \
            const Transform3<T>*               transform,           \
            const Kinematics<T>*               velocity,            \
            Torce<T>*                          torce,               \
            uint*                              particleId,          \
            uint*                              particleCellHash,    \
            uint*                              cellHashStart,       \
            uint*                              cellHashEnd,         \
            int                                nParticles);                                        \
                                                                    \
    template __GLOBAL__ void addExternalForces_kernel(              \
        const RigidBody<T, U>* const* particleRB,                   \
        const uint*                   rigidBodyId,                  \
        const T                       gX,                           \
        const T                       gY,                           \
        const T                       gZ,                           \
        Torce<T>*                     torce,                        \
        const uint                    nParticles);                                     \
                                                                    \
    template __GLOBAL__ void moveParticles_kernel(                  \
        const RigidBody<T, U>* const*   RB,                         \
        const TimeIntegrator<T>* const* TI,                         \
        uint*                           rigidBodyId,                \
        Transform3<T>*                  transform,                  \
        Kinematics<T>*                  velocity,                   \
        Torce<T>*                       torce,                      \
        int                             nParticles);
X(float, float)
X(double, float)
X(double, double)
#undef X