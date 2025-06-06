// #include <curand.h>
#include "thrust/device_ptr.h"
#include "thrust/for_each.h"
#include "thrust/iterator/zip_iterator.h"
#include "thrust/sort.h"
#include <cooperative_groups.h>

#include "ComponentManagerGPU.hh"
#include "ComponentManagerGPU_Kernels.hh"
#include "LinkedCellGPUWrapper.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU() = default;

// -----------------------------------------------------------------------------
// Constructor with the number of particles, number of obstacles, and number of
// cells.
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU(
    GrainsMemBuffer<RigidBody<T, T>*, MemType::DEVICE>* particleRB,
    GrainsMemBuffer<RigidBody<T, T>*, MemType::DEVICE>* obstacleRB,
    uint                                                nParticles,
    uint                                                nObstacles,
    uint                                                nCells)
    : ComponentManager<T, MemType::DEVICE>(
          particleRB, obstacleRB, nParticles, nObstacles, nCells)
{
    allocate();
    initialize();
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
ComponentManagerGPU<T>::~ComponentManagerGPU() = default;

// -----------------------------------------------------------------------------
// Allocates memory for the component manager
template <typename T>
void ComponentManagerGPU<T>::allocate()
{
    m_particleCellHash.reserve(m_nParticles);
    m_cellHashStart.reserve(m_nCells + 1);
    m_cellHashEnd.reserve(m_nCells + 1);
}

// -------------------------------------------------------------------------
// Initializes data members to default values
template <typename T>
void ComponentManagerGPU<T>::initialize()
{
}

// -----------------------------------------------------------------------------
// Updates links between particles and linked cell
template <typename T>
void ComponentManagerGPU<T>::updateLinks(
    const GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>& LC)
{
    using GP = GrainsParameters<T>;
    // Kernel launch parameters
    const uint numThreads = GP::m_numThreadsPerBlock;
    const uint numBlocks  = GP::m_numBlocksPerGrid;
    uint       sMemSize   = sizeof(uint) * (numThreads + 1);

    // Zeroing out the arrays
    zeroOutArray_kernel<<<numBlocks, numThreads>>>(m_cellHashStart.getData(),
                                                   m_nCells + 1);
    zeroOutArray_kernel<<<numBlocks, numThreads>>>(m_cellHashEnd.getData(),
                                                   m_nCells + 1);

    // First - finding the cell hash for each particle
    computeLinearLinkedCellHashGPU_kernel<<<numBlocks, numThreads>>>(
        LC.getData(),
        m_transform.getData(),
        m_nParticles,
        m_particleCellHash.getData());

    // Second - sorting the particle ids according to the cell hash
    thrust::sort_by_key(
        thrust::device_ptr<uint>(m_particleCellHash.getData()),
        thrust::device_ptr<uint>(m_particleCellHash.getData() + m_nParticles),
        thrust::device_ptr<uint>(m_particleId.getData()));

    // Third - reseting the cellStart array and finding the start location of
    // each hash
    sortComponentsAndFindCellStart_kernel<<<numBlocks, numThreads, sMemSize>>>(
        m_particleCellHash.getData(),
        m_nParticles,
        m_cellHashStart.getData(),
        m_cellHashEnd.getData());
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between particles and obstacles
template <typename T>
void ComponentManagerGPU<T>::detectCollisionAndComputeContactForcesObstacles(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
{
    using GP = GrainsParameters<T>;
    // Kernel launch parameters
    // const uint numThreads = GP::m_numThreadsPerBlock;
    // const uint numBlocks  = GP::m_numBlocksPerGrid;

    // Invoke the kernel
    // detectCollisionAndComputeContactForcesObstacles_kernel<<<numBlocks,
    //                                                          numThreads>>>(
    //     particleRB,
    //     obstacleRB,
    //     CF,
    //     m_rigidBodyId,
    //     m_transform,
    //     m_velocity,
    //     m_torce,
    //     m_obstacleRigidBodyId,
    //     m_obstacleTransform,
    //     m_nParticles,
    //     m_nObstacles);
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between particles and particles
template <typename T>
void ComponentManagerGPU<T>::detectCollisionAndComputeContactForcesParticles(
    const GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>&        LC,
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
{
    using GP = GrainsParameters<T>;
    // Kernel launch parameters
    const uint numThreads = GP::m_numThreadsPerBlock;
    const uint numBlocks  = GP::m_numBlocksPerGrid;

    // Invoke the kernel
    detectCollisionAndComputeContactForcesParticles_kernel<<<numBlocks,
                                                             numThreads>>>(
        m_particleRB->getData(),
        LC.getData(),
        CF.getData(),
        m_rigidBodyId.getData(),
        m_transform.getData(),
        m_velocity.getData(),
        m_torce.getData(),
        m_particleId.getData(),
        m_particleCellHash.getData(),
        m_cellHashStart.getData(),
        m_cellHashEnd.getData(),
        m_nParticles);
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between all components
template <typename T>
void ComponentManagerGPU<T>::detectCollisionAndComputeContactForces(
    const GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>&        LC,
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
{
    // Updates links between components and linked cell
    updateLinks(LC);

    // Particle-particle interactions
    detectCollisionAndComputeContactForcesParticles(LC, CF);

    // Particle-obstacle interactions
    detectCollisionAndComputeContactForcesObstacles(CF);
}

// -----------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerGPU<T>::addExternalForces()
{
    using GP = GrainsParameters<T>;
    // Kernel launch parameters
    const uint numThreads = GP::m_numThreadsPerBlock;
    const uint numBlocks  = GP::m_numBlocksPerGrid;

    // since g is a host-side vector, we need to break it into three components
    // to be able to pass it to the kernel
    const T gX = GP::m_gravity[X];
    const T gY = GP::m_gravity[Y];
    const T gZ = GP::m_gravity[Z];

    // Invoke the kernel
    addExternalForces_kernel<<<numBlocks, numThreads>>>(gX,
                                                        gY,
                                                        gZ,
                                                        m_particleRB->getData(),
                                                        m_rigidBodyId.getData(),
                                                        m_torce.getData(),
                                                        m_nParticles);
}

// -----------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerGPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI)
{
    using GP = GrainsParameters<T>;
    // Kernel launch parameters
    const uint numThreads = GP::m_numThreadsPerBlock;
    const uint numBlocks  = GP::m_numBlocksPerGrid;

    // Invoke the kernel
    moveParticles_kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                    m_particleRB->getData(),
                                                    m_rigidBodyId.getData(),
                                                    m_transform.getData(),
                                                    m_velocity.getData(),
                                                    m_torce.getData(),
                                                    m_nParticles);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerGPU<float>;
template class ComponentManagerGPU<double>;