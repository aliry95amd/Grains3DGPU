// #include <curand.h>
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
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* particleRB,
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* obstacleRB,
    uint                                             nParticles,
    uint                                             nObstacles,
    uint                                             nCells)
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
    m_cellHashStart.reserve(m_nCells);
    m_cellHashEnd.reserve(m_nCells);
}

// -------------------------------------------------------------------------
// Initializes data members to default values
template <typename T>
void ComponentManagerGPU<T>::initialize()
{
    m_neighborList->createNeighborList(m_transform.getData(), m_nParticles);
}

// -----------------------------------------------------------------------------
// Updates links between particles and linked cell
template <typename T>
void ComponentManagerGPU<T>::updateNeighborList()
{
    if(m_neighborList->needsUpdate(m_transform.getData(), m_nParticles) == true)
        m_neighborList->updateNeighborList(m_transform.getData(), m_nParticles);
}

// -----------------------------------------------------------------------------
// Computes relative transformations
template <typename T>
void ComponentManagerGPU<T>::computeRelativeTransformations()
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_neighborList->getSize(),
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    computeRelativeTransformations_Kernel<<<numBlocks, numThreads>>>(
        m_neighborList->getData(),
        m_transform.getData(),
        m_relTransform.getData(),
        m_nPairs);
}

// -----------------------------------------------------------------------------
// Detects collision between particles and obstacles
template <typename T>
void ComponentManagerGPU<T>::detectCollisionsObstacles()
{
    using GP = GrainsParameters<T>;
    // Kernel launch parameters
    // const uint numThreads = GP::m_numThreads;
    // const uint numBlocks  = GP::m_numBlocks;

    // Invoke the kernel
    // detectCollisionAndComputeContactForcesObstacles_Kernel<<<numBlocks,
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
// Detects collisions between particles and particles
template <typename T>
void ComponentManagerGPU<T>::detectCollisionsParticles()
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_neighborList->getSize(),
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    detectCollisionsParticles_Kernel<<<numBlocks, numThreads>>>(
        m_neighborList->getData(),
        m_particleRB->getData(),
        m_relTransform.getData(),
        m_contactInfo.getData(),
        m_nPairs);
}

// -----------------------------------------------------------------------------
// Detects collision between all components
template <typename T>
void ComponentManagerGPU<T>::detectCollisions()
{
    // Updates links between components and linked cell
    updateNeighborList();

    // Computes the relative transformations
    computeRelativeTransformations();

    // Particle-particle interactions
    detectCollisionsParticles();

    // Particle-obstacle interactions
    detectCollisionsObstacles();
}

// -----------------------------------------------------------------------------
// Computes contact forces
template <typename T>
void ComponentManagerGPU<T>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_neighborList->getSize(),
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    computeContactForces_Kernel<<<numBlocks, numThreads>>>(
        CF.getData(),
        m_neighborList->getData(),
        m_contactInfo.getData(),
        m_particleRB->getData(),
        m_velocity.getData(),
        m_torce.getData(),
        m_relTransform.getData(),
        m_nPairs);
}

// -----------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerGPU<T>::addExternalForces()
{
    using GP = GrainsParameters<T>;

    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(GP::m_numParticles,
                                   GP::m_GPU,
                                   numBlocks,
                                   numThreads);

    // since g is a host-side vector, we need to break it into three components
    // to be able to pass it to the kernel
    const T gX = GP::m_gravity[X];
    const T gY = GP::m_gravity[Y];
    const T gZ = GP::m_gravity[Z];

    addExternalForces_Kernel<<<numBlocks, numThreads>>>(gX,
                                                        gY,
                                                        gZ,
                                                        m_particleRB->getData(),
                                                        m_torce.getData(),
                                                        m_nParticles);
}

// -----------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerGPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI)
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(GrainsParameters<T>::m_numParticles,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    moveParticles_Kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                    m_particleRB->getData(),
                                                    m_transform.getData(),
                                                    m_quaternion.getData(),
                                                    m_velocity.getData(),
                                                    m_torce.getData(),
                                                    m_rigidBodyId.getData(),
                                                    m_nParticles);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerGPU<float>;
template class ComponentManagerGPU<double>;