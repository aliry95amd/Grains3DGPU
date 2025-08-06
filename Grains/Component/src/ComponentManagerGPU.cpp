// #include <curand.h>
#include "ComponentManagerGPU.hh"
#include "ComponentManagerGPU_Kernels.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU() = default;

// -----------------------------------------------------------------------------
// Constructor with the number of particles, and obstacles
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU(
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* particleRB,
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* obstacleRB,
    uint                                             nParticles,
    uint                                             nObstacles)
    : ComponentManager<T, MemType::DEVICE>(
          particleRB, obstacleRB, nParticles, nObstacles)
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
}

// -------------------------------------------------------------------------
// Initializes data members to default values
template <typename T>
void ComponentManagerGPU<T>::initialize()
{
}

// -----------------------------------------------------------------------------
// Updates the neighbor list if needed
template <typename T>
void ComponentManagerGPU<T>::updateNeighborList()
{
    if(m_neighborList->needsUpdate())
    {
        m_neighborList->updateNeighborList(m_position);

        // Resize pair-dependent buffers to match actual number of pairs
        this->resizePairBuffers();
    }
}

// -----------------------------------------------------------------------------
// Computes relative transformations
template <typename T>
void ComponentManagerGPU<T>::computeRelativeTransformations()
{
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    computeRelativeTransformations_Kernel<<<numBlocks, numThreads>>>(
        m_neighborList->getData(),
        m_position.getData(),
        m_quaternion.getData(),
        m_relPosition.getData(),
        m_relQuaternion.getData(),
        nPairs);
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
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    detectCollisionsParticles_Kernel<<<numBlocks, numThreads>>>(
        m_neighborList->getData(),
        m_particleRB->getData(),
        m_relPosition.getData(),
        m_relQuaternion.getData(),
        m_contactInfo.getData(),
        nPairs);
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
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs,
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
        m_relPosition.getData(),
        nPairs);
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
                                                    m_position.getData(),
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