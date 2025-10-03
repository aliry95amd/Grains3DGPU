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
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* rigidBody,
    uint                                             nObstacles,
    uint                                             nParticles)
    : ComponentManager<T, MemType::DEVICE>(rigidBody, nObstacles, nParticles)
{
    uint maxPairs
        = m_nObstacles * m_nParticles + m_nParticles * (m_nParticles - 1) / 2;
    m_prefixScan.initialize(maxPairs);
    m_prefixScan.fill();
    m_activeIndex.initialize(maxPairs);
    m_activeIndex.fill();
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
ComponentManagerGPU<T>::~ComponentManagerGPU() = default;

// -----------------------------------------------------------------------------
// Resizes pair-dependent buffers based on current neighbor list size
template <typename T>
void ComponentManagerGPU<T>::resizePairBuffers()
{
    uint nPairs = m_neighborList->getSize();
    m_prefixScan.setSize(nPairs);
    m_activeIndex.setSize(nPairs);
}

// -----------------------------------------------------------------------------
// Updates the neighbor list if needed
template <typename T>
void ComponentManagerGPU<T>::updateNeighborList()
{
    if(m_neighborList->needsUpdate())
    {
        m_neighborList->updateNeighborList(m_position,
                                           m_nObstacles,
                                           m_nParticles);

        // Resize pair-dependent buffers in base, then GPU-specific buffers
        ComponentManager<T, MemType::DEVICE>::resizePairBuffers();
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
// Detects collisions between components
template <typename T>
void ComponentManagerGPU<T>::detectCollisionsComponents()
{
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    detectCollisionsComponents_Kernel<<<numBlocks, numThreads>>>(
        m_neighborList->getData(),
        m_rigidBody->getData(),
        m_relPosition.getData(),
        m_relQuaternion.getData(),
        m_contactInfo.getData(),
        nPairs);
}

// -----------------------------------------------------------------------------
// Transforms contact information to world frame
template <typename T>
void ComponentManagerGPU<T>::transformContactInfoToWorld()
{
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);
    transformContactInfo_Kernel<<<numBlocks, numThreads>>>(
        m_neighborList->getData(),
        m_position.getData(),
        m_quaternion.getData(),
        m_contactInfo.getData(),
        m_contactInfoWorld.getData(),
        m_activePairs.getData(),
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

    // Interactions
    detectCollisionsComponents();

    // Transforms contact info to world frame
    transformContactInfoToWorld();
}

// -----------------------------------------------------------------------------
// Computes contact forces
template <typename T>
void ComponentManagerGPU<T>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
{
    uint nPairs = m_neighborList->getSize();
    // Build compact index of active pairs using the shared helper and persistent buffers
    const uint nActive = buildCompactActiveIndex(m_activePairs.getData(),
                                                 nPairs,
                                                 m_prefixScan.getData(),
                                                 m_activeIndex.getData());

    // Launch compact forces kernel
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nActive,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);
    computeContactForcesCompact_Kernel<<<numBlocks, numThreads>>>(
        CF.getData(),
        m_neighborList->getData(),
        m_contactInfoWorld.getData(),
        m_activeIndex.getData(),
        m_rigidBody->getData(),
        m_position.getData(),
        m_velocity.getData(),
        m_torce.getData(),
        nActive);
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
                                                        m_rigidBody->getData(),
                                                        m_torce.getData(),
                                                        m_nObstacles,
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
                                                    m_rigidBody->getData(),
                                                    m_position.getData(),
                                                    m_quaternion.getData(),
                                                    m_velocity.getData(),
                                                    m_torce.getData(),
                                                    m_nObstacles,
                                                    m_nParticles);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerGPU<float>;
template class ComponentManagerGPU<double>;