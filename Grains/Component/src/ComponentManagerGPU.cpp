// #include <curand.h>
#include "ComponentManagerGPU.hh"
#include "ComponentManagerGPU_Kernels.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU() = default;

// -------------------------------------------------------------------------------------------------
// Constructor with the number of particles, and obstacles
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU(
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* rigidBody, uint nObstacles, uint nParticles)
    : ComponentManager<T, MemType::DEVICE>(rigidBody, nObstacles, nParticles)
{
    this->initialize();
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
ComponentManagerGPU<T>::~ComponentManagerGPU() = default;

// -------------------------------------------------------------------------------------------------
// Initializes buffers for pair-dependent data
template <typename T>
void ComponentManagerGPU<T>::initialize()
{
    ComponentManager<T, MemType::DEVICE>::initialize();
    uint numPairs = m_relPosition.getSize();
    m_prefixScan.initialize(numPairs);
    m_activeIndex.initialize(numPairs);
}

// -------------------------------------------------------------------------------------------------
// Resizes pair-dependent buffers based on current neighbor list size
template <typename T>
void ComponentManagerGPU<T>::resizePairBuffers(const uint size)
{
    ComponentManager<T, MemType::DEVICE>::resizePairBuffers(size);
    m_prefixScan.resize(size);
    m_activeIndex.resize(size);
    m_intermediateTorceA.resize(size);
    m_intermediateTorceB.resize(size);
}

// -------------------------------------------------------------------------------------------------
// Updates the neighbor list if needed
template <typename T>
void ComponentManagerGPU<T>::updateNeighborList()
{
    // Static reference to simulation state
    auto& SS = GrainsParameters<T>::m_simulationState;

    bool updated = m_neighborList->updateNeighborList(m_position, m_numObstacles, m_numParticles);
    if(updated)
    {
        // Resize pair-dependent buffers in base, then GPU-specific buffers
        this->resizePairBuffers(m_neighborList->getSize());
        // Increment update counter and sort if needed
        SS.neighborListUpdateCount++;
    }
}

// -------------------------------------------------------------------------------------------------
// Computes relative transformations
template <typename T>
void ComponentManagerGPU<T>::computeRelativeTransformations()
{
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);

    computeRelativeTransformations_Kernel<<<numBlocks, numThreads>>>(m_neighborList->getData(),
                                                                     m_position.getData(),
                                                                     m_quaternion.getData(),
                                                                     m_relPosition.getData(),
                                                                     m_relQuaternion.getData(),
                                                                     nPairs);
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------------------------------
// Detects collisions between components
template <typename T>
// template <GJKType GJKVARIANT, bool GJKACC>
void ComponentManagerGPU<T>::detectCollisionsComponents()
{
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);

    detectCollisionsComponents_Kernel<<<numBlocks, numThreads>>>(m_neighborList->getData(),
                                                                 m_rigidBody->getData(),
                                                                 m_relPosition.getData(),
                                                                 m_relQuaternion.getData(),
                                                                 m_contactInfo.getData(),
                                                                 nPairs);
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------------------------------
// Transforms contact information to world frame
template <typename T>
void ComponentManagerGPU<T>::transformContactInfoToWorld()
{
    uint nPairs = m_neighborList->getSize();
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);
    transformContactInfo_Kernel<<<numBlocks, numThreads>>>(m_neighborList->getData(),
                                                           m_position.getData(),
                                                           m_quaternion.getData(),
                                                           m_contactInfo.getData(),
                                                           m_contactInfoWorld.getData(),
                                                           nPairs);
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------------------------------
// Detects collision between all components
template <typename T>
void ComponentManagerGPU<T>::detectCollisions()
{
    // Perform contact table cleanup periodically
    this->cleanupContactTable();

    // Sorts particles by Morton codes for improved cache efficiency
    this->sortParticles();

    // Updates links between components and linked cell
    updateNeighborList();

    // Computes the relative transformations
    computeRelativeTransformations();

    // Interactions
    detectCollisionsComponents();

    // Transforms contact info to world frame
    transformContactInfoToWorld();
}

// -------------------------------------------------------------------------------------------------
// Computes contact forces
template <typename T>
void ComponentManagerGPU<T>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& CF)
{
    uint nPairs = m_neighborList->getSize();
    // // Build compact index of active pairs using the shared helper and
    // persistent buffers const uint nActive =
    // buildCompactActiveIndex(m_activePairs.getData(),
    //                                              nPairs,
    //                                              m_prefixScan.getData(),
    //                                              m_activeIndex.getData());

    // // Launch compact forces kernel
    // uint numThreads, numBlocks;
    // computeOptimalThreadsAndBlocks(nActive,
    //                                GrainsParameters<T>::m_GPU,
    //                                numBlocks,
    //                                numThreads);
    // computeContactForcesCompact_Kernel<<<numBlocks, numThreads>>>(
    //     CF.getData(),
    //     m_neighborList->getData(),
    //     m_contactInfoWorld.getData(),
    //     m_activeIndex.getData(),
    //     m_rigidBody->getData(),
    //     m_position.getData(),
    //     m_velocity.getData(),
    //     m_torce.getData(),
    //     nActive);

    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);

    // Get the contact memory view (hash table + history data)
    ContactMemoryView<T> contactMemory = this->getContactMemoryView();

    // Kernel 1: Compute forces and write to intermediate per-pair storage (no race conditions)
    computeContactForces_Kernel<<<numBlocks, numThreads>>>(CF.getData(),
                                                           m_neighborList->getData(),
                                                           m_contactInfoWorld.getData(),
                                                           nullptr,
                                                           m_position.getData(),
                                                           m_velocity.getData(),
                                                           m_intermediateTorceA.getData(),
                                                           m_intermediateTorceB.getData(),
                                                           contactMemory,
                                                           nPairs);

    // Kernel 2: Reduce per-pair forces to per-particle torces using atomics
    reduceTorces_Kernel<<<numBlocks, numThreads>>>(m_neighborList->getData(),
                                                   nullptr,
                                                   m_intermediateTorceA.getData(),
                                                   m_intermediateTorceB.getData(),
                                                   m_torce.getData(),
                                                   nPairs);
}

// -------------------------------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerGPU<T>::addExternalForces()
{
    using GP = GrainsParameters<T>;

    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_numParticles, GP::m_GPU, numBlocks, numThreads);

    // since g is a host-side vector, we need to break it into three components to be able to pass
    // it to the kernel
    const T gX = GP::m_gravity[X];
    const T gY = GP::m_gravity[Y];
    const T gZ = GP::m_gravity[Z];

    addExternalForces_Kernel<<<numBlocks, numThreads>>>(gX,
                                                        gY,
                                                        gZ,
                                                        m_rigidBody->getData(),
                                                        m_torce.getData(),
                                                        m_numObstacles,
                                                        m_numParticles);
}

// -------------------------------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerGPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI)
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_numParticles,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    moveParticles_Kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                    m_rigidBody->getData(),
                                                    m_position.getData(),
                                                    m_quaternion.getData(),
                                                    m_velocity.getData(),
                                                    m_torce.getData(),
                                                    m_numObstacles,
                                                    m_numParticles);
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerGPU<float>;
template class ComponentManagerGPU<double>;