#include "ForceModule.hh"
#include "ForceModule_Kernels.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"

// =================================================================================================
// ForceModule – DEVICE (MemType::DEVICE) implementation
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// Constructor: allocates GPU compaction buffers and optionally the contact history table
template <typename T, MemType M>
ForceModule<T, M>::ForceModule(size_t pairCapacity, bool isContactWithMemory)
{
    // Persistent GPU compaction buffers (sized to pair capacity)
    m_prefixScan.initialize(pairCapacity);
    m_activeIndex.initialize(pairCapacity);

    if(isContactWithMemory)
    {
        uint hashCapacity = static_cast<uint>(pairCapacity / 0.7);
        m_contactTable.allocate(hashCapacity, static_cast<uint>(pairCapacity));
    }
}

// -------------------------------------------------------------------------------------------------
// resizeBuffers: resize GPU compaction buffers when pair buffer capacity grows
template <typename T, MemType M>
void ForceModule<T, M>::resizeBuffers(size_t newPairCapacity)
{
    m_prefixScan.resize(newPairCapacity);
    m_activeIndex.resize(newPairCapacity);
}

// -------------------------------------------------------------------------------------------------
// cleanupContactTable: periodic mark-and-sweep
template <typename T, MemType M>
void ForceModule<T, M>::cleanupContactTable()
{
    using GP = GrainsParameters<T>;
    if(!GP::m_isContactWithMemory)
        return;

    auto& SS = GP::m_simulationState;
    if(SS.neighborListUpdateCount % 1000 == 0)
        m_contactTable.markAndSweep();
}

// -------------------------------------------------------------------------------------------------
// run: DEVICE force computation pipeline
//   1. cleanup contact table
//   2. lazily resize intermediate per-pair torce buffers
//   3. computeContactForces_Kernel (per-pair → intermediate storage, no race conditions)
//   4. reduceTorces_Kernel (intermediate storage → per-particle torce via atomics)
//   5. addExternalForces_Kernel (gravity)
template <typename T, MemType M>
void ForceModule<T, M>::run(const GrainsMemBuffer<ContactForceModel<T>*, M>& CF,
                            const GrainsMemBuffer<ContactInfo<T>, M>&        contactInfo,
                            const GrainsMemBuffer<uint2, M>&                 pairList,
                            uint                                             numPairs,
                            const GrainsMemBuffer<Vector3<T>, M>&            position,
                            const GrainsMemBuffer<Kinematics<T>, M>&         velocity,
                            GrainsMemBuffer<Torce<T>, M>&                    torce,
                            const GrainsMemBuffer<RigidBody<T>*, M>*         rigidBody,
                            uint                                             numObstacles,
                            uint                                             numParticles)
{
    using GP = GrainsParameters<T>;

    // 1. Periodic cleanup of contact hash table
    cleanupContactTable();

    // 2. Lazily resize per-pair intermediate buffers to current pair count
    m_intermediateTorceA.resize(numPairs);
    m_intermediateTorceB.resize(numPairs);

    // 3. Compute contact forces → per-pair intermediate storage (no write-write races)
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(numPairs, GP::m_GPU, numBlocks, numThreads);

    ContactMemoryView<T> contactMemory = m_contactTable.getView();

    computeContactForces_Kernel<<<numBlocks, numThreads>>>(CF.getData(),
                                                           pairList.getData(),
                                                           contactInfo.getData(),
                                                           nullptr,
                                                           position.getData(),
                                                           velocity.getData(),
                                                           m_intermediateTorceA.getData(),
                                                           m_intermediateTorceB.getData(),
                                                           contactMemory,
                                                           numPairs);

    // 4. Reduce per-pair forces to per-particle torces using atomics
    reduceTorces_Kernel<<<numBlocks, numThreads>>>(pairList.getData(),
                                                   nullptr,
                                                   m_intermediateTorceA.getData(),
                                                   m_intermediateTorceB.getData(),
                                                   torce.getData(),
                                                   numPairs);

    // 5. Add external forces (gravity) to moving particles
    computeOptimalThreadsAndBlocks(numParticles, GP::m_GPU, numBlocks, numThreads);

    const T gX = GP::m_gravity[X];
    const T gY = GP::m_gravity[Y];
    const T gZ = GP::m_gravity[Z];

    addExternalForces_Kernel<<<numBlocks, numThreads>>>(gX,
                                                        gY,
                                                        gZ,
                                                        rigidBody->getData(),
                                                        torce.getData(),
                                                        numObstacles,
                                                        numParticles);
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation – DEVICE
template class ForceModule<float, MemType::DEVICE>;
template class ForceModule<double, MemType::DEVICE>;
