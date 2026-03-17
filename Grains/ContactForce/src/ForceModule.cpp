#include "ForceModule.hh"
#include "ForceModuleCommon.hh"
#include "ForceModule_Kernels.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"

// -------------------------------------------------------------------------------------------------
// Constructor
template <typename T, MemType M>
ForceModule<T, M>::ForceModule(size_t pairCapacity, bool isContactWithMemory)
{
    if constexpr(M == MemType::DEVICE)
    {
        // Compaction buffers sized to pair capacity
        m_activeIndex.initialize(pairCapacity);
        m_activeFlags.initialize(pairCapacity);
        m_numActivePairs.initialize(1);
        const size_t cubBytes = queryCubSelectTempStorageBytes(static_cast<uint>(pairCapacity),
                                                               m_activeIndex.getData(),
                                                               m_numActivePairs.getData());
        m_cubSelectTempStorage.initialize(cubBytes);
    }

    if(isContactWithMemory)
    {
        uint hashCapacity = static_cast<uint>(pairCapacity / 0.7);
        m_contactTable.allocate(hashCapacity, static_cast<uint>(pairCapacity));
    }
}

// -------------------------------------------------------------------------------------------------
// Resizes GPU compaction buffers when pair buffer capacity grows; no-op on the HOST path.
template <typename T, MemType M>
void ForceModule<T, M>::resizeBuffers(size_t newPairCapacity)
{
    if constexpr(M == MemType::DEVICE)
    {
        m_activeIndex.resize(newPairCapacity);
        m_activeFlags.resize(newPairCapacity);
        const size_t cubBytes = queryCubSelectTempStorageBytes(static_cast<uint>(newPairCapacity),
                                                               m_activeIndex.getData(),
                                                               m_numActivePairs.getData());
        m_cubSelectTempStorage.resize(cubBytes);
    }
}

// -------------------------------------------------------------------------------------------------
// Periodic mark-and-sweep
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
// Runs the complete force computation pipeline.
template <typename T, MemType M>
void ForceModule<T, M>::run(const GrainsMemBuffer<ContactForceModel<T>*, M>& CF,
                            const GrainsMemBuffer<RigidBody<T>*, M>*         rigidBody,
                            const GrainsMemBuffer<Vector3<T>, M>&            position,
                            const GrainsMemBuffer<Kinematics<T>, M>&         velocity,
                            const GrainsMemBuffer<uint2, M>&                 pairList,
                            const GrainsMemBuffer<ContactInfo<T>, M>&        contactInfo,
                            uint                                             numPairs,
                            GrainsMemBuffer<Torce<T>, M>&                    torce,
                            uint                                             numObstacles,
                            uint                                             numParticles)
{
    // 1. Periodic cleanup of contact hash table
    cleanupContactTable();

    if constexpr(M == MemType::HOST)
    {
        // 2. Compute contact forces (sequential per-pair)
        ContactMemoryView<T> contactMemory = m_contactTable.getView();
        for(uint i = 0; i < numPairs; ++i)
        {
            computeContactForces_common(CF.getData(),
                                        pairList.getData(),
                                        contactInfo.getData(),
                                        position.getData(),
                                        velocity.getData(),
                                        torce.getData(),
                                        contactMemory,
                                        i);
        }

        // 3. Add external forces (gravity) to moving particles
        for(uint pID = numObstacles; pID < numObstacles + numParticles; ++pID)
        {
            addExternalForces_common(GrainsParameters<T>::m_gravity,
                                     rigidBody->getData(),
                                     torce.getData(),
                                     pID);
        }
    }
    else if constexpr(M == MemType::DEVICE)
    {
        using GP = GrainsParameters<T>;

        uint numThreads, numBlocks;

        // 2. Flag active pairs (overlap < 0)
        computeOptimalThreadsAndBlocks(numPairs, GP::m_GPU, numBlocks, numThreads);
        flagActivePairs_Kernel<<<numBlocks, numThreads>>>(contactInfo.getData(),
                                                          m_activeFlags.getData(),
                                                          numPairs);

        // 3. Compact active pair indices using CUB DeviceSelect::Flagged
        const uint nActive = buildCompactActiveIndex(m_activeFlags.getData(),
                                                     numPairs,
                                                     m_activeIndex.getData(),
                                                     m_numActivePairs.getData(),
                                                     m_cubSelectTempStorage.getData(),
                                                     m_cubSelectTempStorage.getSize());

        if(nActive > 0)
        {
            // 4. Lazily resize per-pair intermediate buffers (indexed by original pair ID)
            m_intermediateTorceA.resize(numPairs);
            m_intermediateTorceB.resize(numPairs);

            // 5. Compute contact forces for active pairs only
            computeOptimalThreadsAndBlocks(nActive, GP::m_GPU, numBlocks, numThreads);
            ContactMemoryView<T> contactMemory = m_contactTable.getView();
            computeContactForces_Kernel<<<numBlocks, numThreads>>>(CF.getData(),
                                                                   pairList.getData(),
                                                                   contactInfo.getData(),
                                                                   m_activeIndex.getData(),
                                                                   position.getData(),
                                                                   velocity.getData(),
                                                                   m_intermediateTorceA.getData(),
                                                                   m_intermediateTorceB.getData(),
                                                                   contactMemory,
                                                                   nActive);

            // 6. Reduce per-pair forces to per-particle torces using atomics
            reduceTorces_Kernel<<<numBlocks, numThreads>>>(pairList.getData(),
                                                           m_activeIndex.getData(),
                                                           m_intermediateTorceA.getData(),
                                                           m_intermediateTorceB.getData(),
                                                           torce.getData(),
                                                           nActive);
        }

        // 7. Add external forces (gravity) to moving particles
        computeOptimalThreadsAndBlocks(numParticles, GP::m_GPU, numBlocks, numThreads);
        addExternalForces_Kernel<<<numBlocks, numThreads>>>(GP::m_gravity[X],
                                                            GP::m_gravity[Y],
                                                            GP::m_gravity[Z],
                                                            rigidBody->getData(),
                                                            torce.getData(),
                                                            numObstacles,
                                                            numParticles);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiations
template class ForceModule<float, MemType::HOST>;
template class ForceModule<double, MemType::HOST>;
template class ForceModule<float, MemType::DEVICE>;
template class ForceModule<double, MemType::DEVICE>;
