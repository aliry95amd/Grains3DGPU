#include "ForceModule.hh"
#include "ForceModuleCommon.hh"
#include "GrainsParameters.hh"

// =================================================================================================
// ForceModule – HOST (MemType::HOST) implementation
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// Constructor: optionally allocates the contact history table
template <typename T, MemType M>
ForceModule<T, M>::ForceModule(size_t pairCapacity, bool isContactWithMemory)
{
    // GPU compaction / intermediate buffers are not needed on the HOST path
    if(isContactWithMemory)
    {
        uint hashCapacity = static_cast<uint>(pairCapacity / 0.7);
        m_contactTable.allocate(hashCapacity, static_cast<uint>(pairCapacity));
    }
}

// -------------------------------------------------------------------------------------------------
// resizeBuffers: no-op on HOST (no GPU compaction buffers to resize)
template <typename T, MemType M>
void ForceModule<T, M>::resizeBuffers(size_t /*newPairCapacity*/)
{
    // Nothing to resize for the HOST path
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
// run: HOST force computation pipeline
//   1. cleanup contact table
//   2. sequential contact force loop
//   3. external forces (gravity) loop
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
    // 1. Periodic cleanup of contact hash table
    cleanupContactTable();

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

// -------------------------------------------------------------------------------------------------
// Explicit instantiation – HOST
template class ForceModule<float, MemType::HOST>;
template class ForceModule<double, MemType::HOST>;
