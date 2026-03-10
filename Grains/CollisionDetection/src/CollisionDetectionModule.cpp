#include <cub/cub.cuh>

#include "CollisionDetectionCommon.hh"
#include "CollisionDetectionModule.hh"
#include "CollisionDetectionModule_Kernels.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"

// -------------------------------------------------------------------------------------------------
// Constructor: builds the NeighborList object, then allocates pair-indexed buffers
template <typename T, MemType M>
CollisionDetectionModule<T, M>::CollisionDetectionModule(
    const GrainsMemBuffer<RigidBody<T>*, M>* rigidBody,
    const GrainsMemBuffer<Vector3<T>, M>&    positions,
    const GrainsMemBuffer<Quaternion<T>, M>& orientations,
    const CollisionDetectionParameters<T>&   CD,
    uint                                     nObstacles,
    uint                                     nParticles)
    : m_particleSorter(nObstacles, nParticles)
    , m_neighborList(NeighborListFactory<T, M>::create(
          rigidBody, positions, orientations, CD, nObstacles, nParticles))
{
    size_t freeMem;
    if constexpr(M == MemType::HOST)
        freeMem = getAvailableHostMemory();
    else
        freeMem = getAvailableDeviceMemory();

    const auto& LCD       = GrainsParameters<T>::m_collisionDetection.linkedCellParameters;
    size_t estimatedPairs = static_cast<size_t>(nParticles) * LCD.initialNumberOfPairsPerParticle;
    size_t maxPairs       = static_cast<size_t>(nObstacles) * nParticles
                      + static_cast<size_t>(nParticles) * (nParticles - 1) / 2;
    estimatedPairs = std::min(estimatedPairs, maxPairs);

    size_t sizePerPair = sizeof(Vector3<T>) + sizeof(Quaternion<T>) + sizeof(ContactInfo<T>)
                         + sizeof(uint8_t) + sizeof(uint);
    size_t sizeNeeded = estimatedPairs * sizePerPair;
    GAssert(sizeNeeded < freeMem,
            "Not enough memory for pair-dependent buffers in CollisionDetectionModule!");

    m_relPosition.initialize(estimatedPairs);
    m_relQuaternion.initialize(estimatedPairs);
    m_contactInfoLocal.initialize(estimatedPairs);
    if constexpr(M == MemType::DEVICE)
    {
        m_bvPassFlags.initialize(estimatedPairs);
        m_bvPassPairIndices.initialize(estimatedPairs);
        m_bvPassPairCountDevice.initialize(1);
        // Query CUB scratch size for DeviceSelect::Flagged over estimatedPairs elements
        cub::CountingInputIterator<uint> countIter(0);
        cub::DeviceSelect::Flagged(nullptr,
                                   m_cubTempStorageBytes,
                                   countIter,
                                   (uint8_t*)nullptr,
                                   (uint*)nullptr,
                                   (int*)nullptr,
                                   (int)estimatedPairs);
        m_cubTempStorage.initialize(m_cubTempStorageBytes);
    }
}

// -------------------------------------------------------------------------------------------------
// Gets the current pair list pointer from the neighbor list
template <typename T, MemType M>
const uint2* CollisionDetectionModule<T, M>::getPairList() const
{
    return m_neighborList->getData();
}

// -------------------------------------------------------------------------------------------------
// Gets the current number of active pairs in the neighbor list
template <typename T, MemType M>
uint CollisionDetectionModule<T, M>::getPairCount() const
{
    return m_neighborList->getSize();
}

// -------------------------------------------------------------------------------------------------
// Gets the allocated size of the pair buffers (may exceed getPairCount())
template <typename T, MemType M>
size_t CollisionDetectionModule<T, M>::getPairBufferSize() const
{
    return m_relPosition.getSize();
}

// -------------------------------------------------------------------------------------------------
// Gets the neighbor list (read-only)
template <typename T, MemType M>
const NeighborList<T, M>* CollisionDetectionModule<T, M>::getNeighborList() const
{
    return m_neighborList.get();
}

// -------------------------------------------------------------------------------------------------
// Runs the full collision detection pipeline
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::run(const RigidBody<T>* const*          rigidBodies,
                                         GrainsMemBuffer<Vector3<T>, M>&     positions,
                                         GrainsMemBuffer<Quaternion<T>, M>&  orientations,
                                         GrainsMemBuffer<Kinematics<T>, M>&  velocities,
                                         GrainsMemBuffer<Torce<T>, M>&       torces,
                                         GrainsMemBuffer<uint, M>&           rigidBodyIds,
                                         GrainsMemBuffer<uint, M>&           componentIds,
                                         GrainsMemBuffer<ContactInfo<T>, M>& contactInfo,
                                         GrainsMemBuffer<uint2, M>&          pairList,
                                         uint&                               numPairs,
                                         uint                                nObstacles,
                                         uint                                nParticles)
{
    sortParticles(positions,
                  orientations,
                  velocities,
                  torces,
                  rigidBodyIds,
                  componentIds,
                  nObstacles,
                  nParticles);
    updateNeighborList(positions, pairList, contactInfo, numPairs, nObstacles, nParticles);
    computeRelativeTransformations(positions, orientations, pairList);
    detectCollisionsComponents(rigidBodies, pairList);
    transformContactInfo(positions, orientations, pairList, contactInfo);
}

// -------------------------------------------------------------------------------------------------
// Resizes per-pair buffers and ComponentManager's contactInfo / pairList
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::resizePairBuffers(
    GrainsMemBuffer<uint2, M>& pairList, GrainsMemBuffer<ContactInfo<T>, M>& contactInfo, uint size)
{
    m_relPosition.resize(size);
    m_relQuaternion.resize(size);
    m_contactInfoLocal.resize(size);
    if constexpr(M == MemType::DEVICE)
    {
        m_bvPassFlags.resize(size);
        m_bvPassPairIndices.resize(size);
        // Re-query CUB scratch size for the new capacity
        cub::CountingInputIterator<uint> countIter(0);
        size_t                           newBytes = 0;
        cub::DeviceSelect::Flagged(nullptr,
                                   newBytes,
                                   countIter,
                                   (uint8_t*)nullptr,
                                   (uint*)nullptr,
                                   (int*)nullptr,
                                   (int)size);
        if(newBytes > m_cubTempStorageBytes)
        {
            m_cubTempStorageBytes = newBytes;
            m_cubTempStorage.resize(m_cubTempStorageBytes);
        }
    }
    contactInfo.resize(size);
    pairList.resize(size);
}

// -------------------------------------------------------------------------------------------------
// Updates the neighbor list; resizes buffers and increments global counter when rebuilt
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::updateNeighborList(
    GrainsMemBuffer<Vector3<T>, M>&     positions,
    GrainsMemBuffer<uint2, M>&          pairList,
    GrainsMemBuffer<ContactInfo<T>, M>& contactInfo,
    uint&                               numPairs,
    uint                                nObstacles,
    uint                                nParticles)
{
    auto& SS = GrainsParameters<T>::m_simulationState;

    bool updated = m_neighborList->updateNeighborList(positions, nObstacles, nParticles);
    if(updated)
    {
        const uint newSize = m_neighborList->getSize();
        resizePairBuffers(pairList, contactInfo, newSize);
        m_neighborList->getBuffer().copyTo(pairList);
        numPairs = newSize;
        SS.neighborListUpdateCount++;
    }
}

// -------------------------------------------------------------------------------------------------
// Computes per-pair relative position / quaternion (B in A-local frame)
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::computeRelativeTransformations(
    const GrainsMemBuffer<Vector3<T>, M>&    positions,
    const GrainsMemBuffer<Quaternion<T>, M>& orientations,
    const GrainsMemBuffer<uint2, M>&         pairList)
{
    const uint nPairs = m_neighborList->getSize();

    if constexpr(M == MemType::HOST)
    {
        for(uint i = 0; i < nPairs; ++i)
            computeRelativeTransformations_common(pairList.getData(),
                                                  positions.getData(),
                                                  orientations.getData(),
                                                  m_relPosition.getData(),
                                                  m_relQuaternion.getData(),
                                                  i);
    }
    else
    {
        uint numThreads, numBlocks;
        computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);

        computeRelativeTransformations_Kernel<<<numBlocks, numThreads>>>(positions.getData(),
                                                                         orientations.getData(),
                                                                         pairList.getData(),
                                                                         m_relPosition.getData(),
                                                                         m_relQuaternion.getData(),
                                                                         nPairs);
        cudaDeviceSynchronize();
    }
}

// -------------------------------------------------------------------------------------------------
// Performs a narrow-phase GJK-based collision detection for each pair and writes contact info in
// A-local frame. Might also opionally perform a BV pre-filter.
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::detectCollisionsComponents(
    const RigidBody<T>* const* rigidBodies, const GrainsMemBuffer<uint2, M>& pairList)
{
    const uint               nPairs = m_neighborList->getSize();
    const BoundingVolumeType bvType = GrainsParameters<T>::m_collisionDetection.boundingVolumeType;

    if constexpr(M == MemType::HOST)
    {
        for(uint i = 0; i < nPairs; ++i)
        {
            if(bvType == BoundingVolumeType::OBB)
                detectCollisionsComponents_common<T,
                                                  GJKType::JOHNSON,
                                                  false,
                                                  BoundingVolumeType::OBB>(
                    pairList.getData(),
                    rigidBodies,
                    m_relPosition.getData(),
                    m_relQuaternion.getData(),
                    m_contactInfoLocal.getData(),
                    i);
            else
                detectCollisionsComponents_common(pairList.getData(),
                                                  rigidBodies,
                                                  m_relPosition.getData(),
                                                  m_relQuaternion.getData(),
                                                  m_contactInfoLocal.getData(),
                                                  i);
        }
    }
    else
    {
        if(bvType == BoundingVolumeType::OBB)
        {
            filterPairsBV(rigidBodies, pairList, m_contactInfoLocal);
            uint numThreads, numBlocks;
            computeOptimalThreadsAndBlocks((uint)m_bvPassPairCount,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            detectCollisionsComponents_Kernel<T, GJKType::JOHNSON, false>
                <<<numBlocks, numThreads>>>(rigidBodies,
                                            pairList.getData(),
                                            m_bvPassPairIndices.getData(),
                                            m_relPosition.getData(),
                                            m_relQuaternion.getData(),
                                            m_contactInfoLocal.getData(),
                                            (uint)m_bvPassPairCount);
        }
        else
        {
            uint numThreads, numBlocks;
            computeOptimalThreadsAndBlocks(nPairs,
                                           GrainsParameters<T>::m_GPU,
                                           numBlocks,
                                           numThreads);
            detectCollisionsComponents_Kernel<T, GJKType::JOHNSON, false>
                <<<numBlocks, numThreads>>>(rigidBodies,
                                            pairList.getData(),
                                            nullptr,
                                            m_relPosition.getData(),
                                            m_relQuaternion.getData(),
                                            m_contactInfoLocal.getData(),
                                            nPairs);
        }
        cudaDeviceSynchronize();
    }
}

// -------------------------------------------------------------------------------------------------
// Launches a BV kernel to write flags, then uses CUB DeviceSelect::Flagged to compact passing pair
// indices into m_bvPassPairIndices. Only invoked on the DEVICE path.
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::filterPairsBV(
    const RigidBody<T>* const*          rigidBodies,
    const GrainsMemBuffer<uint2, M>&    pairList,
    GrainsMemBuffer<ContactInfo<T>, M>& contactInfoLocal)
{
    if constexpr(M == MemType::DEVICE)
    {
        const uint nPairs = m_neighborList->getSize();
        uint       numThreads, numBlocks;
        computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);

        // Step 1: BV pass/fail flags + no-contact sentinels for rejected pairs
        filterPairsBV_Kernel<T, BoundingVolumeType::OBB>
            <<<numBlocks, numThreads>>>(rigidBodies,
                                        pairList.getData(),
                                        m_relPosition.getData(),
                                        m_relQuaternion.getData(),
                                        contactInfoLocal.getData(),
                                        m_bvPassFlags.getData(),
                                        nPairs);
        cudaDeviceSynchronize();

        // Step 2: CUB compaction - select original pair indices where flag == 1
        cub::CountingInputIterator<uint> countIter(0);
        cub::DeviceSelect::Flagged(m_cubTempStorage.getData(),
                                   m_cubTempStorageBytes,
                                   countIter,
                                   m_bvPassFlags.getData(),
                                   m_bvPassPairIndices.getData(),
                                   m_bvPassPairCountDevice.getData(),
                                   (int)nPairs);
        cudaDeviceSynchronize();

        // Step 3: copy passing count to host
        cudaMemcpy(&m_bvPassPairCount,
                   m_bvPassPairCountDevice.getData(),
                   sizeof(int),
                   cudaMemcpyDeviceToHost);
    }
}

// -------------------------------------------------------------------------------------------------
// Transforms contact info from A-local frame to world frame
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::transformContactInfo(
    const GrainsMemBuffer<Vector3<T>, M>&    positions,
    const GrainsMemBuffer<Quaternion<T>, M>& orientations,
    const GrainsMemBuffer<uint2, M>&         pairList,
    GrainsMemBuffer<ContactInfo<T>, M>&      contactInfo)
{
    const uint nPairs = m_neighborList->getSize();

    if constexpr(M == MemType::HOST)
    {
        for(uint i = 0; i < nPairs; ++i)
            transformContactInfo_common(pairList.getData(),
                                        positions.getData(),
                                        orientations.getData(),
                                        m_contactInfoLocal.getData(),
                                        contactInfo.getData(),
                                        i);
    }
    else
    {
        uint numThreads, numBlocks;
        computeOptimalThreadsAndBlocks(nPairs, GrainsParameters<T>::m_GPU, numBlocks, numThreads);

        transformContactInfo_Kernel<<<numBlocks, numThreads>>>(positions.getData(),
                                                               orientations.getData(),
                                                               pairList.getData(),
                                                               m_contactInfoLocal.getData(),
                                                               contactInfo.getData(),
                                                               nPairs);
        cudaDeviceSynchronize();
    }
}

// -------------------------------------------------------------------------------------------------
// Sorts particles by Morton codes
template <typename T, MemType M>
void CollisionDetectionModule<T, M>::sortParticles(GrainsMemBuffer<Vector3<T>, M>&    positions,
                                                   GrainsMemBuffer<Quaternion<T>, M>& orientations,
                                                   GrainsMemBuffer<Kinematics<T>, M>& velocities,
                                                   GrainsMemBuffer<Torce<T>, M>&      torces,
                                                   GrainsMemBuffer<uint, M>&          rigidBodyIds,
                                                   GrainsMemBuffer<uint, M>&          componentIds,
                                                   uint                               nObstacles,
                                                   uint                               nParticles)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& LC = GP::m_collisionDetection.linkedCellParameters;

    if(LC.sortFrequency > 0 && SS.neighborListUpdateCount % LC.sortFrequency == 0)
    {
        m_particleSorter.sortParticles(positions,
                                       velocities,
                                       orientations,
                                       torces,
                                       rigidBodyIds,
                                       componentIds,
                                       nObstacles,
                                       nParticles);
        SS.particlesSorted = true;
    }
    else
        SS.particlesSorted = false;
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiations
template class CollisionDetectionModule<float, MemType::HOST>;
template class CollisionDetectionModule<double, MemType::HOST>;
template class CollisionDetectionModule<float, MemType::DEVICE>;
template class CollisionDetectionModule<double, MemType::DEVICE>;
