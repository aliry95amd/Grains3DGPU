#include "ParticleMigrator.hh"
#include "Basic.hh"

#include <algorithm>
#include <numeric>
#include <set>

// =================================================================================================
template <typename T>
ParticleMigrator<T>::ParticleMigrator()
    : m_decomp(nullptr)
{
}

// =================================================================================================
template <typename T>
ParticleMigrator<T>::ParticleMigrator(const DomainDecomposition<T>* decomp)
    : m_decomp(decomp)
{
}

// =================================================================================================
template <typename T>
void ParticleMigrator<T>::identifyEmigrants(const Vector3<T>* hostPos,
                                             uint              numObstacles,
                                             uint              numLocalParticles)
{
    m_emigrateIndicesLower.clear();
    m_emigrateIndicesUpper.clear();

    for(uint p = 0; p < numLocalParticles; ++p)
    {
        uint idx = numObstacles + p;
        const Vector3<T>& pos = hostPos[idx];

        if(!m_decomp->isLocal(pos))
        {
            int ownerRank = m_decomp->getOwnerRank(pos);
            if(ownerRank == m_decomp->getLowerNeighbor())
                m_emigrateIndicesLower.push_back(idx);
            else if(ownerRank == m_decomp->getUpperNeighbor())
                m_emigrateIndicesUpper.push_back(idx);
        }
    }
}

// =================================================================================================
template <typename T>
void ParticleMigrator<T>::ensureStagingCapacity(uint capacity)
{
    if(capacity == 0)
        return;

    auto ensureSize = [capacity](auto& buf) {
        if(buf.getCapacity() < capacity)
            buf.initialize(capacity);
        else
            buf.setSize(capacity);
    };

    ensureSize(m_sendPosBuf);
    ensureSize(m_sendQuatBuf);
    ensureSize(m_sendVelBuf);
    ensureSize(m_sendTorceBuf);
    ensureSize(m_sendRbIdBuf);
    ensureSize(m_sendCompIdBuf);

    ensureSize(m_recvPosBuf);
    ensureSize(m_recvQuatBuf);
    ensureSize(m_recvVelBuf);
    ensureSize(m_recvTorceBuf);
    ensureSize(m_recvRbIdBuf);
    ensureSize(m_recvCompIdBuf);
}

// =================================================================================================
// Compact local arrays by removing emigrated particles.
// Uses a host-side copy-and-writeback approach: copy all to host, remove holes, copy back.
template <typename T>
void ParticleMigrator<T>::compactLocal(
    GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
    GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
    GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
    GrainsMemBuffer<Torce<T>, MemType::DEVICE>&      torces,
    GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
    GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
    uint                                             numObstacles,
    uint                                             numLocalParticles)
{
    std::set<uint> removeSet;
    for(uint idx : m_emigrateIndicesLower) removeSet.insert(idx);
    for(uint idx : m_emigrateIndicesUpper) removeSet.insert(idx);

    if(removeSet.empty())
        return;

    uint totalOld = numObstacles + numLocalParticles;

    // Temporary host buffers for compaction
    GrainsMemBuffer<Vector3<T>, MemType::PINNED>    hPos(totalOld);
    GrainsMemBuffer<Quaternion<T>, MemType::PINNED>  hQuat(totalOld);
    GrainsMemBuffer<Kinematics<T>, MemType::PINNED>  hVel(totalOld);
    GrainsMemBuffer<Torce<T>, MemType::PINNED>       hTorce(totalOld);
    GrainsMemBuffer<uint, MemType::PINNED>           hRbId(totalOld);
    GrainsMemBuffer<uint, MemType::PINNED>           hCompId(totalOld);

    cudaErrCheck(cudaMemcpy(hPos.getData(), positions.getData(),
                            totalOld * sizeof(Vector3<T>), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hQuat.getData(), quaternions.getData(),
                            totalOld * sizeof(Quaternion<T>), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hVel.getData(), velocities.getData(),
                            totalOld * sizeof(Kinematics<T>), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hTorce.getData(), torces.getData(),
                            totalOld * sizeof(Torce<T>), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hRbId.getData(), rigidBodyIds.getData(),
                            totalOld * sizeof(uint), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(hCompId.getData(), componentIds.getData(),
                            totalOld * sizeof(uint), cudaMemcpyDeviceToHost));

    // Compact: shift remaining particles to fill holes left by emigrants
    uint writeIdx = numObstacles;
    for(uint readIdx = numObstacles; readIdx < totalOld; ++readIdx)
    {
        if(removeSet.count(readIdx))
            continue;
        if(writeIdx != readIdx)
        {
            hPos[writeIdx]    = hPos[readIdx];
            hQuat[writeIdx]   = hQuat[readIdx];
            hVel[writeIdx]    = hVel[readIdx];
            hTorce[writeIdx]  = hTorce[readIdx];
            hRbId[writeIdx]   = hRbId[readIdx];
            hCompId[writeIdx] = hCompId[readIdx];
        }
        ++writeIdx;
    }

    uint newTotal = writeIdx;

    cudaErrCheck(cudaMemcpy(positions.getData() + numObstacles,
                            hPos.getData() + numObstacles,
                            (newTotal - numObstacles) * sizeof(Vector3<T>),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(quaternions.getData() + numObstacles,
                            hQuat.getData() + numObstacles,
                            (newTotal - numObstacles) * sizeof(Quaternion<T>),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(velocities.getData() + numObstacles,
                            hVel.getData() + numObstacles,
                            (newTotal - numObstacles) * sizeof(Kinematics<T>),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(torces.getData() + numObstacles,
                            hTorce.getData() + numObstacles,
                            (newTotal - numObstacles) * sizeof(Torce<T>),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(rigidBodyIds.getData() + numObstacles,
                            hRbId.getData() + numObstacles,
                            (newTotal - numObstacles) * sizeof(uint),
                            cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(componentIds.getData() + numObstacles,
                            hCompId.getData() + numObstacles,
                            (newTotal - numObstacles) * sizeof(uint),
                            cudaMemcpyHostToDevice));

    positions.setSize(newTotal);
    quaternions.setSize(newTotal);
    velocities.setSize(newTotal);
    torces.setSize(newTotal);
    rigidBodyIds.setSize(newTotal);
    componentIds.setSize(newTotal);
}

// =================================================================================================
template <typename T>
uint ParticleMigrator<T>::migrate(
    GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
    GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
    GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
    GrainsMemBuffer<Torce<T>, MemType::DEVICE>&      torces,
    GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
    GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
    uint                                             numObstacles,
    uint                                             numLocalParticles)
{
#ifndef GRAINS_USE_MPI
    (void)positions; (void)quaternions; (void)velocities; (void)torces;
    (void)rigidBodyIds; (void)componentIds;
    (void)numObstacles;
    return numLocalParticles;
#else
    uint totalLocal = numObstacles + numLocalParticles;

    // 1. Copy positions D2H
    if(m_hostPositions.getCapacity() < totalLocal)
        m_hostPositions.initialize(totalLocal);
    else
        m_hostPositions.setSize(totalLocal);
    cudaErrCheck(cudaMemcpy(m_hostPositions.getData(), positions.getData(),
                            totalLocal * sizeof(Vector3<T>), cudaMemcpyDeviceToHost));

    // 2. Identify emigrants
    identifyEmigrants(m_hostPositions.getData(), numObstacles, numLocalParticles);

    uint numEmigrateLower = static_cast<uint>(m_emigrateIndicesLower.size());
    uint numEmigrateUpper = static_cast<uint>(m_emigrateIndicesUpper.size());
    uint totalEmigrate    = numEmigrateLower + numEmigrateUpper;

    // 3. Exchange counts
    uint numImmigrateLower = 0, numImmigrateUpper = 0;
    MPI_Request reqs[4];
    int nReqs = 0;

    if(m_decomp->getLowerNeighbor() >= 0)
    {
        MPI_Isend(&numEmigrateLower, 1, MPI_UNSIGNED, m_decomp->getLowerNeighbor(),
                  100, MPI_COMM_WORLD, &reqs[nReqs++]);
        MPI_Irecv(&numImmigrateLower, 1, MPI_UNSIGNED, m_decomp->getLowerNeighbor(),
                  100, MPI_COMM_WORLD, &reqs[nReqs++]);
    }
    if(m_decomp->getUpperNeighbor() >= 0)
    {
        MPI_Isend(&numEmigrateUpper, 1, MPI_UNSIGNED, m_decomp->getUpperNeighbor(),
                  101, MPI_COMM_WORLD, &reqs[nReqs++]);
        MPI_Irecv(&numImmigrateUpper, 1, MPI_UNSIGNED, m_decomp->getUpperNeighbor(),
                  101, MPI_COMM_WORLD, &reqs[nReqs++]);
    }
    MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

    uint totalImmigrate = numImmigrateLower + numImmigrateUpper;

    if(totalEmigrate == 0 && totalImmigrate == 0)
        return numLocalParticles;

    // 4. Ensure staging buffers
    uint maxBuf = std::max({numEmigrateLower, numEmigrateUpper,
                            numImmigrateLower, numImmigrateUpper, 1u});
    ensureStagingCapacity(maxBuf);

    // 5. Pack emigrants from device to pinned buffers and exchange
    auto packAndSend = [&](const std::vector<uint>& indices, uint count,
                           int neighbor, int tagBase) {
        if(count == 0 || neighbor < 0)
            return;

        for(uint i = 0; i < count; ++i)
        {
            uint srcIdx = indices[i];
            cudaErrCheck(cudaMemcpy(&m_sendPosBuf[i], positions.getData() + srcIdx,
                                    sizeof(Vector3<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendQuatBuf[i], quaternions.getData() + srcIdx,
                                    sizeof(Quaternion<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendVelBuf[i], velocities.getData() + srcIdx,
                                    sizeof(Kinematics<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendTorceBuf[i], torces.getData() + srcIdx,
                                    sizeof(Torce<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendRbIdBuf[i], rigidBodyIds.getData() + srcIdx,
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendCompIdBuf[i], componentIds.getData() + srcIdx,
                                    sizeof(uint), cudaMemcpyDeviceToHost));
        }

        MPI_Request sendReqs[6];
        MPI_Isend(m_sendPosBuf.getData(), count * sizeof(Vector3<T>),
                  MPI_BYTE, neighbor, tagBase, MPI_COMM_WORLD, &sendReqs[0]);
        MPI_Isend(m_sendQuatBuf.getData(), count * sizeof(Quaternion<T>),
                  MPI_BYTE, neighbor, tagBase + 1, MPI_COMM_WORLD, &sendReqs[1]);
        MPI_Isend(m_sendVelBuf.getData(), count * sizeof(Kinematics<T>),
                  MPI_BYTE, neighbor, tagBase + 2, MPI_COMM_WORLD, &sendReqs[2]);
        MPI_Isend(m_sendTorceBuf.getData(), count * sizeof(Torce<T>),
                  MPI_BYTE, neighbor, tagBase + 3, MPI_COMM_WORLD, &sendReqs[3]);
        MPI_Isend(m_sendRbIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase + 4, MPI_COMM_WORLD, &sendReqs[4]);
        MPI_Isend(m_sendCompIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase + 5, MPI_COMM_WORLD, &sendReqs[5]);
        MPI_Waitall(6, sendReqs, MPI_STATUSES_IGNORE);
    };

    auto recvImmigrants = [&](uint count, int neighbor, int tagBase) {
        if(count == 0 || neighbor < 0)
            return;

        MPI_Request recvReqs[6];
        MPI_Irecv(m_recvPosBuf.getData(), count * sizeof(Vector3<T>),
                  MPI_BYTE, neighbor, tagBase, MPI_COMM_WORLD, &recvReqs[0]);
        MPI_Irecv(m_recvQuatBuf.getData(), count * sizeof(Quaternion<T>),
                  MPI_BYTE, neighbor, tagBase + 1, MPI_COMM_WORLD, &recvReqs[1]);
        MPI_Irecv(m_recvVelBuf.getData(), count * sizeof(Kinematics<T>),
                  MPI_BYTE, neighbor, tagBase + 2, MPI_COMM_WORLD, &recvReqs[2]);
        MPI_Irecv(m_recvTorceBuf.getData(), count * sizeof(Torce<T>),
                  MPI_BYTE, neighbor, tagBase + 3, MPI_COMM_WORLD, &recvReqs[3]);
        MPI_Irecv(m_recvRbIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase + 4, MPI_COMM_WORLD, &recvReqs[4]);
        MPI_Irecv(m_recvCompIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase + 5, MPI_COMM_WORLD, &recvReqs[5]);
        MPI_Waitall(6, recvReqs, MPI_STATUSES_IGNORE);
    };

    // Send and receive with lower neighbor
    packAndSend(m_emigrateIndicesLower, numEmigrateLower,
                m_decomp->getLowerNeighbor(), 110);
    recvImmigrants(numImmigrateLower, m_decomp->getLowerNeighbor(), 110);

    // Send and receive with upper neighbor
    packAndSend(m_emigrateIndicesUpper, numEmigrateUpper,
                m_decomp->getUpperNeighbor(), 120);
    recvImmigrants(numImmigrateUpper, m_decomp->getUpperNeighbor(), 120);

    // 6. Remove emigrated particles (compact local arrays)
    compactLocal(positions, quaternions, velocities, torces,
                 rigidBodyIds, componentIds, numObstacles, numLocalParticles);

    uint newLocalCount = numLocalParticles - totalEmigrate;

    // 7. Append immigrants to the end of local arrays
    if(totalImmigrate > 0)
    {
        uint appendStart = numObstacles + newLocalCount;
        uint newTotal    = appendStart + totalImmigrate;

        positions.resize(newTotal);
        quaternions.resize(newTotal);
        velocities.resize(newTotal);
        torces.resize(newTotal);
        rigidBodyIds.resize(newTotal);
        componentIds.resize(newTotal);

        // Copy immigrants from pinned receive buffers to device
        // Lower immigrants first, then upper
        uint offset = appendStart;
        if(numImmigrateLower > 0)
        {
            cudaErrCheck(cudaMemcpy(positions.getData() + offset, m_recvPosBuf.getData(),
                                    numImmigrateLower * sizeof(Vector3<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(quaternions.getData() + offset, m_recvQuatBuf.getData(),
                                    numImmigrateLower * sizeof(Quaternion<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(velocities.getData() + offset, m_recvVelBuf.getData(),
                                    numImmigrateLower * sizeof(Kinematics<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(torces.getData() + offset, m_recvTorceBuf.getData(),
                                    numImmigrateLower * sizeof(Torce<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(rigidBodyIds.getData() + offset, m_recvRbIdBuf.getData(),
                                    numImmigrateLower * sizeof(uint),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(componentIds.getData() + offset, m_recvCompIdBuf.getData(),
                                    numImmigrateLower * sizeof(uint),
                                    cudaMemcpyHostToDevice));
            offset += numImmigrateLower;
        }
        if(numImmigrateUpper > 0)
        {
            cudaErrCheck(cudaMemcpy(positions.getData() + offset, m_recvPosBuf.getData(),
                                    numImmigrateUpper * sizeof(Vector3<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(quaternions.getData() + offset, m_recvQuatBuf.getData(),
                                    numImmigrateUpper * sizeof(Quaternion<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(velocities.getData() + offset, m_recvVelBuf.getData(),
                                    numImmigrateUpper * sizeof(Kinematics<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(torces.getData() + offset, m_recvTorceBuf.getData(),
                                    numImmigrateUpper * sizeof(Torce<T>),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(rigidBodyIds.getData() + offset, m_recvRbIdBuf.getData(),
                                    numImmigrateUpper * sizeof(uint),
                                    cudaMemcpyHostToDevice));
            cudaErrCheck(cudaMemcpy(componentIds.getData() + offset, m_recvCompIdBuf.getData(),
                                    numImmigrateUpper * sizeof(uint),
                                    cudaMemcpyHostToDevice));
        }

        newLocalCount += totalImmigrate;
    }

    return newLocalCount;
#endif
}

// =================================================================================================
template class ParticleMigrator<float>;
template class ParticleMigrator<double>;
