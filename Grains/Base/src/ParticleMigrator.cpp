#include "ParticleMigrator.hh"
#include "Basic.hh"
#include <set>

// =================================================================================================
template <typename T, MemType M>
ParticleMigrator<T, M>::ParticleMigrator() : m_decomp(nullptr) {}

template <typename T, MemType M>
ParticleMigrator<T, M>::ParticleMigrator(const DomainDecomposition<T>* decomp)
    : m_decomp(decomp) {}

// =================================================================================================
template <typename T, MemType M>
void ParticleMigrator<T, M>::identifyEmigrants(
    const GrainsMemBuffer<Vector3<T>, M>& positions,
    uint numObstacles, uint numLocalParticles)
{
    m_emigrateIndicesLower.clear();
    m_emigrateIndicesUpper.clear();

    uint totalLocal = numObstacles + numLocalParticles;

    if constexpr(M == MemType::DEVICE)
    {
        if(m_hostPositions.getCapacity() < totalLocal)
            m_hostPositions.initialize(totalLocal);
        else
            m_hostPositions.setSize(totalLocal);
        cudaErrCheck(cudaMemcpy(m_hostPositions.getData(), positions.getData(),
                                totalLocal * sizeof(Vector3<T>), cudaMemcpyDeviceToHost));

        for(uint p = 0; p < numLocalParticles; ++p)
        {
            uint idx = numObstacles + p;
            const Vector3<T>& pos = m_hostPositions[idx];
            if(!m_decomp->isLocal(pos))
            {
                int owner = m_decomp->getOwnerRank(pos);
                if(owner == m_decomp->getLowerNeighbor())
                    m_emigrateIndicesLower.push_back(idx);
                else if(owner == m_decomp->getUpperNeighbor())
                    m_emigrateIndicesUpper.push_back(idx);
            }
        }
    }
    else
    {
        for(uint p = 0; p < numLocalParticles; ++p)
        {
            uint idx = numObstacles + p;
            const Vector3<T>& pos = positions[idx];
            if(!m_decomp->isLocal(pos))
            {
                int owner = m_decomp->getOwnerRank(pos);
                if(owner == m_decomp->getLowerNeighbor())
                    m_emigrateIndicesLower.push_back(idx);
                else if(owner == m_decomp->getUpperNeighbor())
                    m_emigrateIndicesUpper.push_back(idx);
            }
        }
    }
}

// =================================================================================================
template <typename T, MemType M>
void ParticleMigrator<T, M>::ensureStagingCapacity(uint capacity)
{
    if(capacity == 0) return;
    auto ensure = [capacity](auto& buf) {
        if(buf.getCapacity() < capacity) buf.initialize(capacity);
        else buf.setSize(capacity);
    };
    ensure(m_sendPosBuf);    ensure(m_recvPosBuf);
    ensure(m_sendQuatBuf);   ensure(m_recvQuatBuf);
    ensure(m_sendVelBuf);    ensure(m_recvVelBuf);
    ensure(m_sendTorceBuf);  ensure(m_recvTorceBuf);
    ensure(m_sendRbIdBuf);   ensure(m_recvRbIdBuf);
    ensure(m_sendCompIdBuf); ensure(m_recvCompIdBuf);
}

// =================================================================================================
template <typename T, MemType M>
void ParticleMigrator<T, M>::compactLocal(
    GrainsMemBuffer<Vector3<T>, M>&    positions,
    GrainsMemBuffer<Quaternion<T>, M>& quaternions,
    GrainsMemBuffer<Kinematics<T>, M>& velocities,
    GrainsMemBuffer<Torce<T>, M>&      torces,
    GrainsMemBuffer<uint, M>&          rigidBodyIds,
    GrainsMemBuffer<uint, M>&          componentIds,
    uint numObstacles, uint numLocalParticles)
{
    std::set<uint> removeSet;
    for(uint idx : m_emigrateIndicesLower) removeSet.insert(idx);
    for(uint idx : m_emigrateIndicesUpper) removeSet.insert(idx);
    if(removeSet.empty()) return;

    uint totalOld = numObstacles + numLocalParticles;

    if constexpr(M == MemType::DEVICE)
    {
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

        uint writeIdx = numObstacles;
        for(uint readIdx = numObstacles; readIdx < totalOld; ++readIdx)
        {
            if(removeSet.count(readIdx)) continue;
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

        uint newCount = writeIdx - numObstacles;
        cudaErrCheck(cudaMemcpy(positions.getData() + numObstacles,
                                hPos.getData() + numObstacles,
                                newCount * sizeof(Vector3<T>), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(quaternions.getData() + numObstacles,
                                hQuat.getData() + numObstacles,
                                newCount * sizeof(Quaternion<T>), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(velocities.getData() + numObstacles,
                                hVel.getData() + numObstacles,
                                newCount * sizeof(Kinematics<T>), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(torces.getData() + numObstacles,
                                hTorce.getData() + numObstacles,
                                newCount * sizeof(Torce<T>), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(rigidBodyIds.getData() + numObstacles,
                                hRbId.getData() + numObstacles,
                                newCount * sizeof(uint), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(componentIds.getData() + numObstacles,
                                hCompId.getData() + numObstacles,
                                newCount * sizeof(uint), cudaMemcpyHostToDevice));

        positions.setSize(writeIdx);
        quaternions.setSize(writeIdx);
        velocities.setSize(writeIdx);
        torces.setSize(writeIdx);
        rigidBodyIds.setSize(writeIdx);
        componentIds.setSize(writeIdx);
    }
    else
    {
        uint writeIdx = numObstacles;
        for(uint readIdx = numObstacles; readIdx < totalOld; ++readIdx)
        {
            if(removeSet.count(readIdx)) continue;
            if(writeIdx != readIdx)
            {
                positions[writeIdx]    = positions[readIdx];
                quaternions[writeIdx]  = quaternions[readIdx];
                velocities[writeIdx]   = velocities[readIdx];
                torces[writeIdx]       = torces[readIdx];
                rigidBodyIds[writeIdx] = rigidBodyIds[readIdx];
                componentIds[writeIdx] = componentIds[readIdx];
            }
            ++writeIdx;
        }
        positions.setSize(writeIdx);
        quaternions.setSize(writeIdx);
        velocities.setSize(writeIdx);
        torces.setSize(writeIdx);
        rigidBodyIds.setSize(writeIdx);
        componentIds.setSize(writeIdx);
    }
}

// =================================================================================================
template <typename T, MemType M>
uint ParticleMigrator<T, M>::migrate(
    GrainsMemBuffer<Vector3<T>, M>&    positions,
    GrainsMemBuffer<Quaternion<T>, M>& quaternions,
    GrainsMemBuffer<Kinematics<T>, M>& velocities,
    GrainsMemBuffer<Torce<T>, M>&      torces,
    GrainsMemBuffer<uint, M>&          rigidBodyIds,
    GrainsMemBuffer<uint, M>&          componentIds,
    uint                               numObstacles,
    uint                               numLocalParticles)
{
#ifndef GRAINS_USE_MPI
    (void)positions; (void)quaternions; (void)velocities; (void)torces;
    (void)rigidBodyIds; (void)componentIds; (void)numObstacles;
    return numLocalParticles;
#else
    identifyEmigrants(positions, numObstacles, numLocalParticles);

    uint numEmigLower = static_cast<uint>(m_emigrateIndicesLower.size());
    uint numEmigUpper = static_cast<uint>(m_emigrateIndicesUpper.size());
    uint totalEmig    = numEmigLower + numEmigUpper;

    uint numImmigLower = 0, numImmigUpper = 0;
    MPI_Request reqs[4];
    int nReqs = 0;

    if(m_decomp->getLowerNeighbor() >= 0)
    {
        MPI_Isend(&numEmigLower, 1, MPI_UNSIGNED, m_decomp->getLowerNeighbor(),
                  100, MPI_COMM_WORLD, &reqs[nReqs++]);
        MPI_Irecv(&numImmigLower, 1, MPI_UNSIGNED, m_decomp->getLowerNeighbor(),
                  100, MPI_COMM_WORLD, &reqs[nReqs++]);
    }
    if(m_decomp->getUpperNeighbor() >= 0)
    {
        MPI_Isend(&numEmigUpper, 1, MPI_UNSIGNED, m_decomp->getUpperNeighbor(),
                  101, MPI_COMM_WORLD, &reqs[nReqs++]);
        MPI_Irecv(&numImmigUpper, 1, MPI_UNSIGNED, m_decomp->getUpperNeighbor(),
                  101, MPI_COMM_WORLD, &reqs[nReqs++]);
    }
    MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

    uint totalImmig = numImmigLower + numImmigUpper;
    if(totalEmig == 0 && totalImmig == 0)
        return numLocalParticles;

    uint maxBuf = std::max({numEmigLower, numEmigUpper, numImmigLower, numImmigUpper, 1u});
    ensureStagingCapacity(maxBuf);

    // Pack and send helper
    auto packAndSend = [&](const std::vector<uint>& indices, uint count,
                           int neighbor, int tagBase) {
        if(count == 0 || neighbor < 0) return;

        for(uint i = 0; i < count; ++i)
        {
            uint src = indices[i];
            if constexpr(M == MemType::DEVICE)
            {
                cudaErrCheck(cudaMemcpy(&m_sendPosBuf[i], positions.getData() + src,
                                        sizeof(Vector3<T>), cudaMemcpyDeviceToHost));
                cudaErrCheck(cudaMemcpy(&m_sendQuatBuf[i], quaternions.getData() + src,
                                        sizeof(Quaternion<T>), cudaMemcpyDeviceToHost));
                cudaErrCheck(cudaMemcpy(&m_sendVelBuf[i], velocities.getData() + src,
                                        sizeof(Kinematics<T>), cudaMemcpyDeviceToHost));
                cudaErrCheck(cudaMemcpy(&m_sendTorceBuf[i], torces.getData() + src,
                                        sizeof(Torce<T>), cudaMemcpyDeviceToHost));
                cudaErrCheck(cudaMemcpy(&m_sendRbIdBuf[i], rigidBodyIds.getData() + src,
                                        sizeof(uint), cudaMemcpyDeviceToHost));
                cudaErrCheck(cudaMemcpy(&m_sendCompIdBuf[i], componentIds.getData() + src,
                                        sizeof(uint), cudaMemcpyDeviceToHost));
            }
            else
            {
                m_sendPosBuf[i]    = positions[src];
                m_sendQuatBuf[i]   = quaternions[src];
                m_sendVelBuf[i]    = velocities[src];
                m_sendTorceBuf[i]  = torces[src];
                m_sendRbIdBuf[i]   = rigidBodyIds[src];
                m_sendCompIdBuf[i] = componentIds[src];
            }
        }

        MPI_Request sr[6];
        MPI_Isend(m_sendPosBuf.getData(), count * sizeof(Vector3<T>),
                  MPI_BYTE, neighbor, tagBase, MPI_COMM_WORLD, &sr[0]);
        MPI_Isend(m_sendQuatBuf.getData(), count * sizeof(Quaternion<T>),
                  MPI_BYTE, neighbor, tagBase+1, MPI_COMM_WORLD, &sr[1]);
        MPI_Isend(m_sendVelBuf.getData(), count * sizeof(Kinematics<T>),
                  MPI_BYTE, neighbor, tagBase+2, MPI_COMM_WORLD, &sr[2]);
        MPI_Isend(m_sendTorceBuf.getData(), count * sizeof(Torce<T>),
                  MPI_BYTE, neighbor, tagBase+3, MPI_COMM_WORLD, &sr[3]);
        MPI_Isend(m_sendRbIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase+4, MPI_COMM_WORLD, &sr[4]);
        MPI_Isend(m_sendCompIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase+5, MPI_COMM_WORLD, &sr[5]);
        MPI_Waitall(6, sr, MPI_STATUSES_IGNORE);
    };

    auto recvImmigrants = [&](uint count, int neighbor, int tagBase) {
        if(count == 0 || neighbor < 0) return;
        MPI_Request rr[6];
        MPI_Irecv(m_recvPosBuf.getData(), count * sizeof(Vector3<T>),
                  MPI_BYTE, neighbor, tagBase, MPI_COMM_WORLD, &rr[0]);
        MPI_Irecv(m_recvQuatBuf.getData(), count * sizeof(Quaternion<T>),
                  MPI_BYTE, neighbor, tagBase+1, MPI_COMM_WORLD, &rr[1]);
        MPI_Irecv(m_recvVelBuf.getData(), count * sizeof(Kinematics<T>),
                  MPI_BYTE, neighbor, tagBase+2, MPI_COMM_WORLD, &rr[2]);
        MPI_Irecv(m_recvTorceBuf.getData(), count * sizeof(Torce<T>),
                  MPI_BYTE, neighbor, tagBase+3, MPI_COMM_WORLD, &rr[3]);
        MPI_Irecv(m_recvRbIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase+4, MPI_COMM_WORLD, &rr[4]);
        MPI_Irecv(m_recvCompIdBuf.getData(), count * sizeof(uint),
                  MPI_BYTE, neighbor, tagBase+5, MPI_COMM_WORLD, &rr[5]);
        MPI_Waitall(6, rr, MPI_STATUSES_IGNORE);
    };

    packAndSend(m_emigrateIndicesLower, numEmigLower, m_decomp->getLowerNeighbor(), 110);
    recvImmigrants(numImmigLower, m_decomp->getLowerNeighbor(), 110);
    packAndSend(m_emigrateIndicesUpper, numEmigUpper, m_decomp->getUpperNeighbor(), 120);
    recvImmigrants(numImmigUpper, m_decomp->getUpperNeighbor(), 120);

    compactLocal(positions, quaternions, velocities, torces,
                 rigidBodyIds, componentIds, numObstacles, numLocalParticles);

    uint newLocal = numLocalParticles - totalEmig;

    // Append immigrants
    auto appendFromRecv = [&](uint count, uint& offset) {
        if(count == 0) return;
        for(uint i = 0; i < count; ++i)
        {
            if constexpr(M == MemType::DEVICE)
            {
                cudaErrCheck(cudaMemcpy(positions.getData() + offset + i,
                    &m_recvPosBuf[i], sizeof(Vector3<T>), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(quaternions.getData() + offset + i,
                    &m_recvQuatBuf[i], sizeof(Quaternion<T>), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(velocities.getData() + offset + i,
                    &m_recvVelBuf[i], sizeof(Kinematics<T>), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(torces.getData() + offset + i,
                    &m_recvTorceBuf[i], sizeof(Torce<T>), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(rigidBodyIds.getData() + offset + i,
                    &m_recvRbIdBuf[i], sizeof(uint), cudaMemcpyHostToDevice));
                cudaErrCheck(cudaMemcpy(componentIds.getData() + offset + i,
                    &m_recvCompIdBuf[i], sizeof(uint), cudaMemcpyHostToDevice));
            }
            else
            {
                positions[offset + i]    = m_recvPosBuf[i];
                quaternions[offset + i]  = m_recvQuatBuf[i];
                velocities[offset + i]   = m_recvVelBuf[i];
                torces[offset + i]       = m_recvTorceBuf[i];
                rigidBodyIds[offset + i] = m_recvRbIdBuf[i];
                componentIds[offset + i] = m_recvCompIdBuf[i];
            }
        }
        offset += count;
    };

    if(totalImmig > 0)
    {
        uint appendStart = numObstacles + newLocal;
        uint newTotal    = appendStart + totalImmig;
        positions.resize(newTotal);
        quaternions.resize(newTotal);
        velocities.resize(newTotal);
        torces.resize(newTotal);
        rigidBodyIds.resize(newTotal);
        componentIds.resize(newTotal);

        uint offset = appendStart;
        appendFromRecv(numImmigLower, offset);
        appendFromRecv(numImmigUpper, offset);
        newLocal += totalImmig;
    }

    return newLocal;
#endif
}

// =================================================================================================
template class ParticleMigrator<float, MemType::HOST>;
template class ParticleMigrator<double, MemType::HOST>;
template class ParticleMigrator<float, MemType::DEVICE>;
template class ParticleMigrator<double, MemType::DEVICE>;
