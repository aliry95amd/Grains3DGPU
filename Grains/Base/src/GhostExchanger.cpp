#include "GhostExchanger.hh"
#include "Basic.hh"

// =================================================================================================
template <typename T>
GhostExchanger<T>::GhostExchanger()
    : m_decomp(nullptr)
    , m_numGhostsReceived(0)
{
}

// =================================================================================================
template <typename T>
GhostExchanger<T>::GhostExchanger(const DomainDecomposition<T>* decomp)
    : m_decomp(decomp)
    , m_numGhostsReceived(0)
{
}

// =================================================================================================
template <typename T>
void GhostExchanger<T>::identifyGhosts(const Vector3<T>* hostPos,
                                        uint              numObstacles,
                                        uint              numLocalParticles)
{
    m_sendIndicesLower.clear();
    m_sendIndicesUpper.clear();

    for(uint p = 0; p < numLocalParticles; ++p)
    {
        uint idx = numObstacles + p;
        const Vector3<T>& pos = hostPos[idx];

        if(m_decomp->needsGhostLower(pos))
            m_sendIndicesLower.push_back(idx);
        if(m_decomp->needsGhostUpper(pos))
            m_sendIndicesUpper.push_back(idx);
    }
}

// =================================================================================================
template <typename T>
void GhostExchanger<T>::ensureStagingCapacity(uint capacity)
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
    ensureSize(m_sendRbIdBuf);
    ensureSize(m_sendCompIdBuf);

    ensureSize(m_recvPosBuf);
    ensureSize(m_recvQuatBuf);
    ensureSize(m_recvVelBuf);
    ensureSize(m_recvRbIdBuf);
    ensureSize(m_recvCompIdBuf);
}

// =================================================================================================
template <typename T>
void GhostExchanger<T>::gatherToHost(
    const GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
    const GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
    const GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
    const GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
    const GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
    const std::vector<uint>&                               indices,
    uint                                                   /*offset*/,
    GrainsMemBuffer<Vector3<T>, MemType::PINNED>&          posBuf,
    GrainsMemBuffer<Quaternion<T>, MemType::PINNED>&       quatBuf,
    GrainsMemBuffer<Kinematics<T>, MemType::PINNED>&       velBuf,
    GrainsMemBuffer<uint, MemType::PINNED>&                rbIdBuf,
    GrainsMemBuffer<uint, MemType::PINNED>&                compIdBuf)
{
    uint count = static_cast<uint>(indices.size());
    if(count == 0)
        return;

    for(uint i = 0; i < count; ++i)
    {
        uint srcIdx = indices[i];
        cudaErrCheck(cudaMemcpy(&posBuf[i], positions.getData() + srcIdx,
                                sizeof(Vector3<T>), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(&quatBuf[i], quaternions.getData() + srcIdx,
                                sizeof(Quaternion<T>), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(&velBuf[i], velocities.getData() + srcIdx,
                                sizeof(Kinematics<T>), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(&rbIdBuf[i], rigidBodyIds.getData() + srcIdx,
                                sizeof(uint), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(&compIdBuf[i], componentIds.getData() + srcIdx,
                                sizeof(uint), cudaMemcpyDeviceToHost));
    }
}

// =================================================================================================
template <typename T>
void GhostExchanger<T>::scatterToDevice(
    GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
    GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
    GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
    GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
    GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
    uint                                             destOffset,
    uint                                             count)
{
    if(count == 0)
        return;

    cudaErrCheck(cudaMemcpy(positions.getData() + destOffset, m_recvPosBuf.getData(),
                            count * sizeof(Vector3<T>), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(quaternions.getData() + destOffset, m_recvQuatBuf.getData(),
                            count * sizeof(Quaternion<T>), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(velocities.getData() + destOffset, m_recvVelBuf.getData(),
                            count * sizeof(Kinematics<T>), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(rigidBodyIds.getData() + destOffset, m_recvRbIdBuf.getData(),
                            count * sizeof(uint), cudaMemcpyHostToDevice));
    cudaErrCheck(cudaMemcpy(componentIds.getData() + destOffset, m_recvCompIdBuf.getData(),
                            count * sizeof(uint), cudaMemcpyHostToDevice));
}

// =================================================================================================
template <typename T>
uint GhostExchanger<T>::exchange(
    GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
    GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
    GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
    GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
    GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
    uint                                             numObstacles,
    uint                                             numLocalParticles)
{
#ifndef GRAINS_USE_MPI
    (void)positions; (void)quaternions; (void)velocities;
    (void)rigidBodyIds; (void)componentIds;
    (void)numObstacles; (void)numLocalParticles;
    m_numGhostsReceived = 0;
    return 0;
#else
    uint totalLocal = numObstacles + numLocalParticles;

    // 1. Copy positions D2H for boundary scanning
    if(m_hostPositions.getCapacity() < totalLocal)
        m_hostPositions.initialize(totalLocal);
    else
        m_hostPositions.setSize(totalLocal);
    cudaErrCheck(cudaMemcpy(m_hostPositions.getData(), positions.getData(),
                            totalLocal * sizeof(Vector3<T>), cudaMemcpyDeviceToHost));

    // 2. Identify which local particles are ghosts for each neighbor
    identifyGhosts(m_hostPositions.getData(), numObstacles, numLocalParticles);

    uint numSendLower = static_cast<uint>(m_sendIndicesLower.size());
    uint numSendUpper = static_cast<uint>(m_sendIndicesUpper.size());
    uint maxSend      = std::max(numSendLower, numSendUpper);

    // 3. Exchange counts with neighbors to know how much to receive
    uint numRecvLower = 0, numRecvUpper = 0;
    MPI_Request reqs[4];
    int nReqs = 0;

    if(m_decomp->getLowerNeighbor() >= 0)
    {
        MPI_Isend(&numSendLower, 1, MPI_UNSIGNED, m_decomp->getLowerNeighbor(),
                  0, MPI_COMM_WORLD, &reqs[nReqs++]);
        MPI_Irecv(&numRecvLower, 1, MPI_UNSIGNED, m_decomp->getLowerNeighbor(),
                  0, MPI_COMM_WORLD, &reqs[nReqs++]);
    }
    if(m_decomp->getUpperNeighbor() >= 0)
    {
        MPI_Isend(&numSendUpper, 1, MPI_UNSIGNED, m_decomp->getUpperNeighbor(),
                  1, MPI_COMM_WORLD, &reqs[nReqs++]);
        MPI_Irecv(&numRecvUpper, 1, MPI_UNSIGNED, m_decomp->getUpperNeighbor(),
                  1, MPI_COMM_WORLD, &reqs[nReqs++]);
    }
    MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

    uint totalRecv = numRecvLower + numRecvUpper;
    m_numGhostsReceived = totalRecv;

    // 4. Ensure staging buffers are large enough
    uint maxBufSize = std::max({maxSend, numRecvLower, numRecvUpper, 1u});
    ensureStagingCapacity(maxBufSize);

    // 5. Resize device arrays to accommodate ghosts
    uint newTotal = totalLocal + totalRecv;
    positions.resize(newTotal);
    quaternions.resize(newTotal);
    velocities.resize(newTotal);
    rigidBodyIds.resize(newTotal);
    componentIds.resize(newTotal);

    // 6. Gather, exchange, and scatter — lower neighbor
    if(m_decomp->getLowerNeighbor() >= 0)
    {
        nReqs = 0;
        int neighbor = m_decomp->getLowerNeighbor();

        if(numSendLower > 0)
        {
            gatherToHost(positions, quaternions, velocities, rigidBodyIds, componentIds,
                         m_sendIndicesLower, 0,
                         m_sendPosBuf, m_sendQuatBuf, m_sendVelBuf,
                         m_sendRbIdBuf, m_sendCompIdBuf);

            MPI_Isend(m_sendPosBuf.getData(), numSendLower * sizeof(Vector3<T>),
                      MPI_BYTE, neighbor, 10, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendQuatBuf.getData(), numSendLower * sizeof(Quaternion<T>),
                      MPI_BYTE, neighbor, 11, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendVelBuf.getData(), numSendLower * sizeof(Kinematics<T>),
                      MPI_BYTE, neighbor, 12, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendRbIdBuf.getData(), numSendLower * sizeof(uint),
                      MPI_BYTE, neighbor, 13, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendCompIdBuf.getData(), numSendLower * sizeof(uint),
                      MPI_BYTE, neighbor, 14, MPI_COMM_WORLD, &reqs[nReqs++]);
        }

        if(numRecvLower > 0)
        {
            MPI_Irecv(m_recvPosBuf.getData(), numRecvLower * sizeof(Vector3<T>),
                      MPI_BYTE, neighbor, 10, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvQuatBuf.getData(), numRecvLower * sizeof(Quaternion<T>),
                      MPI_BYTE, neighbor, 11, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvVelBuf.getData(), numRecvLower * sizeof(Kinematics<T>),
                      MPI_BYTE, neighbor, 12, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvRbIdBuf.getData(), numRecvLower * sizeof(uint),
                      MPI_BYTE, neighbor, 13, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvCompIdBuf.getData(), numRecvLower * sizeof(uint),
                      MPI_BYTE, neighbor, 14, MPI_COMM_WORLD, &reqs[nReqs++]);
        }

        MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

        if(numRecvLower > 0)
            scatterToDevice(positions, quaternions, velocities, rigidBodyIds, componentIds,
                            totalLocal, numRecvLower);
    }

    // 7. Gather, exchange, and scatter — upper neighbor
    if(m_decomp->getUpperNeighbor() >= 0)
    {
        nReqs = 0;
        int neighbor = m_decomp->getUpperNeighbor();

        if(numSendUpper > 0)
        {
            gatherToHost(positions, quaternions, velocities, rigidBodyIds, componentIds,
                         m_sendIndicesUpper, 0,
                         m_sendPosBuf, m_sendQuatBuf, m_sendVelBuf,
                         m_sendRbIdBuf, m_sendCompIdBuf);

            MPI_Isend(m_sendPosBuf.getData(), numSendUpper * sizeof(Vector3<T>),
                      MPI_BYTE, neighbor, 20, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendQuatBuf.getData(), numSendUpper * sizeof(Quaternion<T>),
                      MPI_BYTE, neighbor, 21, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendVelBuf.getData(), numSendUpper * sizeof(Kinematics<T>),
                      MPI_BYTE, neighbor, 22, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendRbIdBuf.getData(), numSendUpper * sizeof(uint),
                      MPI_BYTE, neighbor, 23, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendCompIdBuf.getData(), numSendUpper * sizeof(uint),
                      MPI_BYTE, neighbor, 24, MPI_COMM_WORLD, &reqs[nReqs++]);
        }

        if(numRecvUpper > 0)
        {
            MPI_Irecv(m_recvPosBuf.getData(), numRecvUpper * sizeof(Vector3<T>),
                      MPI_BYTE, neighbor, 20, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvQuatBuf.getData(), numRecvUpper * sizeof(Quaternion<T>),
                      MPI_BYTE, neighbor, 21, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvVelBuf.getData(), numRecvUpper * sizeof(Kinematics<T>),
                      MPI_BYTE, neighbor, 22, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvRbIdBuf.getData(), numRecvUpper * sizeof(uint),
                      MPI_BYTE, neighbor, 23, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvCompIdBuf.getData(), numRecvUpper * sizeof(uint),
                      MPI_BYTE, neighbor, 24, MPI_COMM_WORLD, &reqs[nReqs++]);
        }

        MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

        if(numRecvUpper > 0)
            scatterToDevice(positions, quaternions, velocities, rigidBodyIds, componentIds,
                            totalLocal + numRecvLower, numRecvUpper);
    }

    return totalRecv;
#endif
}

// =================================================================================================
template <typename T>
void GhostExchanger<T>::removeGhosts(
    GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
    GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
    GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
    GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
    GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
    uint                                             totalLocal)
{
    positions.setSize(totalLocal);
    quaternions.setSize(totalLocal);
    velocities.setSize(totalLocal);
    rigidBodyIds.setSize(totalLocal);
    componentIds.setSize(totalLocal);
}

// =================================================================================================
template class GhostExchanger<float>;
template class GhostExchanger<double>;
