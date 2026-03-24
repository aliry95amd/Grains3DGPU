#include "GhostExchanger.hh"
#include "Basic.hh"

// =================================================================================================
template <typename T, MemType M>
GhostExchanger<T, M>::GhostExchanger()
    : m_decomp(nullptr), m_numGhostsReceived(0) {}

template <typename T, MemType M>
GhostExchanger<T, M>::GhostExchanger(const DomainDecomposition<T>* decomp)
    : m_decomp(decomp), m_numGhostsReceived(0) {}

// =================================================================================================
template <typename T, MemType M>
void GhostExchanger<T, M>::identifyGhosts(const GrainsMemBuffer<Vector3<T>, M>& positions,
                                           uint numObstacles, uint numLocalParticles)
{
    m_sendIndicesLower.clear();
    m_sendIndicesUpper.clear();

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
            if(m_decomp->needsGhostLower(pos)) m_sendIndicesLower.push_back(idx);
            if(m_decomp->needsGhostUpper(pos)) m_sendIndicesUpper.push_back(idx);
        }
    }
    else
    {
        for(uint p = 0; p < numLocalParticles; ++p)
        {
            uint idx = numObstacles + p;
            const Vector3<T>& pos = positions[idx];
            if(m_decomp->needsGhostLower(pos)) m_sendIndicesLower.push_back(idx);
            if(m_decomp->needsGhostUpper(pos)) m_sendIndicesUpper.push_back(idx);
        }
    }
}

// =================================================================================================
template <typename T, MemType M>
void GhostExchanger<T, M>::ensureStagingCapacity(uint capacity)
{
    if(capacity == 0) return;

    auto ensure = [capacity](auto& buf) {
        if(buf.getCapacity() < capacity) buf.initialize(capacity);
        else buf.setSize(capacity);
    };

    ensure(m_sendPosBuf);    ensure(m_recvPosBuf);
    ensure(m_sendQuatBuf);   ensure(m_recvQuatBuf);
    ensure(m_sendVelBuf);    ensure(m_recvVelBuf);
    ensure(m_sendRbIdBuf);   ensure(m_recvRbIdBuf);
    ensure(m_sendCompIdBuf); ensure(m_recvCompIdBuf);
}

// =================================================================================================
template <typename T, MemType M>
void GhostExchanger<T, M>::gatherToStaging(
    const GrainsMemBuffer<Vector3<T>, M>&    positions,
    const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
    const GrainsMemBuffer<Kinematics<T>, M>& velocities,
    const GrainsMemBuffer<uint, M>&          rigidBodyIds,
    const GrainsMemBuffer<uint, M>&          componentIds,
    const std::vector<uint>&                 indices)
{
    uint count = static_cast<uint>(indices.size());
    if(count == 0) return;

    if constexpr(M == MemType::DEVICE)
    {
        for(uint i = 0; i < count; ++i)
        {
            uint src = indices[i];
            cudaErrCheck(cudaMemcpy(&m_sendPosBuf[i], positions.getData() + src,
                                    sizeof(Vector3<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendQuatBuf[i], quaternions.getData() + src,
                                    sizeof(Quaternion<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendVelBuf[i], velocities.getData() + src,
                                    sizeof(Kinematics<T>), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendRbIdBuf[i], rigidBodyIds.getData() + src,
                                    sizeof(uint), cudaMemcpyDeviceToHost));
            cudaErrCheck(cudaMemcpy(&m_sendCompIdBuf[i], componentIds.getData() + src,
                                    sizeof(uint), cudaMemcpyDeviceToHost));
        }
    }
    else
    {
        for(uint i = 0; i < count; ++i)
        {
            uint idx = indices[i];
            m_sendPosBuf[i]    = positions[idx];
            m_sendQuatBuf[i]   = quaternions[idx];
            m_sendVelBuf[i]    = velocities[idx];
            m_sendRbIdBuf[i]   = rigidBodyIds[idx];
            m_sendCompIdBuf[i] = componentIds[idx];
        }
    }
}

// =================================================================================================
template <typename T, MemType M>
void GhostExchanger<T, M>::scatterFromStaging(
    GrainsMemBuffer<Vector3<T>, M>&    positions,
    GrainsMemBuffer<Quaternion<T>, M>& quaternions,
    GrainsMemBuffer<Kinematics<T>, M>& velocities,
    GrainsMemBuffer<uint, M>&          rigidBodyIds,
    GrainsMemBuffer<uint, M>&          componentIds,
    uint destOffset, uint count)
{
    if(count == 0) return;

    if constexpr(M == MemType::DEVICE)
    {
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
    else
    {
        for(uint i = 0; i < count; ++i)
        {
            positions[destOffset + i]    = m_recvPosBuf[i];
            quaternions[destOffset + i]  = m_recvQuatBuf[i];
            velocities[destOffset + i]   = m_recvVelBuf[i];
            rigidBodyIds[destOffset + i] = m_recvRbIdBuf[i];
            componentIds[destOffset + i] = m_recvCompIdBuf[i];
        }
    }
}

// =================================================================================================
template <typename T, MemType M>
uint GhostExchanger<T, M>::exchange(
    GrainsMemBuffer<Vector3<T>, M>&    positions,
    GrainsMemBuffer<Quaternion<T>, M>& quaternions,
    GrainsMemBuffer<Kinematics<T>, M>& velocities,
    GrainsMemBuffer<uint, M>&          rigidBodyIds,
    GrainsMemBuffer<uint, M>&          componentIds,
    uint                               numObstacles,
    uint                               numLocalParticles)
{
#ifndef GRAINS_USE_MPI
    (void)positions; (void)quaternions; (void)velocities;
    (void)rigidBodyIds; (void)componentIds;
    (void)numObstacles; (void)numLocalParticles;
    m_numGhostsReceived = 0;
    return 0;
#else
    uint totalLocal = numObstacles + numLocalParticles;

    identifyGhosts(positions, numObstacles, numLocalParticles);

    uint numSendLower = static_cast<uint>(m_sendIndicesLower.size());
    uint numSendUpper = static_cast<uint>(m_sendIndicesUpper.size());

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

    uint maxBuf = std::max({numSendLower, numSendUpper, numRecvLower, numRecvUpper, 1u});
    ensureStagingCapacity(maxBuf);

    uint newTotal = totalLocal + totalRecv;
    positions.resize(newTotal);
    quaternions.resize(newTotal);
    velocities.resize(newTotal);
    rigidBodyIds.resize(newTotal);
    componentIds.resize(newTotal);

    // --- Lower neighbor ---
    if(m_decomp->getLowerNeighbor() >= 0)
    {
        nReqs = 0;
        int nb = m_decomp->getLowerNeighbor();

        if(numSendLower > 0)
        {
            gatherToStaging(positions, quaternions, velocities,
                            rigidBodyIds, componentIds, m_sendIndicesLower);
            MPI_Isend(m_sendPosBuf.getData(), numSendLower * sizeof(Vector3<T>),
                      MPI_BYTE, nb, 10, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendQuatBuf.getData(), numSendLower * sizeof(Quaternion<T>),
                      MPI_BYTE, nb, 11, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendVelBuf.getData(), numSendLower * sizeof(Kinematics<T>),
                      MPI_BYTE, nb, 12, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendRbIdBuf.getData(), numSendLower * sizeof(uint),
                      MPI_BYTE, nb, 13, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendCompIdBuf.getData(), numSendLower * sizeof(uint),
                      MPI_BYTE, nb, 14, MPI_COMM_WORLD, &reqs[nReqs++]);
        }
        if(numRecvLower > 0)
        {
            MPI_Irecv(m_recvPosBuf.getData(), numRecvLower * sizeof(Vector3<T>),
                      MPI_BYTE, nb, 10, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvQuatBuf.getData(), numRecvLower * sizeof(Quaternion<T>),
                      MPI_BYTE, nb, 11, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvVelBuf.getData(), numRecvLower * sizeof(Kinematics<T>),
                      MPI_BYTE, nb, 12, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvRbIdBuf.getData(), numRecvLower * sizeof(uint),
                      MPI_BYTE, nb, 13, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvCompIdBuf.getData(), numRecvLower * sizeof(uint),
                      MPI_BYTE, nb, 14, MPI_COMM_WORLD, &reqs[nReqs++]);
        }
        MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

        if(numRecvLower > 0)
            scatterFromStaging(positions, quaternions, velocities,
                               rigidBodyIds, componentIds, totalLocal, numRecvLower);
    }

    // --- Upper neighbor ---
    if(m_decomp->getUpperNeighbor() >= 0)
    {
        nReqs = 0;
        int nb = m_decomp->getUpperNeighbor();

        if(numSendUpper > 0)
        {
            gatherToStaging(positions, quaternions, velocities,
                            rigidBodyIds, componentIds, m_sendIndicesUpper);
            MPI_Isend(m_sendPosBuf.getData(), numSendUpper * sizeof(Vector3<T>),
                      MPI_BYTE, nb, 20, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendQuatBuf.getData(), numSendUpper * sizeof(Quaternion<T>),
                      MPI_BYTE, nb, 21, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendVelBuf.getData(), numSendUpper * sizeof(Kinematics<T>),
                      MPI_BYTE, nb, 22, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendRbIdBuf.getData(), numSendUpper * sizeof(uint),
                      MPI_BYTE, nb, 23, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Isend(m_sendCompIdBuf.getData(), numSendUpper * sizeof(uint),
                      MPI_BYTE, nb, 24, MPI_COMM_WORLD, &reqs[nReqs++]);
        }
        if(numRecvUpper > 0)
        {
            MPI_Irecv(m_recvPosBuf.getData(), numRecvUpper * sizeof(Vector3<T>),
                      MPI_BYTE, nb, 20, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvQuatBuf.getData(), numRecvUpper * sizeof(Quaternion<T>),
                      MPI_BYTE, nb, 21, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvVelBuf.getData(), numRecvUpper * sizeof(Kinematics<T>),
                      MPI_BYTE, nb, 22, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvRbIdBuf.getData(), numRecvUpper * sizeof(uint),
                      MPI_BYTE, nb, 23, MPI_COMM_WORLD, &reqs[nReqs++]);
            MPI_Irecv(m_recvCompIdBuf.getData(), numRecvUpper * sizeof(uint),
                      MPI_BYTE, nb, 24, MPI_COMM_WORLD, &reqs[nReqs++]);
        }
        MPI_Waitall(nReqs, reqs, MPI_STATUSES_IGNORE);

        if(numRecvUpper > 0)
            scatterFromStaging(positions, quaternions, velocities,
                               rigidBodyIds, componentIds,
                               totalLocal + numRecvLower, numRecvUpper);
    }

    return totalRecv;
#endif
}

// =================================================================================================
template <typename T, MemType M>
void GhostExchanger<T, M>::removeGhosts(
    GrainsMemBuffer<Vector3<T>, M>&    positions,
    GrainsMemBuffer<Quaternion<T>, M>& quaternions,
    GrainsMemBuffer<Kinematics<T>, M>& velocities,
    GrainsMemBuffer<uint, M>&          rigidBodyIds,
    GrainsMemBuffer<uint, M>&          componentIds,
    uint                               totalLocal)
{
    positions.setSize(totalLocal);
    quaternions.setSize(totalLocal);
    velocities.setSize(totalLocal);
    rigidBodyIds.setSize(totalLocal);
    componentIds.setSize(totalLocal);
}

// =================================================================================================
template class GhostExchanger<float, MemType::HOST>;
template class GhostExchanger<double, MemType::HOST>;
template class GhostExchanger<float, MemType::DEVICE>;
template class GhostExchanger<double, MemType::DEVICE>;
