#include "GhostExchangerCPU.hh"
#include "Basic.hh"

// =================================================================================================
template <typename T>
GhostExchangerCPU<T>::GhostExchangerCPU()
    : m_decomp(nullptr)
    , m_numGhostsReceived(0)
{
}

// =================================================================================================
template <typename T>
GhostExchangerCPU<T>::GhostExchangerCPU(const DomainDecomposition<T>* decomp)
    : m_decomp(decomp)
    , m_numGhostsReceived(0)
{
}

// =================================================================================================
template <typename T>
void GhostExchangerCPU<T>::identifyGhosts(const GrainsMemBuffer<Vector3<T>>& positions,
                                           uint numObstacles,
                                           uint numLocalParticles)
{
    m_sendIndicesLower.clear();
    m_sendIndicesUpper.clear();

    for(uint p = 0; p < numLocalParticles; ++p)
    {
        uint idx = numObstacles + p;
        const Vector3<T>& pos = positions[idx];

        if(m_decomp->needsGhostLower(pos))
            m_sendIndicesLower.push_back(idx);
        if(m_decomp->needsGhostUpper(pos))
            m_sendIndicesUpper.push_back(idx);
    }
}

// =================================================================================================
template <typename T>
void GhostExchangerCPU<T>::ensureStagingCapacity(uint capacity)
{
    if(capacity == 0)
        return;

    auto ensure = [capacity](auto& buf) {
        if(buf.getCapacity() < capacity)
            buf.initialize(capacity);
        else
            buf.setSize(capacity);
    };

    ensure(m_sendPosBuf);    ensure(m_recvPosBuf);
    ensure(m_sendQuatBuf);   ensure(m_recvQuatBuf);
    ensure(m_sendVelBuf);    ensure(m_recvVelBuf);
    ensure(m_sendRbIdBuf);   ensure(m_recvRbIdBuf);
    ensure(m_sendCompIdBuf); ensure(m_recvCompIdBuf);
}

// =================================================================================================
template <typename T>
void GhostExchangerCPU<T>::gatherToStaging(
    const GrainsMemBuffer<Vector3<T>>&    positions,
    const GrainsMemBuffer<Quaternion<T>>& quaternions,
    const GrainsMemBuffer<Kinematics<T>>& velocities,
    const GrainsMemBuffer<uint>&          rigidBodyIds,
    const GrainsMemBuffer<uint>&          componentIds,
    const std::vector<uint>&              indices)
{
    for(uint i = 0; i < static_cast<uint>(indices.size()); ++i)
    {
        uint idx = indices[i];
        m_sendPosBuf[i]    = positions[idx];
        m_sendQuatBuf[i]   = quaternions[idx];
        m_sendVelBuf[i]    = velocities[idx];
        m_sendRbIdBuf[i]   = rigidBodyIds[idx];
        m_sendCompIdBuf[i] = componentIds[idx];
    }
}

// =================================================================================================
template <typename T>
void GhostExchangerCPU<T>::scatterFromStaging(
    GrainsMemBuffer<Vector3<T>>&    positions,
    GrainsMemBuffer<Quaternion<T>>& quaternions,
    GrainsMemBuffer<Kinematics<T>>& velocities,
    GrainsMemBuffer<uint>&          rigidBodyIds,
    GrainsMemBuffer<uint>&          componentIds,
    uint destOffset, uint count)
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

// =================================================================================================
template <typename T>
uint GhostExchangerCPU<T>::exchange(
    GrainsMemBuffer<Vector3<T>>&    positions,
    GrainsMemBuffer<Quaternion<T>>& quaternions,
    GrainsMemBuffer<Kinematics<T>>& velocities,
    GrainsMemBuffer<uint>&          rigidBodyIds,
    GrainsMemBuffer<uint>&          componentIds,
    uint                            numObstacles,
    uint                            numLocalParticles)
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

    // Exchange counts
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

    // Resize host arrays to accommodate ghosts
    uint newTotal = totalLocal + totalRecv;
    positions.resize(newTotal);
    quaternions.resize(newTotal);
    velocities.resize(newTotal);
    rigidBodyIds.resize(newTotal);
    componentIds.resize(newTotal);

    // Lower neighbor exchange
    if(m_decomp->getLowerNeighbor() >= 0)
    {
        nReqs = 0;
        int neighbor = m_decomp->getLowerNeighbor();

        if(numSendLower > 0)
        {
            gatherToStaging(positions, quaternions, velocities,
                            rigidBodyIds, componentIds, m_sendIndicesLower);

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
            scatterFromStaging(positions, quaternions, velocities,
                               rigidBodyIds, componentIds, totalLocal, numRecvLower);
    }

    // Upper neighbor exchange
    if(m_decomp->getUpperNeighbor() >= 0)
    {
        nReqs = 0;
        int neighbor = m_decomp->getUpperNeighbor();

        if(numSendUpper > 0)
        {
            gatherToStaging(positions, quaternions, velocities,
                            rigidBodyIds, componentIds, m_sendIndicesUpper);

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
            scatterFromStaging(positions, quaternions, velocities,
                               rigidBodyIds, componentIds,
                               totalLocal + numRecvLower, numRecvUpper);
    }

    return totalRecv;
#endif
}

// =================================================================================================
template <typename T>
void GhostExchangerCPU<T>::removeGhosts(
    GrainsMemBuffer<Vector3<T>>&    positions,
    GrainsMemBuffer<Quaternion<T>>& quaternions,
    GrainsMemBuffer<Kinematics<T>>& velocities,
    GrainsMemBuffer<uint>&          rigidBodyIds,
    GrainsMemBuffer<uint>&          componentIds,
    uint                            totalLocal)
{
    positions.setSize(totalLocal);
    quaternions.setSize(totalLocal);
    velocities.setSize(totalLocal);
    rigidBodyIds.setSize(totalLocal);
    componentIds.setSize(totalLocal);
}

// =================================================================================================
template class GhostExchangerCPU<float>;
template class GhostExchangerCPU<double>;
