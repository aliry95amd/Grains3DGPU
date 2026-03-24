#include "ParticleMigratorCPU.hh"
#include "Basic.hh"

#include <set>

// =================================================================================================
template <typename T>
ParticleMigratorCPU<T>::ParticleMigratorCPU()
    : m_decomp(nullptr)
{
}

// =================================================================================================
template <typename T>
ParticleMigratorCPU<T>::ParticleMigratorCPU(const DomainDecomposition<T>* decomp)
    : m_decomp(decomp)
{
}

// =================================================================================================
template <typename T>
void ParticleMigratorCPU<T>::identifyEmigrants(const GrainsMemBuffer<Vector3<T>>& positions,
                                                uint numObstacles,
                                                uint numLocalParticles)
{
    m_emigrateIndicesLower.clear();
    m_emigrateIndicesUpper.clear();

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

// =================================================================================================
template <typename T>
void ParticleMigratorCPU<T>::ensureStagingCapacity(uint capacity)
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
    ensure(m_sendTorceBuf);  ensure(m_recvTorceBuf);
    ensure(m_sendRbIdBuf);   ensure(m_recvRbIdBuf);
    ensure(m_sendCompIdBuf); ensure(m_recvCompIdBuf);
}

// =================================================================================================
template <typename T>
void ParticleMigratorCPU<T>::compactLocal(
    GrainsMemBuffer<Vector3<T>>&    positions,
    GrainsMemBuffer<Quaternion<T>>& quaternions,
    GrainsMemBuffer<Kinematics<T>>& velocities,
    GrainsMemBuffer<Torce<T>>&      torces,
    GrainsMemBuffer<uint>&          rigidBodyIds,
    GrainsMemBuffer<uint>&          componentIds,
    uint numObstacles, uint numLocalParticles)
{
    std::set<uint> removeSet;
    for(uint idx : m_emigrateIndicesLower) removeSet.insert(idx);
    for(uint idx : m_emigrateIndicesUpper) removeSet.insert(idx);

    if(removeSet.empty())
        return;

    uint totalOld = numObstacles + numLocalParticles;
    uint writeIdx = numObstacles;

    for(uint readIdx = numObstacles; readIdx < totalOld; ++readIdx)
    {
        if(removeSet.count(readIdx))
            continue;
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

// =================================================================================================
template <typename T>
uint ParticleMigratorCPU<T>::migrate(
    GrainsMemBuffer<Vector3<T>>&    positions,
    GrainsMemBuffer<Quaternion<T>>& quaternions,
    GrainsMemBuffer<Kinematics<T>>& velocities,
    GrainsMemBuffer<Torce<T>>&      torces,
    GrainsMemBuffer<uint>&          rigidBodyIds,
    GrainsMemBuffer<uint>&          componentIds,
    uint                            numObstacles,
    uint                            numLocalParticles)
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

    // Exchange counts
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

    // Pack and send/receive helper lambdas
    auto packAndSend = [&](const std::vector<uint>& indices, uint count,
                           int neighbor, int tagBase) {
        if(count == 0 || neighbor < 0) return;

        for(uint i = 0; i < count; ++i)
        {
            uint idx = indices[i];
            m_sendPosBuf[i]    = positions[idx];
            m_sendQuatBuf[i]   = quaternions[idx];
            m_sendVelBuf[i]    = velocities[idx];
            m_sendTorceBuf[i]  = torces[idx];
            m_sendRbIdBuf[i]   = rigidBodyIds[idx];
            m_sendCompIdBuf[i] = componentIds[idx];
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

    packAndSend(m_emigrateIndicesLower, numEmigLower,
                m_decomp->getLowerNeighbor(), 110);
    recvImmigrants(numImmigLower, m_decomp->getLowerNeighbor(), 110);

    packAndSend(m_emigrateIndicesUpper, numEmigUpper,
                m_decomp->getUpperNeighbor(), 120);
    recvImmigrants(numImmigUpper, m_decomp->getUpperNeighbor(), 120);

    // Compact: remove emigrants
    compactLocal(positions, quaternions, velocities, torces,
                 rigidBodyIds, componentIds, numObstacles, numLocalParticles);

    uint newLocal = numLocalParticles - totalEmig;

    // Append immigrants
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
        auto appendFromRecv = [&](uint count) {
            for(uint i = 0; i < count; ++i)
            {
                positions[offset + i]    = m_recvPosBuf[i];
                quaternions[offset + i]  = m_recvQuatBuf[i];
                velocities[offset + i]   = m_recvVelBuf[i];
                torces[offset + i]       = m_recvTorceBuf[i];
                rigidBodyIds[offset + i] = m_recvRbIdBuf[i];
                componentIds[offset + i] = m_recvCompIdBuf[i];
            }
            offset += count;
        };

        appendFromRecv(numImmigLower);
        appendFromRecv(numImmigUpper);

        newLocal += totalImmig;
    }

    return newLocal;
#endif
}

// =================================================================================================
template class ParticleMigratorCPU<float>;
template class ParticleMigratorCPU<double>;
