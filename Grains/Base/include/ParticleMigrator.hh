#ifndef _PARTICLEMIGRATOR_HH_
#define _PARTICLEMIGRATOR_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "DomainDecomposition.hh"
#include "GrainsMemBuffer.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "Torce.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief Unified particle migrator templated on MemType.

    Detects particles that left the local subdomain, sends them to the new owner rank via MPI,
    and compacts local arrays.  When M == DEVICE, uses cudaMemcpy + pinned staging; when M == HOST,
    operates directly via operator[].

    @tparam T scalar type (float / double)
    @tparam M memory type of the simulation buffers (HOST or DEVICE)
    @author Multi-rank extension — 2026 */
// =================================================================================================
template <typename T, MemType M = MemType::HOST>
class ParticleMigrator
{
    static constexpr MemType StagingMem = (M == MemType::DEVICE) ? MemType::PINNED : MemType::HOST;

    const DomainDecomposition<T>* m_decomp;

    std::vector<uint> m_emigrateIndicesLower;
    std::vector<uint> m_emigrateIndicesUpper;

    GrainsMemBuffer<Vector3<T>, StagingMem>    m_sendPosBuf,  m_recvPosBuf;
    GrainsMemBuffer<Quaternion<T>, StagingMem>  m_sendQuatBuf, m_recvQuatBuf;
    GrainsMemBuffer<Kinematics<T>, StagingMem>  m_sendVelBuf,  m_recvVelBuf;
    GrainsMemBuffer<Torce<T>, StagingMem>       m_sendTorceBuf, m_recvTorceBuf;
    GrainsMemBuffer<uint, StagingMem>           m_sendRbIdBuf, m_recvRbIdBuf;
    GrainsMemBuffer<uint, StagingMem>           m_sendCompIdBuf, m_recvCompIdBuf;

    GrainsMemBuffer<Vector3<T>, StagingMem> m_hostPositions;

public:
    ParticleMigrator();
    explicit ParticleMigrator(const DomainDecomposition<T>* decomp);
    ~ParticleMigrator() = default;

    uint migrate(GrainsMemBuffer<Vector3<T>, M>&    positions,
                 GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                 GrainsMemBuffer<Kinematics<T>, M>& velocities,
                 GrainsMemBuffer<Torce<T>, M>&      torces,
                 GrainsMemBuffer<uint, M>&          rigidBodyIds,
                 GrainsMemBuffer<uint, M>&          componentIds,
                 uint                               numObstacles,
                 uint                               numLocalParticles);

private:
    void identifyEmigrants(const GrainsMemBuffer<Vector3<T>, M>& positions,
                           uint numObstacles, uint numLocalParticles);
    void ensureStagingCapacity(uint capacity);
    void compactLocal(GrainsMemBuffer<Vector3<T>, M>&    positions,
                      GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                      GrainsMemBuffer<Kinematics<T>, M>& velocities,
                      GrainsMemBuffer<Torce<T>, M>&      torces,
                      GrainsMemBuffer<uint, M>&          rigidBodyIds,
                      GrainsMemBuffer<uint, M>&          componentIds,
                      uint numObstacles, uint numLocalParticles);
};

#endif
