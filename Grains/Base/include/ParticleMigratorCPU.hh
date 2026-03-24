#ifndef _PARTICLEMIGRATORCPU_HH_
#define _PARTICLEMIGRATORCPU_HH_

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
/** @brief Particle migration for multi-CPU (HOST memory) DEM.

    Same role as ParticleMigrator but operates entirely on HOST-resident GrainsMemBuffers.

    @author Multi-CPU extension — 2026 */
// =================================================================================================
template <typename T>
class ParticleMigratorCPU
{
private:
    const DomainDecomposition<T>* m_decomp;

    std::vector<uint> m_emigrateIndicesLower;
    std::vector<uint> m_emigrateIndicesUpper;

    GrainsMemBuffer<Vector3<T>>    m_sendPosBuf, m_recvPosBuf;
    GrainsMemBuffer<Quaternion<T>> m_sendQuatBuf, m_recvQuatBuf;
    GrainsMemBuffer<Kinematics<T>> m_sendVelBuf, m_recvVelBuf;
    GrainsMemBuffer<Torce<T>>      m_sendTorceBuf, m_recvTorceBuf;
    GrainsMemBuffer<uint>          m_sendRbIdBuf, m_recvRbIdBuf;
    GrainsMemBuffer<uint>          m_sendCompIdBuf, m_recvCompIdBuf;

public:
    ParticleMigratorCPU();
    explicit ParticleMigratorCPU(const DomainDecomposition<T>* decomp);
    ~ParticleMigratorCPU() = default;

    /** @brief Migrate particles that left the local domain (HOST buffers).
        @return new number of local particles */
    uint migrate(GrainsMemBuffer<Vector3<T>>&    positions,
                 GrainsMemBuffer<Quaternion<T>>& quaternions,
                 GrainsMemBuffer<Kinematics<T>>& velocities,
                 GrainsMemBuffer<Torce<T>>&      torces,
                 GrainsMemBuffer<uint>&          rigidBodyIds,
                 GrainsMemBuffer<uint>&          componentIds,
                 uint                            numObstacles,
                 uint                            numLocalParticles);

private:
    void identifyEmigrants(const GrainsMemBuffer<Vector3<T>>& positions,
                           uint numObstacles, uint numLocalParticles);
    void ensureStagingCapacity(uint capacity);

    void compactLocal(GrainsMemBuffer<Vector3<T>>&    positions,
                      GrainsMemBuffer<Quaternion<T>>& quaternions,
                      GrainsMemBuffer<Kinematics<T>>& velocities,
                      GrainsMemBuffer<Torce<T>>&      torces,
                      GrainsMemBuffer<uint>&          rigidBodyIds,
                      GrainsMemBuffer<uint>&          componentIds,
                      uint numObstacles, uint numLocalParticles);
};

#endif
