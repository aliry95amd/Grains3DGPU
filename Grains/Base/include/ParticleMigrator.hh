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
/** @brief Migrates particles that have left the local subdomain to their new owner rank.

    After time integration some particles may have moved across the subdomain boundary.  This class
    detects them, packs their full state (position, orientation, velocity, torce, ids), sends them
    to the neighboring rank via MPI, and compacts the local arrays.

    Like GhostExchanger, it uses pinned staging buffers and non-blocking MPI for efficiency.

    @author Multi-GPU extension — 2026 */
// =================================================================================================
template <typename T>
class ParticleMigrator
{
private:
    const DomainDecomposition<T>* m_decomp;

    // Indices of particles leaving to each neighbor (host-side)
    std::vector<uint> m_emigrateIndicesLower;
    std::vector<uint> m_emigrateIndicesUpper;

    // Pinned staging buffers for the full particle state
    GrainsMemBuffer<Vector3<T>, MemType::PINNED>    m_sendPosBuf;
    GrainsMemBuffer<Quaternion<T>, MemType::PINNED>  m_sendQuatBuf;
    GrainsMemBuffer<Kinematics<T>, MemType::PINNED>  m_sendVelBuf;
    GrainsMemBuffer<Torce<T>, MemType::PINNED>       m_sendTorceBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_sendRbIdBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_sendCompIdBuf;

    GrainsMemBuffer<Vector3<T>, MemType::PINNED>    m_recvPosBuf;
    GrainsMemBuffer<Quaternion<T>, MemType::PINNED>  m_recvQuatBuf;
    GrainsMemBuffer<Kinematics<T>, MemType::PINNED>  m_recvVelBuf;
    GrainsMemBuffer<Torce<T>, MemType::PINNED>       m_recvTorceBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_recvRbIdBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_recvCompIdBuf;

    GrainsMemBuffer<Vector3<T>, MemType::PINNED> m_hostPositions;

public:
    ParticleMigrator();
    explicit ParticleMigrator(const DomainDecomposition<T>* decomp);
    ~ParticleMigrator() = default;

    /** @brief Migrate particles that left the local domain.
        Returns the new local particle count after migration (may have grown or shrunk).
        @param positions        device position buffer
        @param quaternions      device quaternion buffer
        @param velocities       device velocity buffer
        @param torces           device torce buffer
        @param rigidBodyIds     device rigid-body-id buffer
        @param componentIds     device component-id buffer
        @param numObstacles     number of obstacles (fixed offset)
        @param numLocalParticles current local particle count (updated in-place)
        @return new number of local particles */
    uint migrate(GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                 GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                 GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
                 GrainsMemBuffer<Torce<T>, MemType::DEVICE>&      torces,
                 GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
                 GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
                 uint                                             numObstacles,
                 uint                                             numLocalParticles);

private:
    void identifyEmigrants(const Vector3<T>* hostPos, uint numObstacles, uint numLocalParticles);
    void ensureStagingCapacity(uint capacity);

    void compactLocal(GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                      GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                      GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
                      GrainsMemBuffer<Torce<T>, MemType::DEVICE>&      torces,
                      GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
                      GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
                      uint                                             numObstacles,
                      uint                                             numLocalParticles);
};

#endif
