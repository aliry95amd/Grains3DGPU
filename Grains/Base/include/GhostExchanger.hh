#ifndef _GHOSTEXCHANGER_HH_
#define _GHOSTEXCHANGER_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "DomainDecomposition.hh"
#include "GrainsMemBuffer.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief Manages ghost-particle exchange between MPI ranks for the multi-GPU DEM solver.

    Before each neighbor-list rebuild the halo region must be populated with copies of particles
    that live on adjacent ranks but are close enough to interact with local particles.

    Workflow per step:
      1. copyPositionsToHost()  — D2H of positions for boundary scanning
      2. identifyGhosts()       — CPU scan to build per-neighbor send lists
      3. packAndExchange()      — gather from device, stage to pinned, MPI Isend/Irecv
      4. unpackGhosts()         — append received data to device arrays
      5. removeGhosts()         — shrink arrays back to local-only size after the step

    Uses pinned (page-locked) staging buffers for overlap of MPI and GPU transfers.

    @author Multi-GPU extension — 2026 */
// =================================================================================================
template <typename T>
class GhostExchanger
{
private:
    const DomainDecomposition<T>* m_decomp;

    // Per-neighbor send/receive index lists (host)
    std::vector<uint> m_sendIndicesLower;
    std::vector<uint> m_sendIndicesUpper;

    // Pinned staging buffers for SoA fields
    GrainsMemBuffer<Vector3<T>, MemType::PINNED>    m_sendPosBuf;
    GrainsMemBuffer<Quaternion<T>, MemType::PINNED>  m_sendQuatBuf;
    GrainsMemBuffer<Kinematics<T>, MemType::PINNED>  m_sendVelBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_sendRbIdBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_sendCompIdBuf;

    GrainsMemBuffer<Vector3<T>, MemType::PINNED>    m_recvPosBuf;
    GrainsMemBuffer<Quaternion<T>, MemType::PINNED>  m_recvQuatBuf;
    GrainsMemBuffer<Kinematics<T>, MemType::PINNED>  m_recvVelBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_recvRbIdBuf;
    GrainsMemBuffer<uint, MemType::PINNED>           m_recvCompIdBuf;

    // Host-side position mirror for boundary scanning
    GrainsMemBuffer<Vector3<T>, MemType::PINNED> m_hostPositions;

    uint m_numGhostsReceived;

public:
    /** @name Constructors */
    //@{
    GhostExchanger();
    explicit GhostExchanger(const DomainDecomposition<T>* decomp);
    ~GhostExchanger() = default;
    //@}

    /** @name Exchange operations */
    //@{
    /** @brief Full ghost exchange cycle.  Call once per step before collision detection.
        Appends ghost particles after local particles in each device array.
        @param positions      device position buffer (resized to include ghosts)
        @param quaternions    device quaternion buffer
        @param velocities     device velocity buffer
        @param rigidBodyIds   device rigid-body-id buffer
        @param componentIds   device component-id buffer
        @param numObstacles   number of obstacles (offset for particle data)
        @param numLocalParticles  number of locally owned particles
        @return number of ghost particles appended */
    uint exchange(GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                  GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                  GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
                  GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
                  GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
                  uint                                             numObstacles,
                  uint                                             numLocalParticles);

    /** @brief Shrinks arrays back to local-only size (removes appended ghosts). */
    static void removeGhosts(GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                             GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                             GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
                             GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
                             GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
                             uint                                             totalLocal);

    uint getNumGhostsReceived() const { return m_numGhostsReceived; }
    //@}

private:
    void identifyGhosts(const Vector3<T>* hostPos, uint numObstacles, uint numLocalParticles);
    void ensureStagingCapacity(uint capacity);

    void gatherToHost(const GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                      const GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                      const GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
                      const GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
                      const GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
                      const std::vector<uint>&                               indices,
                      uint                                                   offset,
                      GrainsMemBuffer<Vector3<T>, MemType::PINNED>&          posBuf,
                      GrainsMemBuffer<Quaternion<T>, MemType::PINNED>&       quatBuf,
                      GrainsMemBuffer<Kinematics<T>, MemType::PINNED>&       velBuf,
                      GrainsMemBuffer<uint, MemType::PINNED>&                rbIdBuf,
                      GrainsMemBuffer<uint, MemType::PINNED>&                compIdBuf);

    void scatterToDevice(GrainsMemBuffer<Vector3<T>, MemType::DEVICE>&    positions,
                         GrainsMemBuffer<Quaternion<T>, MemType::DEVICE>& quaternions,
                         GrainsMemBuffer<Kinematics<T>, MemType::DEVICE>& velocities,
                         GrainsMemBuffer<uint, MemType::DEVICE>&          rigidBodyIds,
                         GrainsMemBuffer<uint, MemType::DEVICE>&          componentIds,
                         uint                                             destOffset,
                         uint                                             count);
};

#endif
