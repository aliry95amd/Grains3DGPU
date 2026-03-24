#ifndef _GHOSTEXCHANGERCPU_HH_
#define _GHOSTEXCHANGERCPU_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "DomainDecomposition.hh"
#include "GrainsMemBuffer.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief Ghost-particle exchange for multi-CPU (HOST memory) DEM.

    Same role as GhostExchanger but operates entirely on HOST-resident GrainsMemBuffers.
    No CUDA transfers are needed — positions are scanned directly, and MPI exchanges read/write
    from host staging buffers.

    @author Multi-CPU extension — 2026 */
// =================================================================================================
template <typename T>
class GhostExchangerCPU
{
private:
    const DomainDecomposition<T>* m_decomp;

    std::vector<uint> m_sendIndicesLower;
    std::vector<uint> m_sendIndicesUpper;

    // Staging buffers (plain host memory)
    GrainsMemBuffer<Vector3<T>>    m_sendPosBuf, m_recvPosBuf;
    GrainsMemBuffer<Quaternion<T>> m_sendQuatBuf, m_recvQuatBuf;
    GrainsMemBuffer<Kinematics<T>> m_sendVelBuf, m_recvVelBuf;
    GrainsMemBuffer<uint>          m_sendRbIdBuf, m_recvRbIdBuf;
    GrainsMemBuffer<uint>          m_sendCompIdBuf, m_recvCompIdBuf;

    uint m_numGhostsReceived;

public:
    GhostExchangerCPU();
    explicit GhostExchangerCPU(const DomainDecomposition<T>* decomp);
    ~GhostExchangerCPU() = default;

    /** @brief Full ghost exchange cycle for HOST buffers.
        @return number of ghost particles appended */
    uint exchange(GrainsMemBuffer<Vector3<T>>&    positions,
                  GrainsMemBuffer<Quaternion<T>>& quaternions,
                  GrainsMemBuffer<Kinematics<T>>& velocities,
                  GrainsMemBuffer<uint>&          rigidBodyIds,
                  GrainsMemBuffer<uint>&          componentIds,
                  uint                            numObstacles,
                  uint                            numLocalParticles);

    static void removeGhosts(GrainsMemBuffer<Vector3<T>>&    positions,
                             GrainsMemBuffer<Quaternion<T>>& quaternions,
                             GrainsMemBuffer<Kinematics<T>>& velocities,
                             GrainsMemBuffer<uint>&          rigidBodyIds,
                             GrainsMemBuffer<uint>&          componentIds,
                             uint                            totalLocal);

    uint getNumGhostsReceived() const { return m_numGhostsReceived; }

private:
    void identifyGhosts(const GrainsMemBuffer<Vector3<T>>& positions,
                        uint numObstacles, uint numLocalParticles);
    void ensureStagingCapacity(uint capacity);

    void gatherToStaging(const GrainsMemBuffer<Vector3<T>>&    positions,
                         const GrainsMemBuffer<Quaternion<T>>& quaternions,
                         const GrainsMemBuffer<Kinematics<T>>& velocities,
                         const GrainsMemBuffer<uint>&          rigidBodyIds,
                         const GrainsMemBuffer<uint>&          componentIds,
                         const std::vector<uint>&              indices);

    void scatterFromStaging(GrainsMemBuffer<Vector3<T>>&    positions,
                            GrainsMemBuffer<Quaternion<T>>& quaternions,
                            GrainsMemBuffer<Kinematics<T>>& velocities,
                            GrainsMemBuffer<uint>&          rigidBodyIds,
                            GrainsMemBuffer<uint>&          componentIds,
                            uint destOffset, uint count);
};

#endif
