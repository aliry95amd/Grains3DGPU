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
/** @brief Unified ghost-particle exchanger templated on MemType.

    Works for both HOST (multi-CPU) and DEVICE (multi-GPU) particle buffers.  When M == DEVICE,
    positions are copied D2H for boundary scanning, and gather/scatter use cudaMemcpy with PINNED
    staging buffers.  When M == HOST, positions are scanned directly and data is copied with
    operator[].

    @tparam T scalar type (float / double)
    @tparam M memory type of the simulation buffers (HOST or DEVICE)
    @author Multi-rank extension — 2026 */
// =================================================================================================
template <typename T, MemType M = MemType::HOST>
class GhostExchanger
{
    static constexpr MemType StagingMem = (M == MemType::DEVICE) ? MemType::PINNED : MemType::HOST;

    const DomainDecomposition<T>* m_decomp;

    std::vector<uint> m_sendIndicesLower;
    std::vector<uint> m_sendIndicesUpper;

    GrainsMemBuffer<Vector3<T>, StagingMem>    m_sendPosBuf,  m_recvPosBuf;
    GrainsMemBuffer<Quaternion<T>, StagingMem>  m_sendQuatBuf, m_recvQuatBuf;
    GrainsMemBuffer<Kinematics<T>, StagingMem>  m_sendVelBuf,  m_recvVelBuf;
    GrainsMemBuffer<uint, StagingMem>           m_sendRbIdBuf, m_recvRbIdBuf;
    GrainsMemBuffer<uint, StagingMem>           m_sendCompIdBuf, m_recvCompIdBuf;

    GrainsMemBuffer<Vector3<T>, StagingMem> m_hostPositions;

    uint m_numGhostsReceived;

public:
    GhostExchanger();
    explicit GhostExchanger(const DomainDecomposition<T>* decomp);
    ~GhostExchanger() = default;

    uint exchange(GrainsMemBuffer<Vector3<T>, M>&    positions,
                  GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                  GrainsMemBuffer<Kinematics<T>, M>& velocities,
                  GrainsMemBuffer<uint, M>&          rigidBodyIds,
                  GrainsMemBuffer<uint, M>&          componentIds,
                  uint                               numObstacles,
                  uint                               numLocalParticles);

    static void removeGhosts(GrainsMemBuffer<Vector3<T>, M>&    positions,
                             GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                             GrainsMemBuffer<Kinematics<T>, M>& velocities,
                             GrainsMemBuffer<uint, M>&          rigidBodyIds,
                             GrainsMemBuffer<uint, M>&          componentIds,
                             uint                               totalLocal);

    uint getNumGhostsReceived() const { return m_numGhostsReceived; }

private:
    void identifyGhosts(const GrainsMemBuffer<Vector3<T>, M>& positions,
                        uint numObstacles, uint numLocalParticles);
    void ensureStagingCapacity(uint capacity);

    void gatherToStaging(const GrainsMemBuffer<Vector3<T>, M>&    positions,
                         const GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                         const GrainsMemBuffer<Kinematics<T>, M>& velocities,
                         const GrainsMemBuffer<uint, M>&          rigidBodyIds,
                         const GrainsMemBuffer<uint, M>&          componentIds,
                         const std::vector<uint>&                 indices);

    void scatterFromStaging(GrainsMemBuffer<Vector3<T>, M>&    positions,
                            GrainsMemBuffer<Quaternion<T>, M>& quaternions,
                            GrainsMemBuffer<Kinematics<T>, M>& velocities,
                            GrainsMemBuffer<uint, M>&          rigidBodyIds,
                            GrainsMemBuffer<uint, M>&          componentIds,
                            uint destOffset, uint count);
};

#endif
