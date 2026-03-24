#ifndef _GRAINSMULTICPU_HH_
#define _GRAINSMULTICPU_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "DomainDecomposition.hh"
#include "GhostExchangerCPU.hh"
#include "GrainsCPU.hh"
#include "ParticleMigratorCPU.hh"

// =================================================================================================
/** @brief Multi-CPU DEM solver using MPI.

    Extends GrainsCPU with spatial domain decomposition, ghost-particle exchange and particle
    migration — exactly the same algorithm as GrainsMultiGPU but operating entirely on
    HOST-resident buffers.  No GPU is involved; the solver runs across MPI ranks using CPU
    collision detection and force computation.

    @author Multi-CPU extension — 2026 */
// =================================================================================================
template <typename T>
class GrainsMultiCPU : public GrainsCPU<T>
{
private:
    std::unique_ptr<DomainDecomposition<T>> m_decomp;
    std::unique_ptr<GhostExchangerCPU<T>>   m_ghostExchanger;
    std::unique_ptr<ParticleMigratorCPU<T>> m_migrator;

    uint m_numLocalParticles;
    uint m_numGhostParticles;
    int  m_mpiRank;
    int  m_mpiSize;

public:
    GrainsMultiCPU();
    ~GrainsMultiCPU();

    /** @brief Tasks to perform before time-stepping.
        @param rootElement XML root */
    void initialize(DOMElement* rootElement) final;

    /** @brief Multi-CPU simulation loop with ghost exchange and migration. */
    void simulate() final;

    /** @brief Gather and finalize post-processing. */
    void finalize() final;

private:
    void partitionParticlesToLocal();
    void stepWithCommunication();

    template <MemType M>
    void postProcessParallel(const std::unique_ptr<ComponentManager<T, M>>& cm);
};

#endif
