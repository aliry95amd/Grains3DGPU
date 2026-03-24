#ifndef _GRAINSMULTIGPU_HH_
#define _GRAINSMULTIGPU_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "DomainDecomposition.hh"
#include "GhostExchanger.hh"
#include "GrainsGPU.hh"
#include "ParticleMigrator.hh"

// =================================================================================================
/** @brief Multi-GPU DEM solver using MPI + CUDA.

    Extends GrainsGPU with:
      - 1-D slab domain decomposition (DomainDecomposition)
      - Ghost-particle halo exchange  (GhostExchanger)
      - Particle migration across rank boundaries (ParticleMigrator)
      - Per-rank GPU binding  (each MPI rank owns one GPU)
      - Parallel post-processing I/O  (rank 0 gathers or each rank writes its local data)

    The existing single-GPU kernels (collision detection, contact forces, time integration) are
    reused unchanged.  Only the outer simulation loop is modified to insert communication phases.

    @author Multi-GPU extension — 2026 */
// =================================================================================================
template <typename T>
class GrainsMultiGPU : public GrainsGPU<T>
{
private:
    std::unique_ptr<DomainDecomposition<T>> m_decomp;
    std::unique_ptr<GhostExchanger<T>>      m_ghostExchanger;
    std::unique_ptr<ParticleMigrator<T>>    m_migrator;

    uint m_numLocalParticles;
    uint m_numGhostParticles;
    int  m_mpiRank;
    int  m_mpiSize;

public:
    /** @name Constructors & Destructor */
    //@{
    GrainsMultiGPU();
    ~GrainsMultiGPU();
    //@}

    /** @name High-level methods */
    //@{
    /** @brief Per-rank GPU setup + domain decomposition initialization. */
    void setupGPUDevice();

    /** @brief Full initialization: reads XML, partitions domain, builds local structures. */
    void initialize(DOMElement* rootElement) final;

    /** @brief Multi-GPU simulation loop with ghost exchange and migration. */
    void simulate() final;

    /** @brief Gather and finalize post-processing across ranks. */
    void finalize() final;
    //@}

private:
    /** @brief Construction phase: partitions the rigid bodies, creates local component manager.
        @param rootElement XML root */
    void Construction(DOMElement* rootElement);

    /** @brief Partitions particles to the local subdomain (host side, before H2D copy). */
    void partitionParticlesToLocal();

    /** @brief Performs one ghost exchange + collision + force + integrate cycle. */
    void stepWithCommunication();

    /** @brief Parallel post-processing: each rank writes its local data. */
    template <MemType M>
    void postProcessParallel(const std::unique_ptr<ComponentManager<T, M>>& cm);
};

#endif
