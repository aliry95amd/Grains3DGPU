#ifndef _GRAINSMPI_HH_
#define _GRAINSMPI_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include <chrono>
#include <type_traits>

#include "DomainDecomposition.hh"
#include "GhostExchanger.hh"
#include "GrainsCPU.hh"
#include "GrainsGPU.hh"
#include "LoadBalancer.hh"
#include "ParticleMigrator.hh"

// =================================================================================================
/** @brief Unified multi-rank DEM solver for both GPU (DEVICE) and CPU (HOST).

    Uses conditional inheritance: when M == DEVICE inherits GrainsGPU<T>; when M == HOST inherits
    GrainsCPU<T>.  The simulation loop, ghost exchange, particle migration, and dynamic load
    balancing are shared in a single implementation using if constexpr to handle the few spots
    where DEVICE and HOST differ.

    @tparam T scalar type (float / double)
    @tparam M memory type: MemType::DEVICE for multi-GPU, MemType::HOST for multi-CPU
    @author Multi-rank extension — 2026 */
// =================================================================================================
template <typename T, MemType M>
class GrainsMPI : public std::conditional_t<M == MemType::DEVICE, GrainsGPU<T>, GrainsCPU<T>>
{
    using Base = std::conditional_t<M == MemType::DEVICE, GrainsGPU<T>, GrainsCPU<T>>;

    std::unique_ptr<DomainDecomposition<T>> m_decomp;
    std::unique_ptr<GhostExchanger<T, M>>   m_ghostExchanger;
    std::unique_ptr<ParticleMigrator<T, M>> m_migrator;
    std::unique_ptr<LoadBalancer<T>>        m_loadBalancer;

    uint m_numLocalParticles;
    uint m_numGhostParticles;
    int  m_mpiRank;
    int  m_mpiSize;

    auto& getActiveCM()
    {
        if constexpr(M == MemType::DEVICE)
            return GrainsGPU<T>::m_d_components;
        else
            return Grains<T>::m_components;
    }

    auto& getContactForce()
    {
        if constexpr(M == MemType::DEVICE)
            return GrainsGPU<T>::m_d_contactForce;
        else
            return Grains<T>::m_contactForce;
    }

    auto& getTimeIntegrator()
    {
        if constexpr(M == MemType::DEVICE)
            return GrainsGPU<T>::m_d_timeIntegrator;
        else
            return Grains<T>::m_timeIntegrator;
    }

public:
    GrainsMPI();
    ~GrainsMPI();

    void initialize(DOMElement* rootElement) final;
    void simulate() final;
    void finalize() final;

private:
    void constructionMPI(DOMElement* rootElement);
    void partitionParticlesToLocal();
    void stepWithCommunication();

    template <MemType MM>
    void postProcessParallel(const std::unique_ptr<ComponentManager<T, MM>>& cm);
};

#endif
