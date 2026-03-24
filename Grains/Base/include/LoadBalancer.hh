#ifndef _LOADBALANCER_HH_
#define _LOADBALANCER_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "DomainDecomposition.hh"
#include <vector>

// =================================================================================================
/** @brief Dynamic load balancer for 1-D slab domain decomposition.

    Periodically measures per-rank wall-clock step time, gathers load across all ranks, and shifts
    slab boundaries to equalize work.  Boundary shifts are clamped to prevent oscillation.

    After rebalancing, the existing ParticleMigrator naturally handles redistribution on the
    next timestep since particles outside the new boundaries are detected as emigrants.

    @author Multi-rank extension — 2026 */
// =================================================================================================
template <typename T>
class LoadBalancer
{
private:
    uint m_rebalanceFrequency;
    T    m_maxShiftFraction;
    T    m_imbalanceThreshold;
    bool m_enabled;

    std::vector<double> m_allLoads;

public:
    /** @brief Construct with rebalancing parameters.
        @param frequency       rebalance every N steps (0 = disabled)
        @param maxShiftFrac    max boundary shift as fraction of slab width per rebalance
        @param threshold       only rebalance if max/min load ratio exceeds this */
    LoadBalancer(uint frequency    = 0,
                 T    maxShiftFrac = T(0.1),
                 T    threshold    = T(1.2));

    ~LoadBalancer() = default;

    bool isEnabled()                     const { return m_enabled; }
    bool shouldRebalance(uint stepCount) const;

    /** @brief Perform load-based boundary adjustment.
        Each rank provides its local step time; the balancer gathers all times via MPI,
        computes ideal boundaries, and updates the decomposition.
        @param decomp          domain decomposition to update (boundaries are shifted in-place)
        @param localStepTime   wall-clock time of the last step on this rank */
    void rebalance(DomainDecomposition<T>& decomp, double localStepTime);
};

#endif
