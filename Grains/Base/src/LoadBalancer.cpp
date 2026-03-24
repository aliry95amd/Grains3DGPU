#include "LoadBalancer.hh"
#include "Basic.hh"

#include <algorithm>
#include <numeric>

// =================================================================================================
template <typename T>
LoadBalancer<T>::LoadBalancer(uint frequency, T maxShiftFrac, T threshold)
    : m_rebalanceFrequency(frequency)
    , m_maxShiftFraction(maxShiftFrac)
    , m_imbalanceThreshold(threshold)
    , m_enabled(frequency > 0)
{
}

// =================================================================================================
template <typename T>
bool LoadBalancer<T>::shouldRebalance(uint stepCount) const
{
    if(!m_enabled || m_rebalanceFrequency == 0)
        return false;
    return (stepCount % m_rebalanceFrequency == 0);
}

// =================================================================================================
template <typename T>
void LoadBalancer<T>::rebalance(DomainDecomposition<T>& decomp, double localStepTime)
{
#ifndef GRAINS_USE_MPI
    (void)decomp;
    (void)localStepTime;
    return;
#else
    int rank     = decomp.getRank();
    int numRanks = decomp.getNumRanks();

    if(numRanks <= 1)
        return;

    m_allLoads.resize(numRanks);
    MPI_Allgather(&localStepTime, 1, MPI_DOUBLE,
                  m_allLoads.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);

    double maxLoad = *std::max_element(m_allLoads.begin(), m_allLoads.end());
    double minLoad = *std::min_element(m_allLoads.begin(), m_allLoads.end());

    if(minLoad <= 0.0 || maxLoad / minLoad < static_cast<double>(m_imbalanceThreshold))
        return;

    double totalLoad = std::accumulate(m_allLoads.begin(), m_allLoads.end(), 0.0);
    double idealLoad = totalLoad / numRanks;

    int splitAxis = decomp.getSplitAxis();
    T globalMin   = decomp.getGlobalMin()[splitAxis];
    T globalMax   = decomp.getGlobalMax()[splitAxis];
    T globalExtent = globalMax - globalMin;

    // Compute cumulative load fractions → ideal slab boundaries
    // Each boundary i (between rank i-1 and rank i) should sit where
    // cumulative load up to rank i-1 equals i * idealLoad.
    std::vector<T> newBoundaries(numRanks + 1);
    newBoundaries[0]         = globalMin;
    newBoundaries[numRanks]  = globalMax;

    double cumLoad = 0.0;
    for(int r = 0; r < numRanks - 1; ++r)
    {
        cumLoad += m_allLoads[r];
        T idealFraction = static_cast<T>(cumLoad / totalLoad);
        newBoundaries[r + 1] = globalMin + idealFraction * globalExtent;
    }

    // Clamp: limit each boundary shift to maxShiftFraction of the current slab width
    T currentSlabWidth = globalExtent / static_cast<T>(numRanks);
    T maxShift = m_maxShiftFraction * currentSlabWidth;

    // Current uniform boundaries for clamping reference
    for(int r = 1; r < numRanks; ++r)
    {
        T uniformBoundary = globalMin + r * currentSlabWidth;
        T shift = newBoundaries[r] - uniformBoundary;
        if(shift > maxShift)       newBoundaries[r] = uniformBoundary + maxShift;
        else if(shift < -maxShift) newBoundaries[r] = uniformBoundary - maxShift;
    }

    // Ensure monotonicity
    for(int r = 1; r < numRanks; ++r)
    {
        if(newBoundaries[r] <= newBoundaries[r - 1])
            newBoundaries[r] = newBoundaries[r - 1] + globalExtent * T(0.001);
    }

    T newLocalMin = newBoundaries[rank];
    T newLocalMax = newBoundaries[rank + 1];

    decomp.updateBoundaries(newLocalMin, newLocalMax);

    if(rank == 0)
    {
        Gout("[LoadBalancer] Rebalanced: max/min ratio was",
             maxLoad / minLoad, "-> shifted boundaries");
    }
#endif
}

// =================================================================================================
template class LoadBalancer<float>;
template class LoadBalancer<double>;
