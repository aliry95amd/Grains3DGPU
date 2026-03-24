#include "DomainDecomposition.hh"
#include "Basic.hh"

// =================================================================================================
// MPI constructor
// =================================================================================================
#ifdef GRAINS_USE_MPI
template <typename T>
DomainDecomposition<T>::DomainDecomposition(MPI_Comm          comm,
                                            const Vector3<T>& globalMin,
                                            const Vector3<T>& globalMax,
                                            T                 ghostWidth)
    : m_globalMin(globalMin)
    , m_globalMax(globalMax)
    , m_ghostWidth(ghostWidth)
    , m_comm(comm)
{
    MPI_Comm_rank(m_comm, &m_rank);
    MPI_Comm_size(m_comm, &m_numRanks);
    partition();
}
#endif

// =================================================================================================
// Single-rank fallback constructor
// =================================================================================================
template <typename T>
DomainDecomposition<T>::DomainDecomposition(const Vector3<T>& globalMin,
                                            const Vector3<T>& globalMax,
                                            T                 ghostWidth)
    : m_rank(0)
    , m_numRanks(1)
    , m_globalMin(globalMin)
    , m_globalMax(globalMax)
    , m_ghostWidth(ghostWidth)
{
    partition();
}

// =================================================================================================
// Uniform 1-D slab partitioning along the longest axis
// =================================================================================================
template <typename T>
void DomainDecomposition<T>::partition()
{
    T extentX = m_globalMax[X] - m_globalMin[X];
    T extentY = m_globalMax[Y] - m_globalMin[Y];
    T extentZ = m_globalMax[Z] - m_globalMin[Z];

    if(extentX >= extentY && extentX >= extentZ)
        m_splitAxis = X;
    else if(extentY >= extentX && extentY >= extentZ)
        m_splitAxis = Y;
    else
        m_splitAxis = Z;

    T globalExtent = m_globalMax[m_splitAxis] - m_globalMin[m_splitAxis];
    m_slabWidth    = globalExtent / static_cast<T>(m_numRanks);

    m_localMin = m_globalMin;
    m_localMax = m_globalMax;
    m_localMin[m_splitAxis] = m_globalMin[m_splitAxis] + m_rank * m_slabWidth;
    m_localMax[m_splitAxis] = m_globalMin[m_splitAxis] + (m_rank + 1) * m_slabWidth;

    if(m_rank == m_numRanks - 1)
        m_localMax[m_splitAxis] = m_globalMax[m_splitAxis];

    m_lowerNeighbor = (m_rank > 0)              ? m_rank - 1 : -1;
    m_upperNeighbor = (m_rank < m_numRanks - 1) ? m_rank + 1 : -1;

    Gout("[Rank", m_rank, "] Domain decomposition: split axis =", m_splitAxis,
         ", slab width =", m_slabWidth);
    Gout("[Rank", m_rank, "] Local domain: [",
         m_localMin[m_splitAxis], ",", m_localMax[m_splitAxis], "]");
    Gout("[Rank", m_rank, "] Neighbors: lower =", m_lowerNeighbor,
         ", upper =", m_upperNeighbor);
}

// =================================================================================================
template <typename T>
Vector3<T> DomainDecomposition<T>::getGhostMin() const
{
    Vector3<T> gmin = m_localMin;
    if(m_lowerNeighbor >= 0)
        gmin[m_splitAxis] -= m_ghostWidth;
    return gmin;
}

// =================================================================================================
template <typename T>
Vector3<T> DomainDecomposition<T>::getGhostMax() const
{
    Vector3<T> gmax = m_localMax;
    if(m_upperNeighbor >= 0)
        gmax[m_splitAxis] += m_ghostWidth;
    return gmax;
}

// =================================================================================================
template <typename T>
bool DomainDecomposition<T>::isLocal(const Vector3<T>& pos) const
{
    T coord = pos[m_splitAxis];
    return coord >= m_localMin[m_splitAxis] && coord < m_localMax[m_splitAxis];
}

// =================================================================================================
template <typename T>
bool DomainDecomposition<T>::isInGhostLower(const Vector3<T>& pos) const
{
    if(m_lowerNeighbor < 0)
        return false;
    T coord = pos[m_splitAxis];
    return coord >= (m_localMin[m_splitAxis] - m_ghostWidth) && coord < m_localMin[m_splitAxis];
}

// =================================================================================================
template <typename T>
bool DomainDecomposition<T>::isInGhostUpper(const Vector3<T>& pos) const
{
    if(m_upperNeighbor < 0)
        return false;
    T coord = pos[m_splitAxis];
    return coord >= m_localMax[m_splitAxis] && coord < (m_localMax[m_splitAxis] + m_ghostWidth);
}

// =================================================================================================
template <typename T>
bool DomainDecomposition<T>::needsGhostLower(const Vector3<T>& pos) const
{
    if(m_lowerNeighbor < 0)
        return false;
    T coord = pos[m_splitAxis];
    return coord < (m_localMin[m_splitAxis] + m_ghostWidth);
}

// =================================================================================================
template <typename T>
bool DomainDecomposition<T>::needsGhostUpper(const Vector3<T>& pos) const
{
    if(m_upperNeighbor < 0)
        return false;
    T coord = pos[m_splitAxis];
    return coord >= (m_localMax[m_splitAxis] - m_ghostWidth);
}

// =================================================================================================
template <typename T>
int DomainDecomposition<T>::getOwnerRank(const Vector3<T>& pos) const
{
    T coord = pos[m_splitAxis];
    if(coord < m_globalMin[m_splitAxis] || coord >= m_globalMax[m_splitAxis])
        return -1;
    int rank = static_cast<int>((coord - m_globalMin[m_splitAxis]) / m_slabWidth);
    if(rank >= m_numRanks)
        rank = m_numRanks - 1;
    return rank;
}

// =================================================================================================
// Explicit instantiation
template class DomainDecomposition<float>;
template class DomainDecomposition<double>;
