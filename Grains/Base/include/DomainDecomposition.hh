#ifndef _DOMAINDECOMPOSITION_HH_
#define _DOMAINDECOMPOSITION_HH_

#ifdef GRAINS_USE_MPI
#include <mpi.h>
#endif

#include "Vector3.hh"
#include <vector>

// =================================================================================================
/** @brief 1-D slab domain decomposition for multi-GPU DEM.

    Splits the global simulation domain along its longest axis into contiguous slabs, one per MPI
    rank.  Each rank knows its local subdomain bounds, ghost-layer extent, and the ranks of its
    two neighbors (lower / upper along the split axis).

    The decomposition is intentionally kept simple (uniform 1-D) so that the rest of the solver
    (linked cells, collision detection, force computation) can remain unchanged — each rank just
    operates on a smaller local domain padded by a ghost layer.

    @author Multi-GPU extension — 2026 */
// =================================================================================================
template <typename T>
class DomainDecomposition
{
private:
    int m_rank;
    int m_numRanks;

    Vector3<T> m_globalMin;
    Vector3<T> m_globalMax;

    Vector3<T> m_localMin;
    Vector3<T> m_localMax;

    T   m_ghostWidth;
    int m_splitAxis;  // Direction enum value: X=0, Y=1, Z=2

    int m_lowerNeighbor;  // rank on the "low" side of split axis, -1 if boundary
    int m_upperNeighbor;  // rank on the "high" side of split axis, -1 if boundary

    T m_slabWidth;  // width of each slab along split axis

#ifdef GRAINS_USE_MPI
    MPI_Comm m_comm;
#endif

public:
    /** @name Constructors & Destructor */
    //@{
#ifdef GRAINS_USE_MPI
    /** @brief Construct from MPI communicator and global domain.
        @param comm MPI communicator (typically MPI_COMM_WORLD)
        @param globalMin origin of the simulation domain
        @param globalMax maximum coordinate of the simulation domain
        @param ghostWidth ghost-layer thickness (>= max particle diameter + skin) */
    DomainDecomposition(MPI_Comm           comm,
                        const Vector3<T>&  globalMin,
                        const Vector3<T>&  globalMax,
                        T                  ghostWidth);
#endif

    /** @brief Single-rank fallback (no MPI) — local domain equals global domain. */
    DomainDecomposition(const Vector3<T>& globalMin,
                        const Vector3<T>& globalMax,
                        T                 ghostWidth);

    ~DomainDecomposition() = default;
    //@}

    /** @name Accessors */
    //@{
    int               getRank()          const { return m_rank; }
    int               getNumRanks()      const { return m_numRanks; }
    const Vector3<T>& getGlobalMin()     const { return m_globalMin; }
    const Vector3<T>& getGlobalMax()     const { return m_globalMax; }
    const Vector3<T>& getLocalMin()      const { return m_localMin; }
    const Vector3<T>& getLocalMax()      const { return m_localMax; }
    T                 getGhostWidth()    const { return m_ghostWidth; }
    int               getSplitAxis()     const { return m_splitAxis; }
    int               getLowerNeighbor() const { return m_lowerNeighbor; }
    int               getUpperNeighbor() const { return m_upperNeighbor; }
    //@}

    /** @name Query methods */
    //@{
    /** @brief Local domain extended by the ghost layer on each side of the split axis. */
    Vector3<T> getGhostMin() const;
    Vector3<T> getGhostMax() const;

    /** @brief True if @p pos lies strictly inside the local subdomain (not in the ghost layer). */
    bool isLocal(const Vector3<T>& pos) const;

    /** @brief True if @p pos is in the lower ghost layer (belongs to the lower neighbor). */
    bool isInGhostLower(const Vector3<T>& pos) const;

    /** @brief True if @p pos is in the upper ghost layer (belongs to the upper neighbor). */
    bool isInGhostUpper(const Vector3<T>& pos) const;

    /** @brief True if a local particle at @p pos must be sent as a ghost to the lower neighbor. */
    bool needsGhostLower(const Vector3<T>& pos) const;

    /** @brief True if a local particle at @p pos must be sent as a ghost to the upper neighbor. */
    bool needsGhostUpper(const Vector3<T>& pos) const;

    /** @brief Returns the rank that owns @p pos (-1 if outside global domain). */
    int getOwnerRank(const Vector3<T>& pos) const;

    /** @brief Dynamically update local boundaries (called by LoadBalancer).
        @param newLocalMin new lower bound along split axis
        @param newLocalMax new upper bound along split axis */
    void updateBoundaries(T newLocalMin, T newLocalMax);
    //@}

private:
    void partition();
};

#endif
