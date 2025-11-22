#ifndef _GRAINSPARAMETERS_HH_
#define _GRAINSPARAMETERS_HH_

#include "Vector3.hh"

/** @brief Type of neighbor list */
enum class NeighborListType
{
    /** @brief N-Squared neighbor list */
    NSQ = 0,
    /** @brief Linked cell neighbor list */
    LINKEDCELL = 1
};

/** @brief Type of linked cell */
enum class LinkedCellType
{
    /** @brief Host linked cells */
    HOST = 0,
    /** @brief Sort-based linked cells for device */
    SORTBASED = 1,
    /** @brief Atomic linked cells for device */
    ATOMIC = 2
};

/** @brief Type of bounding volume */
enum class BoundingVolumeType
{
    /** @brief No bounding volume */
    OFF = 0,
    /** @brief Oriented Bounding Box */
    OBB = 1,
    /** @brief Oriented Bounding Cylinder */
    OBC = 2
};

/** @brief Type of narrow-phase detection */
enum class NarrowPhaseType
{
    /** @brief Gilbert-Johnson-Keerthi algorithm */
    GJK = 0
};

/** @brief Type of cell ordering */
enum class CellOrdering
{
    /** @brief Linear ordering */
    LINEAR = 0,
    /** @brief Morton ordering (Z-curve) */
    MORTON = 1
};

/** @brief Parameters for linked cell configuration */
template <typename T>
struct LinkedCellParameters
{
    /** \brief Type of linked cell */
    LinkedCellType type = LinkedCellType::HOST;
    /** \brief Cell ordering */
    CellOrdering cellOrdering = CellOrdering::LINEAR;
    /** \brief Linked cell size factor */
    T cellSizeFactor = 1;
    /** \brief If using adaptive skin, this is the desired number of 
        iterations that the skin should be valid for.
        If it is set to 0, then we don't use adaptive skin. */
    uint updateFrequency = 1;
    /** \brief If using Morton ordering, this is the number of iterations 
        between each sorting */
    uint sortFrequency = 0;
    /** \brief Minimum corner of the linked cell domain */
    Vector3<T> minCorner = Vector3<T>(0, 0, 0);
    /** \brief Maximum corner of the linked cell domain */
    Vector3<T> maxCorner = Vector3<T>(0, 0, 0);
};

// =============================================================================
/** @brief Parameters needed for Grains.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
template <typename T>
class GrainsParameters
{
public:
    /** @name Parameters */
    //@{
    /* Spatial */
    /** @brief Global domain origin */
    static Vector3<T> m_origin;
    /** @brief Global domain dimension */
    static Vector3<T> m_maxCoordinate;
    /** @brief Is simulation periodic? */
    static bool m_isPeriodic;

    /* Temporal */
    /** @brief Initial simulation time */
    static T m_tStart;
    /** @brief End simulation time */
    static T m_tEnd;
    /** @brief Simulation time step */
    static T m_dt;
    /** @brief Physical time */
    static T m_time;

    /* Numbers */
    /** @brief Number of particles in simulation */
    static uint m_numParticles;
    /** @brief Number of obstacles in simulation */
    static uint m_numObstacles;

    /* Physical */
    /** \brief Gravity vector */
    static Vector3<T> m_gravity;

    /* Material */
    /** \brief Map from material name to an uint ID */
    static std::unordered_map<std::string, uint> m_materialMap;
    /** \brief Number of different possible contact pairs (incl. obs-obs) */
    static uint m_numContactPairs;

    /* Post-Processing */
    /** \brief Queue of simulation time to write Post-Processing */
    static std::queue<T> m_tSave;

    /* GPU */
    /** \brief is simulation on GPU? */
    static bool m_isGPU;
    /** \brief GPU device properties */
    static cudaDeviceProp m_GPU;

    /* Collision Detection */
    struct CollisionDetectionParameters
    {
        /** \brief Type of neighbor list */
        NeighborListType neighborListType = NeighborListType::NSQ;
        /** \brief LinkedCell parameters */
        LinkedCellParameters<T> linkedCellParameters;
        /** \brief Type of bounding volume */
        BoundingVolumeType boundingVolumeType = BoundingVolumeType::OFF;
        /** \brief Type of narrow-phase detection */
        NarrowPhaseType narrowPhaseType = NarrowPhaseType::GJK;
    };

    /** \brief Collision detection parameters */
    static CollisionDetectionParameters m_collisionDetection;
    //@}
};

#endif