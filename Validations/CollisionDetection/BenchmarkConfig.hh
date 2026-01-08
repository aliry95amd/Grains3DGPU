#ifndef _BENCHMARKCONFIG_HH_
#define _BENCHMARKCONFIG_HH_

#include "LinkedCell.hh"
#include "Vector3.hh"

enum class ParticleShapeType
{
    SPHERE       = 0,
    BOX          = 1,
    SUPERQUADRIC = 2
};

enum class GJKRepresentationType
{
    TRANSFORM  = 0,
    QUATERNION = 1
};

enum class GJKVariantType
{
    JOHNSON      = 0,
    SIGNEDVOLUME = 1
};

enum class PLATFORM
{
    CPU  = 0,
    GPU  = 1,
    BOTH = 2
};

// =================================================================================================
/** @brief Configuration for collision detection benchmarks */
template <typename T>
struct BenchmarkConfig
{
    // Platform parameters
    PLATFORM platform;

    // Particle parameters
    uint              numParticles;
    ParticleShapeType shapeType;
    T                 particleSize;
    T                 aspectRatio;

    // Domain parameters
    Vector3<T> domainMin;
    Vector3<T> domainMax;

    // Neighbor list parameters
    NeighborListType neighborListType;
    LinkedCellType   linkedCellType;
    bool             sort;
    bool             adaptiveSkin;

    // GJK parameters
    GJKRepresentationType gjkRepresentation;
    GJKVariantType        gjkVariant;
    bool                  testRelative;

    // Test parameters
    uint numTrials;
    uint randomSeed;
    bool validateContacts;

    BenchmarkConfig()
        : platform(PLATFORM::BOTH)
        , numParticles(1000)
        , shapeType(ParticleShapeType::BOX)
        , particleSize(0.05)
        , aspectRatio(1.0)
        , domainMin(Vector3<T>(-1, -1, -1))
        , domainMax(Vector3<T>(1, 1, 1))
        , neighborListType(NeighborListType::NSQ)
        , linkedCellType(LinkedCellType::SORTBASED)
        , sort(false)
        , adaptiveSkin(false)
        , gjkRepresentation(GJKRepresentationType::TRANSFORM)
        , gjkVariant(GJKVariantType::JOHNSON)
        , testRelative(true)
        , numTrials(5)
        , randomSeed(42)
        , validateContacts(false)
    {
    }
};

#endif
