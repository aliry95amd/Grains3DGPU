#ifndef _BENCHMARKCONFIG_HH_
#define _BENCHMARKCONFIG_HH_

#include "LinkedCell.hh"
#include "Vector3.hh"

enum class PrecisionType
{
    SINGLE = 0,  // float
    DOUBLE = 1   // double
};

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
struct BenchmarkConfig
{
    // Precision
    PrecisionType precision;

    // Platform parameters
    PLATFORM platform;

    // Particle parameters
    uint              numParticles;
    ParticleShapeType shapeType;
    Vector3<double>   particleSize;
    double            aspectRatio;

    // Domain parameters
    Vector3<double> domainMin;
    Vector3<double> domainMax;

    // Test parameters
    uint numTrials;
    uint randomSeed;
    bool validateContacts;

    BenchmarkConfig()
        : precision(PrecisionType::SINGLE)
        , platform(PLATFORM::BOTH)
        , numParticles(1000)
        , shapeType(ParticleShapeType::BOX)
        , particleSize(0.05)
        , aspectRatio(1.0)
        , domainMin(Vector3<double>(-1, -1, -1))
        , domainMax(Vector3<double>(1, 1, 1))
        , numTrials(5)
        , randomSeed(42)
        , validateContacts(false){};
};

#endif
