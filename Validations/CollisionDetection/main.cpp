#include "CollisionDetectionBenchmark.hh"
#include <iostream>
#include <vector>

template <typename T>
void runBenchmarkWithPrecision(const BenchmarkConfig& config,
                               const std::string&     csvFilename,
                               bool                   appendToCSV)
{
    CollisionDetectionBenchmark<T> benchmark(config, csvFilename, appendToCSV);
    benchmark.runBenchmark();
}

int main()
{
    Gout(std::string(80, '='));
    Gout("Collision Detection Performance Benchmark");
    Gout(std::string(80, '='));

    // =============================================================================================
    // Configure particle parameters
    // =============================================================================================
    struct ParticleConfig
    {
        uint              numParticles;
        Vector3<double>   domainMin;
        Vector3<double>   domainMax;
        ParticleShapeType shapeType;
        Vector3<double>   particleSize;
        double            aspectRatio;
    };

    double r = 0.05;                                // Base particle size
    double A = cos(1.0 / 3.0 * acos(-1.0 / 64.0));  // Approximately 0.8634

    std::vector<ParticleConfig> particleConfigs = {
        // Spheres - S1
        {128,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(24 * r, 24 * r, 24 * r),
         ParticleShapeType::SPHERE,
         Vector3<double>(r, r, r),
         1.0},
        {4096,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(48 * r, 48 * r, 48 * r),
         ParticleShapeType::SPHERE,
         Vector3<double>(r, r, r),
         1.0},
        {32768,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(96 * r, 96 * r, 96 * r),
         ParticleShapeType::SPHERE,
         Vector3<double>(r, r, r),
         1.0},
        {524288,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(192 * r, 192 * r, 192 * r),
         ParticleShapeType::SPHERE,
         Vector3<double>(r, r, r),
         1.0},
        // Boxes - B1
        {1024,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(24 * r, 24 * r, 24 * r),
         ParticleShapeType::BOX,
         Vector3<double>(2. / sqrt(3.) * r, 2. / sqrt(3.) * r, 2. / sqrt(3.) * r),
         1.0},
        {4096,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(48 * r, 48 * r, 48 * r),
         ParticleShapeType::BOX,
         Vector3<double>(2. / sqrt(3.) * r, 2. / sqrt(3.) * r, 2. / sqrt(3.) * r),
         1.0},
        {16384,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(96 * r, 96 * r, 96 * r),
         ParticleShapeType::BOX,
         Vector3<double>(2. / sqrt(3.) * r, 2. / sqrt(3.) * r, 2. / sqrt(3.) * r),
         1.0},
        {65536,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(192 * r, 192 * r, 192 * r),
         ParticleShapeType::BOX,
         Vector3<double>(2. / sqrt(3.) * r, 2. / sqrt(3.) * r, 2. / sqrt(3.) * r),
         1.0},
        // Superquadrics - S4
        {1024,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(24 * r, 24 * r, 24 * r),
         ParticleShapeType::SUPERQUADRIC,
         Vector3<double>(r / 2., r / 2., 4. * r),
         4.0},
        {4096,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(48 * r, 48 * r, 48 * r),
         ParticleShapeType::SUPERQUADRIC,
         Vector3<double>(r / 2., r / 2., 4. * r),
         4.0},
        {16384,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(96 * r, 96 * r, 96 * r),
         ParticleShapeType::SUPERQUADRIC,
         Vector3<double>(r / 2., r / 2., 4. * r),
         4.0},
        {65536,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(192 * r, 192 * r, 192 * r),
         ParticleShapeType::SUPERQUADRIC,
         Vector3<double>(r / 2., r / 2., 4. * r),
         4.0},
        // Boxes - B4
        {1024,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(24 * r, 24 * r, 24 * r),
         ParticleShapeType::BOX,
         Vector3<double>(1. / sqrt(6. * A) * r, 1. / sqrt(6. * A) * r, 16. / sqrt(3.) * A * r),
         4.0},
        {4096,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(48 * r, 48 * r, 48 * r),
         ParticleShapeType::BOX,
         Vector3<double>(1. / sqrt(6. * A) * r, 1. / sqrt(6. * A) * r, 16. / sqrt(3.) * A * r),
         4.0},
        {16384,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(96 * r, 96 * r, 96 * r),
         ParticleShapeType::BOX,
         Vector3<double>(1. / sqrt(6. * A) * r, 1. / sqrt(6. * A) * r, 16. / sqrt(3.) * A * r),
         4.0},
        {65536,
         Vector3<double>(0.0, 0.0, 0.0),
         Vector3<double>(192 * r, 192 * r, 192 * r),
         ParticleShapeType::BOX,
         Vector3<double>(1. / sqrt(6. * A) * r, 1. / sqrt(6. * A) * r, 16. / sqrt(3.) * A * r),
         4.0},
    };

    // Test parameters
    uint numTrials        = 3;      // Number of trials per configuration
    uint randomSeed       = 42;     // Base random seed
    bool validateContacts = false;  // Write contact info to file for validation

    // =============================================================================================
    // Generate all test configurations
    // =============================================================================================
    std::vector<PrecisionType>         precisions = {PrecisionType::SINGLE, PrecisionType::DOUBLE};
    std::vector<PLATFORM>              platforms  = {PLATFORM::CPU, PLATFORM::GPU};
    std::vector<GJKRepresentationType> representations
        = {GJKRepresentationType::QUATERNION, GJKRepresentationType::TRANSFORM};
    std::vector<GJKVariantType> gjkVariants
        = {GJKVariantType::JOHNSON, GJKVariantType::SIGNEDVOLUME};
    std::vector<bool> transformModes = {true, false};  // relative vs global

    std::string csvFilename = "data/collision_benchmark_comprehensive.csv";

    int totalConfigs = precisions.size() * platforms.size() * representations.size()
                       * gjkVariants.size() * transformModes.size() * particleConfigs.size();
    int configCount = 0;

    Gout("\nRunning comprehensive benchmark with " + std::to_string(totalConfigs)
         + " different configurations...\n");

    // =========================================================================================
    // Iterate through all combinations
    // =========================================================================================
    for(const auto& particleConfig : particleConfigs)
    {
        for(auto precision : precisions)
        {
            for(auto platform : platforms)
            {
                for(auto representation : representations)
                {
                    for(auto gjkVariant : gjkVariants)
                    {
                        for(bool useRelativeTransform : transformModes)
                        {
                            configCount++;

                            // Create configuration
                            BenchmarkConfig config;
                            config.precision            = precision;
                            config.platform             = platform;
                            config.numParticles         = particleConfig.numParticles;
                            config.shapeType            = particleConfig.shapeType;
                            config.particleSize         = particleConfig.particleSize;
                            config.aspectRatio          = particleConfig.aspectRatio;
                            config.domainMin            = particleConfig.domainMin;
                            config.domainMax            = particleConfig.domainMax;
                            config.gjkRepresentation    = representation;
                            config.gjkVariant           = gjkVariant;
                            config.useRelativeTransform = useRelativeTransform;
                            config.numTrials            = numTrials;
                            config.randomSeed           = randomSeed;
                            config.validateContacts     = validateContacts && (configCount == 1);

                            // Print configuration info
                            Gout("\n" + std::string(80, '-'));
                            Gout("Configuration " + std::to_string(configCount) + "/"
                                 + std::to_string(totalConfigs));
                            Gout("  Particles: " + std::to_string(particleConfig.numParticles));
                            Gout("  Domain: [" + std::to_string(particleConfig.domainMin[0])
                                 + " to " + std::to_string(particleConfig.domainMax[0]) + "]");
                            Gout("  Shape: "
                                 + std::string(particleConfig.shapeType == ParticleShapeType::SPHERE
                                                   ? "Sphere"
                                               : particleConfig.shapeType == ParticleShapeType::BOX
                                                   ? "Box"
                                                   : "Superquadric"));
                            Gout("  Aspect: " + std::to_string(particleConfig.aspectRatio));
                            Gout("  Precision: "
                                 + std::string(precision == PrecisionType::SINGLE ? "Single"
                                                                                  : "Double"));
                            Gout("  Platform: "
                                 + std::string(platform == PLATFORM::CPU ? "CPU" : "GPU"));
                            Gout("  Representation: "
                                 + std::string(representation == GJKRepresentationType::QUATERNION
                                                   ? "Quaternion"
                                                   : "Transform3"));
                            Gout("  GJK Variant: "
                                 + std::string(gjkVariant == GJKVariantType::JOHNSON
                                                   ? "Johnson"
                                                   : "SignedVolume"));
                            Gout("  Transform Mode: "
                                 + std::string(useRelativeTransform ? "Relative" : "Global"));
                            Gout(std::string(80, '-'));

                            // Run benchmark with the configured precision
                            bool appendToCSV = (configCount > 1);
                            if(config.precision == PrecisionType::SINGLE)
                            {
                                runBenchmarkWithPrecision<float>(config, csvFilename, appendToCSV);
                            }
                            else
                            {
                                runBenchmarkWithPrecision<double>(config, csvFilename, appendToCSV);
                            }
                        }
                    }
                }
            }
        }
    }

    Gout("\n" + std::string(80, '='));
    Gout("All " + std::to_string(totalConfigs) + " configurations completed!");
    Gout("Results written to: " + csvFilename);
    Gout(std::string(80, '='));

    return 0;
}
