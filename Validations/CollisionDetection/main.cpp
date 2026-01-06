#include "CollisionDetectionBenchmark.hh"
#include <iostream>

int main()
{
    Gout(std::string(80, '='));
    Gout("Collision Detection Performance Benchmark");
    Gout(std::string(80, '='));

    // =========================================================================================
    // Configure benchmark parameters
    // =========================================================================================
    BenchmarkConfig<float> config;

    // Platform parameters
    config.platform = PLATFORM::GPU;

    // Particle parameters
    config.numParticles = 256;
    config.shapeType    = ParticleShapeType::SPHERE;
    config.particleSize = 0.05;

    // Domain parameters (adjust for desired packing density)
    config.domainMin = Vector3<float>(-1.0, -1.0, -1.0);
    config.domainMax = Vector3<float>(1.0, 1.0, 1.0);

    // Neighbor list configuration
    config.neighborListType = NeighborListType::NSQ;
    config.linkedCellType   = LinkedCellType::SORTBASED;  // HOST, SORTBASED, ATOMIC, ATOMICFIXED
    config.sort             = 0;  // 0 = no sorting, N = sort every N updates
    config.adaptiveSkin     = 0;  // 0 = always update, N = update every N timesteps

    // GJK configuration
    config.gjkRepresentation = GJKRepresentationType::QUATERNION;
    config.gjkVariant        = GJKVariantType::JOHNSON;
    config.testRelative      = true;

    // Test parameters
    config.numTrials  = 5;   // Number of trials per configuration
    config.randomSeed = 42;  // Base random seed

    // =========================================================================================
    // Run benchmark with current configuration
    // =========================================================================================
    std::string csvFilename = "collision_benchmark_results.csv";

    CollisionDetectionBenchmark<float> benchmark(config, csvFilename);
    benchmark.runBenchmark();

    std::cout << "\nAll performance tests completed!\n";
    std::cout << "Results written to: " << csvFilename << "\n";
    return 0;
}
