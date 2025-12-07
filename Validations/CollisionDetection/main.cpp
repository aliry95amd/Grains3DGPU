#include "GJKPerformanceComparison.hh"
#include <iostream>

int main()
{
    Gout(std::string(80, '='));
    Gout("GJK Performance Comparison Test");
    Gout(std::string(80, '='));

    // Test with double precision
    // GJKPerformanceComparison<double>         performanceTest;
    // std::vector<std::pair<int, std::string>> shapeTypes
    //     = {{0, "Box"}, {1, "Sphere"}, {2, "Superquadric"}};
    // std::vector<int> numParticlesList = {1000, 2000, 4000, 8000};
    GJKPerformanceComparison<float>          performanceTest;
    std::vector<std::pair<int, std::string>> shapeTypes       = {{0, "Box"}};
    std::vector<int>                         numParticlesList = {16000};

    for(auto& shapeType : shapeTypes)
    {
        for(int numParticles : numParticlesList)
        {
            Gout("\nTest Begins:");
            Gout(std::string(80, '='));
            performanceTest.setTestParameters(numParticles,      // numParticles
                                              2 * numParticles,  // numPairs
                                              shapeType.first,
                                              256,
                                              true);
            performanceTest.setRandomSeed(42);  // For reproducible results
            performanceTest.run();
        }
    }

    std::cout << "\nAll performance tests completed!\n";
    return 0;
}
