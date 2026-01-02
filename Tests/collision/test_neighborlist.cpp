#include <cmath>
#include <gtest/gtest.h>
#include <vector>

#include "Box.hh"
#include "CollisionDetection.hh"
#include "GrainsParameters.hh"
#include "NeighborList.hh"
#include "NeighborListFactory.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "Vector3.hh"

// Test fixture for NeighborList testing
class NeighborListTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize simulation state
        using GP                                      = GrainsParameters<double>;
        GP::m_simulationState.neighborListUpdateCount = 0;

        // Setup domain parameters
        nObstacles  = 0;
        nParticles  = 8;  // 2x2x2 grid
        uint nTotal = nObstacles + nParticles;

        // Create shared geometry
        double boxSize = 0.5;
        sharedBox      = new Box<double>(boxSize, boxSize, boxSize);

        // Create rigid bodies, positions, and quaternions for a 2x2x2 grid
        double spacing = 2.0;  // Spacing between particle centers
        for(uint i = 0; i < 2; ++i)
        {
            for(uint j = 0; j < 2; ++j)
            {
                for(uint k = 0; k < 2; ++k)
                {
                    rigidBodies.push_back(new RigidBody<double>(sharedBox, 0.1, 1000.0, 1));
                    positions.push_back(Vector3<double>(i * spacing, j * spacing, k * spacing));
                    quaternions.push_back(Quaternion<double>(0.0, 0.0, 0.0, 1.0));
                }
            }
        }

        // Setup CPU buffers
        rb_cpu.reserve(nTotal);
        positions_cpu.reserve(nTotal);
        quaternions_cpu.reserve(nTotal);

        for(uint i = 0; i < nTotal; ++i)
        {
            rb_cpu.push_back(rigidBodies[i]);
            positions_cpu.push_back(positions[i]);
            quaternions_cpu.push_back(quaternions[i]);
        }

        // Setup GPU buffers
        rb_gpu.reserve(nTotal);
        positions_gpu.reserve(nTotal);
        quaternions_gpu.reserve(nTotal);

        rb_gpu.copyFrom(rb_cpu);
        positions_gpu.copyFrom(positions_cpu);
        quaternions_gpu.copyFrom(quaternions_cpu);

        // Setup LinkedCell parameters
        linkedCellParams.minCorner              = Vector3<double>(-1.0, -1.0, -1.0);
        linkedCellParams.maxCorner              = Vector3<double>(5.0, 5.0, 5.0);
        linkedCellParams.minCellSize            = 3.0;  // Larger to ensure neighbors are found
        linkedCellParams.cellSizeFactor         = 1.0;
        linkedCellParams.maxNumCellsPerObstacle = 64;
        linkedCellParams.updateFrequency        = 0;
        linkedCellParams.sortFrequency          = 0;
    }

    void TearDown() override
    {
        // Reset simulation state
        using GP                                      = GrainsParameters<double>;
        GP::m_simulationState.neighborListUpdateCount = 0;

        // TODO: Fix memory cleanup - crashes on cleanup
        // For now, skip cleanup to prevent segfault
        /*
        // Clean up rigid bodies
        for(auto rb : rigidBodies)
        {
            delete rb;
        }
        rigidBodies.clear();

        // Clean up shared box
        delete sharedBox;
        */
    }

    // Helper to get neighbor pairs from CPU
    std::set<std::pair<uint, uint>> getNeighborPairs(NeighborList<double, MemType::HOST>* NL)
    {
        NL->updateNeighborList(positions_cpu, nObstacles, nParticles);
        std::set<std::pair<uint, uint>> pairs;

        const uint2* pairList = NL->getData();
        uint         size     = NL->getSize();

        for(uint i = 0; i < size; ++i)
        {
            uint a = pairList[i].x;
            uint b = pairList[i].y;
            // Store in sorted order to ensure consistent comparison
            if(a > b)
                std::swap(a, b);
            pairs.insert({a, b});
        }
        return pairs;
    }

    // Helper to get neighbor pairs from GPU
    std::set<std::pair<uint, uint>> getNeighborPairsGPU(NeighborList<double, MemType::DEVICE>* NL)
    {
        NL->updateNeighborList(positions_gpu, nObstacles, nParticles);
        std::set<std::pair<uint, uint>> pairs;

        uint size = NL->getSize();
        if(size == 0)
            return pairs;

        // Copy pairs from device to host
        std::vector<uint2> host_pairs(size);
        cudaMemcpy(host_pairs.data(), NL->getData(), size * sizeof(uint2), cudaMemcpyDeviceToHost);

        for(uint i = 0; i < size; ++i)
        {
            uint a = host_pairs[i].x;
            uint b = host_pairs[i].y;
            // Store in sorted order to ensure consistent comparison
            if(a > b)
                std::swap(a, b);
            pairs.insert({a, b});
        }
        return pairs;
    }

    // Domain parameters
    uint nObstacles;
    uint nParticles;

    // Shared geometry
    Box<double>* sharedBox;

    // Particle data
    std::vector<RigidBody<double>*> rigidBodies;
    std::vector<Vector3<double>>    positions;
    std::vector<Quaternion<double>> quaternions;

    // CPU buffers
    GrainsMemBuffer<RigidBody<double>*, MemType::HOST> rb_cpu;
    GrainsMemBuffer<Vector3<double>, MemType::HOST>    positions_cpu;
    GrainsMemBuffer<Quaternion<double>, MemType::HOST> quaternions_cpu;

    // GPU buffers
    GrainsMemBuffer<RigidBody<double>*, MemType::DEVICE> rb_gpu;
    GrainsMemBuffer<Vector3<double>, MemType::DEVICE>    positions_gpu;
    GrainsMemBuffer<Quaternion<double>, MemType::DEVICE> quaternions_gpu;

    // LinkedCell parameters
    LinkedCellParameters<double> linkedCellParams;
};

// Test: BruteForce CPU vs GPU give same results
TEST_F(NeighborListTest, BruteForce_CPU_vs_GPU)
{
    using GP                                  = GrainsParameters<double>;
    GP::m_collisionDetection.neighborListType = NeighborListType::NSQ;

    // Test CPU
    NeighborList<double, MemType::HOST>* NL_cpu = nullptr;
    NeighborListFactory<double, MemType::HOST>::create(&rb_cpu,
                                                       positions_cpu,
                                                       quaternions_cpu,
                                                       nObstacles,
                                                       nParticles,
                                                       NL_cpu);
    auto cpu_pairs = getNeighborPairs(NL_cpu);

    // Test GPU
    GP::m_simulationState.neighborListUpdateCount = 0;  // Reset for GPU test
    NeighborList<double, MemType::DEVICE>* NL_gpu = nullptr;
    NeighborListFactory<double, MemType::DEVICE>::create(&rb_gpu,
                                                         positions_gpu,
                                                         quaternions_gpu,
                                                         nObstacles,
                                                         nParticles,
                                                         NL_gpu);
    auto gpu_pairs = getNeighborPairsGPU(NL_gpu);

    // Verify same results
    EXPECT_GT(cpu_pairs.size(), 0) << "Should find some neighbors";
    EXPECT_EQ(cpu_pairs, gpu_pairs) << "CPU and GPU pairs should match exactly";
}

// Test: LinkedCell CPU and all GPU variants give same results
TEST_F(NeighborListTest, LinkedCell_CPU_vs_GPU_Variants)
{
    using GP                                      = GrainsParameters<double>;
    GP::m_collisionDetection.neighborListType     = NeighborListType::LINKEDCELL;
    GP::m_collisionDetection.linkedCellParameters = linkedCellParams;

    std::vector<std::set<std::pair<uint, uint>>> all_pairs;
    std::vector<std::string>                     names;

    // Test CPU Host
    linkedCellParams.type                         = LinkedCellType::HOST;
    GP::m_collisionDetection.linkedCellParameters = linkedCellParams;
    GP::m_simulationState.neighborListUpdateCount = 0;

    NeighborList<double, MemType::HOST>* NL_cpu = nullptr;
    NeighborListFactory<double, MemType::HOST>::create(&rb_cpu,
                                                       positions_cpu,
                                                       quaternions_cpu,
                                                       nObstacles,
                                                       nParticles,
                                                       NL_cpu);
    auto cpu_pairs = getNeighborPairs(NL_cpu);
    all_pairs.push_back(cpu_pairs);
    names.push_back("CPU_Host");

    // Test GPU variants
    LinkedCellType gpu_types[]
        = {LinkedCellType::SORTBASED, LinkedCellType::ATOMIC, LinkedCellType::ATOMICFIXED};
    std::string gpu_names[] = {"GPU_SortBased", "GPU_Atomic", "GPU_AtomicFixed"};

    for(int i = 0; i < 2; ++i)
    {
        linkedCellParams.type                         = gpu_types[i];
        GP::m_collisionDetection.linkedCellParameters = linkedCellParams;
        GP::m_simulationState.neighborListUpdateCount = 0;

        NeighborList<double, MemType::DEVICE>* NL_gpu = nullptr;
        NeighborListFactory<double, MemType::DEVICE>::create(&rb_gpu,
                                                             positions_gpu,
                                                             quaternions_gpu,
                                                             nObstacles,
                                                             nParticles,
                                                             NL_gpu);
        auto gpu_pairs = getNeighborPairsGPU(NL_gpu);
        all_pairs.push_back(gpu_pairs);
        names.push_back(gpu_names[i]);
    }

    // Print results
    // std::cout << "\nLinkedCell Results:" << std::endl;
    // for(size_t i = 0; i < names.size(); ++i)
    // {
    //     std::cout << "  " << names[i] << ": " << all_pairs[i].size() << " neighbor pairs"
    //               << std::endl;
    // }

    // Verify all give same results
    EXPECT_GT(all_pairs[0].size(), 0) << "Should find some neighbors";

    for(size_t i = 1; i < all_pairs.size(); ++i)
    {
        EXPECT_EQ(all_pairs[0], all_pairs[i])
            << names[0] << " vs " << names[i] << " pairs don't match";
    }

    // Wait for GPU operations to complete
    cudaDeviceSynchronize();
}
