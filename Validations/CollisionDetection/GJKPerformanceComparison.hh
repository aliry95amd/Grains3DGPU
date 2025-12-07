#ifndef _GJK_PERFORMANCE_COMPARISON_HH_
#define _GJK_PERFORMANCE_COMPARISON_HH_

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include "Box.hh"
#include "ConvexFactory.hh"
#include "Cylinder.hh"
#include "GJK.hh"
#include "Grains.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "RigidBodyFactory.hh"
#include "Sphere.hh"
#include "Superquadric.hh"
#include "Transform3.hh"
#include "Vector3.hh"
#include "VectorMath.hh"

/* TODO:
    Transform3 Generation random distribution
    Sort pairList and Compare
    Make sure GPU is working
*/

// =================================================================================================
/** @brief Performance Comparison Tool for GJK Algorithms

    This class provides comprehensive performance analysis comparing various GJK
    algorithms across different shape combinations, both on CPU and GPU
    platforms.

    @author A.Yazdani - 2025 - GJK Performance Validation */
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// CPU Performance Functions
namespace GJKPerformanceCPU
{
    // ---------------------------------------------------------------------------------------------
    // Run distance algorithm on Transform3 arrays
    template <typename T, GJKType type>
    void collisionDetection(
        RigidBody<T>** rb, Transform3<T>* tr, uint2* pairList, T* dist, uint* iter, int const N)
    {
        for(int p = 0; p < N; p++)
        {
            uint       i = pairList[p].x;
            uint       j = pairList[p].y;
            Vector3<T> pa, pb;
            dist[p] = computeClosestPoints_GJK<T, type, false, EPS<T>>(*(rb[i]->getConvex()),
                                                                       *(rb[j]->getConvex()),
                                                                       tr[i],
                                                                       tr[j],
                                                                       rb[i]->getCrustThickness(),
                                                                       rb[j]->getCrustThickness(),
                                                                       pa,
                                                                       pb,
                                                                       iter[p]);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Run distance algorithm on Quaternion+Position arrays
    template <typename T, GJKType type>
    void collisionDetection(RigidBody<T>** rb,
                            Vector3<T>*    pos,
                            Quaternion<T>* quat,
                            uint2*         pairList,
                            T*             dist,
                            uint*          iter,
                            int const      N)
    {
        for(int p = 0; p < N; p++)
        {
            uint       i = pairList[p].x;
            uint       j = pairList[p].y;
            Vector3<T> pa, pb;
            dist[p] = computeClosestPoints_GJK<T, type, false, EPS<T>>(*(rb[i]->getConvex()),
                                                                       *(rb[j]->getConvex()),
                                                                       pos[i],
                                                                       pos[j],
                                                                       quat[i],
                                                                       quat[j],
                                                                       rb[i]->getCrustThickness(),
                                                                       rb[j]->getCrustThickness(),
                                                                       pa,
                                                                       pb,
                                                                       iter[p]);
        }
    }
}  // namespace GJKPerformanceCPU

// -------------------------------------------------------------------------------------------------
// GPU Performance Functions
namespace GJKPerformanceGPU
{
    // ---------------------------------------------------------------------------------------------
    // Johnson algorithm GPU kernel (Transform3)
    template <typename T, GJKType type>
    __global__ void collisionDetection(
        RigidBody<T>** rb, Transform3<T>* tr, uint2* pairList, T* dist, uint* iter, int const N)
    {
        int bID = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
        int tID = bID * blockDim.x + threadIdx.x;

        if(tID < N)
        {
            uint       i = pairList[tID].x;
            uint       j = pairList[tID].y;
            Vector3<T> pa, pb;
            dist[tID] = computeClosestPoints_GJK<T, type, false, EPS<T>>(*(rb[i]->getConvex()),
                                                                         *(rb[j]->getConvex()),
                                                                         tr[i],
                                                                         tr[j],
                                                                         rb[i]->getCrustThickness(),
                                                                         rb[j]->getCrustThickness(),
                                                                         pa,
                                                                         pb,
                                                                         iter[tID]);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Johnson algorithm GPU kernel (Quaternion)
    template <typename T, GJKType type>
    __global__ void collisionDetection(RigidBody<T>** rb,
                                       Vector3<T>*    pos,
                                       Quaternion<T>* quat,
                                       uint2*         pairList,
                                       T*             dist,
                                       uint*          iter,
                                       int const      N)
    {
        int bID = gridDim.x * gridDim.y * blockIdx.z + blockIdx.y * gridDim.x + blockIdx.x;
        int tID = bID * blockDim.x + threadIdx.x;

        if(tID < N)
        {
            uint       i = pairList[tID].x;
            uint       j = pairList[tID].y;
            Vector3<T> pa, pb;
            dist[tID] = computeClosestPoints_GJK<T, type, false, EPS<T>>(*(rb[i]->getConvex()),
                                                                         *(rb[j]->getConvex()),
                                                                         pos[i],
                                                                         pos[j],
                                                                         quat[i],
                                                                         quat[j],
                                                                         rb[i]->getCrustThickness(),
                                                                         rb[j]->getCrustThickness(),
                                                                         pa,
                                                                         pb,
                                                                         iter[tID]);
        }
    }
}

// -------------------------------------------------------------------------------------------------
/** @brief GJK Performance Comparison Class */
template <typename T>
class GJKPerformanceComparison
{
private:
    uint m_numParticles;
    uint m_numPairs;
    uint m_shapeType;
    uint m_seed;
    uint m_numThreads;
    bool m_runGPUTests;
    struct PerformanceResults
    {
        double time_CJT;
        double time_CJQ;
        double time_CST;
        double time_CSQ;
        double time_GJT;
        double time_GJQ;
        double time_GST;
        double time_GSQ;

        double acc_CJT;
        double acc_CJQ;
        double acc_CST;
        double acc_CSQ;
        double acc_GJT;
        double acc_GJQ;
        double acc_GST;
        double acc_GSQ;

        double iter_CJT;
        double iter_CJQ;
        double iter_CST;
        double iter_CSQ;
        double iter_GJT;
        double iter_GJQ;
        double iter_GST;
        double iter_GSQ;
    } m_results;

public:
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor */
    GJKPerformanceComparison()
        : m_numParticles(1)
        , m_numPairs(1)
        , m_shapeType(0)
        , m_seed(42)
        , m_numThreads(256)
        , m_runGPUTests(true)
    {
        // Initialize results
        m_results.time_CJT = 0.0;
        m_results.time_CJQ = 0.0;
        m_results.time_CST = 0.0;
        m_results.time_CSQ = 0.0;
        m_results.time_GJT = 0.0;
        m_results.time_GJQ = 0.0;
        m_results.time_GST = 0.0;
        m_results.time_GSQ = 0.0;

        m_results.acc_CJT = 0.0;
        m_results.acc_CJQ = 0.0;
        m_results.acc_CST = 0.0;
        m_results.acc_CSQ = 0.0;
        m_results.acc_GJT = 0.0;
        m_results.acc_GJQ = 0.0;
        m_results.acc_GST = 0.0;
        m_results.acc_GSQ = 0.0;

        m_results.iter_CJT = 0.0;
        m_results.iter_CJQ = 0.0;
        m_results.iter_CST = 0.0;
        m_results.iter_CSQ = 0.0;
        m_results.iter_GJT = 0.0;
        m_results.iter_GJQ = 0.0;
        m_results.iter_GST = 0.0;
        m_results.iter_GSQ = 0.0;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor */
    ~GJKPerformanceComparison() {}

    // ---------------------------------------------------------------------------------------------
    /** @brief Runs the test */
    void run()
    {
        Gout("Number of particles: " + std::to_string(m_numParticles));
        Gout("Number of pairs: " + std::to_string(m_numPairs));
        Gout("Shape type: " + std::to_string(m_shapeType));
        Gout("GPU tests: ", (m_runGPUTests ? "Enabled" : "Disabled"));

        // ---------------------------------------------------------------------
        // Creating two random Transform3 arrays for collision pairs
        std::default_random_engine        generator(m_seed);
        std::uniform_real_distribution<T> distribution(-0.5, 0.5);
        std::uniform_real_distribution<T> rotation(0, 2.0 * M_PI);

        // Allocating memory using GrainsMemBuffer
        GrainsMemBuffer<Transform3<T>, MemType::HOST> h_tr(m_numParticles);
        GrainsMemBuffer<Vector3<T>, MemType::HOST>    h_pos(m_numParticles);
        GrainsMemBuffer<Quaternion<T>, MemType::HOST> h_quat(m_numParticles);

        // Randomize arrays on host
        for(uint i = 0; i < m_numParticles; ++i)
        {
            h_tr[i].setBasis(rotation(generator), rotation(generator), rotation(generator));
            h_tr[i].setOrigin(Vector3<T>(distribution(generator),
                                         distribution(generator),
                                         distribution(generator)));
            h_quat[i] = h_tr[i].getRotation();
            h_pos[i]  = h_tr[i].getOrigin();
        }

        // Copy to device if needed
        GrainsMemBuffer<Transform3<T>, MemType::DEVICE> d_tr   = h_tr;
        GrainsMemBuffer<Vector3<T>, MemType::DEVICE>    d_pos  = h_pos;
        GrainsMemBuffer<Quaternion<T>, MemType::DEVICE> d_quat = h_quat;

        // ---------------------------------------------------------------------
        // Creating Particles based on shape type
        T          r1       = T(0.05);
        T          r2       = T(0.05);
        T          r3       = T(0.05);
        Convex<T>* h_convex = nullptr;

        switch(m_shapeType)
        {
        case 0:  // Box
            h_convex = new Box<T>(2 * r1, 2 * r2, 2 * r3);
            break;
        case 1:  // Sphere
            h_convex = new Sphere<T>(r1);
            break;
        case 2:  // Superquadric
            h_convex = new Superquadric<T>(r1, r2, r3, T(3.0), T(3.0));
            break;
        default:
            std::cerr << "Invalid shape type!" << std::endl;
            return;
        }

        // Create rigid bodies using GrainsMemBuffer
        GrainsMemBuffer<RigidBody<T>*> h_rb(m_numParticles);
        for(uint i = 0; i < m_numParticles; ++i)
        {
            Convex<T>* cvx = h_convex->clone();
            h_rb[i]        = new RigidBody<T>(cvx, T(0), 0, 1);
        }

        // Copy to device
        GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE> d_rb(m_numParticles);
        RigidBodyFactory<T>::copyHostToDevice(h_rb, d_rb);

        // ---------------------------------------------------------------------
        // Generate random collision pairs
        std::uniform_int_distribution<int>    particleDist(0, m_numParticles - 1);
        GrainsMemBuffer<uint2, MemType::HOST> h_pairList(m_numPairs);
        for(uint p = 0; p < m_numPairs; ++p)
        {
            int i = particleDist(generator);
            int j = particleDist(generator);
            while(i == j)
                j = particleDist(generator);
            h_pairList[p] = make_uint2(i, j);
        }

        // Copy pairlist to device
        GrainsMemBuffer<uint2, MemType::DEVICE> d_pairList = h_pairList;

        // ---------------------------------------------------------------------
        // Host memory for results
        auto                              start_timer = std::chrono::high_resolution_clock::now();
        auto                              end_timer   = std::chrono::high_resolution_clock::now();
        GrainsMemBuffer<T, MemType::HOST> h_distance_base(m_numPairs, 0);
        GrainsMemBuffer<uint, MemType::HOST>   h_iterations_base(m_numPairs, 0);
        GrainsMemBuffer<T, MemType::HOST>      h_distance(m_numPairs, 0);
        GrainsMemBuffer<uint, MemType::HOST>   h_iterations(m_numPairs, 0);
        GrainsMemBuffer<T, MemType::DEVICE>    d_distance_base(m_numPairs, 0);
        GrainsMemBuffer<uint, MemType::DEVICE> d_iterations_base(m_numPairs, 0);
        GrainsMemBuffer<T, MemType::DEVICE>    d_distance(m_numPairs, 0);
        GrainsMemBuffer<uint, MemType::DEVICE> d_iterations(m_numPairs, 0);

        // ---------------------------------------------------------------------
        // Helper functions for result comparison and iteration calculation
        auto compareResultsCJT = [&](const auto& data, const auto& base) -> double {
            constexpr double tolerance     = EPS<T>;
            uint             numMismatches = 0;

            // Get pointers to data for comparison
            const T* data_ptr;
            const T* base_ptr = base.getData();

            // Handle different memory types - create temporary host copy for
            // device data
            std::unique_ptr<GrainsMemBuffer<T, MemType::HOST>> h_data_copy;
            if constexpr(std::is_same_v<std::decay_t<decltype(data)>,
                                        GrainsMemBuffer<T, MemType::HOST>>)
                data_ptr = data.getData();
            else
            {
                h_data_copy = std::make_unique<GrainsMemBuffer<T, MemType::HOST>>(
                    data);  // Copy device to host
                data_ptr = h_data_copy->getData();
            }

            for(uint p = 0; p < m_numPairs; ++p)
            {
                if(fabs(data_ptr[p] - base_ptr[p]) > tolerance)
                    numMismatches++;
            }
            return (1 - static_cast<double>(numMismatches) / m_numPairs) * 100.0;
        };

        // Helper function to compute average iterations
        auto computeAverageIterations = [&](const auto& iter) -> double {
            double      totalIterations = 0.0;
            const uint* iter_ptr;
            // Handle different memory types - create temporary host copy for
            // device data
            std::unique_ptr<GrainsMemBuffer<uint, MemType::HOST>> h_iter_copy;
            if constexpr(std::is_same_v<std::decay_t<decltype(iter)>,
                                        GrainsMemBuffer<uint, MemType::HOST>>)
                iter_ptr = iter.getData();
            else
            {
                h_iter_copy = std::make_unique<GrainsMemBuffer<uint, MemType::HOST>>(
                    iter);  // Copy device to host
                iter_ptr = h_iter_copy->getData();
            }

            for(uint i = 0; i < m_numPairs; ++i)
                totalIterations += iter_ptr[i];
            return static_cast<double>(totalIterations) / m_numPairs;
        };

        // ---------------------------------------------------------------------
        // CPU tests
        start_timer = std::chrono::high_resolution_clock::now();
        GJKPerformanceCPU::collisionDetection<T, GJKType::JOHNSON>(h_rb.getData(),
                                                                   h_tr.getData(),
                                                                   h_pairList.getData(),
                                                                   h_distance_base.getData(),
                                                                   h_iterations_base.getData(),
                                                                   m_numPairs);
        end_timer          = std::chrono::high_resolution_clock::now();
        m_results.time_CJT = std::chrono::duration<double>(end_timer - start_timer).count();
        m_results.acc_CJT  = compareResultsCJT(h_distance_base, h_distance_base);
        m_results.iter_CJT = computeAverageIterations(h_iterations_base);

        start_timer = std::chrono::high_resolution_clock::now();
        GJKPerformanceCPU::collisionDetection<T, GJKType::SIGNEDVOLUME>(h_rb.getData(),
                                                                        h_tr.getData(),
                                                                        h_pairList.getData(),
                                                                        h_distance.getData(),
                                                                        h_iterations.getData(),
                                                                        m_numPairs);
        end_timer          = std::chrono::high_resolution_clock::now();
        m_results.time_CST = std::chrono::duration<double>(end_timer - start_timer).count();
        m_results.acc_CST  = compareResultsCJT(h_distance, h_distance_base);
        m_results.iter_CST = computeAverageIterations(h_iterations);

        start_timer = std::chrono::high_resolution_clock::now();
        GJKPerformanceCPU::collisionDetection<T, GJKType::JOHNSON>(h_rb.getData(),
                                                                   h_pos.getData(),
                                                                   h_quat.getData(),
                                                                   h_pairList.getData(),
                                                                   h_distance.getData(),
                                                                   h_iterations.getData(),
                                                                   m_numPairs);
        end_timer          = std::chrono::high_resolution_clock::now();
        m_results.time_CJQ = std::chrono::duration<double>(end_timer - start_timer).count();
        m_results.acc_CJQ  = compareResultsCJT(h_distance, h_distance_base);
        m_results.iter_CJQ = computeAverageIterations(h_iterations);

        start_timer = std::chrono::high_resolution_clock::now();
        GJKPerformanceCPU::collisionDetection<T, GJKType::SIGNEDVOLUME>(h_rb.getData(),
                                                                        h_pos.getData(),
                                                                        h_quat.getData(),
                                                                        h_pairList.getData(),
                                                                        h_distance.getData(),
                                                                        h_iterations.getData(),
                                                                        m_numPairs);
        end_timer          = std::chrono::high_resolution_clock::now();
        m_results.time_CSQ = std::chrono::duration<double>(end_timer - start_timer).count();
        m_results.acc_CSQ  = compareResultsCJT(h_distance, h_distance_base);
        m_results.iter_CSQ = computeAverageIterations(h_iterations);

        // ---------------------------------------------------------------------
        // GPU tests
        if(m_runGPUTests)
        {
            uint numBlocks = (m_numPairs + m_numThreads - 1) / m_numThreads;
            start_timer    = std::chrono::high_resolution_clock::now();
            GJKPerformanceGPU::collisionDetection<T, GJKType::JOHNSON>
                <<<numBlocks, m_numThreads>>>(d_rb.getData(),
                                              d_tr.getData(),
                                              d_pairList.getData(),
                                              d_distance.getData(),
                                              d_iterations.getData(),
                                              m_numPairs);
            cudaDeviceSynchronize();
            end_timer          = std::chrono::high_resolution_clock::now();
            m_results.time_GJT = std::chrono::duration<double>(end_timer - start_timer).count();
            m_results.acc_GJT  = compareResultsCJT(d_distance, h_distance_base);
            m_results.iter_GJT = computeAverageIterations(d_iterations);

            start_timer = std::chrono::high_resolution_clock::now();
            GJKPerformanceGPU::collisionDetection<T, GJKType::SIGNEDVOLUME>
                <<<numBlocks, m_numThreads>>>(d_rb.getData(),
                                              d_tr.getData(),
                                              d_pairList.getData(),
                                              d_distance.getData(),
                                              d_iterations.getData(),
                                              m_numPairs);
            cudaDeviceSynchronize();
            end_timer          = std::chrono::high_resolution_clock::now();
            m_results.time_GST = std::chrono::duration<double>(end_timer - start_timer).count();
            m_results.acc_GST  = compareResultsCJT(d_distance, h_distance_base);
            m_results.iter_GST = computeAverageIterations(d_iterations);

            start_timer = std::chrono::high_resolution_clock::now();
            GJKPerformanceGPU::collisionDetection<T, GJKType::JOHNSON>
                <<<numBlocks, m_numThreads>>>(d_rb.getData(),
                                              d_pos.getData(),
                                              d_quat.getData(),
                                              d_pairList.getData(),
                                              d_distance.getData(),
                                              d_iterations.getData(),
                                              m_numPairs);
            cudaDeviceSynchronize();
            end_timer          = std::chrono::high_resolution_clock::now();
            m_results.time_GJQ = std::chrono::duration<double>(end_timer - start_timer).count();
            m_results.acc_GJQ  = compareResultsCJT(d_distance, h_distance_base);
            m_results.iter_GJQ = computeAverageIterations(d_iterations);

            start_timer = std::chrono::high_resolution_clock::now();
            GJKPerformanceGPU::collisionDetection<T, GJKType::SIGNEDVOLUME>
                <<<numBlocks, m_numThreads>>>(d_rb.getData(),
                                              d_pos.getData(),
                                              d_quat.getData(),
                                              d_pairList.getData(),
                                              d_distance.getData(),
                                              d_iterations.getData(),
                                              m_numPairs);
            cudaDeviceSynchronize();
            end_timer          = std::chrono::high_resolution_clock::now();
            m_results.time_GSQ = std::chrono::duration<double>(end_timer - start_timer).count();
            m_results.acc_GSQ  = compareResultsCJT(d_distance, h_distance_base);
            m_results.iter_GSQ = computeAverageIterations(d_iterations);
        }

        // Print Results
        Gout("\nPERFORMANCE RESULTS");
        Gout(std::string(80, '-'));
        std::cout << std::fixed << std::setprecision(6);
        Gout("CJT time wrt CJT [-]: ", m_results.time_CJT / m_results.time_CJT);
        Gout("CST time wrt CJT [-]: ", m_results.time_CST / m_results.time_CJT);
        Gout("CJQ time wrt CJT [-]: ", m_results.time_CJQ / m_results.time_CJT);
        Gout("CSQ time wrt CJT [-]: ", m_results.time_CSQ / m_results.time_CJT);
        Gout("GJT time wrt CJT [-]: ", m_results.time_GJT / m_results.time_CJT);
        Gout("GST time wrt CJT [-]: ", m_results.time_GST / m_results.time_CJT);
        Gout("GJQ time wrt CJT [-]: ", m_results.time_GJQ / m_results.time_CJT);
        Gout("GSQ time wrt CJT [-]: ", m_results.time_GSQ / m_results.time_CJT);

        Gout("\nACCURACY ANALYSIS");
        Gout(std::string(80, '-'));
        std::cout << std::fixed << std::setprecision(2);
        Gout("CJT accuracy wrt CJT [%]: ", m_results.acc_CJT);
        Gout("CST accuracy wrt CJT [%]: ", m_results.acc_CST);
        Gout("CJQ accuracy wrt CJT [%]: ", m_results.acc_CJQ);
        Gout("CSQ accuracy wrt CJT [%]: ", m_results.acc_CSQ);
        Gout("GJT accuracy wrt CJT [%]: ", m_results.acc_GJT);
        Gout("GST accuracy wrt CJT [%]: ", m_results.acc_GST);
        Gout("GJQ accuracy wrt CJT [%]: ", m_results.acc_GJQ);
        Gout("GSQ accuracy wrt CJT [%]: ", m_results.acc_GSQ);

        Gout("\nCONVERGENCE ANALYSIS");
        Gout(std::string(80, '-'));
        Gout("CJT average iterations [-]: ", m_results.iter_CJT);
        Gout("CST average iterations [-]: ", m_results.iter_CST);
        Gout("CJQ average iterations [-]: ", m_results.iter_CJQ);
        Gout("CSQ average iterations [-]: ", m_results.iter_CSQ);
        Gout("GJT average iterations [-]: ", m_results.iter_GJT);
        Gout("GST average iterations [-]: ", m_results.iter_GST);
        Gout("GJQ average iterations [-]: ", m_results.iter_GJQ);
        Gout("GSQ average iterations [-]: ", m_results.iter_GSQ);

        // Cleanup
        delete h_convex;
        for(uint i = 0; i < m_numParticles; ++i)
        {
            delete h_rb[i];
        }
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Set test parameters */
    void setTestParameters(uint numParticles = 100000,
                           uint numPairs     = 100000,
                           uint shapeType    = 0,
                           uint numThreads   = 256,
                           bool runGPUTests  = true)
    {
        m_numParticles = numParticles;
        m_numPairs     = numPairs;
        m_shapeType    = shapeType;
        m_numThreads   = numThreads;
        m_runGPUTests  = runGPUTests;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Set random seed for reproducible tests */
    void setRandomSeed(unsigned int seed)
    {
        m_seed = seed;
    }
};

#endif
