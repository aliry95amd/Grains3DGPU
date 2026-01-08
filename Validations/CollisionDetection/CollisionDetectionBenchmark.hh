#ifndef _COLLISIONDETECTIONBENCHMARK_HH_
#define _COLLISIONDETECTIONBENCHMARK_HH_

#include <algorithm>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "BenchmarkConfig.hh"
#include "Box.hh"
#include "CSVWriter.hh"
#include "ComponentManager.hh"
#include "ComponentManagerCPU.hh"
#include "ComponentManagerGPU.hh"
#include "GJK.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "Insertion.hh"
#include "InsertionWindow.hh"
#include "Kinematics.hh"
#include "LinkedCell.hh"
#include "LinkedCellFactory.hh"
#include "NeighborList.hh"
#include "NeighborListFactory.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "RigidBodyFactory.hh"
#include "Sphere.hh"
#include "StepTimer.hh"
#include "Superquadric.hh"
#include "Transform3.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief Comprehensive Collision Detection Benchmark

    This class provides full pipeline performance analysis including particle insertion,
    broad-phase (NeighborList/LinkedCell), and narrow-phase (GJK) collision detection.
    Results are exported to CSV for post-processing.

    @author A.Yazdani - 2026 - Collision Detection Performance Validation */
// =================================================================================================
template <typename T>
class CollisionDetectionBenchmark
{
private:
    BenchmarkConfig<T>         m_config;
    std::unique_ptr<CSVWriter> m_csvWriter;

    // Storage for particles
    std::unique_ptr<Convex<T>>                    m_shapeTemplate;
    GrainsMemBuffer<RigidBody<T>*, MemType::HOST> m_rigidBodies;
    GrainsMemBuffer<Vector3<T>, MemType::HOST>    m_positions;
    GrainsMemBuffer<Quaternion<T>, MemType::HOST> m_quaternions;
    GrainsMemBuffer<Kinematics<T>, MemType::HOST> m_kinematics;

    // GPU-side storage
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE> m_d_rigidBodies;
    GrainsMemBuffer<Vector3<T>, MemType::DEVICE>    m_d_positions;
    GrainsMemBuffer<Quaternion<T>, MemType::DEVICE> m_d_quaternions;

public:
    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor with configuration */
    CollisionDetectionBenchmark(const BenchmarkConfig<T>& config, const std::string& csvFilename)
        : m_config(config)
    {
        m_csvWriter = std::make_unique<CSVWriter>(csvFilename);
        initializeCSVHeader();
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor */
    ~CollisionDetectionBenchmark()
    {
        cleanup();
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Run benchmark with current configuration */
    void runBenchmark()
    {
        // Create CollisionDetectionParameters from config
        CollisionDetectionParameters<T> cdParams;
        cdParams.neighborListType                     = m_config.neighborListType;
        cdParams.linkedCellParameters.type            = m_config.linkedCellType;
        cdParams.linkedCellParameters.minCorner       = m_config.domainMin;
        cdParams.linkedCellParameters.maxCorner       = m_config.domainMax;
        cdParams.linkedCellParameters.minCellSize     = m_config.particleSize * 2;
        cdParams.linkedCellParameters.cellSizeFactor  = 1.0;
        cdParams.linkedCellParameters.updateFrequency = m_config.adaptiveSkin ? 10 : 0;
        cdParams.linkedCellParameters.sortFrequency   = m_config.sort ? 10 : 0;

        // Insert particles
        insertParticles(cdParams.linkedCellParameters, m_config.randomSeed);

        auto compMgrCPU = std::make_unique<ComponentManagerCPU<T>>(&m_rigidBodies,
                                                                   0,  // numObstacles
                                                                   m_config.numParticles);
        compMgrCPU->setPosition(m_positions);
        compMgrCPU->setQuaternion(m_quaternions);

        // GPU - copy all data to device
        RigidBodyFactory<T>::copyHostToDevice(m_rigidBodies, m_d_rigidBodies);
        m_d_positions.copyFrom(m_positions);
        m_d_quaternions.copyFrom(m_quaternions);

        auto compMgrGPU = std::make_unique<ComponentManagerGPU<T>>(&m_d_rigidBodies,
                                                                   0,  // numObstacles
                                                                   m_config.numParticles);
        compMgrCPU->template copyTo<MemType::DEVICE>(
            reinterpret_cast<std::unique_ptr<ComponentManager<T, MemType::DEVICE>>&>(compMgrGPU));
        // Create ComponentManager based on platform
        for(uint trial = 0; trial < m_config.numTrials; ++trial)
        {
            if(m_config.platform == PLATFORM::CPU || m_config.platform == PLATFORM::BOTH)
            {
                runBenchmark(cdParams, compMgrCPU.get(), trial);
            }
            if(m_config.platform == PLATFORM::GPU || m_config.platform == PLATFORM::BOTH)
            {
                // GPU
                runBenchmark(cdParams, compMgrGPU.get(), trial);
            }
        }

        Gout("\nAll " + std::to_string(m_config.numTrials) + " trials complete!");
    }

private:
    // ---------------------------------------------------------------------------------------------
    /** @brief Cleanup allocated memory */
    void cleanup()
    {
        // Delete all rigid bodies
        for(uint i = 0; i < m_config.numParticles; ++i)
        {
            if(m_rigidBodies[i] != nullptr)
            {
                delete m_rigidBodies[i];
                m_rigidBodies[i] = nullptr;
            }
        }

        // Clear all containers
        m_rigidBodies.clear();
        m_positions.clear();
        m_quaternions.clear();
        m_kinematics.clear();
    }

private:
    // ---------------------------------------------------------------------------------------------
    /** @brief Initialize CSV header */
    void initializeCSVHeader()
    {
        std::vector<std::string> columns = {"TrialID",
                                            "Platform",
                                            "ParticleCount",
                                            "ShapeType",
                                            "ParticleSize",
                                            "AspectRatio",
                                            "NeighborListType",
                                            "LCVariant",
                                            "Sort",
                                            "AdaptiveSkin",
                                            "GJKAlgo",
                                            "GJKRepresentation",
                                            "TestRelative",
                                            "SortingTime_ms",
                                            "NeighborUpdateTime_ms",
                                            "RelativeTransformTime_ms",
                                            "GJKTime_ms",
                                            "TotalTime_ms",
                                            "PairCount"};
        m_csvWriter->writeHeader(columns);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Insert particles using Insertion class */
    void insertParticles(const LinkedCellParameters<T>& linkedCellParams, uint seed)
    {
        // Create shape template
        T r = m_config.particleSize;
        switch(m_config.shapeType)
        {
        case ParticleShapeType::BOX:
            m_shapeTemplate = std::make_unique<Box<T>>(2 * r, 2 * r, 2 * r);
            break;
        case ParticleShapeType::SPHERE:
            m_shapeTemplate = std::make_unique<Sphere<T>>(r);
            break;
        case ParticleShapeType::SUPERQUADRIC:
            m_shapeTemplate = std::make_unique<Superquadric<T>>(r, r, r, T(3.0), T(3.0));
            break;
        default:
            GAbort("Invalid shape type!");
        }

        // Allocate memory
        uint nTotal = m_config.numParticles;
        m_rigidBodies.initialize(nTotal);
        m_positions.initialize(nTotal);
        m_quaternions.initialize(nTotal);
        m_kinematics.initialize(nTotal);
        if(m_config.platform == PLATFORM::GPU || m_config.platform == PLATFORM::BOTH)
        {
            m_d_rigidBodies.initialize(nTotal);
            m_d_positions.initialize(nTotal);
            m_d_quaternions.initialize(nTotal);
        }

        // Create rigid bodies (all particles, no obstacles)
        for(uint i = 0; i < nTotal; ++i)
        {
            Convex<T>* cvx   = m_shapeTemplate->clone();
            m_rigidBodies[i] = new RigidBody<T>(cvx, T(0), 0, 1);
            m_positions[i]   = Vector3<T>(T(0), T(0), T(0));
            m_quaternions[i] = Quaternion<T>(T(1), T(0), T(0), T(0));
            m_kinematics[i]  = Kinematics<T>();
        }

        // Use Insertion class for particle placement
        // Create insertion window covering the domain for positions
        std::vector<InsertionWindow<T>> positionWindows;
        positionWindows.emplace_back(m_config.domainMin, m_config.domainMax, seed);

        // Create insertion windows for random orientations (Euler angles: 0 to 2π)
        std::vector<InsertionWindow<T>> orientationWindows;
        Vector3<T>                      angleMin(T(0), T(0), T(0));
        Vector3<T>                      angleMax(T(2.0 * M_PI), T(2.0 * M_PI), T(2.0 * M_PI));
        orientationWindows.emplace_back(angleMin, angleMax, seed + 1);

        // Create Insertion object
        auto insertion = std::make_unique<Insertion<T>>();
        insertion->setPositionInsertionInfo(positionWindows);
        insertion->setOrientationInsertionInfo(orientationWindows);

        // Perform insertion
        insertion->insert(&m_rigidBodies,
                          m_positions,
                          m_quaternions,
                          m_kinematics,
                          linkedCellParams,
                          0,
                          nTotal);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Run ComponentManager pipeline with timing (templated for CPU/GPU) */
    template <MemType M>
    void runBenchmark(const CollisionDetectionParameters<T>& cdParams,
                      ComponentManager<T, M>*                compMgr,
                      uint                                   trialID)
    {
        GrainsParameters<T>::m_simulationState.neighborListUpdateCount = 0;

        // Select appropriate buffers based on memory type
        const GrainsMemBuffer<RigidBody<T>*, M>* rbBuffer;
        const GrainsMemBuffer<Vector3<T>, M>*    posBuffer;
        const GrainsMemBuffer<Quaternion<T>, M>* quatBuffer;

        if constexpr(M == MemType::HOST)
        {
            rbBuffer   = &m_rigidBodies;
            posBuffer  = &m_positions;
            quatBuffer = &m_quaternions;
        }
        else
        {
            rbBuffer   = &m_d_rigidBodies;
            posBuffer  = &m_d_positions;
            quatBuffer = &m_d_quaternions;
        }

        auto neighborList = NeighborListFactory<T, M>::create(rbBuffer,
                                                              *posBuffer,
                                                              *quatBuffer,
                                                              cdParams,
                                                              0,
                                                              m_config.numParticles);
        compMgr->setNeighborList(std::move(neighborList));

        // Step 4: Time sorting (if enabled)
        StepTimer sortTimer;
        double    sortingTime = 0.0;

        if(m_config.sort)
        {
            sortTimer.start();
            compMgr->sortParticles();
            sortTimer.stop();
            sortingTime = sortTimer.getElapsedMilliseconds();
        }

        // Step 5: Time neighbor list update (includes LC construction)
        StepTimer nlTimer;
        nlTimer.start();
        compMgr->updateNeighborList();
        nlTimer.stop();
        double nlUpdateTime = nlTimer.getElapsedMilliseconds();

        // Get pair count for reporting
        const NeighborList<T, M>* NL        = compMgr->getNeighborList();
        uint                      pairCount = NL->getSize();

        // Step 6: Time relative transformation computation (if enabled)
        StepTimer relTransformTimer;
        double    relTransformTime = 0.0;
        if(m_config.testRelative && pairCount > 0)
        {
            relTransformTimer.start();
            compMgr->computeRelativeTransformations();
            relTransformTimer.stop();
            relTransformTime = relTransformTimer.getElapsedMilliseconds();
        }

        // Step 7: Time GJK collision detection
        StepTimer gjkTimer;
        double    gjkTime = 0.0;
        if(pairCount > 0)
        {
            gjkTimer.start();
            // Call detectCollisionsComponents - currently only JOHNSON variant supported
            // TODO: Add runtime dispatch for different GJK variants
            compMgr->detectCollisionsComponents();
            gjkTimer.stop();
            gjkTime = gjkTimer.getElapsedMilliseconds();
        }

        if(m_config.validateContacts && trialID == 0)
        {
            writeContactsToFile(compMgr, trialID);
        }

        // Step 8: Calculate total time
        double totalTime = sortingTime + nlUpdateTime + relTransformTime + gjkTime;
        Gout("Trial " + std::to_string(trialID)
             + ": Total pipeline time: " + std::to_string(totalTime) + " ms");

        // Step 9: Write results to CSV
        writeResultRow(trialID,
                       m_config.platform == PLATFORM::CPU   ? "CPU"
                       : m_config.platform == PLATFORM::GPU ? "GPU"
                                                            : "BOTH",
                       m_config.numParticles,
                       m_config.shapeType == ParticleShapeType::BOX            ? "Box"
                       : m_config.shapeType == ParticleShapeType::SPHERE       ? "Sphere"
                       : m_config.shapeType == ParticleShapeType::SUPERQUADRIC ? "Superquadric"
                                                                               : "Unknown",
                       m_config.particleSize,
                       m_config.aspectRatio,
                       m_config.neighborListType == NeighborListType::NSQ ? "NSQ" : "LINKEDCELL",
                       m_config.linkedCellType == LinkedCellType::HOST        ? "Host"
                       : m_config.linkedCellType == LinkedCellType::SORTBASED ? "SortBased"
                       : m_config.linkedCellType == LinkedCellType::ATOMIC    ? "Atomic"
                                                                              : "AtomicFixed",
                       m_config.sort,
                       m_config.adaptiveSkin,
                       m_config.gjkVariant == GJKVariantType::JOHNSON ? "Johnson" : "SignedVolume",
                       m_config.gjkRepresentation == GJKRepresentationType::TRANSFORM
                           ? "Transform"
                           : "Quaternion",
                       m_config.testRelative,
                       sortingTime,
                       nlUpdateTime,
                       relTransformTime,
                       gjkTime,
                       totalTime,
                       pairCount);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Write contact information to file
        @param compMgr Component manager containing contact info
        @param trialID Trial identifier */
    template <MemType M>
    void writeContactsToFile(ComponentManager<T, M>* compMgr, uint trialID)
    {
        auto platform = (M == MemType::HOST) ? "CPU" : "GPU";

        // Get pair list
        const NeighborList<T, M>*             NL = compMgr->getNeighborList();
        GrainsMemBuffer<uint2, MemType::HOST> pairList(NL->getSize());
        NL->getBuffer().copyTo(pairList);

        // Get contact info
        GrainsMemBuffer<ContactInfo<T>, MemType::HOST> contactInfo(NL->getSize());
        compMgr->getContactInfo(contactInfo);

        uint numContacts = NL->getSize();

        // Create filename
        std::string filename = "contacts_trial" + std::to_string(trialID) + "_" + platform + ".txt";
        std::ofstream outFile(filename);

        if(!outFile.is_open())
        {
            GAbort(("Failed to open file: " + filename).c_str());
        }

        // Write header
        outFile << "# Contact Information for Trial " << trialID << " (" << platform << ")\n";
        outFile << "# PairID Body1 Body2 ContactPoint_X ContactPoint_Y ContactPoint_Z "
                << "ContactVector_X ContactVector_Y ContactVector_Z OverlapDistance\n";

        // Create indices for sorting
        std::vector<uint> sortedIndices(numContacts);
        for(uint i = 0; i < numContacts; ++i)
        {
            sortedIndices[i] = i;
        }

        // Sort indices by Body1 (pair.x) ascending, then Body2 (pair.y) ascending
        std::sort(sortedIndices.begin(), sortedIndices.end(), [&pairList](uint a, uint b) {
            if(pairList[a].x != pairList[b].x)
                return pairList[a].x < pairList[b].x;
            return pairList[a].y < pairList[b].y;
        });

        // Write contact data in sorted order
        for(uint idx = 0; idx < numContacts; ++idx)
        {
            uint        i       = sortedIndices[idx];
            const auto& contact = contactInfo[i];
            const auto& pair    = pairList[i];

            Vector3<T> contactPt  = contact.getContactPoint();
            Vector3<T> contactVec = contact.getContactVector();
            T          overlap    = contact.getOverlapDistance();

            outFile << i << " " << pair.x << " " << pair.y << " " << contactPt[X] << " "
                    << contactPt[Y] << " " << contactPt[Z] << " " << contactVec[X] << " "
                    << contactVec[Y] << " " << contactVec[Z] << " " << overlap << "\n";
        }

        outFile.close();
        Gout("Wrote " + std::to_string(numContacts) + " contacts to " + filename);
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Write result row to CSV */
    void writeResultRow(uint               trialID,
                        const std::string& platform,
                        uint               particleCount,
                        const std::string& shapeType,
                        double             particleSize,
                        double             aspectRatio,
                        const std::string& nlType,
                        const std::string& lcVariant,
                        bool               sort,
                        bool               adaptiveSkin,
                        const std::string& gjkAlgo,
                        const std::string& gjkRep,
                        bool               testRelative,
                        double             sortingTime,
                        double             updateTime,
                        double             relTransformTime,
                        double             gjkTime,
                        double             totalTime,
                        uint               pairCount)
    {
        // Write row data
        m_csvWriter->writeRow(trialID,
                              platform,
                              particleCount,
                              shapeType,
                              particleSize,
                              aspectRatio,
                              nlType,
                              lcVariant,
                              sort ? 1 : 0,
                              adaptiveSkin ? 1 : 0,
                              gjkAlgo,
                              gjkRep,
                              testRelative,
                              sortingTime,
                              updateTime,
                              relTransformTime,
                              gjkTime,
                              totalTime,
                              pairCount);
    }
};

#endif
