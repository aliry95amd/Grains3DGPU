#include "Grains.hh"
#include "ComponentManagerCPU.hh"
#include "ContactForceModelFactory.hh"
#include "PostProcessingWriterFactory.hh"
#include "RigidBodyFactory.hh"
#include "TimeIntegratorFactory.hh"
#include "VectorMath.hh"

/* ============================================================================================== */
/* High-Level Methods                                                                             */
/* ============================================================================================== */
// Default constructor
template <typename T>
Grains<T>::Grains()
{
    Gout(std::string(80, '='));
    Gout("Starting Grains3D ...");
    Gout(std::string(80, '='));
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
Grains<T>::~Grains()
{
}

// -------------------------------------------------------------------------------------------------
// Initializes the simulation using the XML input
template <typename T>
void Grains<T>::initialize(DOMElement* rootElement)
{
    // Reading different blocks of the input XML
    Gout(std::string(80, '='));
    Gout("Reading the input file ...");
    Gout(std::string(80, '='));
    Construction(rootElement);
    Forces(rootElement);
    AdditionalFeatures(rootElement);
    Gout(std::string(80, '='));
    Gout("Reading the input file completed!");
    Gout(std::string(80, '='));

    // Post-processing start
    for(auto& pp : m_postProcessor)
        pp->PostProcessing_start();
}

// -------------------------------------------------------------------------------------------------
// Performs post-processing
template <typename T>
template <MemType MT>
void Grains<T>::postProcess(const std::unique_ptr<ComponentManager<T, MT>>& cm)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    if(GP::m_tSave.empty())
        return;

    if(GP::m_tSave.front() - SS.time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();
        if constexpr(MT == MemType::DEVICE)
        {
            cm->copyTo_PostProcessing(m_components);
            for(auto& pp : m_postProcessor)
                pp->PostProcessing(m_rigidBodyList, m_components, SS.time);
        }
        else
        {
            for(auto& pp : m_postProcessor)
                pp->PostProcessing(m_rigidBodyList, cm, SS.time);
        }
    }
    // In case we get past the saveTime, we need to remove it from the queue
    if(!GP::m_tSave.empty() && SS.time > GP::m_tSave.front())
        GP::m_tSave.pop();
}

// -------------------------------------------------------------------------------------------------
// Performs tasks after time-stepping
template <typename T>
void Grains<T>::finalize()
{
    for(auto& pp : m_postProcessor)
        pp->PostProcessing_end();
}

/* ============================================================================================== */
/* Low-Level Methods                                                                              */
/* ============================================================================================== */
// Constructs the simulation -- Reads the Construction part of the XML input to
// set the parameters
template <typename T>
void Grains<T>::Construction(DOMElement* rootElement)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& CD = GP::m_collisionDetection;
    auto& LC = CD.linkedCellParameters;

    // Output message
    GoutWI(3, "Construction");
    // ---------------------------------------------------------------------------------------------
    // Checking if Construction node is available
    DOMNode* root = ReaderXML::getNode(rootElement, "Construction");
    GAssert(root, "Construction node is mandatory!");

    // ---------------------------------------------------------------------------------------------
    // Domain size: origin, max coordinates and periodicity
    DOMNode* nOrigin = ReaderXML::getNode(root, "Origin");
    if(nOrigin)
        GP::m_origin.setValue(T(ReaderXML::getNodeAttr_Double(nOrigin, "X")),
                              T(ReaderXML::getNodeAttr_Double(nOrigin, "Y")),
                              T(ReaderXML::getNodeAttr_Double(nOrigin, "Z")));
    else
        GP::m_origin.setValue(T(0), T(0), T(0));

    DOMNode* nDomain = ReaderXML::getNode(root, "MaxCoordinate");
    GP::m_maxCoordinate.setValue(T(ReaderXML::getNodeAttr_Double(nDomain, "X")),
                                 T(ReaderXML::getNodeAttr_Double(nDomain, "Y")),
                                 T(ReaderXML::getNodeAttr_Double(nDomain, "Z")));

    // if the simulation is periodic
    DOMNode* nPeriodicity = ReaderXML::getNode(root, "Periodicity");
    if(nPeriodicity)
    {
        int PX = ReaderXML::getNodeAttr_Int(nPeriodicity, "PX");
        int PY = ReaderXML::getNodeAttr_Int(nPeriodicity, "PY");
        int PZ = ReaderXML::getNodeAttr_Int(nPeriodicity, "PZ");
        GAssert(PX * PY * PZ == 0, "Periodicity is not implemented!");
        GP::m_isPeriodic = false;
    }

    // ---------------------------------------------------------------------------------------------
    // Components
    // Particle variables
    DOMNode*                       particles = ReaderXML::getNode(root, "Particles");
    GrainsMemBuffer<RigidBody<T>*> refParticleRigidBodyList;
    GrainsMemBuffer<Vector3<T>>    refParticleInitialPosition;
    GrainsMemBuffer<Quaternion<T>> refParticleInitialOrientation;
    GrainsMemBuffer<uint>          numEachRefParticle;
    uint                           numParticles = 0;
    // Obstacle variables
    DOMNode*                       obstacles = ReaderXML::getNode(root, "Obstacles");
    GrainsMemBuffer<RigidBody<T>*> refObstacleRigidBodyList;
    GrainsMemBuffer<Vector3<T>>    refObstacleInitialPosition;
    GrainsMemBuffer<Quaternion<T>> refObstacleInitialOrientation;
    GrainsMemBuffer<uint>          numEachRefObstacle;
    uint                           numObstacles = 0;
    GoutWI(6, "Reading rigid bodies ...");
    RigidBodyFactory<T>::create(obstacles,
                                particles,
                                refObstacleRigidBodyList,
                                refParticleRigidBodyList,
                                refObstacleInitialPosition,
                                refParticleInitialPosition,
                                refObstacleInitialOrientation,
                                refParticleInitialOrientation,
                                numEachRefObstacle,
                                numEachRefParticle,
                                numObstacles,
                                numParticles);
    GoutWI(6, "Reading rigid bodies completed!");

    // Setting up rigid bodies buffer
    const uint totalNumComponents = numObstacles + numParticles;
    GAssert(totalNumComponents > 0, "No components found in the simulation!");
    m_rigidBodyList.initialize(totalNumComponents);
    GrainsMemBuffer<Vector3<T>>    initialPosition(totalNumComponents);
    GrainsMemBuffer<Quaternion<T>> initialOrientation(totalNumComponents);
    {
        uint offset = 0;
        for(uint i = 0; i < refObstacleRigidBodyList.getSize(); ++i)
        {
            for(uint j = 0; j < numEachRefObstacle[i]; j++)
            {
                // Deep copy of the rigid body
                m_rigidBodyList[offset + j] = new RigidBody<T>(*refObstacleRigidBodyList[i]);
                // Initial transformation of the rigid body
                initialPosition[offset + j]    = refObstacleInitialPosition[i];
                initialOrientation[offset + j] = refObstacleInitialOrientation[i];
            }
            // Increment the starting position
            offset += numEachRefObstacle[i];
        }

        for(uint i = 0; i < refParticleRigidBodyList.getSize(); ++i)
        {
            for(uint j = 0; j < numEachRefParticle[i]; j++)
            {
                // Deep copy of the rigid body
                m_rigidBodyList[offset + j] = new RigidBody<T>(*refParticleRigidBodyList[i]);
                // Initial transformation of the rigid body
                initialPosition[offset + j]    = refParticleInitialPosition[i];
                initialOrientation[offset + j] = refParticleInitialOrientation[i];
            }
            // Increment the starting position
            offset += numEachRefParticle[i];
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Setting up some parameters
    SS.numObstacles = numObstacles;
    SS.numParticles = numParticles;

    // Calculate minCellSize based on maximum particle radius (needed for insertion checks)
    T maxParticleRadius = 0;
    for(uint i = 0; i < refParticleRigidBodyList.getSize(); ++i)
    {
        auto refParticle = refParticleRigidBodyList[i];
        T    radius      = refParticle->getCircumscribedRadius();
        if(radius > maxParticleRadius)
            maxParticleRadius = radius;
    }

    // Calculate maximum obstacle radius for cell occupancy
    T maxObstacleRadius = 0;
    for(uint i = 0; i < refObstacleRigidBodyList.getSize(); ++i)
    {
        auto refObstacle = refObstacleRigidBodyList[i];
        T    radius      = refObstacle->getCircumscribedRadius();
        if(radius > maxObstacleRadius)
            maxObstacleRadius = radius;
    }

    // ---------------------------------------------------------------------------------------------
    // Setting up collision detection
    GoutWI(6, "Reading collision detection ...");
    DOMNode* collisionDetection = ReaderXML::getNode(root, "CollisionDetection");
    GAssert(collisionDetection, "CollisionDetection node is mandatory!");

    // Neighbor list
    DOMNode* nNeighborList = ReaderXML::getNode(collisionDetection, "NeighborList");
    GAssert(nNeighborList, "NeighborList node is mandatory!");
    std::string neighborListType = ReaderXML::getNodeAttr_String(nNeighborList, "Type");
    if(neighborListType == "BruteForce")
        CD.neighborListType = NeighborListType::NSQ;
    else if(neighborListType == "LinkedCell")
        CD.neighborListType = NeighborListType::LINKEDCELL;
    else
        GAbort("Unknown NeighborList type! Aborting Grains!");
    GoutWI(9, "NeighborList: " + neighborListType);

    // Linked cell
    if(CD.neighborListType == NeighborListType::LINKEDCELL)
    {
        DOMNode* nLinkedCell = ReaderXML::getNode(collisionDetection, "LinkedCell");
        GAssert(nLinkedCell, "LinkedCell node is mandatory when using LinkedCell neighbor list!");
        std::string linkedCellType = ReaderXML::getNodeAttr_String(nLinkedCell, "Type");
        if(linkedCellType == "Host")
            LC.type = LinkedCellType::HOST;
        else if(linkedCellType == "Device_SortBased")
            LC.type = LinkedCellType::SORTBASED;
        else if(linkedCellType == "Device_Atomic")
            LC.type = LinkedCellType::ATOMIC;
        else
            GAbort("Unknown LinkedCell type! Aborting Grains!");

        // Minimum cell size
        LC.minCellSize = 2 * maxParticleRadius;

        // Cell size factor
        LC.cellSizeFactor = T(ReaderXML::getNodeAttr_Double(nLinkedCell, "CellSizeFactor"));

        // Adjusting maxNumCellsPerObstacle
        uint adjustedMaxNumCellsPerObstaclePerDim = static_cast<uint>(
            std::ceil((2 * maxObstacleRadius) / (LC.cellSizeFactor * LC.minCellSize)));
        LC.maxNumCellsPerObstacle = adjustedMaxNumCellsPerObstaclePerDim
                                    * adjustedMaxNumCellsPerObstaclePerDim
                                    * adjustedMaxNumCellsPerObstaclePerDim;

        // Update and sort frequency
        LC.updateFrequency = ReaderXML::getNodeAttr_Int(nLinkedCell, "UpdatingFrequency");
        LC.sortFrequency   = ReaderXML::getNodeAttr_Int(nLinkedCell, "SortingFrequency");

        // TODO: Take from the input file
        LC.minCorner = GP::m_origin;
        LC.maxCorner = GP::m_maxCoordinate;

        GoutWI(9,
               "LinkedCell: " + linkedCellType + +", cell size factor "
                   + std::to_string(LC.cellSizeFactor) + ", updating frequency "
                   + std::to_string(LC.updateFrequency) + ", sorting frequency "
                   + std::to_string(LC.sortFrequency) + " ...");
    }
    else  // BruteForce - still need basic LinkedCell params for insertion checks
    {
        LC.type                                   = LinkedCellType::HOST;
        LC.cellSizeFactor                         = T(1.0);
        LC.minCorner                              = GP::m_origin;
        LC.maxCorner                              = GP::m_maxCoordinate;
        LC.minCellSize                            = 2 * maxParticleRadius;
        uint adjustedMaxNumCellsPerObstaclePerDim = static_cast<uint>(
            std::ceil((2 * maxObstacleRadius) / (LC.cellSizeFactor * LC.minCellSize)));
        LC.maxNumCellsPerObstacle = adjustedMaxNumCellsPerObstaclePerDim
                                    * adjustedMaxNumCellsPerObstaclePerDim
                                    * adjustedMaxNumCellsPerObstaclePerDim;
    }

    // Bounding volume
    DOMNode* nBoundingVolume = ReaderXML::getNode(collisionDetection, "BoundingVolume");
    if(nBoundingVolume)
    {
        std::string boundingVolumeType = ReaderXML::getNodeAttr_String(nBoundingVolume, "Type");
        if(boundingVolumeType == "OFF")
            CD.boundingVolumeType = BoundingVolumeType::OFF;
        else if(boundingVolumeType == "OBB")
            CD.boundingVolumeType = BoundingVolumeType::OBB;
        else if(boundingVolumeType == "OBC")
            CD.boundingVolumeType = BoundingVolumeType::OBC;
        else
            GAbort("Unknown bounding volume type! Aborting Grains!");
        GoutWI(9, "BoundingVolume: " + boundingVolumeType);
    }

    // Narrow phase detection
    DOMNode* nNarrowPhase = ReaderXML::getNode(collisionDetection, "NarrowPhase");
    if(nNarrowPhase)
    {
        std::string narrowPhaseType = ReaderXML::getNodeAttr_String(nNarrowPhase, "Type");
        if(narrowPhaseType == "GJK")
            CD.narrowPhaseType = NarrowPhaseType::GJK;
        else
            GAbort("Unknown narrow phase type! Aborting Grains!");
        GoutWI(9, "NarrowPhase: " + narrowPhaseType);
    }
    GoutWI(6, "Reading collision detection completed!");

    // ---------------------------------------------------------------------------------------------
    // Contact force models
    // Calculating the proper array size for the contact force array
    // We might need to redute it by removing the obstacle-obstacle pairs,
    // but it should be fine by now
    uint numMaterials     = GP::m_materialMap.size();
    GP::m_numContactPairs = numMaterials * (numMaterials + 1) / 2;
    DOMNode* contacts     = ReaderXML::getNode(root, "ContactForceModels");
    if(contacts)
    {
        GoutWI(6, "Reading contact force models ...");
        ContactForceModelFactory<T>::create(rootElement, m_contactForce);
        GoutWI(6, "Reading contact force models completed!");
    }

    // ---------------------------------------------------------------------------------------------
    // Temporal setting and time integration
    DOMNode* tempSetting = ReaderXML::getNode(root, "TemporalSetting");
    if(tempSetting)
    {
        DOMNode* nTime  = ReaderXML::getNode(tempSetting, "TimeInterval");
        T        tStart = ReaderXML::getNodeAttr_Double(nTime, "Start");
        T        tEnd   = ReaderXML::getNodeAttr_Double(nTime, "End");
        T        tStep  = ReaderXML::getNodeAttr_Double(nTime, "dt");
        GP::m_tStart    = tStart;
        GP::m_tEnd      = tEnd + HIGHEPS<T>;
        GP::m_dt        = tStep;
        DOMNode* nTI    = ReaderXML::getNode(tempSetting, "TimeIntegration");
        if(nTI)
        {
            GoutWI(6, "Reading time integration model ...");
            TimeIntegratorFactory<T>::create(nTI, GP::m_dt, m_timeIntegrator);
            GoutWI(6, "Reading time integration model completed!");
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Setting up the component managers
    m_components = std::make_unique<ComponentManagerCPU<T>>(&m_rigidBodyList,
                                                            SS.numObstacles,
                                                            SS.numParticles);
    // Initialize components
    m_components->initializeComponents(initialPosition, initialOrientation);
}

// -------------------------------------------------------------------------------------------------
// External force definition
template <typename T>
void Grains<T>::Forces(DOMElement* rootElement)
{
    assert(rootElement != NULL);
    DOMNode* root = ReaderXML::getNode(rootElement, "Forces");

    // Output message
    GoutWI(3, "Forces");

    // Read the forces
    if(root)
    {
        // Gravity
        DOMNode* nGravity = ReaderXML::getNode(root, "Gravity");
        GAssert(nGravity, "Gravity node is mandatory!");
        GrainsParameters<T>::m_gravity[X] = T(ReaderXML::getNodeAttr_Double(nGravity, "GX"));
        GrainsParameters<T>::m_gravity[Y] = T(ReaderXML::getNodeAttr_Double(nGravity, "GY"));
        GrainsParameters<T>::m_gravity[Z] = T(ReaderXML::getNodeAttr_Double(nGravity, "GZ"));
        GoutWI(6, "Gravity =", Vector3ToString(GrainsParameters<T>::m_gravity));
    }
}

// -------------------------------------------------------------------------------------------------
// Additional features of the simulation: insertion, post-processing
template <typename T>
void Grains<T>::AdditionalFeatures(DOMElement* rootElement)
{
    // Output message
    GoutWI(3, "Simulation");
    // ---------------------------------------------------------------------------------------------
    // Checking if Simulation node is available
    assert(rootElement != NULL);
    DOMNode* root = ReaderXML::getNode(rootElement, "Simulation");
    GAssert(root, "Simulation node is mandatory!");

    // ---------------------------------------------------------------------------------------------
    // Insertion policies
    DOMNode* nInsertion = ReaderXML::getNode(root, "ParticleInsertion");
    GoutWI(6, "Reading insertion policies ...");
    if(nInsertion)
        m_insertion = std::make_unique<Insertion<T>>(nInsertion);
    else
    {
        GoutWI(9, "No policy found, setting the insertion policy to default");
        m_insertion = std::make_unique<Insertion<T>>();
    }
    GoutWI(6, "Reading insertion policies completed.");

    // ---------------------------------------------------------------------------------------------
    // Post-processing writers
    DOMNode* nPostProcessing = ReaderXML::getNode(root, "PostProcessing");
    if(nPostProcessing)
    {
        GoutWI(3, "Post-processing");
        // Post-processing save time
        DOMNode* nTime = ReaderXML::getNode(nPostProcessing, "TimeSave");
        T        tStart, tEnd;
        if(ReaderXML::hasNodeAttr(nTime, "Start"))
            tStart = ReaderXML::getNodeAttr_Double(nTime, "Start");
        else
            tStart = GrainsParameters<T>::m_tStart;
        if(ReaderXML::hasNodeAttr(nTime, "End"))
            tEnd = ReaderXML::getNodeAttr_Double(nTime, "End");
        else
            tEnd = GrainsParameters<T>::m_tEnd;
        T tStep = ReaderXML::getNodeAttr_Double(nTime, "dt");
        for(T t = tStart; t <= tEnd; t += tStep)
            GrainsParameters<T>::m_tSave.push(t);
        // Save for tEnd as well
        GrainsParameters<T>::m_tSave.push(tEnd);

        // Post-processing writers
        DOMNode* nWriters = ReaderXML::getNode(nPostProcessing, "Writers");
        if(nWriters)
        {
            GoutWI(6, "Reading the post processing writers ...");
            DOMNodeList* allPPW = ReaderXML::getNodes(nWriters);
            for(uint i = 0; i < allPPW->getLength(); ++i)
            {
                DOMNode* nPPW = allPPW->item(i);
                m_postProcessor.push_back(std::unique_ptr<PostProcessingWriter<T>>(
                    PostProcessingWriterFactory<T>::create(nPPW)));
            }
            GoutWI(6, "Reading the post processing writers completed!");
        }
    }
    else
        GoutWI(6, "No postprocessing writer!");
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class Grains<float>;
template class Grains<double>;
template void Grains<float>::postProcess<MemType::HOST>(
    const std::unique_ptr<ComponentManager<float, MemType::HOST>>&);
template void Grains<float>::postProcess<MemType::DEVICE>(
    const std::unique_ptr<ComponentManager<float, MemType::DEVICE>>&);
template void Grains<double>::postProcess<MemType::HOST>(
    const std::unique_ptr<ComponentManager<double, MemType::HOST>>&);
template void Grains<double>::postProcess<MemType::DEVICE>(
    const std::unique_ptr<ComponentManager<double, MemType::DEVICE>>&);