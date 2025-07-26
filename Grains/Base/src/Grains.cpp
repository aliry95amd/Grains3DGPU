#include "Grains.hh"
#include "ComponentManagerCPU.hh"
#include "ContactForceModelFactory.hh"
#include "PostProcessingWriterFactory.hh"
#include "RigidBodyFactory.hh"
#include "TimeIntegratorFactory.hh"
#include "VectorMath.hh"

/* ========================================================================== */
/*                             High-Level Methods                             */
/* ========================================================================== */
// Default constructor
template <typename T>
Grains<T>::Grains()
{
    Gout(std::string(80, '='));
    Gout("Starting Grains3D ...");
    Gout(std::string(80, '='));
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
Grains<T>::~Grains()
{
}

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Performs post-processing on the host
template <typename T>
void Grains<T>::postProcess(
    const std::unique_ptr<ComponentManager<T, MemType::HOST>>& cm) const
{
    using GP = GrainsParameters<T>;

    if(GP::m_tSave.empty())
        return;

    if(GP::m_tSave.front() - GP::m_time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();
        for(auto& pp : m_postProcessor)
            pp->PostProcessing(m_particleRigidBodyList,
                               m_obstacleRigidBodyList,
                               cm,
                               GP::m_time);
    }
    // In case we get past the saveTime, we need to remove it from the queue
    if(GP::m_time > GP::m_tSave.front())
        GP::m_tSave.pop();
}

// -----------------------------------------------------------------------------
// Performs post-processing on the device
template <typename T>
void Grains<T>::postProcess(
    const std::unique_ptr<ComponentManager<T, MemType::DEVICE>>& cm)
{
    using GP = GrainsParameters<T>;

    if(GP::m_tSave.empty())
        return;

    if(GP::m_tSave.front() - GP::m_time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();
        cm->copyTo_PostProcessing(m_components);
        for(auto& pp : m_postProcessor)
            pp->PostProcessing(m_particleRigidBodyList,
                               m_obstacleRigidBodyList,
                               m_components,
                               GP::m_time);
    }
    // In case we get past the saveTime, we need to remove it from the queue
    if(GP::m_time > GP::m_tSave.front())
        GP::m_tSave.pop();
}

// -----------------------------------------------------------------------------
// Performs tasks after time-stepping
template <typename T>
void Grains<T>::finalize()
{
    for(auto& pp : m_postProcessor)
        pp->PostProcessing_end();
}

/* ========================================================================== */
/*                            Low-Level Methods                               */
/* ========================================================================== */
// Constructs the simulation -- Reads the Construction part of the XML input to
// set the parameters
template <typename T>
void Grains<T>::Construction(DOMElement* rootElement)
{
    using GP = GrainsParameters<T>;

    // Output message
    GoutWI(3, "Construction");
    // -------------------------------------------------------------------------
    // Checking if Construction node is available
    DOMNode* root = ReaderXML::getNode(rootElement, "Construction");
    if(!root)
        GAbort("Construction node is mandatory!");

    // -------------------------------------------------------------------------
    // Domain size: origin, max coordinates and periodicity
    DOMNode* nOrigin = ReaderXML::getNode(root, "Origin");
    if(nOrigin)
        GP::m_origin.setValue(T(ReaderXML::getNodeAttr_Double(nOrigin, "X")),
                              T(ReaderXML::getNodeAttr_Double(nOrigin, "Y")),
                              T(ReaderXML::getNodeAttr_Double(nOrigin, "Z")));
    else
        GP::m_origin.setValue(T(0), T(0), T(0));

    DOMNode* nDomain = ReaderXML::getNode(root, "MaxCoordinate");
    GP::m_maxCoordinate.setValue(
        T(ReaderXML::getNodeAttr_Double(nDomain, "X")),
        T(ReaderXML::getNodeAttr_Double(nDomain, "Y")),
        T(ReaderXML::getNodeAttr_Double(nDomain, "Z")));

    // if the simulation is periodic
    DOMNode* nPeriodicity = ReaderXML::getNode(root, "Periodicity");
    if(nPeriodicity)
    {
        int PX = ReaderXML::getNodeAttr_Int(nPeriodicity, "PX");
        int PY = ReaderXML::getNodeAttr_Int(nPeriodicity, "PY");
        int PZ = ReaderXML::getNodeAttr_Int(nPeriodicity, "PZ");
        if(PX * PY * PZ != 0)
            GAbort("Periodicity is not implemented!");
        GP::m_isPeriodic = false;
    }

    // -------------------------------------------------------------------------
    // Particles
    DOMNode* particles = ReaderXML::getNode(root, "Particles");

    GrainsMemBuffer<RigidBody<T>*, MemType::HOST> m_refParticleRigidBodyList;
    GrainsMemBuffer<Vector3<T>, MemType::HOST>    refParticlesInitialPosition;
    GrainsMemBuffer<Quaternion<T>, MemType::HOST>
                                         refParticlesInitialOrientation;
    GrainsMemBuffer<uint, MemType::HOST> numEachRefParticle;
    uint                                 numParticles = 0;
    if(particles)
    {
        GoutWI(6, "Reading particle types ...");
        RigidBodyFactory<T>::create(particles,
                                    m_refParticleRigidBodyList,
                                    refParticlesInitialPosition,
                                    refParticlesInitialOrientation,
                                    numEachRefParticle,
                                    numParticles);
        GoutWI(6, "Reading particle types completed!");
    }

    m_particleRigidBodyList.reserve(numParticles);
    GrainsMemBuffer<Vector3<T>, MemType::HOST>    particlesInitialPosition;
    GrainsMemBuffer<Quaternion<T>, MemType::HOST> particlesInitialOrientation;
    particlesInitialPosition.allocate(numParticles);
    particlesInitialOrientation.allocate(numParticles);

    if(numParticles)
    {
        uint offset = 0;
        for(uint i = 0; i < m_refParticleRigidBodyList.getSize(); ++i)
        {
            for(uint j = 0; j < numEachRefParticle[i]; j++)
            {
                // Deep copy of the rigid body
                m_particleRigidBodyList[offset + j]
                    = new RigidBody<T>(*m_refParticleRigidBodyList[i]);
                // Initial transformation of the rigid body
                particlesInitialPosition[offset + j]
                    = refParticlesInitialPosition[i];
                particlesInitialOrientation[offset + j]
                    = refParticlesInitialOrientation[i];
            }
            // Increment the starting position
            offset += numEachRefParticle[i];
        }
    }
    GP::m_numParticles = numParticles;

    // Finding max circumscribed radius among all particles.
    T maxRadius = T(0);
    for(uint i = 0; i < m_refParticleRigidBodyList.getSize(); ++i)
    {
        if(m_refParticleRigidBodyList[i]->getCircumscribedRadius() > maxRadius)
            maxRadius = m_refParticleRigidBodyList[i]->getCircumscribedRadius();
    }
    GP::m_maxRadius = maxRadius;

    // -------------------------------------------------------------------------
    // Obstacles
    DOMNode*     obstacles    = ReaderXML::getNode(root, "Obstacles");
    DOMNodeList* allObstacles = ReaderXML::getNodes(rootElement, "Obstacle");
    // Number of unique obstacles in the simulation
    uint numObstacles = allObstacles->getLength();
    // We also store the initial transformations of the rigid bodies to pass to
    // the ComponentManager to create particles with the initial transformation
    // required.
    GrainsMemBuffer<Vector3<T>, MemType::HOST>    obstaclesInitialPosition;
    GrainsMemBuffer<Quaternion<T>, MemType::HOST> obstaclesInitialOrientation;
    obstaclesInitialPosition.allocate(numObstacles);
    obstaclesInitialOrientation.allocate(numObstacles);
    // Memory allocation for m_rigidBodyList with respect to the number of
    // shapes in the simulation.
    m_obstacleRigidBodyList.reserve(numObstacles);
    if(numObstacles)
    {
        GoutWI(6, "Reading obstacles types ...");
        for(uint i = 0; i < numObstacles; i++)
        {
            DOMNode* nObstacle = allObstacles->item(i);
            // Create the Rigid Body
            m_obstacleRigidBodyList[i] = new RigidBody<T>(nObstacle);
            // Initial transformation of the rigid body
            // One draw back is we might end up with the same rigid body shape,
            // but with different initial transformation.
            DOMNode*      tr = ReaderXML::getNode(nObstacle, "Transformation");
            Vector3<T>    centre(T(0), T(0), T(0));
            Quaternion<T> rotation(T(0), T(0), T(0), T(1));
            if(nTransform)
            {
                DOMNode* nCentre = ReaderXML::getNode(nParticle, "Centre");
                if(nCentre)
                    centre = Vector3<T>(nCentre);

                DOMNode* nRotation
                    = ReaderXML::getNode(nParticle, "AngularPosition");
                if(nRotation)
                    rotation = Quaternion<T>(nRotation);
                obstaclesInitialPosition[i]    = centre;
                obstaclesInitialOrientation[i] = rotation;
            }
            else
                GAbort("Transformation node is mandatory for obstacles!");
        }
        GoutWI(6, "Reading obstacles types completed!");
    }

    // -------------------------------------------------------------------------
    // Setting up collision detection
    GoutWI(6, "Reading collision detection ...");
    DOMNode* collisionDetection
        = ReaderXML::getNode(root, "CollisionDetection");
    if(!collisionDetection)
        GAbort("CollisionDetection node is mandatory!");
    // Neighbor list
    DOMNode* nNeighborList
        = ReaderXML::getNode(collisionDetection, "NeighborList");
    if(!nNeighborList)
        GAbort("NeighborList node not found");
    std::string neighborListType
        = ReaderXML::getNodeAttr_String(nNeighborList, "Type");
    if(neighborListType == "BruteForce")
        GP::m_neighborListType = 0;
    else if(neighborListType == "LinkedCell")
        GP::m_neighborListType = 1;
    else
        GAbort("Unknown NeighborList type! Aborting Grains!");
    GP::m_neighborListFrequency
        = ReaderXML::getNodeAttr_Int(nNeighborList, "UpdateFrequency");
    GoutWI(9,
           "NeighborList generation with " + neighborListType
               + " and frequency " + std::to_string(GP::m_neighborListFrequency)
               + " ...");
    // Linked cell
    if(GP::m_neighborListType == 1)
    {
        DOMNode* nLinkedCell
            = ReaderXML::getNode(collisionDetection, "LinkedCell");
        if(!nLinkedCell)
            GAbort("LinkedCell node is mandatory when using LinkedCell "
                   "neighbor list!");
        std::string linkedCellType
            = ReaderXML::getNodeAttr_String(nLinkedCell, "Type");
        if(linkedCellType == "MemoryEfficient")
            GP::m_linkedCellType = 1;
        else
            GAbort("Unknown LinkedCell type! Aborting Grains!");
        GP::m_linkedCellSizeFactor
            = T(ReaderXML::getNodeAttr_Double(nLinkedCell, "CellSizeFactor"));
        GP::m_sortingFrequency
            = ReaderXML::getNodeAttr_Int(nLinkedCell, "SortingFrequency");
        GoutWI(9,
               linkedCellType + " LinkedCell" + " with cell size factor "
                   + std::to_string(GP::m_linkedCellSizeFactor)
                   + " and sorting frequency "
                   + std::to_string(GP::m_sortingFrequency) + " ...");
    }
    // Bounding volume
    DOMNode* nBoundingVolume
        = ReaderXML::getNode(collisionDetection, "BoundingVolume");
    if(nBoundingVolume)
    {
        std::string boundingVolumeType
            = ReaderXML::getNodeAttr_String(nBoundingVolume, "Type");
        if(boundingVolumeType == "OFF")
            GP::m_boundingVolumeType = 0;
        else if(boundingVolumeType == "OBB")
            GP::m_boundingVolumeType = 1;
        else if(boundingVolumeType == "OBC")
            GP::m_boundingVolumeType = 2;
        else
            GAbort("Unknown bounding volume type! Aborting Grains!");
        GoutWI(9, boundingVolumeType + " Bounding volume ...");
    }
    // Narrow phase detection
    DOMNode* nNarrowPhase
        = ReaderXML::getNode(collisionDetection, "NarrowPhase");
    if(nNarrowPhase)
    {
        std::string narrowPhaseType
            = ReaderXML::getNodeAttr_String(nNarrowPhase, "Type");
        if(narrowPhaseType == "GJK")
            GP::m_narrowPhaseType = 0;
        else
            GAbort("Unknown narrow phase type! Aborting Grains!");
        GoutWI(9, narrowPhaseType + " Narrow phase detection ...");
    }
    GoutWI(6, "Reading collision detection completed!");

    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Setting up the component managers
    GP::m_numParticles = numParticles;
    GP::m_numObstacles = numObstacles;
    m_components
        = std::make_unique<ComponentManagerCPU<T>>(&m_particleRigidBodyList,
                                                   &m_obstacleRigidBodyList,
                                                   GP::m_numParticles,
                                                   GP::m_numObstacles);
    // Initialize the particles and obstacles
    m_components->initializeParticles(particlesInitialPosition,
                                      particlesInitialOrientation);
    m_components->initializeObstacles(obstaclesInitialPosition,
                                      obstaclesInitialOrientation);
}

// -----------------------------------------------------------------------------
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
        if(nGravity)
        {
            GrainsParameters<T>::m_gravity[X]
                = T(ReaderXML::getNodeAttr_Double(nGravity, "GX"));
            GrainsParameters<T>::m_gravity[Y]
                = T(ReaderXML::getNodeAttr_Double(nGravity, "GY"));
            GrainsParameters<T>::m_gravity[Z]
                = T(ReaderXML::getNodeAttr_Double(nGravity, "GZ"));
            GoutWI(6,
                   "Gravity =",
                   Vector3ToString(GrainsParameters<T>::m_gravity));
        }
        else
            GAbort("Gravity is mandatory!");
    }
}

// -----------------------------------------------------------------------------
// Additional features of the simulation: insertion, post-processing
template <typename T>
void Grains<T>::AdditionalFeatures(DOMElement* rootElement)
{
    // Output message
    GoutWI(3, "Simulation");
    // -------------------------------------------------------------------------
    // Checking if Simulation node is available
    assert(rootElement != NULL);
    DOMNode* root = ReaderXML::getNode(rootElement, "Simulation");
    if(!root)
        GAbort("Simulation node is mandatory!");

    // -------------------------------------------------------------------------
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

    // -------------------------------------------------------------------------
    // Post-processing writers
    DOMNode* nPostProcessing = ReaderXML::getNode(root, "PostProcessing");
    if(nPostProcessing)
    {
        GoutWI(3, "Post-processing");
        // Post-processing save time
        DOMNode* nTime  = ReaderXML::getNode(nPostProcessing, "TimeSave");
        T        tStart = ReaderXML::getNodeAttr_Double(nTime, "Start");
        T        tEnd   = ReaderXML::getNodeAttr_Double(nTime, "End");
        T        tStep  = ReaderXML::getNodeAttr_Double(nTime, "dt");
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
                m_postProcessor.push_back(
                    std::unique_ptr<PostProcessingWriter<T>>(
                        PostProcessingWriterFactory<T>::create(nPPW)));
            }
            GoutWI(6, "Reading the post processing writers completed!");
        }
    }
    else
        GoutWI(6, "No postprocessing writer!");
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class Grains<float>;
template class Grains<double>;