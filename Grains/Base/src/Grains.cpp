#include <numeric>

#include "ComponentManagerCPU.hh"
#include "ContactForceModelFactory.hh"
#include "Grains.hh"
#include "LinkedCellFactory.hh"
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
// Performs post-processing
template <typename T>
void Grains<T>::postProcess(
    const std::unique_ptr<ComponentManager<T, MemType::HOST>>& cm) const
{
    using GP = GrainsParameters<T>;

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
// Performs post-processing
template <typename T>
void Grains<T>::postProcessDevice(
    const std::unique_ptr<ComponentManager<T, MemType::DEVICE>>& cm) const
{
    using GP = GrainsParameters<T>;

    if(GP::m_tSave.front() - GP::m_time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();
        // for(auto& pp : m_postProcessor)
        //     pp->PostProcessing(m_particleRigidBodyList,
        //                        m_obstacleRigidBodyList,
        //                        cm,
        //                        GP::m_time);
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
    DOMNode* nDomain = ReaderXML::getNode(root, "LinkedCell");
    GP::m_maxCoordinate.setValue(
        T(ReaderXML::getNodeAttr_Double(nDomain, "MX")),
        T(ReaderXML::getNodeAttr_Double(nDomain, "MY")),
        T(ReaderXML::getNodeAttr_Double(nDomain, "MZ")));

    DOMNode* nOrigin = ReaderXML::getNode(root, "Origin");
    if(nOrigin)
        GP::m_origin.setValue(T(ReaderXML::getNodeAttr_Double(nOrigin, "OX")),
                              T(ReaderXML::getNodeAttr_Double(nOrigin, "OY")),
                              T(ReaderXML::getNodeAttr_Double(nOrigin, "OZ")));
    else
        GP::m_origin.setValue(T(0), T(0), T(0));

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

    GrainsMemBuffer<RigidBody<T, T>*, MemType::HOST> m_refParticleRigidBodyList;
    GrainsMemBuffer<Transform3<T>, MemType::HOST> refParticlesInitialTransform;
    GrainsMemBuffer<uint, MemType::HOST>          numEachRefParticle;
    uint                                          numParticles = 0;
    if(particles)
    {
        GoutWI(6, "Reading particle types ...");
        RigidBodyFactory<T>::create(particles,
                                    m_refParticleRigidBodyList,
                                    refParticlesInitialTransform,
                                    numEachRefParticle,
                                    numParticles);
        GoutWI(6, "Reading particle types completed!");
    }

    m_particleRigidBodyList.reserve(numParticles);
    GrainsMemBuffer<Transform3<T>, MemType::HOST> particlesInitialTransform;
    particlesInitialTransform.allocate(numParticles);
    if(numParticles)
    {
        uint offset = 0;
        for(uint i = 0; i < m_refParticleRigidBodyList.getSize(); ++i)
        {
            for(uint j = 0; j < numEachRefParticle[i]; j++)
            {
                // Deep copy of the rigid body
                m_particleRigidBodyList[offset + j]
                    = new RigidBody<T, T>(*m_refParticleRigidBodyList[i]);
                // Initial transformation of the rigid body
                particlesInitialTransform[offset + j]
                    = refParticlesInitialTransform[i];
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
    GrainsMemBuffer<Transform3<T>, MemType::HOST> obstaclesInitialTransform;
    obstaclesInitialTransform.allocate(numObstacles);
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
            m_obstacleRigidBodyList[i] = new RigidBody<T, T>(nObstacle);
            // Initial transformation of the rigid body
            // One draw back is we might end up with the same rigid body shape,
            // but with different initial transformation.
            DOMNode* tr = ReaderXML::getNode(nObstacle, "Transformation");
            obstaclesInitialTransform[i] = Transform3<T>(tr);
        }
        GoutWI(6, "Reading obstacles types completed!");
    }

    // -------------------------------------------------------------------------
    // LinkedCell
    GoutWI(6, "Constructing linked cell ...");
    DOMNode* nColDet     = ReaderXML::getNode(root, "CollisionDetection");
    DOMNode* nLinkedCell = ReaderXML::getNode(nColDet, "LinkedCell");
    uint     numCells    = 0;
    LinkedCellFactory<T>::create(nLinkedCell, m_linkedCell, numCells);
    GP::m_numCells = numCells;
    GoutWI(6, "Constructing linked cell completed!");

    // -------------------------------------------------------------------------
    // Setting up the component managers
    GP::m_numParticles = numParticles;
    GP::m_numObstacles = numObstacles;
    m_components
        = std::make_unique<ComponentManagerCPU<T>>(&m_particleRigidBodyList,
                                                   &m_obstacleRigidBodyList,
                                                   GP::m_numParticles,
                                                   GP::m_numObstacles,
                                                   GP::m_numCells);
    // Initialize the particles and obstacles
    m_components->initializeParticles(particlesInitialTransform);
    m_components->initializeObstacles(obstaclesInitialTransform);

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