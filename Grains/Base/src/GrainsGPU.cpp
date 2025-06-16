#include "GrainsGPU.hh"
#include "ContactForceModelFactory.hh"
#include "Grains.hh"
#include "LinkedCellFactory.hh"
#include "PostProcessingWriterFactory.hh"
#include "RigidBodyFactory.hh"
#include "TimeIntegratorFactory.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
GrainsGPU<T>::GrainsGPU()
{
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
GrainsGPU<T>::~GrainsGPU()
{
}

// -----------------------------------------------------------------------------
// Setup the GPU and its parameters
template <typename T>
void GrainsGPU<T>::setupGPUDevice()
{
    using GP = GrainsParameters<T>;

    // Set the device to the first one
    uint device = 0;
    cudaErrCheck(cudaSetDevice(device));

    // Get the device properties
    cudaDeviceProp prop;
    cudaErrCheck(cudaGetDeviceProperties(&prop, device));
    GoutWI(3, "GPU Device:");
    GoutWI(6, "Device: ", prop.name);
    GoutWI(6, "Compute capability: ", prop.major, ".", prop.minor);
    GoutWI(6, "Number of SMs: ", prop.multiProcessorCount);
    GoutWI(6,
           "Total global memory: ",
           prop.totalGlobalMem / (1024 * 1024),
           " MB");
    GoutWI(6, "Max threads per block: ", prop.maxThreadsPerBlock);
    GoutWI(6,
           "Max threads dimensions: ",
           prop.maxThreadsDim[0],
           "x",
           prop.maxThreadsDim[1],
           "x",
           prop.maxThreadsDim[2]);

    // Since this function is invoked, it means a GPU simulation is requested.
    GP::m_isGPU = true;
    // // Set the number of threads per block
    // GP::m_numThreadsPerBlock = 2 * prop.multiProcessorCount;
    // // Set the number of blocks
    // GP::m_numBlocksPerGrid = (GP::m_numParticles + GP::m_numThreadsPerBlock - 1)
    //                          / GP::m_numThreadsPerBlock;

    // const uint warpSize   = 32;
    const uint maxThreads = 256; // Avoid 1024 unless necessary
    const uint minThreads = 32;
    const uint minBlocks  = 2 * prop.multiProcessorCount;

    // Start with 128 threads and compute how many blocks we need
    uint threads = 128;
    uint blocks  = (GP::m_numParticles + threads - 1) / threads;

    // If we’re not filling the SMs enough, reduce threads
    if(blocks < minBlocks && threads > minThreads)
    {
        blocks  = minBlocks;
        threads = minThreads;
    }

    // If we're oversubscribing the SMs too much, increase thread count
    if(blocks > 4 * prop.multiProcessorCount && threads < maxThreads)
    {
        threads = maxThreads;
        blocks  = (GP::m_numParticles + threads - 1) / threads;
    }

    GP::m_numThreadsPerBlock = threads;
    GP::m_numBlocksPerGrid   = blocks;
}

// -----------------------------------------------------------------------------
// Initializes the simulation using the XML input
template <typename T>
void GrainsGPU<T>::initialize(DOMElement* rootElement)
{
    // We first read using the base class Grains<T>
    Grains<T>::initialize(rootElement);

    // Reading different blocks of the input XML
    // Note that most of the reading is done at Grains<T>::initialize
    // Herein, we initialize specifically for GPU
    Gout(std::string(80, '='));
    Gout("Setting up the simulation on device ...");
    Gout(std::string(80, '='));
    setupGPUDevice();
    Construction(rootElement);
    Forces(rootElement);
    AdditionalFeatures(rootElement);
    Gout(std::string(80, '='));
    Gout("Setting up the simulation on device completed!");
    Gout(std::string(80, '='));
}

// -----------------------------------------------------------------------------
// Runs the simulation over the prescribed time interval
template <typename T>
void GrainsGPU<T>::simulate()
{
    using GP = GrainsParameters<T>;

    Gout(std::string(80, '='));
    Gout("Starting the simulation on GPU");
    Gout(std::string(80, '='));

    // first, inserting partilces
    Grains<T>::m_components->insertParticles(Grains<T>::m_insertion);
    // Copying to device
    cout << "Copying the inserted particles to the device ..." << endl;
    Grains<T>::m_components->copyTo(m_d_components);
    cout << "Copying completed!" << endl;
    cout << "\nTime \t TO \tend \tParticles \tIn \tOut" << endl;
    for(GP::m_time = GP::m_tStart; GP::m_time <= GP::m_tEnd;
        GP::m_time += GP::m_dt)
    {
        // Output time
        ostringstream oss;
        oss.width(10);
        oss << left << GP::m_time;
        std::cout << '\r' << oss.str() << "  \t" << GP::m_tEnd << std::flush;

        m_d_components->detectCollisions();
        m_d_components->computeContactForces(m_d_contactForce);
        m_d_components->addExternalForces();
        m_d_components->moveParticles(m_d_timeIntegrator);

        // Post-Processing
        Grains<T>::postProcess(m_d_components);
    }
    cudaDeviceSynchronize();
}

/* ========================================================================== */
/*                            Low-Level Methods                               */
/* ========================================================================== */
// Constructs the simulation -- Reads the Construction part of the XML input to
// set the parameters
template <typename T>
void GrainsGPU<T>::Construction(DOMElement* rootElement)
{
    using GP = GrainsParameters<T>;

    // Get the Construction block. We don't check if it exists. It has been
    // already checked in the base class
    DOMNode* root = ReaderXML::getNode(rootElement, "Construction");

    // -------------------------------------------------------------------------
    // Particles
    GoutWI(3, "Copying particle types to device ...");
    m_d_particleRigidBodyList.reserve(GP::m_numParticles);
    RigidBodyFactory<T>::copyHostToDevice(Grains<T>::m_particleRigidBodyList,
                                          m_d_particleRigidBodyList);
    GoutWI(3, "Copying particle types to device completed!");

    // -------------------------------------------------------------------------
    // Obstacles
    GoutWI(3, "Copying obstacle types to device ...");
    m_d_obstacleRigidBodyList.reserve(GP::m_numObstacles);
    RigidBodyFactory<T>::copyHostToDevice(Grains<T>::m_obstacleRigidBodyList,
                                          m_d_obstacleRigidBodyList);
    cudaDeviceSynchronize();
    GoutWI(3, "Copying obstacle types to device completed!");

    // -------------------------------------------------------------------------
    // LinkedCell
    GoutWI(3, "Constructing linked cell on device ...");
    LinkedCellFactory<T>::copyHostToDevice(Grains<T>::m_linkedCell,
                                           m_d_linkedCell);
    GoutWI(3, "Constructing linked cell on device completed!");

    // -------------------------------------------------------------------------
    // Setting up the component managers
    m_d_components
        = std::make_unique<ComponentManagerGPU<T>>(&m_d_particleRigidBodyList,
                                                   &m_d_obstacleRigidBodyList,
                                                   GP::m_numParticles,
                                                   GP::m_numObstacles,
                                                   GP::m_numCells);
    // m_d_components->copyFrom(Grains<T>::m_components);

    // -------------------------------------------------------------------------
    // Contact force models
    // It is a GPU simulation, and we have already read contact force models
    // on the host. We allocate memory on device and copy the models over.
    GoutWI(3, "Copying contact force models to device ...");
    m_d_contactForce.reserve(GP::m_numContactPairs);
    ContactForceModelFactory<T>::copyHostToDevice(Grains<T>::m_contactForce,
                                                  m_d_contactForce);
    GoutWI(3, "Copying contact force models to device completed!");

    // -------------------------------------------------------------------------
    // Temporal setting and time integration
    // It is a GPU simulation, and we have already read time integration on the
    // host. We allocate memory on device and copy the scheme over.
    GoutWI(3, "Copying time integration scheme to device ...");
    TimeIntegratorFactory<T>::copyHostToDevice(Grains<T>::m_timeIntegrator,
                                               m_d_timeIntegrator);
    GoutWI(3, "Copying time integration scheme to device completed!");
}

// -----------------------------------------------------------------------------
// External force definition
template <typename T>
void GrainsGPU<T>::Forces(DOMElement* rootElement)
{
}

// -----------------------------------------------------------------------------
// Additional features of the simulation: insertion, post-processing
template <typename T>
void GrainsGPU<T>::AdditionalFeatures(DOMElement* rootElement)
{
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class GrainsGPU<float>;
template class GrainsGPU<double>;