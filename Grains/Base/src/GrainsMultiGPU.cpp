#ifdef GRAINS_USE_MPI

#include "GrainsMultiGPU.hh"
#include "ComponentManagerGPU.hh"
#include "ContactForceModelFactory.hh"
#include "PostProcessingWriterFactory.hh"
#include "RigidBodyFactory.hh"
#include "TimeIntegratorFactory.hh"

// =================================================================================================
template <typename T>
GrainsMultiGPU<T>::GrainsMultiGPU()
    : GrainsGPU<T>()
    , m_numLocalParticles(0)
    , m_numGhostParticles(0)
    , m_mpiRank(0)
    , m_mpiSize(1)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_mpiSize);
}

// =================================================================================================
template <typename T>
GrainsMultiGPU<T>::~GrainsMultiGPU()
{
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::setupGPUDevice()
{
    using GP = GrainsParameters<T>;

    int deviceCount = 0;
    cudaErrCheck(cudaGetDeviceCount(&deviceCount));
    GAssert(deviceCount > 0, "No CUDA devices found!");

    // Bind each rank to a different GPU (round-robin if more ranks than GPUs)
    int localDevice = m_mpiRank % deviceCount;
    cudaErrCheck(cudaSetDevice(localDevice));

    cudaDeviceProp prop;
    cudaErrCheck(cudaGetDeviceProperties(&prop, localDevice));

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '-'));
        Gout("Multi-GPU DEM: ", m_mpiSize, "ranks,", deviceCount, "GPUs per node");
        Gout(std::string(80, '-'));
    }
    Gout("[Rank", m_mpiRank, "] GPU:", prop.name, "( device", localDevice, ")");

    GP::m_isGPU = true;
    GP::m_GPU   = prop;
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::initialize(DOMElement* rootElement)
{
    // Base class reads all XML blocks (Construction, Forces, AdditionalFeatures) on HOST
    Grains<T>::initialize(rootElement);

    Gout(std::string(80, '='));
    Gout("[Rank", m_mpiRank, "] Setting up multi-GPU simulation on device ...");
    Gout(std::string(80, '='));

    setupGPUDevice();
    Construction(rootElement);

    Gout(std::string(80, '='));
    Gout("[Rank", m_mpiRank, "] Multi-GPU setup completed!");
    Gout(std::string(80, '='));
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::Construction(DOMElement* rootElement)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    // -----------------------------------------------------------------------------------------
    // Domain decomposition
    T maxParticleRadius = 0;
    for(uint i = 0; i < Grains<T>::m_rigidBodyList.getSize(); ++i)
    {
        if(Grains<T>::m_rigidBodyList[i])
        {
            T r = Grains<T>::m_rigidBodyList[i]->getCircumscribedRadius();
            if(r > maxParticleRadius)
                maxParticleRadius = r;
        }
    }

    T ghostWidth = maxParticleRadius * T(4);
    m_decomp = std::make_unique<DomainDecomposition<T>>(
        MPI_COMM_WORLD, GP::m_origin, GP::m_maxCoordinate, ghostWidth);

    // Override linked cell domain to local subdomain + ghost margin
    auto& LC       = GP::m_collisionDetection.linkedCellParameters;
    LC.minCorner   = m_decomp->getGhostMin();
    LC.maxCorner   = m_decomp->getGhostMax();

    // -----------------------------------------------------------------------------------------
    // Partition particles to this rank's subdomain
    partitionParticlesToLocal();

    // -----------------------------------------------------------------------------------------
    // Copy rigid bodies to device
    GoutWI(3, "[Rank", m_mpiRank, "] Copying rigid bodies to device ...");
    RigidBodyFactory<T>::copyHostToDevice(Grains<T>::m_rigidBodyList,
                                          GrainsGPU<T>::m_d_rigidBodyList);
    GoutWI(3, "[Rank", m_mpiRank, "] Copying rigid bodies completed!");

    // -----------------------------------------------------------------------------------------
    // Copy contact force models
    GoutWI(3, "[Rank", m_mpiRank, "] Copying contact force models to device ...");
    GrainsGPU<T>::m_d_contactForce.reserve(GP::m_numContactPairs);
    ContactForceModelFactory<T>::copyHostToDevice(Grains<T>::m_contactForce,
                                                   GrainsGPU<T>::m_d_contactForce);
    GoutWI(3, "[Rank", m_mpiRank, "] Copying contact force models completed!");

    // -----------------------------------------------------------------------------------------
    // Copy time integration
    GoutWI(3, "[Rank", m_mpiRank, "] Copying time integration scheme to device ...");
    TimeIntegratorFactory<T>::copyHostToDevice(Grains<T>::m_timeIntegrator,
                                               GrainsGPU<T>::m_d_timeIntegrator);
    GoutWI(3, "[Rank", m_mpiRank, "] Copying time integration completed!");

    // -----------------------------------------------------------------------------------------
    // Component manager with local particles only
    GrainsGPU<T>::m_d_components = std::make_unique<ComponentManagerGPU<T>>(
        &GrainsGPU<T>::m_d_rigidBodyList, SS.numObstacles, m_numLocalParticles);

    // -----------------------------------------------------------------------------------------
    // Ghost exchanger and migrator
    m_ghostExchanger = std::make_unique<GhostExchanger<T>>(m_decomp.get());
    m_migrator       = std::make_unique<ParticleMigrator<T>>(m_decomp.get());
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::partitionParticlesToLocal()
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& hostCM = Grains<T>::m_components;

    uint numObstacles = SS.numObstacles;
    uint numParticles = SS.numParticles;

    // Get positions on host
    const auto& positions = hostCM->getPosition();

    // Count how many particles belong to this rank
    std::vector<uint> localIndices;
    localIndices.reserve(numParticles);

    for(uint p = 0; p < numParticles; ++p)
    {
        uint idx = numObstacles + p;
        if(m_decomp->isLocal(positions[idx]))
            localIndices.push_back(p);
    }

    m_numLocalParticles = static_cast<uint>(localIndices.size());

    Gout("[Rank", m_mpiRank, "] Local particles:", m_numLocalParticles,
         "of", numParticles, "total");

    // If all particles are local (single rank), nothing to do
    if(m_numLocalParticles == numParticles)
        return;

    // Reorder the host component manager so local particles are contiguous
    // (obstacles stay at indices [0, numObstacles), local particles at [numObstacles, ...))
    const auto& hPos  = hostCM->getPosition();
    const auto& hQuat = hostCM->getQuaternion();
    const auto& hVel  = hostCM->getVelocity();

    GrainsMemBuffer<Vector3<T>>    newPos(numObstacles + m_numLocalParticles);
    GrainsMemBuffer<Quaternion<T>> newQuat(numObstacles + m_numLocalParticles);
    GrainsMemBuffer<Kinematics<T>> newVel(numObstacles + m_numLocalParticles);

    // Copy obstacles
    for(uint i = 0; i < numObstacles; ++i)
    {
        newPos[i]  = hPos[i];
        newQuat[i] = hQuat[i];
        newVel[i]  = hVel[i];
    }

    // Copy only local particles
    for(uint i = 0; i < m_numLocalParticles; ++i)
    {
        uint srcIdx = numObstacles + localIndices[i];
        uint dstIdx = numObstacles + i;
        newPos[dstIdx]  = hPos[srcIdx];
        newQuat[dstIdx] = hQuat[srcIdx];
        newVel[dstIdx]  = hVel[srcIdx];
    }

    // Update the host component manager with local-only data
    hostCM->setPosition(newPos);
    hostCM->setQuaternion(newQuat);
    hostCM->setVelocity(newVel);

    // Update simulation state
    SS.numParticles = m_numLocalParticles;
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::simulate()
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& d_cm = GrainsGPU<T>::m_d_components;

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("Starting multi-GPU simulation (", m_mpiSize, "ranks )");
        Gout(std::string(80, '='));
    }

    // Insert particles on host and copy to device
    Grains<T>::m_components->insertParticles(Grains<T>::m_insertion);

    Gout("[Rank", m_mpiRank, "] Copying", m_numLocalParticles,
         "particles to device ...");
    Grains<T>::m_components->copyTo(d_cm);
    Gout("[Rank", m_mpiRank, "] Copy completed!");

    if(m_mpiRank == 0)
        std::cout << "\nTime \t TO \tend \tLocal Particles" << std::endl;

    // Pre-compute forces on initial configuration (KDK warmup)
    if(GP::m_isLeapFrog)
    {
        m_numGhostParticles = m_ghostExchanger->exchange(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getRigidBodyIdBuffer(),
            d_cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        d_cm->detectCollisions();
        d_cm->computeContactForces(GrainsGPU<T>::m_d_contactForce);
        d_cm->addExternalForces();

        GhostExchanger<T>::removeGhosts(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getRigidBodyIdBuffer(),
            d_cm->getComponentIdBuffer(),
            SS.numObstacles + m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles);
    }

    SS.time = GP::m_tStart;
    postProcessParallel(d_cm);

    uint stepCount = 0;
    for(SS.time = GP::m_tStart + GP::m_dt; SS.time <= GP::m_tEnd; SS.time += GP::m_dt)
    {
        stepCount++;

        if(m_mpiRank == 0 && GP::m_verbosityFrequency > 0
           && (stepCount % GP::m_verbosityFrequency == 0))
        {
            std::ostringstream oss;
            oss.width(10);
            oss << std::left << SS.time;
            std::cout << oss.str() << "  \t" << GP::m_tEnd
                      << "  \t" << m_numLocalParticles << std::endl;
        }

        stepWithCommunication();

        postProcessParallel(d_cm);
    }

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::stepWithCommunication()
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& d_cm = GrainsGPU<T>::m_d_components;

    if(GP::m_isLeapFrog)
    {
        // KDK Step 1: half-kick + drift (local particles only)
        d_cm->setNumberOfParticles(m_numLocalParticles);
        d_cm->moveParticles(GrainsGPU<T>::m_d_timeIntegrator);

        // Migrate particles that left local domain
        m_numLocalParticles = m_migrator->migrate(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getTorceBuffer(),
            d_cm->getRigidBodyIdBuffer(), d_cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles);

        // Ghost exchange
        m_numGhostParticles = m_ghostExchanger->exchange(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getRigidBodyIdBuffer(),
            d_cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        // Collisions and forces on local + ghost particles
        d_cm->detectCollisions();
        d_cm->computeContactForces(GrainsGPU<T>::m_d_contactForce);
        d_cm->addExternalForces();

        // Remove ghosts
        GhostExchanger<T>::removeGhosts(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getRigidBodyIdBuffer(),
            d_cm->getComponentIdBuffer(),
            SS.numObstacles + m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles);

        // KDK Step 3: second half-kick (local only)
        d_cm->advanceVelocity(GrainsGPU<T>::m_d_timeIntegrator);
    }
    else
    {
        // Ghost exchange before collision detection
        m_numGhostParticles = m_ghostExchanger->exchange(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getRigidBodyIdBuffer(),
            d_cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        d_cm->detectCollisions();
        d_cm->computeContactForces(GrainsGPU<T>::m_d_contactForce);
        d_cm->addExternalForces();

        // Remove ghosts before integration
        GhostExchanger<T>::removeGhosts(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getRigidBodyIdBuffer(),
            d_cm->getComponentIdBuffer(),
            SS.numObstacles + m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles);

        d_cm->moveParticles(GrainsGPU<T>::m_d_timeIntegrator);

        // Migrate after integration
        m_numLocalParticles = m_migrator->migrate(
            d_cm->getPositionBuffer(), d_cm->getQuaternionBuffer(),
            d_cm->getVelocityBuffer(), d_cm->getTorceBuffer(),
            d_cm->getRigidBodyIdBuffer(), d_cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        d_cm->setNumberOfParticles(m_numLocalParticles);
    }
}

// =================================================================================================
template <typename T>
template <MemType M>
void GrainsMultiGPU<T>::postProcessParallel(
    const std::unique_ptr<ComponentManager<T, M>>& cm)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    if(GP::m_tSave.empty())
        return;

    if(GP::m_tSave.front() - SS.time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();

        // Temporarily set numParticles to local-only for correct D2H copy
        cm->setNumberOfParticles(m_numLocalParticles);

        if constexpr(M == MemType::DEVICE)
        {
            // Ensure host component manager is sized for local data
            cm->copyTo_PostProcessing(Grains<T>::m_components);
            for(auto& pp : Grains<T>::m_postProcessor)
                pp->PostProcessing(Grains<T>::m_rigidBodyList,
                                   Grains<T>::m_components, SS.time);
        }
        else
        {
            for(auto& pp : Grains<T>::m_postProcessor)
                pp->PostProcessing(Grains<T>::m_rigidBodyList, cm, SS.time);
        }
    }

    if(!GP::m_tSave.empty() && SS.time > GP::m_tSave.front())
        GP::m_tSave.pop();
}

// =================================================================================================
template <typename T>
void GrainsMultiGPU<T>::finalize()
{
    MPI_Barrier(MPI_COMM_WORLD);

    for(auto& pp : Grains<T>::m_postProcessor)
        pp->PostProcessing_end();

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("Multi-GPU simulation completed!");
        Gout(std::string(80, '='));
    }
}

// =================================================================================================
// Explicit instantiation
template class GrainsMultiGPU<float>;
template class GrainsMultiGPU<double>;
template void GrainsMultiGPU<float>::postProcessParallel<MemType::DEVICE>(
    const std::unique_ptr<ComponentManager<float, MemType::DEVICE>>&);
template void GrainsMultiGPU<double>::postProcessParallel<MemType::DEVICE>(
    const std::unique_ptr<ComponentManager<double, MemType::DEVICE>>&);

#endif // GRAINS_USE_MPI
