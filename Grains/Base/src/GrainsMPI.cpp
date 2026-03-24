#ifdef GRAINS_USE_MPI

#include "GrainsMPI.hh"
#include "ComponentManagerCPU.hh"
#include "ComponentManagerGPU.hh"
#include "ContactForceModelFactory.hh"
#include "RigidBodyFactory.hh"
#include "TimeIntegratorFactory.hh"

// =================================================================================================
template <typename T, MemType M>
GrainsMPI<T, M>::GrainsMPI()
    : Base()
    , m_numLocalParticles(0)
    , m_numGhostParticles(0)
    , m_mpiRank(0)
    , m_mpiSize(1)
{
    MPI_Comm_rank(MPI_COMM_WORLD, &m_mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &m_mpiSize);
}

// =================================================================================================
template <typename T, MemType M>
GrainsMPI<T, M>::~GrainsMPI() {}

// =================================================================================================
template <typename T, MemType M>
void GrainsMPI<T, M>::initialize(DOMElement* rootElement)
{
    Grains<T>::initialize(rootElement);

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        if constexpr(M == MemType::DEVICE)
            Gout("Multi-GPU DEM:", m_mpiSize, "MPI ranks");
        else
            Gout("Multi-CPU DEM:", m_mpiSize, "MPI ranks");
        Gout(std::string(80, '='));
    }

    if constexpr(M == MemType::DEVICE)
    {
        // Per-rank GPU setup
        using GP = GrainsParameters<T>;
        int deviceCount = 0;
        cudaErrCheck(cudaGetDeviceCount(&deviceCount));
        GAssert(deviceCount > 0, "No CUDA devices found!");
        int localDevice = m_mpiRank % deviceCount;
        cudaErrCheck(cudaSetDevice(localDevice));
        cudaDeviceProp prop;
        cudaErrCheck(cudaGetDeviceProperties(&prop, localDevice));
        Gout("[Rank", m_mpiRank, "] GPU:", prop.name, "( device", localDevice, ")");
        GP::m_isGPU = true;
        GP::m_GPU   = prop;
    }

    constructionMPI(rootElement);

    Gout("[Rank", m_mpiRank, "] MPI setup completed,",
         m_numLocalParticles, "local particles");
}

// =================================================================================================
template <typename T, MemType M>
void GrainsMPI<T, M>::constructionMPI(DOMElement* /*rootElement*/)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

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

    auto& LC     = GP::m_collisionDetection.linkedCellParameters;
    LC.minCorner = m_decomp->getGhostMin();
    LC.maxCorner = m_decomp->getGhostMax();

    partitionParticlesToLocal();

    if constexpr(M == MemType::DEVICE)
    {
        RigidBodyFactory<T>::copyHostToDevice(Grains<T>::m_rigidBodyList,
                                              GrainsGPU<T>::m_d_rigidBodyList);
        GrainsGPU<T>::m_d_contactForce.reserve(GP::m_numContactPairs);
        ContactForceModelFactory<T>::copyHostToDevice(Grains<T>::m_contactForce,
                                                       GrainsGPU<T>::m_d_contactForce);
        TimeIntegratorFactory<T>::copyHostToDevice(Grains<T>::m_timeIntegrator,
                                                   GrainsGPU<T>::m_d_timeIntegrator);
        GrainsGPU<T>::m_d_components = std::make_unique<ComponentManagerGPU<T>>(
            &GrainsGPU<T>::m_d_rigidBodyList, SS.numObstacles, m_numLocalParticles);
    }
    else
    {
        Grains<T>::m_components = std::make_unique<ComponentManagerCPU<T>>(
            &Grains<T>::m_rigidBodyList, SS.numObstacles, m_numLocalParticles);
    }

    m_ghostExchanger = std::make_unique<GhostExchanger<T, M>>(m_decomp.get());
    m_migrator       = std::make_unique<ParticleMigrator<T, M>>(m_decomp.get());
    m_loadBalancer   = std::make_unique<LoadBalancer<T>>(100, T(0.1), T(1.2));
}

// =================================================================================================
template <typename T, MemType M>
void GrainsMPI<T, M>::partitionParticlesToLocal()
{
    using GP = GrainsParameters<T>;
    auto& SS     = GP::m_simulationState;
    auto& hostCM = Grains<T>::m_components;

    uint numObstacles = SS.numObstacles;
    uint numParticles = SS.numParticles;

    const auto& positions = hostCM->getPosition();

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

    if(m_numLocalParticles == numParticles)
    {
        SS.numParticles = m_numLocalParticles;
        return;
    }

    const auto& hPos  = hostCM->getPosition();
    const auto& hQuat = hostCM->getQuaternion();
    const auto& hVel  = hostCM->getVelocity();

    GrainsMemBuffer<Vector3<T>>    newPos(numObstacles + m_numLocalParticles);
    GrainsMemBuffer<Quaternion<T>> newQuat(numObstacles + m_numLocalParticles);
    GrainsMemBuffer<Kinematics<T>> newVel(numObstacles + m_numLocalParticles);

    for(uint i = 0; i < numObstacles; ++i)
    {
        newPos[i]  = hPos[i];
        newQuat[i] = hQuat[i];
        newVel[i]  = hVel[i];
    }
    for(uint i = 0; i < m_numLocalParticles; ++i)
    {
        uint srcIdx = numObstacles + localIndices[i];
        uint dstIdx = numObstacles + i;
        newPos[dstIdx]  = hPos[srcIdx];
        newQuat[dstIdx] = hQuat[srcIdx];
        newVel[dstIdx]  = hVel[srcIdx];
    }

    hostCM->setPosition(newPos);
    hostCM->setQuaternion(newQuat);
    hostCM->setVelocity(newVel);

    SS.numParticles = m_numLocalParticles;
}

// =================================================================================================
template <typename T, MemType M>
void GrainsMPI<T, M>::simulate()
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& cm = getActiveCM();

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("Starting MPI simulation (", m_mpiSize, "ranks )");
        Gout(std::string(80, '='));
    }

    Grains<T>::m_components->insertParticles(Grains<T>::m_insertion);

    if constexpr(M == MemType::DEVICE)
    {
        Grains<T>::m_components->copyTo(cm);
    }

    if(m_mpiRank == 0)
        std::cout << "\nTime \t TO \tend \tLocal Particles" << std::endl;

    // KDK warmup
    if(GP::m_isLeapFrog)
    {
        m_numGhostParticles = m_ghostExchanger->exchange(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(), SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        cm->detectCollisions();
        cm->computeContactForces(getContactForce());
        cm->addExternalForces();

        GhostExchanger<T, M>::removeGhosts(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(), SS.numObstacles + m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);
    }

    SS.time = GP::m_tStart;
    postProcessParallel(cm);

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

        auto t0 = std::chrono::high_resolution_clock::now();
        stepWithCommunication();
        auto t1 = std::chrono::high_resolution_clock::now();
        double stepTime = std::chrono::duration<double>(t1 - t0).count();

        // Dynamic load balancing
        if(m_loadBalancer->shouldRebalance(stepCount))
        {
            m_loadBalancer->rebalance(*m_decomp, stepTime);
            auto& LC     = GP::m_collisionDetection.linkedCellParameters;
            LC.minCorner = m_decomp->getGhostMin();
            LC.maxCorner = m_decomp->getGhostMax();
        }

        postProcessParallel(cm);
    }

    if constexpr(M == MemType::DEVICE)
        cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
}

// =================================================================================================
template <typename T, MemType M>
void GrainsMPI<T, M>::stepWithCommunication()
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& cm = getActiveCM();

    if(GP::m_isLeapFrog)
    {
        cm->setNumberOfParticles(m_numLocalParticles);
        cm->moveParticles(getTimeIntegrator());

        m_numLocalParticles = m_migrator->migrate(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getTorceBuffer(),
            cm->getRigidBodyIdBuffer(), cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);

        m_numGhostParticles = m_ghostExchanger->exchange(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(), SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        cm->detectCollisions();
        cm->computeContactForces(getContactForce());
        cm->addExternalForces();

        GhostExchanger<T, M>::removeGhosts(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(), SS.numObstacles + m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);

        cm->advanceVelocity(getTimeIntegrator());
    }
    else
    {
        m_numGhostParticles = m_ghostExchanger->exchange(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(), SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        cm->detectCollisions();
        cm->computeContactForces(getContactForce());
        cm->addExternalForces();

        GhostExchanger<T, M>::removeGhosts(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(), SS.numObstacles + m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);

        cm->moveParticles(getTimeIntegrator());

        m_numLocalParticles = m_migrator->migrate(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getTorceBuffer(),
            cm->getRigidBodyIdBuffer(), cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);
    }
}

// =================================================================================================
template <typename T, MemType M>
template <MemType MM>
void GrainsMPI<T, M>::postProcessParallel(
    const std::unique_ptr<ComponentManager<T, MM>>& cm)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    if(GP::m_tSave.empty())
        return;

    if(GP::m_tSave.front() - SS.time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();
        cm->setNumberOfParticles(m_numLocalParticles);

        if constexpr(MM == MemType::DEVICE)
        {
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
template <typename T, MemType M>
void GrainsMPI<T, M>::finalize()
{
    MPI_Barrier(MPI_COMM_WORLD);

    for(auto& pp : Grains<T>::m_postProcessor)
        pp->PostProcessing_end();

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("MPI simulation completed!");
        Gout(std::string(80, '='));
    }
}

// =================================================================================================
template class GrainsMPI<float, MemType::HOST>;
template class GrainsMPI<double, MemType::HOST>;
template class GrainsMPI<float, MemType::DEVICE>;
template class GrainsMPI<double, MemType::DEVICE>;

template void GrainsMPI<float, MemType::HOST>::postProcessParallel<MemType::HOST>(
    const std::unique_ptr<ComponentManager<float, MemType::HOST>>&);
template void GrainsMPI<double, MemType::HOST>::postProcessParallel<MemType::HOST>(
    const std::unique_ptr<ComponentManager<double, MemType::HOST>>&);
template void GrainsMPI<float, MemType::DEVICE>::postProcessParallel<MemType::DEVICE>(
    const std::unique_ptr<ComponentManager<float, MemType::DEVICE>>&);
template void GrainsMPI<double, MemType::DEVICE>::postProcessParallel<MemType::DEVICE>(
    const std::unique_ptr<ComponentManager<double, MemType::DEVICE>>&);

#endif // GRAINS_USE_MPI
