#ifdef GRAINS_USE_MPI

#include "GrainsMultiCPU.hh"
#include "ComponentManagerCPU.hh"

// =================================================================================================
template <typename T>
GrainsMultiCPU<T>::GrainsMultiCPU()
    : GrainsCPU<T>()
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
GrainsMultiCPU<T>::~GrainsMultiCPU()
{
}

// =================================================================================================
template <typename T>
void GrainsMultiCPU<T>::initialize(DOMElement* rootElement)
{
    // Base reads Construction, Forces, AdditionalFeatures on HOST
    Grains<T>::initialize(rootElement);

    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("Multi-CPU DEM:", m_mpiSize, "MPI ranks");
        Gout(std::string(80, '='));
    }

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
    auto& LC     = GP::m_collisionDetection.linkedCellParameters;
    LC.minCorner = m_decomp->getGhostMin();
    LC.maxCorner = m_decomp->getGhostMax();

    // -----------------------------------------------------------------------------------------
    // Partition particles to this rank
    partitionParticlesToLocal();

    // Rebuild the host component manager with local particle count
    Grains<T>::m_components = std::make_unique<ComponentManagerCPU<T>>(
        &Grains<T>::m_rigidBodyList, SS.numObstacles, m_numLocalParticles);

    // Re-initialize transformations for the (now local-only) particles
    // Positions/orientations were already filtered in partitionParticlesToLocal()

    // -----------------------------------------------------------------------------------------
    // Ghost exchanger and migrator
    m_ghostExchanger = std::make_unique<GhostExchangerCPU<T>>(m_decomp.get());
    m_migrator       = std::make_unique<ParticleMigratorCPU<T>>(m_decomp.get());

    Gout("[Rank", m_mpiRank, "] Multi-CPU setup completed,",
         m_numLocalParticles, "local particles");
}

// =================================================================================================
template <typename T>
void GrainsMultiCPU<T>::partitionParticlesToLocal()
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
template <typename T>
void GrainsMultiCPU<T>::simulate()
{
    using G  = Grains<T>;
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& cm = G::m_components;

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("Starting multi-CPU simulation (", m_mpiSize, "ranks )");
        Gout(std::string(80, '='));
    }

    // Insert particles on host
    cm->insertParticles(G::m_insertion);

    if(m_mpiRank == 0)
        std::cout << "\nTime \t TO \tend \tLocal Particles" << std::endl;

    // Pre-compute forces on initial configuration (KDK warmup)
    if(GP::m_isLeapFrog)
    {
        m_numGhostParticles = m_ghostExchanger->exchange(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        cm->detectCollisions();
        cm->computeContactForces(G::m_contactForce);
        cm->addExternalForces();

        GhostExchangerCPU<T>::removeGhosts(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(),
            SS.numObstacles + m_numLocalParticles);
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

        stepWithCommunication();

        postProcessParallel(cm);
    }

    MPI_Barrier(MPI_COMM_WORLD);
}

// =================================================================================================
template <typename T>
void GrainsMultiCPU<T>::stepWithCommunication()
{
    using G  = Grains<T>;
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;
    auto& cm = G::m_components;

    if(GP::m_isLeapFrog)
    {
        // KDK Step 1: half-kick + drift (local only)
        cm->setNumberOfParticles(m_numLocalParticles);
        cm->moveParticles(G::m_timeIntegrator);

        // Migrate particles that left local domain
        m_numLocalParticles = m_migrator->migrate(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getTorceBuffer(),
            cm->getRigidBodyIdBuffer(), cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);

        // Ghost exchange
        m_numGhostParticles = m_ghostExchanger->exchange(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        // Collision detection and forces on local + ghost particles
        cm->detectCollisions();
        cm->computeContactForces(G::m_contactForce);
        cm->addExternalForces();

        // Remove ghosts
        GhostExchangerCPU<T>::removeGhosts(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(),
            SS.numObstacles + m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);

        // KDK Step 3: second half-kick (local only)
        cm->advanceVelocity(G::m_timeIntegrator);
    }
    else
    {
        // Ghost exchange before collision detection
        m_numGhostParticles = m_ghostExchanger->exchange(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles + m_numGhostParticles);

        cm->detectCollisions();
        cm->computeContactForces(G::m_contactForce);
        cm->addExternalForces();

        // Remove ghosts before integration
        GhostExchangerCPU<T>::removeGhosts(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getRigidBodyIdBuffer(),
            cm->getComponentIdBuffer(),
            SS.numObstacles + m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);

        cm->moveParticles(G::m_timeIntegrator);

        // Migrate after integration
        m_numLocalParticles = m_migrator->migrate(
            cm->getPositionBuffer(), cm->getQuaternionBuffer(),
            cm->getVelocityBuffer(), cm->getTorceBuffer(),
            cm->getRigidBodyIdBuffer(), cm->getComponentIdBuffer(),
            SS.numObstacles, m_numLocalParticles);
        cm->setNumberOfParticles(m_numLocalParticles);
    }
}

// =================================================================================================
template <typename T>
template <MemType M>
void GrainsMultiCPU<T>::postProcessParallel(
    const std::unique_ptr<ComponentManager<T, M>>& cm)
{
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    if(GP::m_tSave.empty())
        return;

    if(GP::m_tSave.front() - SS.time < 0.01 * GP::m_dt)
    {
        GP::m_tSave.pop();

        cm->setNumberOfParticles(m_numLocalParticles);

        for(auto& pp : Grains<T>::m_postProcessor)
            pp->PostProcessing(Grains<T>::m_rigidBodyList, cm, SS.time);
    }

    if(!GP::m_tSave.empty() && SS.time > GP::m_tSave.front())
        GP::m_tSave.pop();
}

// =================================================================================================
template <typename T>
void GrainsMultiCPU<T>::finalize()
{
    MPI_Barrier(MPI_COMM_WORLD);

    for(auto& pp : Grains<T>::m_postProcessor)
        pp->PostProcessing_end();

    if(m_mpiRank == 0)
    {
        Gout(std::string(80, '='));
        Gout("Multi-CPU simulation completed!");
        Gout(std::string(80, '='));
    }
}

// =================================================================================================
template class GrainsMultiCPU<float>;
template class GrainsMultiCPU<double>;
template void GrainsMultiCPU<float>::postProcessParallel<MemType::HOST>(
    const std::unique_ptr<ComponentManager<float, MemType::HOST>>&);
template void GrainsMultiCPU<double>::postProcessParallel<MemType::HOST>(
    const std::unique_ptr<ComponentManager<double, MemType::HOST>>&);

#endif // GRAINS_USE_MPI
