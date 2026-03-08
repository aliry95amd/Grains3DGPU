#include "GrainsCPU.hh"
#include "Grains.hh"
#include "GrainsParameters.hh"
#include "VectorMath.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
GrainsCPU<T>::GrainsCPU()
{
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
GrainsCPU<T>::~GrainsCPU()
{
}

// -------------------------------------------------------------------------------------------------
// Runs the simulation over the prescribed time interval
template <typename T>
void GrainsCPU<T>::simulate()
{
    using G  = Grains<T>;
    using GP = GrainsParameters<T>;
    auto& SS = GP::m_simulationState;

    Gout(std::string(80, '='));
    Gout("Starting the simulation on CPU");
    Gout(std::string(80, '='));

    // first, inserting particles on host
    G::m_components->insertParticles(G::m_insertion);

    cout << "Time \t TO \tend \tParticles \tIn \tOut" << endl;

    // Pre-compute forces on the initial configuration so the first
    // moveParticles call (and the KDK first half-kick) has valid accelerations.
    if(GP::m_isLeapFrog)
    {
        G::m_components->detectCollisions();
        G::m_components->computeContactForces(G::m_contactForce);
        G::m_components->addExternalForces();
    }

    // Write initial state (t = tStart) before advancing
    SS.time = GP::m_tStart;
    G::postProcess(G::m_components);

    // time marching
    uint stepCount = 0;
    for(SS.time = GP::m_tStart + GP::m_dt; SS.time <= GP::m_tEnd; SS.time += GP::m_dt)
    {
        stepCount++;
        // Output time
        if(GP::m_verbosityFrequency > 0 && (stepCount % GP::m_verbosityFrequency == 0))
        {
            ostringstream oss;
            oss.width(10);
            oss << left << SS.time;
            std::cout << oss.str() << "  \t" << GP::m_tEnd << std::endl;
        }

        if(GP::m_isLeapFrog)
        {
            // KDK Step 1: half-kick + drift using f_n (from pre-loop or previous step).
            G::m_components->moveParticles(G::m_timeIntegrator);
            // Detect collisions and compute forces at x_{n+1}.
            G::m_components->detectCollisions();
            G::m_components->computeContactForces(G::m_contactForce);
            G::m_components->addExternalForces();
            // KDK Step 3: second half-kick using f_{n+1}.
            G::m_components->advanceVelocity(G::m_timeIntegrator);
        }
        else
        {
            // Single-pass scheme: compute forces at x_n, then advance.
            G::m_components->detectCollisions();
            G::m_components->computeContactForces(G::m_contactForce);
            G::m_components->addExternalForces();
            G::m_components->moveParticles(G::m_timeIntegrator);
        }

        // Post-Processing
        G::postProcess(G::m_components);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class GrainsCPU<float>;
template class GrainsCPU<double>;