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
    // time marching
    for(SS.time = GP::m_tStart; SS.time <= GP::m_tEnd; SS.time += GP::m_dt)
    {
        // Output time
        ostringstream oss;
        oss.width(10);
        oss << left << SS.time;
        std::cout << '\r' << oss.str() << "  \t" << GP::m_tEnd << std::flush;

        G::m_components->detectCollisions();
        G::m_components->computeContactForces(G::m_contactForce);
        G::m_components->addExternalForces();
        G::m_components->moveParticles(G::m_timeIntegrator);

        // Post-Processing
        G::postProcess(G::m_components);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class GrainsCPU<float>;
template class GrainsCPU<double>;