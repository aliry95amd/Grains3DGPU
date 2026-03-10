#include "ComponentManagerCPU.hh"
#include "ComponentManagerCommon.hh"
#include "Quaternion.hh"
#include "VectorMath.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerCPU<T>::ComponentManagerCPU() = default;

// -------------------------------------------------------------------------------------------------
// Constructor with the number of particles, and obstacles
template <typename T>
ComponentManagerCPU<T>::ComponentManagerCPU(
    GrainsMemBuffer<RigidBody<T>*, MemType::HOST>* rigidBody, uint nObstacles, uint nParticles)
    : ComponentManager<T, MemType::HOST>(rigidBody, nObstacles, nParticles)
{
    this->initialize();
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
ComponentManagerCPU<T>::~ComponentManagerCPU() = default;

// -------------------------------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerCPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI)
{
    // #pragma omp parallel for
    for(uint pID = m_numObstacles; pID < m_numObstacles + m_numParticles; ++pID)
    {
        moveParticles_common(TI.getData(),
                             m_rigidBody->getData(),
                             m_position.getData(),
                             m_quaternion.getData(),
                             m_velocity.getData(),
                             m_torce.getData(),
                             pID);
    }
}

// -------------------------------------------------------------------------------------------------
// Performs the second velocity half-kick (KDK Step 3; no-op for single-pass schemes)
template <typename T>
void ComponentManagerCPU<T>::advanceVelocity(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI)
{
    for(uint pID = m_numObstacles; pID < m_numObstacles + m_numParticles; ++pID)
    {
        advanceVelocity_common(TI.getData(),
                               m_rigidBody->getData(),
                               m_quaternion.getData(),
                               m_velocity.getData(),
                               m_torce.getData(),
                               pID);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerCPU<float>;
template class ComponentManagerCPU<double>;