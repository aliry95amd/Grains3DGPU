// #include <curand.h>
#include "ComponentManagerGPU.hh"
#include "ComponentManagerGPU_Kernels.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU() = default;

// -------------------------------------------------------------------------------------------------
// Constructor with the number of particles, and obstacles
template <typename T>
ComponentManagerGPU<T>::ComponentManagerGPU(
    GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>* rigidBody, uint nObstacles, uint nParticles)
    : ComponentManager<T, MemType::DEVICE>(rigidBody, nObstacles, nParticles)
{
    this->initialize();
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
ComponentManagerGPU<T>::~ComponentManagerGPU() = default;

// -------------------------------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerGPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI)
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_numParticles,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    moveParticles_Kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                    m_rigidBody->getData(),
                                                    m_position.getData(),
                                                    m_quaternion.getData(),
                                                    m_velocity.getData(),
                                                    m_torce.getData(),
                                                    m_numObstacles,
                                                    m_numParticles);
}

// -------------------------------------------------------------------------------------------------
// Performs the second velocity half-kick (KDK Step 3; no-op for single-pass schemes)
template <typename T>
void ComponentManagerGPU<T>::advanceVelocity(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::DEVICE>& TI)
{
    uint numThreads, numBlocks;
    computeOptimalThreadsAndBlocks(m_numParticles,
                                   GrainsParameters<T>::m_GPU,
                                   numBlocks,
                                   numThreads);

    advanceVelocity_Kernel<<<numBlocks, numThreads>>>(TI.getData(),
                                                      m_rigidBody->getData(),
                                                      m_quaternion.getData(),
                                                      m_velocity.getData(),
                                                      m_torce.getData(),
                                                      m_numObstacles,
                                                      m_numParticles);
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerGPU<float>;
template class ComponentManagerGPU<double>;