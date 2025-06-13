#include "ComponentManagerCPU.hh"
#include "ComponentManagerCommon.hh"
#include "Quaternion.hh"
#include "VectorMath.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerCPU<T>::ComponentManagerCPU() = default;

// -----------------------------------------------------------------------------
// Constructor with the number of particles, number of obstacles, and number of
// cells with all other data members initialized as default.
template <typename T>
ComponentManagerCPU<T>::ComponentManagerCPU(
    GrainsMemBuffer<RigidBody<T>*, MemType::HOST>* particleRB,
    GrainsMemBuffer<RigidBody<T>*, MemType::HOST>* obstacleRB,
    uint                                           nParticles,
    uint                                           nObstacles,
    uint                                           nCells)
    : ComponentManager<T, MemType::HOST>(
          particleRB, obstacleRB, nParticles, nObstacles, nCells)
{
    allocate();
    initialize();
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
ComponentManagerCPU<T>::~ComponentManagerCPU() = default;

// -----------------------------------------------------------------------------
// Allocates memory for the component manager
template <typename T>
void ComponentManagerCPU<T>::allocate()
{
    m_particleCellHash.reserve(m_nParticles);
    m_cell.resize(m_nCells + 1);
}

// -------------------------------------------------------------------------
// Initializes data members to default values
template <typename T>
void ComponentManagerCPU<T>::initialize()
{
    m_neighborList->createNeighborList(m_transform.getData(), m_nParticles);
}

// -----------------------------------------------------------------------------
// Updates links between particles and linked cell
template <typename T>
void ComponentManagerCPU<T>::updateNeighborList()
{
    if(m_neighborList->needsUpdate(m_transform.getData(), m_nParticles) == true)
        m_neighborList->updateNeighborList(m_transform.getData(), m_nParticles);
}

// -----------------------------------------------------------------------------
// Computes relative transformations
template <typename T>
void ComponentManagerCPU<T>::computeRelativeTransformations()
{
    for(uint pID = 0; pID < m_nPairs; ++pID)
    {
        computeRelativeTransformations_common(m_neighborList->getData(),
                                              m_transform.getData(),
                                              m_relTransform.getData(),
                                              pID);
    }
}

// -----------------------------------------------------------------------------
// Detects collision between particles and obstacles
template <typename T>
void ComponentManagerCPU<T>::detectCollisionsObstacles()
{
}

// -----------------------------------------------------------------------------
// Detects collisions between particles and particles
template <typename T>
void ComponentManagerCPU<T>::detectCollisionsParticles()
{
    for(uint i = 0; i < m_nPairs; ++i)
    {
        detectCollisionsParticles_common(m_neighborList->getData(),
                                         m_particleRB->getData(),
                                         m_relTransform.getData(),
                                         m_contactInfo.getData(),
                                         i);
    }
}

// -----------------------------------------------------------------------------
// Detects collision between all components
template <typename T>
void ComponentManagerCPU<T>::detectCollisions()
{
    // Updates links between components and linked cell
    updateNeighborList();

    // Computes the relative transformations
    computeRelativeTransformations();

    // Particle-particle interactions
    detectCollisionsParticles();

    // Particle-obstacle interactions
    detectCollisionsObstacles();
}

// -----------------------------------------------------------------------------
// Computes contact forces
template <typename T>
void ComponentManagerCPU<T>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    // Invoke the kernel
    // computeContactForces_Kernel<<<numBlocks, numThreads>>>(CF,
    //                                                        pairList,
    //                                                        contactInfo,
    //                                                        particleRB,
    //                                                        velocity,
    //                                                        torce,
    //                                                        transform,
    //                                                        m_nParticles);
}

// -----------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerCPU<T>::addExternalForces()
{
    // #pragma omp parallel for
    for(uint pID = 0; pID < m_nParticles; ++pID)
    {
        addExternalForces_common(GrainsParameters<T>::m_gravity,
                                 m_particleRB->getData(),
                                 m_torce.getData(),
                                 pID);
    }
}

// -----------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerCPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI)
{
    // #pragma omp parallel for
    for(uint pID = 0; pID < m_nParticles; ++pID)
    {
        moveParticles_common(TI.getData(),
                             m_particleRB->getData(),
                             m_transform.getData(),
                             m_velocity.getData(),
                             m_torce.getData(),
                             m_rigidBodyId.getData(),
                             pID);
    }
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerCPU<float>;
template class ComponentManagerCPU<double>;