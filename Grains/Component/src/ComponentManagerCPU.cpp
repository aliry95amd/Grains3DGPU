#include "ComponentManagerCPU.hh"
#include "ComponentManagerCommon.hh"
#include "Quaternion.hh"
#include "VectorMath.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
ComponentManagerCPU<T>::ComponentManagerCPU() = default;

// -----------------------------------------------------------------------------
// Constructor with the number of particles, and obstacles
template <typename T>
ComponentManagerCPU<T>::ComponentManagerCPU(
    GrainsMemBuffer<RigidBody<T>*, MemType::HOST>* rigidBody,
    uint                                           nObstacles,
    uint                                           nParticles)
    : ComponentManager<T, MemType::HOST>(rigidBody, nObstacles, nParticles)
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
}

// -------------------------------------------------------------------------
// Initializes data members to default values
template <typename T>
void ComponentManagerCPU<T>::initialize()
{
}

// -----------------------------------------------------------------------------
// Updates neighbor list if needed
template <typename T>
void ComponentManagerCPU<T>::updateNeighborList()
{
    if(m_neighborList->needsUpdate())
    {
        m_neighborList->updateNeighborList(m_position,
                                           m_nObstacles,
                                           m_nParticles);

        // Resize pair-dependent buffers to match actual number of pairs
        this->resizePairBuffers();
    }
}

// -----------------------------------------------------------------------------
// Computes relative transformations
template <typename T>
void ComponentManagerCPU<T>::computeRelativeTransformations()
{
    uint nPairs = m_neighborList->getSize();
    for(uint cID = 0; cID < nPairs; ++cID)
    {
        computeRelativeTransformations_common(m_neighborList->getData(),
                                              m_position.getData(),
                                              m_quaternion.getData(),
                                              m_relPosition.getData(),
                                              m_relQuaternion.getData(),
                                              cID);
    }
}

// -----------------------------------------------------------------------------
// Detects collisions
template <typename T>
void ComponentManagerCPU<T>::detectCollisionsComponents()
{
    uint nPairs = m_neighborList->getSize();
    for(uint i = 0; i < nPairs; ++i)
    {
        detectCollisionsComponents_common(m_neighborList->getData(),
                                          m_rigidBody->getData(),
                                          m_relPosition.getData(),
                                          m_relQuaternion.getData(),
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

    // Interactions
    detectCollisionsComponents();
}

// -----------------------------------------------------------------------------
// Computes contact forces
template <typename T>
void ComponentManagerCPU<T>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    uint nPairs = m_neighborList->getSize();
    for(uint i = 0; i < nPairs; ++i)
    {
        computeContactForces_common(CF.getData(),
                                    m_neighborList->getData(),
                                    m_contactInfo.getData(),
                                    m_rigidBody->getData(),
                                    m_velocity.getData(),
                                    m_torce.getData(),
                                    m_relPosition.getData(),
                                    i);
    }

    m_position.print("Position");
    m_quaternion.print("Orientation");
    m_relPosition.print("Relative Position");
    m_relQuaternion.print("Relative Quaternion");
    m_contactInfo.print("Contact Information");
    m_torce.print("Torce");
}

// -----------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerCPU<T>::addExternalForces()
{
    // #pragma omp parallel for
    for(uint pID = m_nObstacles; pID < m_nObstacles + m_nParticles; ++pID)
    {
        // Only add to the particles
        addExternalForces_common(GrainsParameters<T>::m_gravity,
                                 m_rigidBody->getData(),
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
    for(uint pID = m_nObstacles; pID < m_nObstacles + m_nParticles; ++pID)
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

// -----------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerCPU<float>;
template class ComponentManagerCPU<double>;