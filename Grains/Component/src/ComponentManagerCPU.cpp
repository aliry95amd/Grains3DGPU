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
// Updates neighbor list if needed
template <typename T>
void ComponentManagerCPU<T>::updateNeighborList()
{
    // Static reference to simulation state
    auto& SS = GrainsParameters<T>::m_simulationState;

    bool updated = m_neighborList->updateNeighborList(m_position, m_numObstacles, m_numParticles);
    if(updated)
    {
        // Resize pair-dependent buffers to match actual number of pairs
        this->resizePairBuffers(m_neighborList->getSize());
        // Increment update counter and sort if needed
        SS.neighborListUpdateCount++;
    }
}

// -------------------------------------------------------------------------------------------------
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

// -------------------------------------------------------------------------------------------------
// Detects collisions
template <typename T>
// template <GJKType GJKVARIANT, bool GJKACC>
void ComponentManagerCPU<T>::detectCollisionsComponents()
{
    uint nPairs = m_neighborList->getSize();
    for(uint i = 0; i < nPairs; ++i)
    {
        detectCollisionsComponents_common<T>(m_neighborList->getData(),
                                             m_rigidBody->getData(),
                                             m_relPosition.getData(),
                                             m_relQuaternion.getData(),
                                             m_contactInfo.getData(),
                                             i);
    }
}

// -------------------------------------------------------------------------------------------------
// Transforms contact information to world frame
template <typename T>
void ComponentManagerCPU<T>::transformContactInfoToWorld()
{
    uint nPairs = m_neighborList->getSize();
    for(uint i = 0; i < nPairs; ++i)
    {
        transformContactInfo_common(m_neighborList->getData(),
                                    m_position.getData(),
                                    m_quaternion.getData(),
                                    m_contactInfo.getData(),
                                    m_contactInfoWorld.getData(),
                                    i);
    }
}

// -------------------------------------------------------------------------------------------------
// Detects collision between all components
template <typename T>
void ComponentManagerCPU<T>::detectCollisions()
{
    // Perform contact table cleanup periodically
    this->cleanupContactTable();

    // Sorts particles by Morton codes for improved cache efficiency
    this->sortParticles();

    // Updates links between components and linked cell
    updateNeighborList();

    // Computes the relative transformations
    computeRelativeTransformations();

    // Interactions
    detectCollisionsComponents();

    // Transforms contact info to world frame
    transformContactInfoToWorld();

    m_contactInfoWorld.print();
}

// -------------------------------------------------------------------------------------------------
// Computes contact forces
template <typename T>
void ComponentManagerCPU<T>::computeContactForces(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    uint nPairs = m_neighborList->getSize();

    // Get the contact memory view
    ContactMemoryView<T> contactMemory = this->getContactMemoryView();

    for(uint i = 0; i < nPairs; ++i)
    {
        computeContactForces_common(CF.getData(),
                                    m_neighborList->getData(),
                                    m_contactInfoWorld.getData(),
                                    m_position.getData(),
                                    m_velocity.getData(),
                                    m_torce.getData(),
                                    contactMemory,
                                    i);
    }
    m_torce.print();
}

// -------------------------------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerCPU<T>::addExternalForces()
{
    // #pragma omp parallel for
    for(uint pID = m_numObstacles; pID < m_numObstacles + m_numParticles; ++pID)
    {
        // Only add to the particles
        addExternalForces_common(GrainsParameters<T>::m_gravity,
                                 m_rigidBody->getData(),
                                 m_torce.getData(),
                                 pID);
    }
}

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
// Explicit instantiation
template class ComponentManagerCPU<float>;
template class ComponentManagerCPU<double>;