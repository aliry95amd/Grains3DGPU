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
    GrainsMemBuffer<RigidBody<T, T>*, MemType::HOST>* particleRB,
    GrainsMemBuffer<RigidBody<T, T>*, MemType::HOST>* obstacleRB,
    uint                                              nParticles,
    uint                                              nObstacles,
    uint                                              nCells)
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
    m_particleCellHash.allocate(m_nParticles);
    m_cell.resize(m_nCells + 1);
}

// -------------------------------------------------------------------------
// Initializes data members to default values
template <typename T>
void ComponentManagerCPU<T>::initialize()
{
}

// -----------------------------------------------------------------------------
// Updates links between components and linked cell
template <typename T>
void ComponentManagerCPU<T>::updateLinks(
    const GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>& LC)
{
    // Reset
    for(int i = 0; i < m_nCells + 1; i++)
        m_cell[i].clear();

    // Updating m_particleCellHash according to the linkedCell. That is,
    // assigning a hash value to each particle based on the cell it belongs to.
    (LC[0])->computeLinearLinkedCellHashCPU(m_transform.getData(),
                                            m_nParticles,
                                            m_particleCellHash.getData());

    // Update cells
    for(int i = 0; i < m_nParticles; i++)
    {
        uint cellId = m_particleCellHash.getData()[i];
        m_cell[cellId].push_back(i);
    }
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between particles and obstacles
template <typename T>
void ComponentManagerCPU<T>::detectCollisionAndComputeContactForcesObstacles(
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    // Loop over all particles
    for(int pId = 0; pId < m_nParticles; pId++)
    {
        // Parameters of the particle
        const RigidBody<T, T>& rbA
            = *(m_particleRB->getData()[m_rigidBodyId[pId]]);
        const Transform3<T>& trA   = m_transform.getData()[pId];
        T                    massA = rbA.getMass();
        uint                 matA  = rbA.getMaterial();

        // Loop over all obstacles
        for(int oId = 0; oId < m_nObstacles; oId++)
        {
            RigidBody<T, T> const& rbB
                = *((m_obstacleRB->getData())[m_obstacleRigidBodyId[oId]]);
            const Transform3<T>& trB = m_obstacleTransform.getData()[oId];
            ContactInfo<T> ci = closestPointsRigidBodies(rbA, rbB, trA, trB);
            if(ci.getOverlapDistance() < T(0))
            {
                // CF ID given materialIDs
                uint contactForceID = ContactForceModelFactory<T>::computeHash(
                    matA,
                    rbB.getMaterial());
                // velocities of the particles
                Kinematics<T> v1(m_velocity[pId]);
                Kinematics<T> v2(m_velocity[oId]);
                // geometric point of contact
                Vector3<T> contactPt(ci.getContactPoint());
                // relative velocity at contact point
                Vector3<T> relVel(v1.kinematicsAtPoint(contactPt)
                                  - v2.kinematicsAtPoint(contactPt));
                // relative angular velocity
                Vector3<T> relAngVel(v1.getAngularComponent()
                                     - v2.getAngularComponent());
                (CF.getData())[contactForceID]->computeForces(
                    ci,
                    relVel,
                    relAngVel,
                    massA,
                    rbB.getMass(),
                    trA.getOrigin(),
                    m_torce.getData()[pId]);
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between particles and particles
template <typename T>
void ComponentManagerCPU<T>::detectCollisionAndComputeContactForcesParticles(
    const GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>&        LC,
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    // Loop over all particles
    // #pragma omp parallel for
    for(int pId = 0; pId < m_nParticles; pId++)
    {
        // Parameters of the primary particle
        const uint             particleId = m_particleId.getData()[pId];
        const uint             cellHash   = m_particleCellHash[pId];
        const RigidBody<T, T>& rbA
            = *(m_particleRB->getData()[m_rigidBodyId[particleId]]);
        const Transform3<T>& trA   = m_transform.getData()[particleId];
        T                    massA = rbA.getMass();
        uint                 matA  = rbA.getMaterial();
        const uint* neighborsList  = (LC.getData()[0])->getNeighbors(cellHash);
        // Loop over all neighboring particles
        for(int i = 0; i < 27; ++i)
        {
            // Get the neighboring cell hash
            uint neighborCellHash = neighborsList[i];
            // Check if the neighboring cell is valid
            if(neighborCellHash == UINT_MAX)
                continue;
            for(auto id : m_cell[neighborCellHash])
            {
                const uint secondaryId = id;
                // To skip self-collision
                if(secondaryId == particleId)
                    continue;
                const RigidBody<T, T>& rbB
                    = *(m_particleRB->getData()[m_rigidBodyId[secondaryId]]);
                const Transform3<T>& trB = m_transform.getData()[secondaryId];
                ContactInfo<T>       ci
                    = closestPointsRigidBodies(rbA, rbB, trA, trB);
                if(ci.getOverlapDistance() < T(0))
                {
                    // CF ID given materialIDs
                    uint contactForceID
                        = ContactForceModelFactory<T>::computeHash(
                            matA,
                            rbB.getMaterial());
                    // velocities of the particles
                    Kinematics<T> v1(CM::m_velocity[particleId]);
                    Kinematics<T> v2(CM::m_velocity[secondaryId]);
                    // geometric point of contact
                    Vector3<T> contactPt(ci.getContactPoint());
                    // relative velocity at contact point
                    Vector3<T> relVel(v1.kinematicsAtPoint(contactPt)
                                      - v2.kinematicsAtPoint(contactPt));
                    // relative angular velocity
                    Vector3<T> relAngVel(v1.getAngularComponent()
                                         - v2.getAngularComponent());
                    (CF.getData())[contactForceID]->computeForces(
                        ci,
                        relVel,
                        relAngVel,
                        massA,
                        rbB.getMass(),
                        trA.getOrigin(),
                        m_torce[particleId]);
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Detects collision and computes forces between all components
template <typename T>
void ComponentManagerCPU<T>::detectCollisionAndComputeContactForces(
    const GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>&        LC,
    const GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    // Updates links between components and linked cell
    updateLinks(LC);

    // Particle-particle interactions
    detectCollisionAndComputeContactForcesParticles(LC, CF);

    // Particle-obstacle interactions
    detectCollisionAndComputeContactForcesObstacles(CF);
}

// -----------------------------------------------------------------------------
// Adds external forces such as gravity
template <typename T>
void ComponentManagerCPU<T>::addExternalForces()
{
    // #pragma omp parallel for
    for(int pId = 0; pId < m_nParticles; pId++)
    {
        addGravity(m_particleRB->getData(),
                   m_rigidBodyId[pId],
                   GrainsParameters<T>::m_gravity,
                   m_torce[pId]);
    }
}

// -----------------------------------------------------------------------------
// Updates the position and velocities of particles
template <typename T>
void ComponentManagerCPU<T>::moveParticles(
    const GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST>& TI)
{
    // #pragma omp parallel for
    for(int pId = 0; pId < m_nParticles; pId++)
    {
        moveParticle(m_particleRB->getData(),
                     TI.getData(),
                     m_transform[pId],
                     m_velocity[pId],
                     m_torce[pId],
                     m_rigidBodyId[pId],
                     pId);
    }
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class ComponentManagerCPU<float>;
template class ComponentManagerCPU<double>;