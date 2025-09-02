#include <limits>

#include "ConvexFactory.hh"
#include "GrainsParameters.hh"
#include "QuaternionMath.hh"
#include "RigidBody.hh"
#include "VectorMath.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
__HOSTDEVICE__ RigidBody<T>::RigidBody()
    : m_convex(NULL)
{
}

// -----------------------------------------------------------------------------
// Constructor with input parameters
template <typename T>
__HOSTDEVICE__
    RigidBody<T>::RigidBody(Convex<T>* convex, T ct, T density, uint material)
    : m_convex(convex)
    , m_crustThickness(ct)
    , m_material(material)
{
    // mass
    m_mass = density * m_convex->computeVolume();
    setInertia();
}

// -----------------------------------------------------------------------------
// Copy constructor
template <typename T>
__HOSTDEVICE__ RigidBody<T>::RigidBody(RigidBody<T> const& rb)
    : m_convex(NULL)
    , m_crustThickness(rb.m_crustThickness)
    , m_mass(rb.m_mass)
    , m_material(rb.m_material)
{
    if(rb.m_convex)
        m_convex = rb.m_convex->clone();
    for(int i = 0; i < 6; ++i)
    {
        m_inertia[i]   = rb.m_inertia[i];
        m_inertia_1[i] = rb.m_inertia_1[i];
    }
}

// -----------------------------------------------------------------------------
// Copy assignment operator
template <typename T>
__HOSTDEVICE__ RigidBody<T>& RigidBody<T>::operator=(const RigidBody<T>& other)
{
    if(this != &other)
    {
        // Free existing resources
        delete m_convex;

        // copy
        m_convex         = other.m_convex ? other.m_convex->clone() : nullptr;
        m_crustThickness = other.m_crustThickness;
        m_mass           = other.m_mass;
        m_material       = other.m_material;
        for(int i = 0; i < 6; ++i)
        {
            m_inertia[i]   = other.m_inertia[i];
            m_inertia_1[i] = other.m_inertia_1[i];
        }
    }
    return *this;
}

// -----------------------------------------------------------------------------
// Move constructor
template <typename T>
__HOSTDEVICE__ RigidBody<T>::RigidBody(RigidBody<T>&& other)
    : m_convex(other.m_convex)
    , m_crustThickness(other.m_crustThickness)
    , m_mass(other.m_mass)
    , m_material(other.m_material)
{
    // Copy arrays
    for(int i = 0; i < 6; ++i)
    {
        m_inertia[i]   = other.m_inertia[i];
        m_inertia_1[i] = other.m_inertia_1[i];
    }

    // Reset moved-from
    other.m_convex         = nullptr;
    other.m_crustThickness = T(0);
    other.m_mass           = T(0);
    other.m_material       = 0u;
    for(int i = 0; i < 6; ++i)
    {
        other.m_inertia[i]   = T(0);
        other.m_inertia_1[i] = T(0);
    }
}

// -----------------------------------------------------------------------------
// Move assignment operator
template <typename T>
__HOSTDEVICE__ RigidBody<T>& RigidBody<T>::operator=(RigidBody<T>&& other)
{
    if(this != &other)
    {
        // Free current resources
        delete m_convex;

        // Move ownership and copy POD fields
        m_convex         = other.m_convex;
        m_crustThickness = other.m_crustThickness;
        m_mass           = other.m_mass;
        m_material       = other.m_material;
        for(int i = 0; i < 6; ++i)
        {
            m_inertia[i]   = other.m_inertia[i];
            m_inertia_1[i] = other.m_inertia_1[i];
        }

        // Reset moved-from
        other.m_convex         = nullptr;
        other.m_crustThickness = T(0);
        other.m_mass           = T(0);
        other.m_material       = 0u;
        for(int i = 0; i < 6; ++i)
        {
            other.m_inertia[i]   = T(0);
            other.m_inertia_1[i] = T(0);
        }
    }
    return *this;
}

// -----------------------------------------------------------------------------
// Constructor with an XML input
template <typename T>
__HOST__ RigidBody<T>::RigidBody(DOMNode* root)
{
    // Convex
    DOMNode* shape = ReaderXML::getNode(root, "Convex");
    m_convex       = ConvexFactory<T>::create(shape);
    // Crust thickenss
    m_crustThickness
        = T(ReaderXML::getNodeAttr_Double(shape, "CrustThickness"));
    // Volume and mass
    T volume  = m_convex->computeVolume();
    T density = T(0);
    if(ReaderXML::hasNodeAttr(root, "Density"))
    {
        density = T(ReaderXML::getNodeAttr_Double(root, "Density"));
        m_mass  = density * volume;
    }
    else
        m_mass = T(0);
    setInertia();
    // Material
    std::string material = ReaderXML::getNodeAttr_String(root, "Material");
    // checking if the material name is already defined.
    // If yes, we access the ID and store it for the rigid body.
    // If it is not, we add the material to the map.
    // Getting the ID of the last material added to the map.
    // This is basically the same as the size of the map.
    if(GrainsParameters<T>::m_materialMap.count(material) == 0)
    {
        uint id = GrainsParameters<T>::m_materialMap.size();
        GrainsParameters<T>::m_materialMap.emplace(material, id);
    }
    m_material = GrainsParameters<T>::m_materialMap[material];
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
__HOSTDEVICE__ RigidBody<T>::~RigidBody()
{
    delete m_convex;
}

// -----------------------------------------------------------------------------
// Gets the rigid body's convex
template <typename T>
__HOSTDEVICE__ Convex<T>* RigidBody<T>::getConvex() const
{
    return (m_convex);
}

// -----------------------------------------------------------------------------
// Gets the rigid body's inertia
template <typename T>
__HOSTDEVICE__ void RigidBody<T>::getInertia(T (&inertia)[6]) const
{
    for(int i = 0; i < 6; ++i)
        inertia[i] = m_inertia[i];
}

// -----------------------------------------------------------------------------
// Gets the inverse of rigid body's inertia
template <typename T>
__HOSTDEVICE__ void RigidBody<T>::getInertia_1(T (&inertia_1)[6]) const
{
    for(int i = 0; i < 6; ++i)
        inertia_1[i] = m_inertia_1[i];
}

// -----------------------------------------------------------------------------
// Gets the rigid body's crust thickness
template <typename T>
__HOSTDEVICE__ T RigidBody<T>::getCrustThickness() const
{
    return (m_crustThickness);
}

// -----------------------------------------------------------------------------
// Gets the rigid body's volume
template <typename T>
__HOSTDEVICE__ T RigidBody<T>::getVolume() const
{
    return (m_convex->computeVolume());
}

// -----------------------------------------------------------------------------
// Gets the rigid body's volume
template <typename T>
__HOSTDEVICE__ T RigidBody<T>::getMass() const
{
    return (m_mass);
}

// -----------------------------------------------------------------------------
// Gets the rigid body's material ID
template <typename T>
__HOSTDEVICE__ uint RigidBody<T>::getMaterial() const
{
    return (m_material);
}

// -----------------------------------------------------------------------------
// Gets the circumcribed radius of the rigid body
template <typename T>
__HOSTDEVICE__ T RigidBody<T>::getCircumscribedRadius() const
{
    return (m_convex->computeCircumscribedRadius());
}

// -----------------------------------------------------------------------------
// Sets the inertia tensor and its inverse
template <typename T>
__HOSTDEVICE__ void RigidBody<T>::setInertia()
{
    if(m_mass == T(0))
    {
        for(int i = 0; i < 6; i++)
        {
            m_inertia[i]   = T(0);
            m_inertia_1[i] = T(0);
        }
    }
    else
    {
        // Storing inertia and inverse of it
        m_convex->computeInertia(m_inertia, m_inertia_1);
        T density = m_mass / getVolume();
        for(int i = 0; i < 6; i++)
        {
            m_inertia[i] *= density;
            m_inertia_1[i] /= density;
        }
    }
}

// -----------------------------------------------------------------------------
// Computes the acceleration of the rigid body given a torce and angular
// velocity in the body-fixed coordinate system -- In the body-fixed coordinate
// system, the moment of inertia tensor is assumed to be diagonal.
template <typename T>
__HOSTDEVICE__ Kinematics<T>
               RigidBody<T>::computeMomentum(const Vector3<T>& omega,
                                  const Torce<T>&   t) const
{
    // Translational momentum
    Vector3<T> transMomentum(t.getForce() / m_mass);

    // Angular momentum
    // Torque
    Vector3<T> angMomentum(t.getTorque());
    // Compute T + (I.w) ^ w in the body-fixed coordinates system
    angMomentum[0] += (m_inertia[3] - m_inertia[5]) * omega[Y] * omega[Z];
    angMomentum[1] += (m_inertia[5] - m_inertia[0]) * omega[X] * omega[Z];
    angMomentum[2] += (m_inertia[0] - m_inertia[3]) * omega[X] * omega[Y];
    // Compute I^-1.(T + w ^ (I.w)) in the body-fixed coordinates system
    angMomentum[0] *= m_inertia_1[0];
    angMomentum[1] *= m_inertia_1[3];
    angMomentum[2] *= m_inertia_1[5];

    return (Kinematics<T>(transMomentum, angMomentum));
}

// -----------------------------------------------------------------------------
// Computes the acceleration of the rigid body given the angular velocity and a
// torce in the space-fixed coordinate system
template <typename T>
__HOSTDEVICE__ Kinematics<T> RigidBody<T>::computeMomentum(
    const Vector3<T>& omega, const Torce<T>& t, const Quaternion<T>& q) const
{
    // Angular momentum
    // Write omega in the body-fixed coordinates system
    Vector3<T> angVelocity = q << omega;
    // Write torque in the body-fixed coordinates system
    Vector3<T> angMomentum = q << t.getTorque();

    // Compute I.w in the body-fixed coordinates system
    Vector3<T> angMomentumTemp(
        m_inertia[0] * angVelocity[0] + m_inertia[1] * angVelocity[1]
            + m_inertia[2] * angVelocity[2],
        m_inertia[1] * angVelocity[0] + m_inertia[3] * angVelocity[1]
            + m_inertia[4] * angVelocity[2],
        m_inertia[2] * angVelocity[0] + m_inertia[4] * angVelocity[1]
            + m_inertia[5] * angVelocity[2]);

    // Compute T + I.w ^ w in the body-fixed coordinates system
    angMomentum += angMomentumTemp ^ angVelocity;

    // Compute I^-1.(T + I.w ^ w) in body-fixed coordinates system
    angMomentumTemp[0] = m_inertia_1[0] * angMomentum[0]
                         + m_inertia_1[1] * angMomentum[1]
                         + m_inertia_1[2] * angMomentum[2];
    angMomentumTemp[1] = m_inertia_1[1] * angMomentum[0]
                         + m_inertia_1[3] * angMomentum[1]
                         + m_inertia_1[4] * angMomentum[2];
    angMomentumTemp[2] = m_inertia_1[2] * angMomentum[0]
                         + m_inertia_1[4] * angMomentum[1]
                         + m_inertia_1[5] * angMomentum[2];
    // Write I^-1.(T + I.w ^ w) in space-fixed coordinates system
    angMomentum = q >> angMomentumTemp;

    // Translational momentum
    Vector3<T> transMomentum(t.getForce() / m_mass);

    return (Kinematics<T>(transMomentum, angMomentum));
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class RigidBody<float>;
template class RigidBody<double>;