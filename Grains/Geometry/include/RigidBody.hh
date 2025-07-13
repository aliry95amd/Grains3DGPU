#ifndef _RIGIDBODY_HH_
#define _RIGIDBODY_HH_

#include "Convex.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "ReaderXML.hh"
#include "Torce.hh"

// =============================================================================
/** @brief The class RigidBody.

    Rigid bodies comprising their shapes and physical attributes. The precision
    is managed by two typenames "T" and "U". "T" corresponds to the precision of
    the rigid body, and "U" represents the precision of the bounding volume
    encapsulating the rigid body. We explicitly instantiate three classes out of
    this template; (T, U) = (double, double), (double, float), (float, float).

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
template <typename T>
class RigidBody
{
protected:
    /**@name Parameters */
    //@{
    /** \brief Convex shape */
    Convex<T>* m_convex;
    /** \brief Crust thickness */
    T m_crustThickness;
    /** \brief Scaling vector related to crust thickness */
    Vector3<T> m_scaling;
    /** \brief Material ID */
    uint m_material;
    /** \brief Volume */
    T m_volume;
    /** \brief Mass */
    T m_mass;
    /** \brief Inertia tensor */
    T m_inertia[6];
    /** \brief Inverse of the inertia tensor */
    T m_inertia_1[6];
    //@}

public:
    /**@name Constructeurs */
    //@{
    /** @brief Default constructor */
    __HOSTDEVICE__
    RigidBody();

    /** @brief Constructor with a convex, crust thickness, material, and 
        density
        @param convex convex
        @param ct crust thickness of the rigid body 
        @param material material ID
        @param density density */
    __HOSTDEVICE__
    RigidBody(Convex<T>* convex, T ct, uint material, T density);

    /** @brief Constructor with an XML input
        @param root XML input */
    __HOST__
    RigidBody(DOMNode* root);

    /** @brief Copy constructor
        @param rb RigidBody object to be copied */
    __HOSTDEVICE__
    RigidBody(RigidBody<T> const& rb);

    /** @brief Copy assignment operator
        @param other RigidBody object to be assigned */
    __HOSTDEVICE__
    RigidBody<T>& operator=(const RigidBody<T>& other);

    /** @brief Destructor */
    __HOSTDEVICE__
    ~RigidBody();
    //@}

    /**@name Get methods */
    //@{
    /** @brief Gets the rigid body's convex */
    __HOSTDEVICE__
    Convex<T>* getConvex() const;

    /** @brief Gets the rigid body's crust thickness */
    __HOSTDEVICE__
    T getCrustThickness() const;

    /** @brief Gets the scaling vector related to crust thickness */
    __HOSTDEVICE__
    Vector3<T> getScalingVector() const;

    /** @brief Gets the rigid body's material ID */
    __HOSTDEVICE__
    uint getMaterial() const;

    /** @brief Gets the rigid body's volume */
    __HOSTDEVICE__
    T getVolume() const;

    /** @brief Gets the rigid body's mass */
    __HOSTDEVICE__
    T getMass() const;

    /** @brief Gets the rigid body's inertia
        @param inertia the destination for inertia */
    __HOSTDEVICE__
    void getInertia(T (&inertia)[6]) const;

    /** @brief Gets the inverse of rigid body's inertia
        @param inertia_1 the destination for the inverse inertia */
    __HOSTDEVICE__
    void getInertia_1(T (&inertia_1)[6]) const;

    /** @brief Gets the circumcribed radius of the rigid body */
    __HOSTDEVICE__
    T getCircumscribedRadius() const;
    //@}

    /**@name Methods */
    //@{
    /** @brief Computes the acceleration of the rigid body as a kinematics
        object after imposing a torce (Torque + Force). The assumption is that
        the torce is given in the body-fixed coordinate system, hence, there is
        no need to have the quaternion
        @param omega angular velocity in the body-fixed coordinate system
        @param t imposed torce in the body-fixed coordinate system */
    __HOSTDEVICE__
    Kinematics<T> computeMomentum(const Vector3<T>& omega,
                                  const Torce<T>&   t) const;

    /** @brief Computes the acceleration of the rigid body as a kinematics
        object after imposing a torce (Torque + Force). The assumption is that
        the torce is given in the space-fixed coordinate system.
        @param omega angular velocity in the space-fixed coordinate system
        @param t imposed torce in the space-fixed coordinate system
        @param q quaternion of rotation from space to body coordinate systems */
    __HOSTDEVICE__
    Kinematics<T> computeMomentum(const Vector3<T>&    omega,
                                  const Torce<T>&      t,
                                  const Quaternion<T>& q) const;
    //@}
};

#endif
