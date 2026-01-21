#ifndef _RIGIDBODY_HH_
#define _RIGIDBODY_HH_

#include <limits>

#include "BitPacker.hh"
#include "Convex.hh"
#include "Kinematics.hh"
#include "Quaternion.hh"
#include "ReaderXML.hh"
#include "Torce.hh"

// =================================================================================================
/** @brief The class RigidBody.

    Rigid bodies comprising their shapes and physical attributes.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
template <typename T>
class RigidBody
{
public:
    /** @name BitPacker constants */
    //@{
    /** \brief Number of bits for each field in the packed properties */
    static constexpr int B_MAT   = 4;
    static constexpr int B_MASS  = 30;
    static constexpr int B_CRUST = 30;
    static_assert(B_MAT + B_MASS + B_CRUST == 64, "Property bits must sum to 64");

    /** \brief Default minimum and maximum values for mass and crust thickness */
    static constexpr T DEFAULT_CRUST_MIN = HIGHEPS<T>;
    static constexpr T DEFAULT_CRUST_MAX = T(1);
    static constexpr T DEFAULT_MASS_MIN  = EPS<T>;
    // Finite sentinel to represent unspecified/obstacle masses without producing inf.
    static constexpr T DEFAULT_MASS_MAX = T(1e3);
    //@}

protected:
    /** @name Parameters */
    //@{
    /** \brief Convex shape */
    Convex<T>* m_convex;
    /** \brief Inertia tensor */
    T m_inertia[6];
    /** \brief Inverse of the inertia tensor */
    T m_inertia_1[6];
    /** \brief Packed properties:
        BitPacker field order is LSB->MSB: material(4), mass(30), crust(30) */
    BitPacker<uint64_t, B_MAT, B_MASS, B_CRUST> m_properties;
    //@}

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor */
    __HOSTDEVICE__
    RigidBody();

    /** @brief Constructor that accepts pre-packed properties produced by BitPacker.
        This is intended for fast device-side construction when the packed properties
        are already available on the caller side. */
    __HOSTDEVICE__
    RigidBody(Convex<T>* convex, uint64_t propertiesRaw);

    /** @brief Constructor with a convex, crust thickness, material, and density.
        @param convex convex
        @param ct crust thickness of the rigid body
        @param density density
        @param material material ID */
    __HOSTDEVICE__
    RigidBody(Convex<T>* convex, T ct, T density, uint material);

    /** @brief Copy constructor
        @param rb RigidBody object to be copied */
    __HOSTDEVICE__
    RigidBody(RigidBody<T> const& rb);

    /** @brief Copy assignment operator
        @param other RigidBody object to be assigned */
    __HOSTDEVICE__
    RigidBody<T>& operator=(const RigidBody<T>& other);

    /** @brief Move constructor
          @param other RigidBody object to be moved */
    __HOSTDEVICE__
    RigidBody(RigidBody<T>&& other);

    /** @brief Move assignment operator
        @param other RigidBody object to be moved */
    __HOSTDEVICE__
    RigidBody<T>& operator=(RigidBody<T>&& other);

    /** @brief Constructor with an XML input
        @param root XML input */
    __HOST__
    RigidBody(DOMNode* root);

    /** @brief Destructor */
    __HOSTDEVICE__
    ~RigidBody();
    //@}

    /** @name Get methods */
    //@{
    /** @brief Gets the rigid body's convex */
    __HOSTDEVICE__
    Convex<T>* getConvex() const;

    /** @brief Gets the rigid body's inertia
        @param inertia the destination for inertia */
    __HOSTDEVICE__
    void getInertia(T (&inertia)[6]) const;

    /** @brief Gets the inverse of rigid body's inertia
        @param inertia_1 the destination for the inverse inertia */
    __HOSTDEVICE__
    void getInertia_1(T (&inertia_1)[6]) const;

    /** @brief Gets the packed properties */
    __HOSTDEVICE__
    const BitPacker<uint64_t, B_MAT, B_MASS, B_CRUST>& getProperties() const;

    /** @brief Gets the packed properties as raw uint64_t */
    __HOSTDEVICE__
    uint64_t getPropertiesRaw() const;

    /** @brief Gets the rigid body's crust thickness */
    __HOSTDEVICE__
    T getCrustThickness() const;

    /** @brief Gets the rigid body's mass */
    __HOSTDEVICE__
    T getMass() const;

    /** @brief Gets the rigid body's material ID */
    __HOSTDEVICE__
    uint getMaterial() const;

    /** @brief Gets the rigid body's volume */
    __HOSTDEVICE__
    T getVolume() const;

    /** @brief Gets the circumcribed radius of the rigid body */
    __HOSTDEVICE__
    T getCircumscribedRadius() const;
    //@}

    /**@name Set methods */
    //@{
    /** @brief Sets the rigid body's inertia and its inverse */
    __HOSTDEVICE__
    void setInertia();

    /** @brief Sets the packed properties directly
        @param p packed properties */
    __HOSTDEVICE__
    void setProperties(BitPacker<uint64_t, B_MAT, B_MASS, B_CRUST> p);

    /** @brief Sets the packed properties from raw uint64_t
        @param p packed properties as uint64_t */
    __HOSTDEVICE__
    void setProperties(uint64_t p);
    //@}

    /**@name Methods */
    //@{
    /** @brief Computes the acceleration of the rigid body as a kinematics object after imposing a
        torce (Torque + Force). The assumption is that the torce is given in the body-fixed
        coordinate system, hence, there is no need to have the quaternion.
        @param omega angular velocity in the body-fixed coordinate system
        @param t imposed torce in the body-fixed coordinate system */
    __HOSTDEVICE__
    Kinematics<T> computeMomentum(const Vector3<T>& omega, const Torce<T>& t) const;

    /** @brief Computes the acceleration of the rigid body as a kinematics object after imposing a
        torce (Torque + Force). The assumption is that the torce is given in the space-fixed
        coordinate system.
        @param omega angular velocity in the space-fixed coordinate system
        @param t imposed torce in the space-fixed coordinate system
        @param q quaternion of rotation from space to body coordinate systems */
    __HOSTDEVICE__
    Kinematics<T>
        computeMomentum(const Vector3<T>& omega, const Torce<T>& t, const Quaternion<T>& q) const;
    //@}
};

#endif
