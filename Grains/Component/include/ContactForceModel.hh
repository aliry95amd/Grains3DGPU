#ifndef _CONTACTFORCEMODEL_HH_
#define _CONTACTFORCEMODEL_HH_

#include "ContactInfo.hh"
#include "Torce.hh"
#include "Vector3.hh"

// ContactForceModel types
enum ContactForceModelType
{
    HOOKE
};

// =============================================================================
/** @brief The class ContactForceModel.

    Defines the contact forces between two colliding components and computes
    these contact forces.

    @author A.YAZDANI - 2024 - Construction */
// =============================================================================
template <typename T>
class ContactForceModel
{
protected:
    /**@name Contructors */
    //@{
    /** @brief Default constructor (forbidden except in derived classes) */
    __HOSTDEVICE__
    ContactForceModel();

    /** @brief Copy constructor
        @param cf ContactForceModel object to be copied */
    __HOSTDEVICE__
    ContactForceModel(ContactForceModel<T> const& cf);
    //@}

public:
    /**@name Contructors */
    //@{
    /** @brief Destructor */
    __HOSTDEVICE__
    virtual ~ContactForceModel();
    //@}

    /** @name Get methods */
    //@{
    /** @brief Returns the ContactForceModel type */
    __HOSTDEVICE__
    virtual ContactForceModelType getContactForceModelType() const = 0;
    //@}

    /** @name Methods */
    //@{
    /** @brief Returns a torce based on the contact information
        @param contactInfos geometric contact features
        @param relVelocityAtContact relative velocity at the contact point
        @param relAngVelocity relative angular velocity
        @param m1 mass of the first component (Particle)
        @param m2 mass of the second component (Particle or Obstacle)
        @param trOrigin transformation origin
        @param torceA computed force and torque for the first component
        @param torceB computed force and torque for the second component */
    __HOSTDEVICE__
    virtual void computeForces(const ContactInfo<T>& contactInfos,
                               const Vector3<T>&     relVelocityAtContact,
                               const Vector3<T>&     relAngVelocity,
                               T                     m1,
                               T                     m2,
                               const Vector3<T>&     trOrigin,
                               Torce<T>&             torceA,
                               Torce<T>&             torceB) const
        = 0;
    //@}
};

#endif
