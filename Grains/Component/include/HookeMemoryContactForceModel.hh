#ifndef _HOOKEMEMORYCONTACTFORCEMODEL_HH_
#define _HOOKEMEMORYCONTACTFORCEMODEL_HH_

#include "ContactForceModel.hh"
#include "ReaderXML.hh"

// =================================================================================================
/** @brief The class HookeMemoryContactForceModel.

    Contact force model involving a Hookean spring and a Dashpot in both the normal and tangential
    directions with tangential history (memory), as well as rolling resistance with memory, and a
    tangential Coulomb friction to compute the force and torque induced by the contact between two
    rigid components.

    This model includes tangential spring effects (kt) and rolling friction (mur) with full memory
    tracking of cumulative displacements.

    @author A.Yazdani - 2025 - Construction (Inspired by D.Huet and A.Wachs) */
// =================================================================================================
template <typename T>
class HookeMemoryContactForceModel : public ContactForceModel<T>
{
private:
    /** @name Parameters */
    //@{
    /** \brief Normal stiffness coefficient */
    T m_kn;
    /** \brief Normal restitution coefficient */
    T m_en;
    /** \brief log(m_en) / sqrt( PI * PI + log(m_en) * log(m_en) ) */
    T m_muen;
    /** \brief Tangential damping coefficient */
    T m_etat;
    /** \brief Tangential stiffness coefficient */
    T m_kt;
    /** \brief Tangential Coulomb friction coefficient */
    T m_muc;
    /** \brief Rolling resistance coefficient */
    T m_mur = T(0);
    /** \brief Rolling friction prefactor */
    T m_etarpf = T(0);
    //@}

public:
    /**@name Constructors */
    //@{
    /** @brief Default constructor */
    __HOSTDEVICE__
    HookeMemoryContactForceModel();

    /** @brief Constructor with an XML node
        @param root XML node */
    __HOST__
    HookeMemoryContactForceModel(DOMNode* root);

    /** @brief Constructor with eight values as contact parameters
        @param kn normal stiffness coefficient
        @param en normal restitution coefficient
        @param kt tangential stiffness coefficient
        @param etat tangential damping coefficient
        @param muc tangential Coulomb friction coefficient
        @param mur rolling resistance coefficient
        @param etarpf rolling friction prefactor */
    __HOSTDEVICE__
    HookeMemoryContactForceModel(T kn, T en, T kt, T etat, T muc, T mur = T(0), T etarpf = T(0));

    /** @brief Destructor */
    __HOSTDEVICE__
    ~HookeMemoryContactForceModel();
    //@}

    /** @name Get methods */
    //@{
    /** @brief Gets the ContactForceModel type */
    __HOSTDEVICE__
    ContactForceModelType getContactForceModelType() const final;

    /** @brief Gets the parameters of the HookeMemory contact force model
        @param kn normal stiffness coefficient
        @param en normal restitution coefficient
        @param kt tangential stiffness coefficient
        @param etat tangential damping coefficient
        @param muc tangential Coulomb friction coefficient
        @param mur rolling resistance coefficient
        @param etarpf rolling friction prefactor */
    __HOSTDEVICE__
    void getContactForceModelParameters(
        T& kn, T& en, T& kt, T& etat, T& muc, T& mur, T& etarpf) const;
    //@}

    /**@name Methods */
    //@{
    /** @brief Returns a force based on the contact information
        @param contactVector geometric contact vector
        @param relVelocityAtContact relative velocity at the contact point
        @param relAngVelocity relative angular velocity
        @param overlapDistance overlap distance
        @param averageMass average mass of the two components
        @param delFN normal force
        @param delFT tangential force
        @param delM torque */
    __HOSTDEVICE__
    void performForcesCalculus(const Vector3<T>& contactVector,
                               const Vector3<T>& relVelocityAtContact,
                               const Vector3<T>& relAngVelocity,
                               const T           overlapDistance,
                               const T           averageMass,
                               Vector3<T>&       delFN,
                               Vector3<T>&       delFT,
                               Vector3<T>&       delM) const;

    /** @brief Returns a force based on the contact information
        @param contactInfos geometric contact features
        @param relVelocityAtContact relative velocity at the contact point
        @param relAngVelocity relative angular velocity
        @param vA position of the first component
        @param vB position of the second component
        @param averageMass average mass of the two components
        @param torceA computed force and torque for the first component
        @param torceB computed force and torque for the second component */
    __HOSTDEVICE__
    void computeForces(const ContactInfo<T>& contactInfos,
                       const Vector3<T>&     relVelocityAtContact,
                       const Vector3<T>&     relAngVelocity,
                       const Vector3<T>&     vA,
                       const Vector3<T>&     vB,
                       const T               averageMass,
                       Torce<T>&             torceA,
                       Torce<T>&             torceB) const final;
    //@}
};

#endif
