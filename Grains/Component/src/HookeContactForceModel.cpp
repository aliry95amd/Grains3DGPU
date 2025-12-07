#include "HookeContactForceModel.hh"
#include "GrainsUtils.hh"
#include "VectorMath.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
__HOSTDEVICE__ HookeContactForceModel<T>::HookeContactForceModel()
{
}

// -------------------------------------------------------------------------------------------------
// Constructor with an XML node
template <typename T>
__HOST__ HookeContactForceModel<T>::HookeContactForceModel(DOMNode* root)
{
    DOMNode* parameter;
    parameter = ReaderXML::getNode(root, "kn");
    GAssert(parameter, "kn not defined! Aborting Grains!");
    m_kn = T(ReaderXML::getNodeValue_Double(parameter));

    parameter = ReaderXML::getNode(root, "en");
    GAssert(parameter, "en not defined! Aborting Grains!");
    m_en   = T(ReaderXML::getNodeValue_Double(parameter));
    m_muen = log(m_en) / sqrt(PI<T> * PI<T> + log(m_en) * log(m_en));

    parameter = ReaderXML::getNode(root, "etat");
    GAssert(parameter, "etat not defined! Aborting Grains!");
    m_etat = T(ReaderXML::getNodeValue_Double(parameter));

    parameter = ReaderXML::getNode(root, "muc");
    GAssert(parameter, "muc not defined! Aborting Grains!");
    m_muc = T(ReaderXML::getNodeValue_Double(parameter));

    parameter = ReaderXML::getNode(root, "kr");
    GAssert(parameter, "kr not defined! Aborting Grains!");
    m_kr = T(ReaderXML::getNodeValue_Double(parameter));
}

// -------------------------------------------------------------------------------------------------
// Constructor with five values as contact parameters
template <typename T>
__HOSTDEVICE__ HookeContactForceModel<T>::HookeContactForceModel(T kn, T en, T etat, T muc, T kr)
    : m_kn(kn)
    , m_en(en)
    , m_etat(etat)
    , m_muc(muc)
    , m_kr(kr)
{
    m_muen = log(m_en) / sqrt(PI<T> * PI<T> + log(m_en) * log(m_en));
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
__HOSTDEVICE__ HookeContactForceModel<T>::~HookeContactForceModel()
{
}

// -------------------------------------------------------------------------------------------------
// Gets the ContactForceModel type
template <typename T>
__HOSTDEVICE__ ContactForceModelType HookeContactForceModel<T>::getContactForceModelType() const
{
    return (HOOKE);
}

// -------------------------------------------------------------------------------------------------
// Gets the parameters of the Hooke contact force model
template <typename T>
__HOSTDEVICE__ void HookeContactForceModel<T>::getContactForceModelParameters(
    T& kn, T& en, T& etat, T& muc, T& kr) const
{
    kn   = m_kn;
    en   = m_en;
    etat = m_etat;
    muc  = m_muc;
    kr   = m_kr;
}

// -------------------------------------------------------------------------------------------------
// Performs forces & torques computation
template <typename T>
__HOSTDEVICE__ void
    HookeContactForceModel<T>::performForcesCalculus(const ContactInfo<T>& contactInfos,
                                                     const Vector3<T>&     relVelocityAtContact,
                                                     const Vector3<T>&     relAngVelocity,
                                                     const T               mA,
                                                     const T               mB,
                                                     Vector3<T>&           delFN,
                                                     Vector3<T>&           delFT,
                                                     Vector3<T>&           delM) const
{
    Vector3<T> geometricPointOfContact = contactInfos.getContactPoint();
    Vector3<T> penetration             = contactInfos.getContactVector();

    // Normal linear elastic force
    // We do this here as we want to modify the penetration vector later
    delFN = m_kn * penetration;

    // Unit normal vector at contact point
    penetration /= norm(penetration);
    round(penetration);

    Vector3<T> v_n = (relVelocityAtContact * penetration) * penetration;
    Vector3<T> v_t = relVelocityAtContact - v_n;

    // Unit tangential vector along relative velocity at contact point
    T          normv_t = norm(v_t);
    Vector3<T> tangent(0, 0, 0);
    if(normv_t > EPS<T>)
        tangent = v_t / normv_t;

    // Normal dissipative force
    T avmass = mA * mB / (mA + mB);
    T omega0 = sqrt(m_kn / avmass);
    if(avmass == T(0))
    {
        avmass = mB == T(0) ? T(0.5) * mA : T(0.5) * mB;
        omega0 = T(2) * sqrt(m_kn / avmass);
    }
    T muen = -omega0 * m_muen;
    delFN += -T(2) * muen * avmass * v_n;
    T normFN = norm(delFN);

    // Tangential dissipative force
    delFT = (-m_etat * T(2) * avmass) * v_t;

    // Tangential Coulomb saturation
    T fn = m_muc * normFN;
    T ft = norm(delFT);
    if(fn < ft)
        delFT = (-fn) * tangent;

    // Rolling resistance moment
    if(m_kr)
    {
        // Relative angular velocity at contact point
        Vector3<T> wn     = (relAngVelocity * penetration) * penetration;
        Vector3<T> wt     = relAngVelocity - wn;
        T          normwt = norm(wt);

        // Anti-spinning effect along the normal wn
        delM = -m_kr * normFN * T(0.001) * wn;

        // Classical rolling resistance moment
        if(normwt > EPS<T>)
            delM -= m_kr * normFN * wt;
    }
}

// -------------------------------------------------------------------------------------------------
// Returns a torce based on the contact information
template <typename T>
__HOSTDEVICE__ void HookeContactForceModel<T>::computeForces(const ContactInfo<T>& contactInfos,
                                                             const Vector3<T>& relVelocityAtContact,
                                                             const Vector3<T>& relAngVelocity,
                                                             const Vector3<T>& vA,
                                                             const Vector3<T>& vB,
                                                             const T           mA,
                                                             const T           mB,
                                                             Torce<T>&         torceA,
                                                             Torce<T>&         torceB) const
{
    // Compute contact force and torque
    Vector3<T> delFN, delFT, delM;
    performForcesCalculus(contactInfos,
                          relVelocityAtContact,
                          relAngVelocity,
                          mA,
                          mB,
                          delFN,
                          delFT,
                          delM);

    const Vector3<T>& geometricPointOfContact = contactInfos.getContactPoint();
    delFN += delFT;
    torceA.addForce(delFN, geometricPointOfContact - vA);
    torceB.addForce(-delFN, geometricPointOfContact - vB);
    if(m_kr)
    {
        torceA.addTorque(delM);
        torceB.addTorque(-delM);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class HookeContactForceModel<float>;
template class HookeContactForceModel<double>;