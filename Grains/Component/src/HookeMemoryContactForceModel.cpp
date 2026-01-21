#include "HookeMemoryContactForceModel.hh"
#include "GrainsUtils.hh"
#include "VectorMath.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
__HOSTDEVICE__ HookeMemoryContactForceModel<T>::HookeMemoryContactForceModel()
{
}

// -------------------------------------------------------------------------------------------------
// Constructor with an XML node
template <typename T>
__HOST__ HookeMemoryContactForceModel<T>::HookeMemoryContactForceModel(DOMNode* root)
{
    GAssert(ReaderXML::hasNodeAttr(root, "kn"), "kn not defined! Aborting Grains!");
    GAssert(ReaderXML::hasNodeAttr(root, "en"), "en not defined! Aborting Grains!");
    GAssert(ReaderXML::hasNodeAttr(root, "kt"), "kt not defined! Aborting Grains!");
    GAssert(ReaderXML::hasNodeAttr(root, "etat"), "etat not defined! Aborting Grains!");
    GAssert(ReaderXML::hasNodeAttr(root, "muc"), "muc not defined! Aborting Grains!");

    m_kn   = T(ReaderXML::getNodeAttr_Double(root, "kn"));
    m_en   = T(ReaderXML::getNodeAttr_Double(root, "en"));
    m_kt   = T(ReaderXML::getNodeAttr_Double(root, "kt"));
    m_etat = T(ReaderXML::getNodeAttr_Double(root, "etat"));
    m_muc  = T(ReaderXML::getNodeAttr_Double(root, "muc"));
    if(ReaderXML::hasNodeAttr(root, "mur"))
        m_mur = T(ReaderXML::getNodeAttr_Double(root, "mur"));
    if(ReaderXML::hasNodeAttr(root, "etarpf"))
        m_etarpf = T(ReaderXML::getNodeAttr_Double(root, "etarpf"));

    m_muen = log(m_en) / sqrt(PI<T> * PI<T> + log(m_en) * log(m_en));
}

// -------------------------------------------------------------------------------------------------
// Constructor with eight values as contact parameters
template <typename T>
__HOSTDEVICE__ HookeMemoryContactForceModel<T>::HookeMemoryContactForceModel(
    T kn, T en, T kt, T etat, T muc, T mur, T etarpf)
    : m_kn(kn)
    , m_en(en)
    , m_kt(kt)
    , m_etat(etat)
    , m_muc(muc)
    , m_mur(mur)
    , m_etarpf(etarpf)
{
    m_muen = log(m_en) / sqrt(PI<T> * PI<T> + log(m_en) * log(m_en));
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
__HOSTDEVICE__ HookeMemoryContactForceModel<T>::~HookeMemoryContactForceModel()
{
}

// -------------------------------------------------------------------------------------------------
// Gets the ContactForceModel type
template <typename T>
__HOSTDEVICE__ ContactForceModelType
    HookeMemoryContactForceModel<T>::getContactForceModelType() const
{
    return (HOOKEMEMORY);
}

// -------------------------------------------------------------------------------------------------
// Gets the parameters of the HookeMemory contact force model
template <typename T>
__HOSTDEVICE__ void HookeMemoryContactForceModel<T>::getContactForceModelParameters(
    T& kn, T& en, T& kt, T& etat, T& muc, T& mur, T& etarpf) const
{
    kn     = m_kn;
    en     = m_en;
    kt     = m_kt;
    etat   = m_etat;
    muc    = m_muc;
    mur    = m_mur;
    etarpf = m_etarpf;
}

// -------------------------------------------------------------------------------------------------
// Performs forces & torques computation
template <typename T>
__HOSTDEVICE__ void
    HookeMemoryContactForceModel<T>::performForcesCalculus(const Vector3<T>& contactVector,
                                                           const Vector3<T>& relVelocityAtContact,
                                                           const Vector3<T>& relAngVelocity,
                                                           const T           overlapDistance,
                                                           const T           averageMass,
                                                           Vector3<T>&       delFN,
                                                           Vector3<T>&       delFT,
                                                           Vector3<T>&       delM) const
{
    // // Notes:
    // // - contactVector is a unit vector pointing from B to A
    // // - overlapDistance is negative when there is penetration

    // // Normal linear elastic force
    // // We do this here as we want to modify the penetration vector later
    // delFN = m_kn * overlapDistance * contactVector;

    // // Unit normal vector at contact point
    // Vector3<T> v_n = (relVelocityAtContact * contactVector) * contactVector;
    // Vector3<T> v_t = relVelocityAtContact - v_n;

    // // Unit tangential vector along relative velocity at contact point
    // T          normv_t = norm(v_t);
    // Vector3<T> tangent(0, 0, 0);
    // if(normv_t > EPS<T>)
    //     tangent = v_t / normv_t;

    // // Normal dissipative force
    // T gamman = - 2. * m_muen * sqrt(averageMass * m_kn);
    // delFN -= gamman * v_n;
    // T normFN = norm(delFN);

    // // Tangential dissipative force
    // delFT = (T(2) * m_etat * averageMass) * v_t;
    // T normFT = norm(delFT);

    // // Tangential Coulomb saturation
    // T fn = m_muc * normFN;
    // if(fn < normFT)
    //     delFT = (-fn) * tangent;

    // // Rolling resistance moment
    // if(m_kr)
    // {
    //     // Relative angular velocity at contact point
    //     Vector3<T> wn     = (relAngVelocity * contactVector) * contactVector;
    //     Vector3<T> wt     = relAngVelocity - wn;
    //     T          normwt = norm(wt);

    //     // Anti-spinning effect along the normal wn
    //     delM = -m_kr * normFN * T(0.001) * wn;

    //     // Classical rolling resistance moment
    //     if(normwt > EPS<T>)
    //         delM -= m_kr * normFN * wt;
    // }

    delFN.setValue(0, 0, 0);
    delFT.setValue(0, 0, 0);
    delM.setValue(0, 0, 0);
}

// -------------------------------------------------------------------------------------------------
// Returns a force based on the contact information
template <typename T>
__HOSTDEVICE__ void
    HookeMemoryContactForceModel<T>::computeForces(const ContactInfo<T>& contactInfos,
                                                   const Vector3<T>&     relVelocityAtContact,
                                                   const Vector3<T>&     relAngVelocity,
                                                   const Vector3<T>&     vA,
                                                   const Vector3<T>&     vB,
                                                   const T               averageMass,
                                                   Torce<T>&             torceA,
                                                   Torce<T>&             torceB) const
{
    // Extract contact info
    Vector3<T> geometricPointOfContact = contactInfos.getContactPoint();
    Vector3<T> contactVector           = contactInfos.getContactVector();
    T          overlapDistance         = contactInfos.getOverlapDistance();

    // Compute contact force and torque
    Vector3<T> delFN, delFT, delM;
    performForcesCalculus(contactVector,
                          relVelocityAtContact,
                          relAngVelocity,
                          overlapDistance,
                          averageMass,
                          delFN,
                          delFT,
                          delM);

    delFN += delFT;
    torceA.addForce(delFN, geometricPointOfContact - vA);
    torceB.addForce(-delFN, geometricPointOfContact - vB);
    if(m_mur)
    {
        torceA.addTorque(delM);
        torceB.addTorque(-delM);
    }
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class HookeMemoryContactForceModel<float>;
template class HookeMemoryContactForceModel<double>;
