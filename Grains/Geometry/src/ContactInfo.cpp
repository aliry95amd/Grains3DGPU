#include "ContactInfo.hh"
#include "GrainsUtils.hh"
#include "VectorMath.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <typename T>
__HOSTDEVICE__ ContactInfo<T>::ContactInfo()
    : m_contactPoint()
    , m_contactVector()
    , m_overlapDistance(T(0))
    , m_contactMetaData()
{
}

// -------------------------------------------------------------------------------------------------
// Constructor with contact point, contact vector, and overlap distance
template <typename T>
__HOSTDEVICE__ ContactInfo<T>::ContactInfo(const Vector3<T>& pt, const Vector3<T>& vec, T overlap)
    : m_contactPoint(pt)
    , m_contactVector(vec)
    , m_overlapDistance(overlap)
    , m_contactMetaData()
{
    // Set overlap sign based on whether overlap is negative
    setOverlapSign(overlap < T(0));
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <typename T>
__HOSTDEVICE__ ContactInfo<T>::~ContactInfo()
{
}

// -------------------------------------------------------------------------------------------------
// Gets the contact point
template <typename T>
__HOSTDEVICE__ Vector3<T> ContactInfo<T>::getContactPoint() const
{
    return (m_contactPoint);
}

// -------------------------------------------------------------------------------------------------
// Gets the contact vector
template <typename T>
__HOSTDEVICE__ Vector3<T> ContactInfo<T>::getContactVector() const
{
    return (m_contactVector);
}

// -------------------------------------------------------------------------------------------------
// Gets the overlap distance
template <typename T>
__HOSTDEVICE__ T ContactInfo<T>::getOverlapDistance() const
{
    return (m_overlapDistance);
}

// -------------------------------------------------------------------------------------------------
// Gets the entire packed contact metadata as BitPacker
template <typename T>
__HOSTDEVICE__ const BitPacker<uint32_t,
                               ContactInfo<T>::B_OVERLAP_SIGN,
                               ContactInfo<T>::B_CONTACT_HASH,
                               ContactInfo<T>::B_AVG_MASS>&
                     ContactInfo<T>::getContactMetaData() const
{
    return m_contactMetaData;
}

// -------------------------------------------------------------------------------------------------
// Gets the entire packed contact metadata as raw value
template <typename T>
__HOSTDEVICE__ uint32_t ContactInfo<T>::getContactMetaDataRaw() const
{
    return m_contactMetaData.getValue();
}

// -------------------------------------------------------------------------------------------------
// Gets the average mass
template <typename T>
__HOSTDEVICE__ T ContactInfo<T>::getAverageMass() const
{
    return m_contactMetaData.template getFixed<2, T>(DEFAULT_AVG_MASS_MIN, DEFAULT_AVG_MASS_MAX);
}

// -------------------------------------------------------------------------------------------------
// Gets the contact hash
template <typename T>
__HOSTDEVICE__ uint ContactInfo<T>::getContactHash() const
{
    return static_cast<uint>(m_contactMetaData.template get<1>());
}

// -------------------------------------------------------------------------------------------------
// Gets the overlap sign
template <typename T>
__HOSTDEVICE__ bool ContactInfo<T>::getOverlapSign() const
{
    return m_contactMetaData.template get<0>() != 0;
}

// -------------------------------------------------------------------------------------------------
// Sets all contact information in one call
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setContactInfo(const Vector3<T>& pt,
                                                   const Vector3<T>& vec,
                                                   T                 dist,
                                                   uint32_t          metadata)
{
    m_contactPoint    = pt;
    m_contactVector   = vec;
    m_overlapDistance = dist;
    m_contactMetaData.setValue(metadata);
}

// -------------------------------------------------------------------------------------------------
// Sets the contact point
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setContactPoint(const Vector3<T>& p)
{
    m_contactPoint = p;
}

// -------------------------------------------------------------------------------------------------
// Sets the contact vector
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setContactVector(const Vector3<T>& v)
{
    m_contactVector = v;
}

// -------------------------------------------------------------------------------------------------
// Sets the overlap distance
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setOverlapDistance(T d)
{
    m_overlapDistance = d;
    // Automatically update overlap sign
    setOverlapSign(d < T(0));
}

// -------------------------------------------------------------------------------------------------
// Sets the entire packed contact metadata from BitPacker
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setContactMetaData(
    const BitPacker<uint32_t, B_OVERLAP_SIGN, B_CONTACT_HASH, B_AVG_MASS>& metadata)
{
    m_contactMetaData = metadata;
}

// -------------------------------------------------------------------------------------------------
// Sets the entire packed contact metadata from raw value
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setContactMetaDataRaw(uint32_t metadata)
{
    m_contactMetaData.setValue(metadata);
}

// -------------------------------------------------------------------------------------------------
// Sets the average mass
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setAverageMass(T avgMass)
{
    bool saturated = false;
    m_contactMetaData.template setFixed<2, T>(avgMass,
                                              DEFAULT_AVG_MASS_MIN,
                                              DEFAULT_AVG_MASS_MAX,
                                              saturated);
}

// -------------------------------------------------------------------------------------------------
// Sets the contact hash
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setContactHash(uint hash)
{
    m_contactMetaData.template set<1>(hash);
}

// -------------------------------------------------------------------------------------------------
// Sets the overlap sign
template <typename T>
__HOSTDEVICE__ void ContactInfo<T>::setOverlapSign(bool isNegative)
{
    m_contactMetaData.template set<0>(isNegative ? 1 : 0);
}

// -------------------------------------------------------------------------------------------------
// Equality operator
template <typename T>
__HOSTDEVICE__ bool ContactInfo<T>::operator==(const ContactInfo<T>& other) const
{
    return (m_contactPoint == other.m_contactPoint) && (m_contactVector == other.m_contactVector)
           && (m_overlapDistance == other.m_overlapDistance)
           && (m_contactMetaData.getValue() == other.m_contactMetaData.getValue());
}

// -------------------------------------------------------------------------------------------------
// Inequality operator
template <typename T>
__HOSTDEVICE__ bool ContactInfo<T>::operator!=(const ContactInfo<T>& other) const
{
    return !(*this == other);
}

// -------------------------------------------------------------------------------------------------
// Output operator
template <typename T>
__HOST__ std::ostream& operator<<(std::ostream& fileOut, const ContactInfo<T>& c)
{
    // Orientation first, followed by the position
    fileOut << "Contact Point: " << c.getContactPoint() << "\n"
            << "Contact Vector: " << c.getContactVector() << "\n"
            << "Overlap Distance: " << c.getOverlapDistance();
    return (fileOut);
}

// -------------------------------------------------------------------------------------------------
// Input operator
template <typename T>
__HOST__ std::istream& operator>>(std::istream& fileIn, ContactInfo<T>& c)
{
    GAbort("Input operator for ContactInfo is not implemented yet!");
    return (fileIn);
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class ContactInfo<float>;
template class ContactInfo<double>;

#define X(T)                                                                                \
    template std::ostream& operator<< <T>(std::ostream & fileOut, const ContactInfo<T>& t); \
                                                                                            \
    template std::istream& operator>> <T>(std::istream & fileIn, ContactInfo<T> & t);
X(float)
X(double)
#undef X