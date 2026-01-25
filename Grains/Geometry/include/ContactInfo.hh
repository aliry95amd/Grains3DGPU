#ifndef _CONTACTINFO_HH_
#define _CONTACTINFO_HH_

#include <limits>
#include <type_traits>

#include "BitPacker.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief The class ContactInfo.

    Contains all the features of a contact point.

    @author A.Yazdani - 2024 - Construction */
// =================================================================================================
template <typename T>
class ContactInfo
{
public:
    /** @name BitPacker constants */
    //@{
    /** \brief Number of bits for each field in the packed contact metadata */
    static constexpr int B_OVERLAP_SIGN = 1;
    static constexpr int B_CONTACT_HASH = 5;
    static constexpr int B_AVG_MASS     = 26;
    static_assert(B_OVERLAP_SIGN + B_CONTACT_HASH + B_AVG_MASS == 32,
                  "Bits must match storage type size");

    /** \brief Default minimum and maximum values for average mass */
    static constexpr T DEFAULT_AVG_MASS_MIN = 0;
    static constexpr T DEFAULT_AVG_MASS_MAX = T(1e5);

    /** \brief Storage type for contact metadata (uint32_t for float/double) */
    using StorageType = uint32_t;
    //@}

protected:
    /** @name Parameters */
    //@{
    /** \brief Contact point */
    Vector3<T> m_contactPoint;
    /** \brief Contact vector */
    Vector3<T> m_contactVector;
    /** \brief Overlap distance */
    T m_overlapDistance;
    /** \brief Packed contact metadata: sign(1), hash(5), avgMass(26) */
    BitPacker<uint32_t, B_OVERLAP_SIGN, B_CONTACT_HASH, B_AVG_MASS> m_contactMetaData;
    //@}

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor */
    __HOSTDEVICE__
    ContactInfo();

    /** @brief Constructor with contact point location in the world reference frame, overlap vector,
        overlap distance and number of iterations of GJK as input parameters.
        @param pt contact point
        @param vec contact vector
        @param distance_ overlap distance */
    __HOSTDEVICE__
    ContactInfo(const Vector3<T>& pt, const Vector3<T>& vec, T overlap);

    /** @brief Destructor */
    __HOSTDEVICE__
    ~ContactInfo();
    //@}

    /** @name Get methods */
    //@{
    /** @brief Gets the contact point */
    __HOSTDEVICE__
    Vector3<T> getContactPoint() const;

    /** @brief Gets the contact vector */
    __HOSTDEVICE__
    Vector3<T> getContactVector() const;

    /** @brief Gets the overlap distance */
    __HOSTDEVICE__
    T getOverlapDistance() const;

    /** @brief Gets the entire packed contact metadata as BitPacker */
    __HOSTDEVICE__
    const BitPacker<uint32_t, B_OVERLAP_SIGN, B_CONTACT_HASH, B_AVG_MASS>&
        getContactMetaData() const;

    /** @brief Gets the entire packed contact metadata as raw value */
    __HOSTDEVICE__
    uint32_t getContactMetaDataRaw() const;

    /** @brief Gets the average mass */
    __HOSTDEVICE__
    T getAverageMass() const;

    /** @brief Gets the contact hash */
    __HOSTDEVICE__
    uint getContactHash() const;

    /** @brief Gets the overlap sign (true if negative) */
    __HOSTDEVICE__
    bool getOverlapSign() const;
    //@}

    /** @name Set methods */
    //@{
    /** @brief Sets all contact information in one call
        @param pt contact point
        @param vec contact vector
        @param dist overlap distance
        @param metadata packed metadata value */
    __HOSTDEVICE__
    void setContactInfo(const Vector3<T>& pt, const Vector3<T>& vec, T dist, uint32_t metadata);

    /** @brief Sets the contact point
        @param p contact point */
    __HOSTDEVICE__
    void setContactPoint(const Vector3<T>& p);

    /** @brief Sets the contact vector
        @param v overlap vector */
    __HOSTDEVICE__
    void setContactVector(const Vector3<T>& v);

    /** @brief Sets the overlap distance
        @param d overlap distance  */
    __HOSTDEVICE__
    void setOverlapDistance(T d);

    /** @brief Sets the entire packed contact metadata from BitPacker
        @param metadata BitPacker containing packed metadata */
    __HOSTDEVICE__
    void setContactMetaData(
        const BitPacker<uint32_t, B_OVERLAP_SIGN, B_CONTACT_HASH, B_AVG_MASS>& metadata);

    /** @brief Sets the entire packed contact metadata from raw value
        @param metadata packed metadata value */
    __HOSTDEVICE__
    void setContactMetaDataRaw(uint32_t metadata);

    /** @brief Sets the average mass
        @param avgMass average mass of contacting particles */
    __HOSTDEVICE__
    void setAverageMass(T avgMass);

    /** @brief Sets the contact hash from two material IDs */
    __HOSTDEVICE__
    void setContactHash(uint hash);

    /** @brief Sets the overlap sign
        @param isNegative true if overlap is negative */
    __HOSTDEVICE__
    void setOverlapSign(bool isNegative);
    //@}

    /** @name Operators */
    //@{
    /** @brief Equality operator
        @param other the ContactInfo to compare with
        @return true if all members are equal */
    __HOSTDEVICE__
    bool operator==(const ContactInfo<T>& other) const;

    /** @brief Inequality operator
        @param other the ContactInfo to compare with
        @return true if any member is different */
    __HOSTDEVICE__
    bool operator!=(const ContactInfo<T>& other) const;
    //@}
};

/** @name External Methods - I/O methods */
//@{
/** @brief Output operator
    @param fileIn input stream
    @param c contact point object */
template <typename T>
__HOST__ std::ostream& operator<<(std::ostream& fileOut, const ContactInfo<T>& c);

/** @brief Input operator
    @param fileIn input stream
    @param c contact point object */
template <typename T>
__HOST__ std::istream& operator>>(std::istream& fileIn, ContactInfo<T>& c);
//@}

#endif