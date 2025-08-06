#include "Vector3.hh"
#include "VectorMath.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
__HOSTDEVICE__ Vector3<T>::Vector3(T def) noexcept
{
    m_comp[X] = m_comp[Y] = m_comp[Z] = def;
}

// -----------------------------------------------------------------------------
// Constructor with the pointer to a buffer
template <typename T>
__HOSTDEVICE__ Vector3<T>::Vector3(T const* buffer) noexcept
{
    setValue(buffer);
}

// -----------------------------------------------------------------------------
// Constructor with 3 components as inputs
template <typename T>
__HOSTDEVICE__ Vector3<T>::Vector3(T x, T y, T z) noexcept
{
    m_comp[X] = x;
    m_comp[Y] = y;
    m_comp[Z] = z;
}

// -----------------------------------------------------------------------------
// Copy constructor
template <typename T>
__HOSTDEVICE__ Vector3<T>::Vector3(const Vector3<T>& vec) noexcept
{
    m_comp[X] = vec.m_comp[X];
    m_comp[Y] = vec.m_comp[Y];
    m_comp[Z] = vec.m_comp[Z];
}

// -----------------------------------------------------------------------------
// Copy assignment operator
template <typename T>
__HOSTDEVICE__ Vector3<T>& Vector3<T>::operator=(const Vector3<T>& vec) noexcept
{
    if(&vec != this)
    {
        for(int i = 0; i < 3; ++i)
            m_comp[i] = vec.m_comp[i];
    }
    return (*this);
}

// -----------------------------------------------------------------------------
// Move constructor
template <typename T>
__HOSTDEVICE__ Vector3<T>::Vector3(Vector3<T>&& vec) noexcept
{
    for(int i = 0; i < 3; ++i)
        m_comp[i] = vec.m_comp[i];
    vec.reset();
}

// -----------------------------------------------------------------------------
// Move assignment operator
template <typename T>
__HOSTDEVICE__ Vector3<T>& Vector3<T>::operator=(Vector3<T>&& vec) noexcept
{
    if(&vec != this)
    {
        for(int i = 0; i < 3; ++i)
            m_comp[i] = vec.m_comp[i];
        vec.reset();
    }
    return (*this);
}

// -----------------------------------------------------------------------------
// Constructor from an XML node
template <typename T>
__HOST__ Vector3<T>::Vector3(DOMNode* root) noexcept
{
    T x = T(ReaderXML::getNodeAttr_Double(root, "X"));
    T y = T(ReaderXML::getNodeAttr_Double(root, "Y"));
    T z = T(ReaderXML::getNodeAttr_Double(root, "Z"));
    setValue(x, y, z);
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
__HOSTDEVICE__ Vector3<T>::~Vector3() noexcept
{
}

// -----------------------------------------------------------------------------
/* Gets the pointer to the buffer */
template <typename T>
__HOSTDEVICE__ T const* Vector3<T>::getBuffer() const noexcept
{
    return (m_comp);
}

// -----------------------------------------------------------------------------
/* Sets the components using a pointer to a buffer */
template <typename T>
__HOSTDEVICE__ void Vector3<T>::setValue(T const* buffer) noexcept
{
    m_comp[X] = buffer[X];
    m_comp[Y] = buffer[Y];
    m_comp[Z] = buffer[Z];
}

// -----------------------------------------------------------------------------
/* Sets the components using three different values */
template <typename T>
__HOSTDEVICE__ void
    Vector3<T>::setValue(const T x, const T y, const T z) noexcept
{
    m_comp[X] = x;
    m_comp[Y] = y;
    m_comp[Z] = z;
}

// -----------------------------------------------------------------------------
// Unitary nomalization operator
template <typename T>
__HOSTDEVICE__ void Vector3<T>::normalize() noexcept
{
    *this /= norm(*this);
}

// -----------------------------------------------------------------------------
// Returns a vector corresponding to the normalized vector
template <typename T>
__HOSTDEVICE__ Vector3<T> Vector3<T>::normalized() const noexcept
{
    return (*this / norm(*this));
}

// -----------------------------------------------------------------------------
// Sets components to zero
template <typename T>
__HOSTDEVICE__ void Vector3<T>::reset() noexcept
{
    m_comp[X] = m_comp[Y] = m_comp[Z] = T(0);
}

// -----------------------------------------------------------------------------
// ith component accessor
template <typename T>
__HOSTDEVICE__ T const& Vector3<T>::operator[](size_t i) const noexcept
{
    return (m_comp[i]);
}

// -----------------------------------------------------------------------------
// ith component accessor - modifiable lvalue
template <typename T>
__HOSTDEVICE__ T& Vector3<T>::operator[](size_t i) noexcept
{
    return (m_comp[i]);
}

// -----------------------------------------------------------------------------
// Conversion operator to float
template <>
__HOSTDEVICE__ Vector3<double>::operator Vector3<float>() const noexcept
{
    return (
        Vector3<float>((float)m_comp[X], (float)m_comp[Y], (float)m_comp[Z]));
}

// -----------------------------------------------------------------------------
// Output operator
template <typename T>
__HOST__ std::ostream& operator<<(std::ostream& fileOut, const Vector3<T>& v)
{
    fileOut << v[X] << " " << v[Y] << " " << v[Z];
    return (fileOut);
}

// -----------------------------------------------------------------------------
// Input operator
template <typename T>
__HOST__ std::istream& operator>>(std::istream& fileIn, Vector3<T>& v)
{
    fileIn >> v[X] >> v[Y] >> v[Z];
    return (fileIn);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class Vector3<float>;
template class Vector3<double>;

#define X(T)                                                      \
    template std::ostream& operator<< <T>(std::ostream & fileOut, \
                                          const Vector3<T>& v);   \
                                                                  \
    template std::istream& operator>> <T>(std::istream & fileIn,  \
                                          Vector3<T> & v);
X(float)
X(double)
#undef X