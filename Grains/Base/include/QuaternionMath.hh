#ifndef _QUATERNIONMATH_HH_
#define _QUATERNIONMATH_HH_

#include "Quaternion.hh"
#include "VectorMath.hh"

// =============================================================================
/** @brief Miscellaneous Quaternion functions and operators as header-only.
   
    Defining important quaternion functions and operators as static functions.
    It will increase the binary size, but the performance gain is much more
    appreciated.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
/** @name Quaternion math functions and operators */
//@{
/** @brief Returns the norm of the quaternion
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE T norm(const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b = q.getBuffer();
    return (sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]));
}

// -----------------------------------------------------------------------------
/** @brief Returns the norm squared of the quaternion
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE T norm2(const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b = q.getBuffer();
    return (b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]);
}

// -----------------------------------------------------------------------------
/** @brief Quaternion conjugate
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
                             conjugate(const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b = q.getBuffer();
    T __RESTRICT__        out[4];
    out[0] = -b[0];
    out[1] = -b[1];
    out[2] = -b[2];
    out[3] = b[3];
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Quaternion conjugate in-place
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void conjugate(Quaternion<T>& q) noexcept
{
    T* __RESTRICT__ b = const_cast<T*>(q.getBuffer());
    b[0]              = -b[0];
    b[1]              = -b[1];
    b[2]              = -b[2];
}

// -----------------------------------------------------------------------------
/** @brief Quaternion inverse
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
                             inverse(const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b = q.getBuffer();
    T __RESTRICT__        out[4];
    T                     norm_inv = T(1) / norm(q);
    out[0]                         = -norm_inv * b[0];
    out[1]                         = -norm_inv * b[1];
    out[2]                         = -norm_inv * b[2];
    out[3]                         = norm_inv * b[3];
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Quaternion inverse in-place
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void inverse(Quaternion<T>& q) noexcept
{
    T* __RESTRICT__ b        = const_cast<T*>(q.getBuffer());
    T               norm_inv = T(1) / norm(q);
    b[0]                     = -norm_inv * b[0];
    b[1]                     = -norm_inv * b[1];
    b[2]                     = -norm_inv * b[2];
    b[3]                     = norm_inv * b[3];
}

// -----------------------------------------------------------------------------
/** @brief Quaternions addition
    @param q1 1st quaternion 
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
    operator+(const Quaternion<T>& q1, const Quaternion<T>& q2) noexcept
{
    const T* __RESTRICT__ b1 = q1.getBuffer();
    const T* __RESTRICT__ b2 = q2.getBuffer();
    T __RESTRICT__        out[4];
    for(uint i = 0; i < 4; ++i)
        out[i] = b1[i] + b2[i];
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Quaternions addition in-place
    @param q1 1st quaternion 
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void operator+=(Quaternion<T>&       q1,
                                             const Quaternion<T>& q2) noexcept
{
    T* __RESTRICT__       b1 = const_cast<T*>(q1.getBuffer());
    const T* __RESTRICT__ b2 = q2.getBuffer();
    for(uint i = 0; i < 4; ++i)
        b1[i] += b2[i];
}

// -----------------------------------------------------------------------------
/** @brief Quaternions subtraction
    @param q1 1st quaternion 
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
    operator-(const Quaternion<T>& q1, const Quaternion<T>& q2) noexcept
{
    const T* __RESTRICT__ b1 = q1.getBuffer();
    const T* __RESTRICT__ b2 = q2.getBuffer();
    T __RESTRICT__        out[4];
    for(uint i = 0; i < 4; ++i)
        out[i] = b1[i] - b2[i];
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Quaternions subtraction in-place
    @param q1 1st quaternion
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void operator-=(Quaternion<T>&       q1,
                                             const Quaternion<T>& q2) noexcept
{
    T* __RESTRICT__       b1 = const_cast<T*>(q1.getBuffer());
    const T* __RESTRICT__ b2 = q2.getBuffer();
    for(uint i = 0; i < 4; ++i)
        b1[i] -= b2[i];
}

// -----------------------------------------------------------------------------
/** @brief Scalar-quaternion multiplication
    @param d the multiplication factor
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
                             operator*(T d, const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b = q.getBuffer();
    T __RESTRICT__        out[4];
    for(uint i = 0; i < 4; ++i)
        out[i] = d * b[i];
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Scalar-quaternion multiplication in-place
    @param d the multiplication factor
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void operator*=(Quaternion<T>& q, T d) noexcept
{
    T* __RESTRICT__ b = const_cast<T*>(q.getBuffer());
    for(uint i = 0; i < 4; ++i)
        b[i] *= d;
}

// -----------------------------------------------------------------------------
/** @brief Quaternion-vector multiplication
    @param q the quaternion
    @param v the vector */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
    operator*(const Quaternion<T>& q, const Vector3<T>& v) noexcept
{
    const T* __RESTRICT__ b1 = q.getBuffer();
    const T* __RESTRICT__ b2 = v.getBuffer();
    T __RESTRICT__        out[4];
    out[0] = (b1[3] * b2[0]) + (b1[1] * b2[2]) - (b1[2] * b2[1]);
    out[1] = (b1[3] * b2[1]) + (b1[2] * b2[0]) - (b1[0] * b2[2]);
    out[2] = (b1[3] * b2[2]) + (b1[0] * b2[1]) - (b1[1] * b2[0]);
    out[3] = -(b1[0] * b2[0]) - (b1[1] * b2[1]) - (b1[2] * b2[2]);
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Quaternion-vector multiplication in-place
    @param q the quaternion
    @param v the vector */
template <typename T>
__HOSTDEVICE__ static INLINE void operator*=(Quaternion<T>&    q,
                                             const Vector3<T>& v) noexcept
{
    T* __RESTRICT__       b1 = const_cast<T*>(q.getBuffer());
    const T* __RESTRICT__ b2 = v.getBuffer();
    T                     out[3];
    out[0] = (b1[3] * b2[0]) + (b1[1] * b2[2]) - (b1[2] * b2[1]);
    out[1] = (b1[3] * b2[1]) + (b1[2] * b2[0]) - (b1[0] * b2[2]);
    out[2] = (b1[3] * b2[2]) + (b1[0] * b2[1]) - (b1[1] * b2[0]);
    b1[3]  = -(b1[0] * b2[0]) - (b1[1] * b2[1]) - (b1[2] * b2[2]);
    b1[0]  = out[0];
    b1[1]  = out[1];
    b1[2]  = out[2];
}

// -----------------------------------------------------------------------------
/** @brief Vector-quaternion multiplication
    @param v the vector
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
    operator*(const Vector3<T>& v, const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b1 = v.getBuffer();
    const T* __RESTRICT__ b2 = q.getBuffer();
    T __RESTRICT__        out[4];
    out[0] = (b1[0] * b2[3]) + (b1[1] * b2[2]) - (b1[2] * b2[1]);
    out[1] = (b1[1] * b2[3]) + (b1[2] * b2[0]) - (b1[0] * b2[2]);
    out[2] = (b1[2] * b2[3]) + (b1[0] * b2[1]) - (b1[1] * b2[0]);
    out[3] = -(b1[0] * b2[0]) - (b1[1] * b2[1]) - (b1[2] * b2[2]);
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Vector-quaternion multiplication in-place. Note that this modifies 
    the quaternion, not the vector.
    @param v the vector
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void operator*=(Vector3<T>&          v,
                                             const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b1 = v.getBuffer();
    T* __RESTRICT__       b2 = const_cast<T*>(q.getBuffer());
    T                     out[3];
    out[0] = (b1[0] * b2[3]) + (b1[1] * b2[2]) - (b1[2] * b2[1]);
    out[1] = (b1[1] * b2[3]) + (b1[2] * b2[0]) - (b1[0] * b2[2]);
    out[2] = (b1[2] * b2[3]) + (b1[0] * b2[1]) - (b1[1] * b2[0]);
    b2[3]  = -(b1[0] * b2[0]) - (b1[1] * b2[1]) - (b1[2] * b2[2]);
    b2[0]  = out[0];
    b2[1]  = out[1];
    b2[2]  = out[2];
}

// -----------------------------------------------------------------------------
/** @brief Quaternion-quaternion multiplication
    @param q1 1st quaternion
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE Quaternion<T>
    operator*(const Quaternion<T>& q1, const Quaternion<T>& q2) noexcept
{
    const T* __RESTRICT__ b1 = q1.getBuffer();
    const T* __RESTRICT__ b2 = q2.getBuffer();
    T __RESTRICT__        out[4];
    out[0]
        = (b1[3] * b2[0]) + (b1[0] * b2[3]) + (b1[1] * b2[2]) - (b1[2] * b2[1]);
    out[1]
        = (b1[3] * b2[1]) + (b1[1] * b2[3]) + (b1[2] * b2[0]) - (b1[0] * b2[2]);
    out[2]
        = (b1[3] * b2[2]) + (b1[2] * b2[3]) + (b1[0] * b2[1]) - (b1[1] * b2[0]);
    out[3]
        = (b1[3] * b2[3]) - (b1[0] * b2[0]) - (b1[1] * b2[1]) - (b1[2] * b2[2]);
    return (Quaternion<T>(out));
}

// -----------------------------------------------------------------------------
/** @brief Quaternion-quaternion multiplication in-place
    @param q1 1st quaternion
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ static INLINE void operator*=(Quaternion<T>&       q1,
                                             const Quaternion<T>& q2) noexcept
{
    T* __RESTRICT__       b1 = const_cast<T*>(q1.getBuffer());
    const T* __RESTRICT__ b2 = q2.getBuffer();
    T                     out[3];
    out[0]
        = (b1[3] * b2[0]) + (b1[0] * b2[3]) + (b1[1] * b2[2]) - (b1[2] * b2[1]);
    out[1]
        = (b1[3] * b2[1]) + (b1[1] * b2[3]) + (b1[2] * b2[0]) - (b1[0] * b2[2]);
    out[2]
        = (b1[3] * b2[2]) + (b1[2] * b2[3]) + (b1[0] * b2[1]) - (b1[1] * b2[0]);
    b1[3]
        = (b1[3] * b2[3]) - (b1[0] * b2[0]) - (b1[1] * b2[1]) - (b1[2] * b2[2]);
    b1[0] = out[0];
    b1[1] = out[1];
    b1[2] = out[2];
}

// -----------------------------------------------------------------------------
/** @brief Quaternion equality operator
    @param q1 1st quaternion
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ bool Quaternion<T>::operator==(const Quaternion<T>& q1,
                                              const Quaternion<T>& q2) noexcept
{
    const T* __RESTRICT__ b1 = q1.getBuffer();
    const T* __RESTRICT__ b2 = q2.getBuffer();
    return (b1[0] == b2[0] && b1[1] == b2[1] && b1[2] == b2[2]
            && b1[3] == b2[3]);
}

// -----------------------------------------------------------------------------
/** @brief Quaternion inequality operator
    @param q1 1st quaternion
    @param q2 2nd quaternion */
template <typename T>
__HOSTDEVICE__ bool Quaternion<T>::operator!=(const Quaternion<T>& q1,
                                              const Quaternion<T>& q2) noexcept
{
    const T* __RESTRICT__ b1 = q1.getBuffer();
    const T* __RESTRICT__ b2 = q2.getBuffer();
    return (b1[0] != b2[0] || b1[1] != b2[1] || b1[2] != b2[2]
            || b1[3] != b2[3]);
}

// -----------------------------------------------------------------------------
/** @brief Quaternion sign flip
    @param q the quaternion */
template <typename T>
__HOSTDEVICE__ Quaternion<T> operator-(const Quaternion<T>& q) noexcept
{
    const T* __RESTRICT__ b = q.getBuffer();
    T __RESTRICT__        out[4];
    for(uint i = 0; i < 4; ++i)
        out[i] = -b[i];
    return (Quaternion<T>(out));
}

#endif