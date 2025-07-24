#include "GJK_JH.hh"
#include "MatrixMath.hh"
#include "MiscMath.hh"

/* ========================================================================== */
/*                             Low-Level Methods                              */
/* ========================================================================== */
template <typename T>
__HOSTDEVICE__ static INLINE void computeDet(uint const       bits,
                                             uint const       last,
                                             uint const       last_bit,
                                             uint const       all_bits,
                                             Vector3<T> const y[4],
                                             T                dp[4][4],
                                             T                det[16][4])
{
    for(uint i = 0, bit = 1; i < 4; ++i, bit <<= 1)
        if(bits & bit)
            dp[i][last] = dp[last][i] = y[i] * y[last];
    dp[last][last] = y[last] * y[last];

    det[last_bit][last] = T(1);
    for(uint j = 0, sj = 1; j < 4; ++j, sj <<= 1)
    {
        if(bits & sj)
        {
            uint s2       = sj | last_bit;
            det[s2][j]    = dp[last][last] - dp[last][j];
            det[s2][last] = dp[j][j] - dp[j][last];
            for(uint k = 0, sk = 1; k < j; ++k, sk <<= 1)
            {
                if(bits & sk)
                {
                    int s3     = sk | s2;
                    det[s3][k] = det[s2][j] * (dp[j][j] - dp[j][k])
                                 + det[s2][last] * (dp[last][j] - dp[last][k]);
                    det[s3][j] = det[sk | last_bit][k] * (dp[k][k] - dp[k][j])
                                 + det[sk | last_bit][last]
                                       * (dp[last][k] - dp[last][j]);
                    det[s3][last]
                        = det[sk | sj][k] * (dp[k][k] - dp[k][last])
                          + det[sk | sj][j] * (dp[j][k] - dp[j][last]);
                }
            }
        }
    }

    if(all_bits == 15)
    {
        det[15][0] = det[14][1] * (dp[1][1] - dp[1][0])
                     + det[14][2] * (dp[2][1] - dp[2][0])
                     + det[14][3] * (dp[3][1] - dp[3][0]);
        det[15][1] = det[13][0] * (dp[0][0] - dp[0][1])
                     + det[13][2] * (dp[2][0] - dp[2][1])
                     + det[13][3] * (dp[3][0] - dp[3][1]);
        det[15][2] = det[11][0] * (dp[0][0] - dp[0][2])
                     + det[11][1] * (dp[1][0] - dp[1][2])
                     + det[11][3] * (dp[3][0] - dp[3][2]);
        det[15][3] = det[7][0] * (dp[0][0] - dp[0][3])
                     + det[7][1] * (dp[1][0] - dp[1][3])
                     + det[7][2] * (dp[2][0] - dp[2][3]);
    }
}

// -----------------------------------------------------------------------------
template <typename T>
__HOSTDEVICE__ static INLINE bool
    valid(uint const s, uint const all_bits, T const det[16][4])
{
    for(uint i = 0, bit = 1; i < 4; ++i, bit <<= 1)
    {
        if(all_bits & bit)
        {
            if(s & bit)
            {
                if(det[s][i] <= EPS<T>)
                    return (false);
            }
            else if(det[s | bit][i] > T(0))
                return (false);
        }
    }
    return (true);
}

// -----------------------------------------------------------------------------
template <typename T>
__HOSTDEVICE__ static INLINE void computeVector(uint const       bits_,
                                                Vector3<T> const y[4],
                                                T const          det[16][4],
                                                Vector3<T>&      v)
{
    T sum = T(0);
    v.setValue(T(0), T(0), T(0));
    for(uint i = 0, bit = 1; i < 4; ++i, bit <<= 1)
    {
        if(bits_ & bit)
        {
            sum += det[bits_][i];
            v += det[bits_][i] * y[i];
        }
    }
    v *= T(1) / sum;
}

// -----------------------------------------------------------------------------
template <typename T>
__HOSTDEVICE__ static INLINE void computePoints(uint const       bits_,
                                                T const          det[16][4],
                                                Vector3<T> const p[4],
                                                Vector3<T> const q[4],
                                                Vector3<T>&      p1,
                                                Vector3<T>&      p2)
{
    T sum = T(0);
    p1.setValue(T(0), T(0), T(0));
    p2.setValue(T(0), T(0), T(0));
    for(uint i = 0, bit = 1; i < 4; ++i, bit <<= 1)
    {
        if(bits_ & bit)
        {
            sum += det[bits_][i];
            p1 += det[bits_][i] * p[i];
            p2 += det[bits_][i] * q[i];
        }
    }
    T s = T(1) / sum;
    p1 *= s;
    p2 *= s;
}

// -----------------------------------------------------------------------------
template <typename T>
__HOSTDEVICE__ static INLINE bool proper(uint const s, T const det[16][4])
{
    for(uint i = 0, bit = 1; i < 4; ++i, bit <<= 1)
        if((s & bit) && det[s][i] <= EPS<T>)
            return (false);
    return (true);
}

// -----------------------------------------------------------------------------
template <typename T>
__HOSTDEVICE__ static INLINE bool closest(uint&            bits,
                                          uint const       last,
                                          uint const       last_bit,
                                          uint const       all_bits,
                                          Vector3<T> const y[4],
                                          T                dp[4][4],
                                          T                det[16][4],
                                          Vector3<T>&      v)
{
    uint s;
    computeDet(bits, last, last_bit, all_bits, y, dp, det);
    for(s = bits; s; --s)
    {
        if((s & bits) == s)
        {
            if(valid(s | last_bit, all_bits, det))
            {
                bits = s | last_bit;
                computeVector(bits, y, det, v);
                return (true);
            }
        }
    }
    if(valid(last_bit, all_bits, det))
    {
        bits = last_bit;
        v    = y[last];
        return (true);
    }
    // Original GJK calls the backup procedure at this point.
    T min_dist2 = INFINITY;
    for(s = all_bits; s; --s)
    {
        if((s & all_bits) == s)
        {
            if(proper(s, det))
            {
                Vector3<T> u;
                computeVector(s, y, det, u);
                T dist2 = norm2(u);
                if(dist2 < min_dist2)
                {
                    min_dist2 = dist2;
                    bits      = s;
                    v         = u;
                }
            }
        }
    }
    return (false);
}

// -----------------------------------------------------------------------------
// The next function is used for detecting degenerate cases that cause
// termination problems due to rounding errors.
template <typename T>
__HOSTDEVICE__ static INLINE bool
    degenerate(uint const all_bits, Vector3<T> const y[4], const Vector3<T>& w)
{
    for(uint i = 0, bit = 1; i < 4; ++i, bit <<= 1)
        if((all_bits & bit) && y[i] == w)
            return (true);
    return (false);
}

// -----------------------------------------------------------------------------
// For num_iterations > 1000
__HOSTDEVICE__
void catch_me()
{
    printf("closestPointsGJK: Out on iteration > 1000\n");
}

/* ========================================================================== */
/*                             High-Level Methods                             */
/* ========================================================================== */
// Returns whether 2 convex shapes intersect using the GJK algorithm - relative
// transformation
template <typename T>
__HOSTDEVICE__ bool intersectGJK(const Convex<T>&     a,
                                 const Convex<T>&     b,
                                 const Transform3<T>& b2a)
{
    uint       bits     = 0; // identifies current simplex
    uint       last     = 0; // identifies last found support point
    uint       last_bit = 0; // last_bit = 1<<last
    uint       all_bits = 0; // all_bits = bits|last_bit
    Vector3<T> y[4]; // support points of A-B in world
    T          det[16][4] = {T(0)}; // cached sub-determinants
    T          dp[4][4]   = {T(0)};

    Vector3<T> v(b2a.getOrigin());
    Vector3<T> w;
    T          prod;

    do
    {
        last     = 0;
        last_bit = 1;
        while(bits & last_bit)
        {
            ++last;
            last_bit <<= 1;
        }
        w    = a.support(-v) - b2a(b.support(v * b2a.getBasis()));
        prod = v * w;
        if(prod > T(0) || fabs(prod) < HIGHEPS<T>)
            return (false);
        if(degenerate(all_bits, y, w))
            return (false);
        y[last]  = w;
        all_bits = bits | last_bit;
        if(!closest(bits, last, last_bit, all_bits, y, dp, det, v))
            return (false);
    } while(bits < 15 && !isApproxZero(v));
    return (true);
}

// -----------------------------------------------------------------------------
// Returns whether 2 convex shapes intersect using the GJK algorithm
template <typename T>
__HOSTDEVICE__ bool intersectGJK(const Convex<T>&     a,
                                 const Convex<T>&     b,
                                 const Transform3<T>& a2w,
                                 const Transform3<T>& b2w)
{
    uint       bits     = 0; // identifies current simplex
    uint       last     = 0; // identifies last found support point
    uint       last_bit = 0; // last_bit = 1<<last
    uint       all_bits = 0; // all_bits = bits|last_bit
    Vector3<T> y[4]; // support points of A-B in world
    T          det[16][4] = {T(0)}; // cached sub-determinants
    T          dp[4][4]   = {T(0)};

    Vector3<T> v(b2w.getOrigin() - a2w.getOrigin());
    Vector3<T> w;
    T          prod;

    do
    {
        last     = 0;
        last_bit = 1;
        while(bits & last_bit)
        {
            ++last;
            last_bit <<= 1;
        }
        w = a2w(a.support((-v) * a2w.getBasis()))
            - b2w(b.support(v * b2w.getBasis()));
        prod = v * w;
        if(prod > T(0) || fabs(prod) < HIGHEPS<T>)
            return (false);
        if(degenerate(all_bits, y, w))
            return (false);
        y[last]  = w;
        all_bits = bits | last_bit;
        if(!closest(bits, last, last_bit, all_bits, y, dp, det, v))
            return (false);
    } while(bits < 15 && !isApproxZero(v));
    return (true);
}

// -----------------------------------------------------------------------------
// Returns the minimal distance between 2 convex shapes and a point per convex
// shape that represents the tips of the minimal distance segment -- relative
// transformation
template <typename T>
__HOSTDEVICE__ T computeClosestPoints_GJK_JH(const Convex<T>&     a,
                                             const Convex<T>&     b,
                                             const Transform3<T>& b2a,
                                             Vector3<T>&          pa,
                                             Vector3<T>&          pb,
                                             uint&                nbIter)
{
    // GJK variables
    uint       bits     = 0; // identifies current simplex
    uint       last     = 0; // identifies last found support point
    uint       last_bit = 0; // last_bit = 1<<last
    uint       all_bits = 0; // all_bits = bits|last_bit
    Vector3<T> p[4]; // support points of A in local
    Vector3<T> q[4]; // support points of B in local
    Vector3<T> y[4]; // support points of A-B in world
    T          mu            = 0.; // optimality gap
    int        numIterations = 0; // No. iterations
    T          det[16][4]    = {T(0)}; // cached sub-determinants
    T          dp[4][4]      = {T(0)};

    // Misc variables, e.g. tolerance, ...
    // relative tolerance
    constexpr T relError = LOWEPS<T>;
    // absolute tolerance
    constexpr T absError = 1.e-4 * relError;
    // isAcceleration?
    // bool acceleration = GrainsExec::m_colDetAcceleration;
    // in case we use acceleration
    // T momentum = T( 0 ), oneMinusMomentum = T( 1 );

    // Initializing vectors
    Vector3<T> v(-b2a.getOrigin());
    Vector3<T> w;
    T          dist = norm(v);

    while(bits < 15 && dist > HIGHEPS<T> && numIterations < 1000)
    {
        // Updating the bits, ...
        last     = 0;
        last_bit = 1;
        while(bits & last_bit)
        {
            ++last;
            last_bit <<= 1;
        }

        p[last] = a.support((-v));
        q[last] = b.support((v)*b2a.getBasis());
        w       = p[last] - b2a(q[last]);

        // termination criteria -- optimiality gap
        // set_max(mu, v * w / dist);
        mu = dist - v * w / dist;
        if(mu <= dist * relError || mu < absError)
        {
            printf("GJK_JH: Optimality gap reached: %d, %f, %f, %f, %f\n",
                   numIterations,
                   dist,
                   v[X],
                   v[Y],
                   v[Z]);
            printf("GJK_JH: Optimality gap reached: %e < %e\n",
                   mu,
                   dist * relError);
            printf("GJK_JH: Optimality gap reached: %e < %e\n", mu, absError);
            break;
        }
        // termination criteria -- degenerate case
        if(degenerate(all_bits, y, w))
        {
            printf("GJK_JH: Degenerate case detected\n");
            printf("GJK_JH: w = (%f, %f, %f)\n",
                   w.getBuffer()[0],
                   w.getBuffer()[1],
                   w.getBuffer()[2]);
            break;
        }
        // if not terminated, get ready for the next iteration
        y[last]  = w;
        all_bits = bits | last_bit;
        ++numIterations;
        if(!closest(bits, last, last_bit, all_bits, y, dp, det, v))
            break;
        dist = norm(v);
    }
    computePoints(bits, det, p, q, pa, pb);
    if(numIterations > 1000)
        catch_me();
    else
        nbIter = numIterations;
    return (dist);
}

// -----------------------------------------------------------------------------
// Returns the minimal distance between 2 convex shapes and a point per convex
// shape that represents the tips of the minimal distance segment
template <typename T>
__HOSTDEVICE__ T computeClosestPoints_GJK_JH(const Convex<T>&     a,
                                             const Convex<T>&     b,
                                             const Transform3<T>& a2w,
                                             const Transform3<T>& b2w,
                                             Vector3<T>&          pa,
                                             Vector3<T>&          pb,
                                             uint&                nbIter)
{
    // GJK variables
    uint       bits     = 0; // identifies current simplex
    uint       last     = 0; // identifies last found support point
    uint       last_bit = 0; // last_bit = 1<<last
    uint       all_bits = 0; // all_bits = bits|last_bit
    Vector3<T> p[4]; // support points of A in local
    Vector3<T> q[4]; // support points of B in local
    Vector3<T> y[4]; // support points of A-B in world
    T          mu            = 0.; // optimality gap
    int        numIterations = 0; // No. iterations
    T          det[16][4]    = {T(0)}; // cached sub-determinants
    T          dp[4][4]      = {T(0)};

    // Misc variables, e.g. tolerance, ...
    // relative tolerance
    constexpr T relError = LOWEPS<T>;
    // absolute tolerance
    constexpr T absError = 1.e-4 * relError;
    // isAcceleration?
    // bool acceleration = GrainsExec::m_colDetAcceleration;
    // in case we use acceleration
    // T momentum = T( 0 ), oneMinusMomentum = T( 1 );

    // Initializing vectors
    Vector3<T> v(a2w.getOrigin() - b2w.getOrigin());
    Vector3<T> w;
    T          dist = norm(v);

    while(bits < 15 && dist > HIGHEPS<T> && numIterations < 1000)
    {
        // Updating the bits, ...
        last     = 0;
        last_bit = 1;
        while(bits & last_bit)
        {
            ++last;
            last_bit <<= 1;
        }

        p[last] = a.support((-v) * a2w.getBasis());
        q[last] = b.support(v * b2w.getBasis());
        w       = a2w(p[last]) - b2w(q[last]);

        // termination criteria -- optimiality gap
        set_max(mu, v * w / dist);
        if(dist - mu <= dist * relError || mu < absError)
            break;
        // termination criteria -- degenerate case
        if(degenerate(all_bits, y, w))
            break;

        // if not terminated, get ready for the next iteration
        y[last]  = w;
        all_bits = bits | last_bit;
        ++numIterations;
        if(!closest(bits, last, last_bit, all_bits, y, dp, det, v))
            break;
        dist = norm(v);
    }
    computePoints(bits, det, p, q, pa, pb);
    if(numIterations > 1000)
        catch_me();
    else
        nbIter = numIterations;
    return (dist);
}

// -----------------------------------------------------------------------------
// Explicit instantiation
#define X(T)                                                             \
    template __HOSTDEVICE__ bool intersectGJK(const Convex<T>&     a,    \
                                              const Convex<T>&     b,    \
                                              const Transform3<T>& b2a); \
    template __HOSTDEVICE__ bool intersectGJK(const Convex<T>&     a,    \
                                              const Convex<T>&     b,    \
                                              const Transform3<T>& a2w,  \
                                              const Transform3<T>& b2w); \
    template __HOSTDEVICE__ T    computeClosestPoints_GJK_JH(            \
           const Convex<T>&     a,                                       \
           const Convex<T>&     b,                                       \
           const Transform3<T>& b2a,                                     \
           Vector3<T>&          pa,                                      \
           Vector3<T>&          pb,                                      \
           uint&                nbIter);                                                \
    template __HOSTDEVICE__ T computeClosestPoints_GJK_JH(               \
        const Convex<T>&     a,                                          \
        const Convex<T>&     b,                                          \
        const Transform3<T>& a2w,                                        \
        const Transform3<T>& b2w,                                        \
        Vector3<T>&          pa,                                         \
        Vector3<T>&          pb,                                         \
        uint&                nbIter);
X(float)
X(double)
#undef X