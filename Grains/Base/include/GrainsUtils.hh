#ifndef _GRAINSUTILS_HH_
#define _GRAINSUTILS_HH_

#include "Basic.hh"
#include "Vector3.hh"

// =============================================================================
/** @brief Miscellaneous functionalities (mostly low-level) for Grains.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
/** @name Miscellaneous functions and utilities for Grains */
//@{
// Macro for outputting CUDA errors
#define cudaErrCheck(ans) cudaAssert((ans), __FILE__, __LINE__);

/** @brief Returns CUDA error
    @param code the error code
    @param file the file name
    @param line the line number
    @param abort whether to abort the program */
__HOST__ static INLINE void
    cudaAssert(cudaError_t code, const char* file, int line, bool abort = false)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr,
                "GPUassert: %s %s %d\n",
                cudaGetErrorString(code),
                file,
                line);
        if(abort)
            exit(code);
    }
}

// -----------------------------------------------------------------------------
/** @brief computes the optimal number of threads and blocks for a given number
    of elements and an architecture
    @param numElements the number of elements
    @param numThreads the number of threads per block
    @param numBlocks the minimum number of blocks
    @param prop the device properties */
__HOST__ static INLINE void
    computeOptimalThreadsAndBlocks(const uint            numElements,
                                   const cudaDeviceProp& prop,
                                   uint&                 numThreads,
                                   uint&                 numBlocks)
{
    constexpr uint maxThreads = 256; // Avoid 1024 unless necessary
    constexpr uint minThreads = 32;
    const uint     minBlocks  = 2 * prop.multiProcessorCount;

    // Start with 128 threads and compute how many blocks we need
    numThreads = 128;
    numBlocks  = (numElements + numThreads - 1) / numThreads;

    // If we’re not filling the SMs enough, reduce threads
    if(numBlocks < minBlocks && numThreads > minThreads)
    {
        numBlocks  = minBlocks;
        numThreads = minThreads;
    }
    // If we're oversubscribing the SMs too much, increase thread count
    if(numBlocks > 4 * prop.multiProcessorCount && numThreads < maxThreads)
    {
        numThreads = maxThreads;
        numBlocks  = (numElements + numThreads - 1) / numThreads;
    }
}

// -----------------------------------------------------------------------------
/** @brief Writes a uint2 object in a string
    @param os the output stream
    @param v the uint2 object */
__HOST__ static INLINE std::ostream& operator<<(std::ostream& os,
                                                const uint2&  v)
{
    os << "(" << v.x << ", " << v.y << ")";
    return os;
}

// -----------------------------------------------------------------------------
/** @brief Writes a real number with a prescribed number of digits in a string
    @param figure the float number
    @param size number of digits */
template <typename T>
__HOST__ static constexpr INLINE std::string realToString(const T&  figure,
                                                          const int size)
{
    std::ostringstream oss;
    oss.width(size);
    oss << std::left << figure;
    return (oss.str());
}

// -----------------------------------------------------------------------------
/** @brief Writes a float number with a prescribed format and a prescribed 
    number of digits after the decimal point in a string
    @param format the format
    @param digits number of digits after the decimal point
    @param number the float number */
template <typename T>
__HOST__ static constexpr INLINE std::string realToString(
    std::ios_base::fmtflags format, const int digits, const T& number)
{
    std::ostringstream oss;
    if(number != T(0))
    {
        oss.setf(format, std::ios::floatfield);
        oss.precision(digits);
    }
    oss << number;
    return (oss.str());
}

// -----------------------------------------------------------------------------
/** @brief Writes a vector3 object in a string
    @param vec the vector3 object */
template <typename T>
__HOST__ static constexpr INLINE std::string
                                 Vector3ToString(const Vector3<T>& vec)
{
    std::ostringstream oss;
    oss << vec;
    return ("[" + oss.str() + "]");
}

// -----------------------------------------------------------------------------
/** @brief Writes a message to stdout
    @param args the output messages */
template <typename... Args>
__HOST__ static constexpr INLINE void Gout(const Args&... args)
{
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
/** @brief Writes a message to stdout with Indent (WI)
    @param numShift the number of shift characters at the beginning
    @param args the output messages */
template <typename... Args>
__HOST__ INLINE void GoutWI(const int numShift, const Args&... args)
{
    // indent
    auto shift = [](int n) { return std::string(n, ' '); };

    std::cout << shift(numShift);
    ((std::cout << args << " "), ...);
    std::cout << std::endl;
}

// -----------------------------------------------------------------------------
/** @brief Writes a message to stdout with Indent (WI)
    @param numShift the number of shift characters at the beginning
    @param args the output messages */
template <typename... Args>
__HOSTDEVICE__ INLINE void GAbort(const Args&... args)
{
#ifdef __CUDA_ARCH__
    printf("[DEVICE] ");
    (printf("%s ", args), ...);
    printf("\n");
    __trap(); // aborts the kernel
#else
    std::cerr << "[HOST] ";
    ((std::cerr << args << " "), ...);
    std::cerr << std::endl;
    std::abort();
#endif
}

// -----------------------------------------------------------------------------
/** @brief Assert function that aborts the program if the condition is false
    @param condition the condition to check
    @param args the message(s) to display if the assertion fails */
template <typename... Args>
__HOSTDEVICE__ INLINE void GAssert(bool condition, const Args&... args)
{
    if(!condition)
        GAbort("GAssert failed:", args...);
}

#endif