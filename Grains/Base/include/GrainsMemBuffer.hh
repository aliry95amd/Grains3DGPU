#ifndef _GRAINSMEMBUFFER_HH_
#define _GRAINSMEMBUFFER_HH_

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>

enum class MemType
{
    HOST,
    DEVICE,
    MANAGED,
    PINNED,
    UNKNOWN
};

// =============================================================================
/** @brief The class GrainsMemBuffer.

    This class provides a buffer that can be allocated in different memory spaces
    (host, device, managed, pinned) and provides methods for memory management
    and data transfer between these spaces.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T, MemType M = MemType::HOST>
class GrainsMemBuffer
{
protected:
    /** @name Parameters */
    //@{
    /** \brief Pointer to the data */
    T* m_ptr = nullptr;
    /** \brief Pointer to the data on device (for zero-copy) */
    T* m_d_ptr = nullptr;
    /** \brief Number of elements in the buffer */
    size_t m_size = 0;
    /** \brief Capacity of the buffer */
    size_t m_capacity = 0;
    /** \brief Next index to write to */
    size_t m_nextIndex = 0;
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor */
    GrainsMemBuffer() = default;

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~GrainsMemBuffer()
    {
        free();
    }

    // -------------------------------------------------------------------------
    /** @brief Copy constructor */
    GrainsMemBuffer(const GrainsMemBuffer&) = delete;

    // -------------------------------------------------------------------------
    /** @brief Copy assignment operator */
    GrainsMemBuffer& operator=(const GrainsMemBuffer&) = delete;

    // -------------------------------------------------------------------------
    /** @brief Move constructor */
    GrainsMemBuffer(GrainsMemBuffer&& other) noexcept
    {
        moveFrom(other);
    }

    // -------------------------------------------------------------------------
    /** @brief Move assignment operator */
    GrainsMemBuffer& operator=(GrainsMemBuffer&& other) noexcept
    {
        if(this != &other)
            moveFrom(other);
        return (*this);
    }
    //@}

    /** @name Get methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Returns the pointer to the data */
    T* getData()
    {
        return m_ptr;
    }

    // -------------------------------------------------------------------------
    /** @brief Returns the pointer to the data on device (for zero-copy) */
    /** @note This is only valid for pinned memory */
    T* getDeviceData()
    {
        if constexpr(M == MemType::PINNED)
            return m_d_ptr ? m_d_ptr : m_ptr;
        else
            return m_ptr;
    }

    // -------------------------------------------------------------------------
    /** @brief Returns the number of elements */
    size_t getSize() const
    {
        return m_size;
    }

    // -------------------------------------------------------------------------
    /** @brief Returns the size in bytes */
    size_t getBytes() const
    {
        return m_size * sizeof(T);
    }

    // -------------------------------------------------------------------------
    /** @brief Returns the type of memory */
    MemType getMemType() const
    {
        if constexpr(M == MemType::Host)
            return MemType::HOST;
        else if constexpr(M == MemType::Device)
            return MemType::DEVICE;
        else if constexpr(M == MemType::Managed)
            return MemType::MANAGED;
        else if constexpr(M == MemType::Pinned)
            return MemType::PINNED;
        else
            return MemType::UNKNOWN;
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Allocates memory of the specified type and size
    @param count number of elements */
    void allocate(size_t count)
    {
        m_size     = count;
        m_capacity = count;
        cudaError_t err;

        if constexpr(M == MemType::HOST)
        {
            m_ptr = static_cast<T*>(std::malloc(sizeof(T) * count));
            if(!m_ptr)
                throw std::bad_alloc();
        }
        else if constexpr(M == MemType::DEVICE)
        {
            err = cudaMalloc(&m_ptr, sizeof(T) * count);
            if(err != cudaSuccess)
                throw std::bad_alloc();
        }
        else if constexpr(M == MemType::MANAGED)
        {
            err = cudaMallocManaged(&m_ptr, sizeof(T) * count);
            if(err != cudaSuccess)
                throw std::bad_alloc();
        }
        else if constexpr(M == MemType::PINNED)
        {
            err = cudaMallocHost(&m_ptr, sizeof(T) * count);
            if(err != cudaSuccess)
                throw std::bad_alloc();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Reserves memory for the buffer
    @param new_capacity new capacity of the buffer */
    void reserve(size_t new_capacity)
    {
        if(new_capacity <= m_capacity)
            return;

        m_capacity = new_capacity;
        GrainsMemBuffer<T, M> new_buf;
        new_buf.allocate(new_capacity);

        cudaMemcpyKind kind;
        if constexpr(M == MemType::HOST)
            kind = cudaMemcpyHostToHost;
        else if constexpr(M == MemType::DEVICE)
            kind = cudaMemcpyDeviceToDevice;
        else if constexpr(M == MemType::MANAGED)
            kind = cudaMemcpyDeviceToDevice;
        else if constexpr(M == MemType::PINNED)
            kind = cudaMemcpyHostToHost;
        else
        {
            std::cerr << "Unsupported memory type for reserve()\n";
            return;
        }

        cudaMemcpy(new_buf.getDeviceData(),
                   getDeviceData(),
                   m_size * sizeof(T),
                   kind);

        *this = std::move(new_buf);
    }

    // -------------------------------------------------------------------------
    /** @brief Resizes the buffer
    @param new_size new size of the buffer */
    void resize(size_t new_size)
    {
        if(new_size <= m_capacity)
        {
            m_size = new_size;
            return;
        }

        T*     old_data = m_ptr;
        size_t old_size = m_size;
        allocate(new_size);

        if(old_data)
        {
            if constexpr(M == MemType::HOST)
            {
                std::memcpy(m_ptr, old_data, old_size * sizeof(T));
                std::free(old_data);
            }
            else if constexpr(M == MemType::DEVICE)
            {
                cudaMemcpy(m_ptr,
                           old_data,
                           old_size * sizeof(T),
                           cudaMemcpyDeviceToDevice);
                cudaFree(old_data);
            }
            else if constexpr(M == MemType::PINNED)
            {
                std::memcpy(m_ptr, old_data, old_size * sizeof(T));
                cudaFreeHost(old_data);
            }
            else if constexpr(M == MemType::MANAGED)
            {
                cudaMemcpy(m_ptr,
                           old_data,
                           old_size * sizeof(T),
                           cudaMemcpyDeviceToDevice);
                cudaFree(old_data);
            }
        }
        m_size     = new_size;
        m_capacity = new_size;
    }

    // -------------------------------------------------------------------------
    /** @brief Pushes a new element to the back of the buffer 
    @param value value to push */
    void push_back(const T& value)
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            if(m_nextIndex >= m_capacity)
            {
                std::cerr << "GrainsMemBuffer::push_back overflow\n";
                return;
            }
            getData()[m_nextIndex++] = value;
            m_size                   = m_nextIndex;
        }
        else
            std::cerr << "GrainsMemBuffer::push_back() only allowed on host or "
                         "pinned memory\n";
    }

    // -------------------------------------------------------------------------
    /** @brief Pushes a bulk of elements to the back of the buffer
    @param values pointer to the array of values
    @param count number of elements to push */
    void push_bulk(const T* values, size_t count)
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            if(m_nextIndex + count > m_capacity)
            {
                std::cerr << "GrainsMemBuffer::push_bulk overflow\n";
                return;
            }
            T* dst = getData() + m_nextIndex;
            std::memcpy(dst, values, count * sizeof(T));
            m_nextIndex += count;
            m_size = m_nextIndex;
        }
        else
            std::cerr << "GrainsMemBuffer::push_bulk() only allowed on host or "
                         "pinned memory\n";
    }

    // -------------------------------------------------------------------------
    /** @brief Shrinks the buffer to fit the current size */
    void shrink_to_fit()
    {
        if(m_size == m_capacity)
            return;

        GrainsMemBuffer<T, M> new_buf;
        new_buf.allocate(m_size);

        if constexpr(M == MemType::HOST || M == MemType::PINNED)
            std::memcpy(new_buf.getData(), getData(), m_size * sizeof(T));
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
            cudaMemcpy(new_buf.getDeviceData(),
                       getDeviceData(),
                       m_size * sizeof(T),
                       kind);
        }
        else
        {
            std::cerr << "Unsupported memory type for shrink_to_fit()\n";
            return;
        }

        *this = std::move(new_buf);
    }

    // -------------------------------------------------------------------------
    /** @brief Copy to another buffer (host/device aware)
    @param dest destination buffer
    @param kind type of copy (host to device, device to host, etc.)
    @param stream CUDA stream to use for asynchronous operations */
    template <MemType M2>
    void copy_to(GrainsMemBuffer<T, M2>& dest, cudaMemcpyKind kind) const
    {
        if(m_size == 0 || !m_ptr)
            return;

        if(dest.m_size < m_size)
        {
            std::cerr << "Destination buffer too small for copy\n";
            return;
        }

        if constexpr(M == MemType::HOST && M2 == MemType::HOST)
        {
            std::memcpy(dest.getData(), getData(), getBytes());
        }
        else
        {
            const T*    src_ptr = this->getDeviceData();
            T*          dst_ptr = dest.getDeviceData();
            cudaError_t err = cudaMemcpy(dst_ptr, src_ptr, getBytes(), kind);
            if(err != cudaSuccess)
                std::cerr << "cudaMemcpy failed in copy_to: "
                          << cudaGetErrorString(err) << "\n";
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Frees the allocated memory */
    void free()
    {
        if(m_ptr)
        {
            if constexpr(M == MemType::HOST)
                std::free(m_ptr);
            else if constexpr(M == MemType::DEVICE)
                cudaFree(m_ptr);
            else if constexpr(M == MemType::PINNED)
                cudaFreeHost(m_ptr);
            else if constexpr(M == MemType::MANAGED)
                cudaFree(m_ptr);
            else
                std::cerr << "Unknown memory type for free()\n";
        }

        m_ptr       = nullptr;
        m_d_ptr     = nullptr;
        m_size      = 0;
        m_capacity  = 0;
        m_nextIndex = 0;
    }
    //@}

    /** @name Operators */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Index operator 
    @param i index of the element */
    T& operator[](size_t i)
    {
        static_assert(M == MemType::HOST || M == MemType::PINNED,
                      "operator[] only available for HOST or PINNED memory");
        assert(i < m_size);
        return m_ptr[i];
    }

    // -------------------------------------------------------------------------
    /** @brief Index operator
    @param i index of the element */
    const T& operator[](size_t i) const
    {
        static_assert(M == MemType::HOST || M == MemType::PINNED,
                      "operator[] only available for HOST or PINNED memory");
        assert(i < m_size);
        return m_ptr[i];
    }

private:
    // -------------------------------------------------------------------------
    /** @brief Move from another buffer
    @param other source buffer */
    void moveFrom(GrainsMemBuffer<T, M>& other)
    {
        m_ptr       = other.m_ptr;
        m_d_ptr     = other.m_d_ptr;
        m_size      = other.m_size;
        m_capacity  = other.m_capacity;
        m_nextIndex = other.m_nextIndex;

        other.m_ptr       = nullptr;
        other.m_d_ptr     = nullptr;
        other.m_size      = 0;
        other.m_capacity  = 0;
        other.m_nextIndex = 0;
    }
};

#endif
