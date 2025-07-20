#ifndef _GRAINSMEMBUFFER_HH_
#define _GRAINSMEMBUFFER_HH_

#include <type_traits>

#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "Misc_Kernels.hh"

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
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor */
    GrainsMemBuffer() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with the size */
    GrainsMemBuffer(size_t size)
    {
        allocate(size);
        fill();
    }

    // -------------------------------------------------------------------------
    /** @brief Constructor with the size and default value
        @param size size of the buffer
        @param value default value to fill the buffer */
    GrainsMemBuffer(size_t size, const T& value)
    {
        allocate(size);
        fill(value);
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~GrainsMemBuffer()
    {
        free();
    }

    // -------------------------------------------------------------------------
    /** @brief Copy constructor
    @param other the other buffer to copy from */
    template <MemType srcM>
    GrainsMemBuffer(const GrainsMemBuffer<T, srcM>& other)
    {
        copyFrom(other);
    }

    // -------------------------------------------------------------------------
    /** @brief Copy assignment operator 
    @param other the other buffer to copy from */
    template <MemType srcM>
    GrainsMemBuffer<T, M>& operator=(const GrainsMemBuffer<T, srcM>& other)
    {
        if(this != &other)
            copyFrom(other);
        return *this;
    }

    // -------------------------------------------------------------------------
    /** @brief Move constructor */
    GrainsMemBuffer(GrainsMemBuffer<T, M>&& other) noexcept
    {
        moveFrom(other);
    }

    // -------------------------------------------------------------------------
    /** @brief Move assignment operator */
    GrainsMemBuffer& operator=(GrainsMemBuffer<T, M>&& other) noexcept
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
    /** @brief Returns the pointer to the data */
    const T* getData() const
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
        if constexpr(M == MemType::HOST)
            return MemType::HOST;
        else if constexpr(M == MemType::DEVICE)
            return MemType::DEVICE;
        else if constexpr(M == MemType::MANAGED)
            return MemType::MANAGED;
        else if constexpr(M == MemType::PINNED)
            return MemType::PINNED;
        else
            return MemType::UNKNOWN;
    }
    //@}

    /** @name Set methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Sets the size of the buffer without reallocation
    @param new_size new size of the buffer (must be <= capacity) */
    void setSize(size_t new_size)
    {
        if(new_size > m_capacity)
        {
            std::cerr
                << "GrainsMemBuffer::setSize() new size exceeds capacity\n";
            return;
        }
        m_size = new_size;
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

        if constexpr(M == MemType::HOST)
        {
            m_ptr = static_cast<T*>(std::malloc(sizeof(T) * count));
            if(!m_ptr)
                throw std::bad_alloc();
        }
        else if constexpr(M == MemType::DEVICE)
        {
            cudaErrCheck(cudaMalloc(&m_ptr, sizeof(T) * count));
        }
        else if constexpr(M == MemType::MANAGED)
        {
            cudaErrCheck(cudaMallocManaged(&m_ptr, sizeof(T) * count));
        }
        else if constexpr(M == MemType::PINNED)
        {
            cudaErrCheck(cudaMallocHost(&m_ptr, sizeof(T) * count));
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

        cudaMemcpyKind kind = getMemcpyKind<M, M>();

        cudaErrCheck(cudaMemcpy(new_buf.getDeviceData(),
                                getDeviceData(),
                                m_size * sizeof(T),
                                kind));

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
                cudaErrCheck(cudaMemcpy(m_ptr,
                                        old_data,
                                        old_size * sizeof(T),
                                        cudaMemcpyDeviceToDevice));
                cudaErrCheck(cudaFree(old_data));
            }
            else if constexpr(M == MemType::PINNED)
            {
                std::memcpy(m_ptr, old_data, old_size * sizeof(T));
                cudaErrCheck(cudaFreeHost(old_data));
            }
            else if constexpr(M == MemType::MANAGED)
            {
                cudaErrCheck(cudaMemcpy(m_ptr,
                                        old_data,
                                        old_size * sizeof(T),
                                        cudaMemcpyDeviceToDevice));
                cudaErrCheck(cudaFree(old_data));
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
            if(m_size >= m_capacity)
            {
                std::cerr << "GrainsMemBuffer::push_back overflow\n";
                return;
            }
            m_ptr[m_size++] = value;
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
            if(m_size + count > m_capacity)
            {
                std::cerr << "GrainsMemBuffer::push_bulk overflow\n";
                return;
            }
            T* dst = m_ptr + m_size;
            std::memcpy(dst, values, count * sizeof(T));
            m_size += count;
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

        GrainsMemBuffer<T, M> new_buf{};
        new_buf.allocate(m_size);

        if constexpr(M == MemType::HOST || M == MemType::PINNED)
            std::memcpy(new_buf.m_ptr, m_ptr, m_size * sizeof(T));
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            cudaMemcpyKind kind = cudaMemcpyDeviceToDevice;
            cudaErrCheck(
                cudaMemcpy(new_buf.getData(), m_ptr, m_size * sizeof(T), kind));
        }
        else
        {
            std::cerr << "Unsupported memory type for shrink_to_fit()\n";
            return;
        }

        *this = std::move(new_buf);
    }

    // -------------------------------------------------------------------------
    /** @brief Returns the kind of memory transfer for the given source and
    destination memory types */
    template <MemType Src, MemType Dst>
    constexpr cudaMemcpyKind getMemcpyKind()
    {
        if constexpr(Src == MemType::HOST && Dst == MemType::HOST)
            return cudaMemcpyHostToHost;
        else if constexpr(Src == MemType::HOST && Dst == MemType::DEVICE)
            return cudaMemcpyHostToDevice;
        else if constexpr(Src == MemType::DEVICE && Dst == MemType::HOST)
            return cudaMemcpyDeviceToHost;
        else if constexpr(Src == MemType::DEVICE && Dst == MemType::DEVICE)
            return cudaMemcpyDeviceToDevice;
        else if constexpr(Src == MemType::HOST && Dst == MemType::MANAGED)
            return cudaMemcpyHostToDevice;
        else if constexpr(Src == MemType::MANAGED && Dst == MemType::HOST)
            return cudaMemcpyDeviceToHost;
        else if constexpr(Src == MemType::DEVICE && Dst == MemType::MANAGED)
            return cudaMemcpyDeviceToDevice;
        else if constexpr(Src == MemType::MANAGED && Dst == MemType::DEVICE)
            return cudaMemcpyDeviceToDevice;
        else if constexpr(Src == MemType::MANAGED && Dst == MemType::MANAGED)
            return cudaMemcpyDeviceToDevice;
        else if constexpr(Src == MemType::PINNED && Dst == MemType::HOST)
            return cudaMemcpyHostToHost;
        else if constexpr(Src == MemType::HOST && Dst == MemType::PINNED)
            return cudaMemcpyHostToHost;
        else if constexpr(Src == MemType::PINNED && Dst == MemType::DEVICE)
            return cudaMemcpyHostToDevice;
        else if constexpr(Src == MemType::DEVICE && Dst == MemType::PINNED)
            return cudaMemcpyDeviceToHost;
        else if constexpr(Src == MemType::PINNED && Dst == MemType::PINNED)
            return cudaMemcpyHostToHost;
        else
            return cudaMemcpyDefault;
    }

    // -------------------------------------------------------------------------
    /** @brief Copy to another buffer (host/device aware)
    @param dest destination buffer */
    template <MemType destM>
    void copyTo(GrainsMemBuffer<T, destM>& dest)
    {
        if(m_size == 0 || !m_ptr)
            return;

        if(dest.getSize() < m_size)
            GAbort("Destination buffer too small for copy");

        if constexpr(M == MemType::HOST && destM == MemType::HOST)
            std::memcpy(dest.getData(), m_ptr, getBytes());
        else
        {
            cudaMemcpyKind kind    = getMemcpyKind<M, destM>();
            const T*       src_ptr = m_ptr;
            T*             dst_ptr = dest.getData();
            cudaErrCheck(cudaMemcpy(dst_ptr, src_ptr, getBytes(), kind));
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Copy from another buffer (host/device aware)
    @param src source buffer */
    template <MemType srcM>
    void copyFrom(const GrainsMemBuffer<T, srcM>& src)
    {
        if constexpr(M == srcM)
        {
            if(this == &src)
                return;
        }

        // Resize this buffer if needed
        if(m_capacity < src.getSize())
            resize(src.getSize());
        m_size = src.getSize();

        if constexpr(M == MemType::HOST && srcM == MemType::HOST)
        {
            std::memcpy(m_ptr, src.m_ptr, src.getBytes());
        }
        else
        {
            cudaMemcpyKind kind    = getMemcpyKind<srcM, M>();
            const T*       src_ptr = src.getData();
            T*             dst_ptr = m_ptr;
            cudaErrCheck(cudaMemcpy(dst_ptr, src_ptr, src.getBytes(), kind));
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Frees the allocated memory */
    void free()
    {
        if(m_ptr)
        {
            if constexpr(M == MemType::HOST)
            {
                std::free(m_ptr);
            }
            else if constexpr(M == MemType::DEVICE)
            {
                cudaErrCheck(cudaFree(m_ptr));
            }
            else if constexpr(M == MemType::PINNED)
            {
                cudaErrCheck(cudaFreeHost(m_ptr));
            }
            else if constexpr(M == MemType::MANAGED)
            {
                cudaErrCheck(cudaFree(m_ptr));
            }
            else
            {
                std::cerr << "Unknown memory type for free()\n";
            }
        }

        m_ptr      = nullptr;
        m_d_ptr    = nullptr;
        m_size     = 0;
        m_capacity = 0;
    }

    // -------------------------------------------------------------------------
    /** @brief Fills the buffer with default values */
    void fill()
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            for(size_t i = 0; i < m_size; ++i)
            {
                m_ptr[i] = T();
            }
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            fill_Kernel<<<(m_size + 255) / 256, 256>>>(m_ptr, m_size);
            cudaDeviceSynchronize();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Fills the buffer with a user-provided value
        @param count number of elements
        @param value value to initialize with */
    void fill(const T& value)
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            for(size_t i = 0; i < m_size; ++i)
                m_ptr[i] = value;
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            static_assert(std::is_fundamental<T>::value,
                          "T must be a primitive type for device init");
            fill_Kernel<<<(m_size + 255) / 256, 256>>>(m_ptr, m_size, value);
            cudaDeviceSynchronize();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Fills the buffer incrementally with values starting from 0 */
    void fillIncremental()
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            for(size_t i = 0; i < m_size; ++i)
                m_ptr[i] = static_cast<T>(i);
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            static_assert(std::is_fundamental<T>::value,
                          "T must be a primitive type for device init");
            fillIncremental_Kernel<<<(m_size + 255) / 256, 256>>>(m_ptr,
                                                                  m_size);
            cudaDeviceSynchronize();
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Prints the buffer contents (host/pinned/device/managed) */
    void print(const std::string& label = "") const
    {
        if constexpr(M == MemType::HOST || M == MemType::PINNED)
        {
            if(!label.empty())
                std::cout << label << ": ";
            for(size_t i = 0; i < m_size; ++i)
                std::cout << m_ptr[i] << " ";
            std::cout << std::endl;
        }
        else if constexpr(M == MemType::DEVICE || M == MemType::MANAGED)
        {
            // Copy to host and print
            std::vector<T> hostBuf(m_size);
            cudaErrCheck(cudaMemcpy(hostBuf.data(),
                                    m_ptr,
                                    getBytes(),
                                    cudaMemcpyDeviceToHost));
            if(!label.empty())
                std::cout << label << ": ";
            for(size_t i = 0; i < m_size; ++i)
                std::cout << hostBuf[i] << " ";
            std::cout << std::endl;
        }
        else
        {
            std::cerr << "print() unsupported for this memory type"
                      << std::endl;
        }
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
        m_ptr      = other.m_ptr;
        m_d_ptr    = other.m_d_ptr;
        m_size     = other.m_size;
        m_capacity = other.m_capacity;

        other.m_ptr      = nullptr;
        other.m_d_ptr    = nullptr;
        other.m_size     = 0;
        other.m_capacity = 0;
    }
};

#endif
