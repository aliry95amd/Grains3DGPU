#include "ContactTable.hh"

// -------------------------------------------------------------------------------------------------
// Default constructor
template <MemType M>
ContactHashTable<M>::ContactHashTable()
    : m_table()
    , m_capacity(0)
    , m_nextIndex(nullptr)
{
}

// -------------------------------------------------------------------------------------------------
// Constructor with specified capacity
template <MemType M>
ContactHashTable<M>::ContactHashTable(uint capacity)
    : m_table()
    , m_capacity(0)
    , m_nextIndex(nullptr)
{
    allocate(capacity);
}

// -------------------------------------------------------------------------------------------------
// Destructor
template <MemType M>
ContactHashTable<M>::~ContactHashTable()
{
    deallocate();
}

// -------------------------------------------------------------------------------------------------
// Gets a lightweight view for passing to kernels
template <MemType M>
ContactHashTableView ContactHashTable<M>::getView()
{
    return ContactHashTableView(m_table.getData(), m_capacity, m_nextIndex);
}

// -------------------------------------------------------------------------------------------------
// Gets the pointer to the table data
template <MemType M>
const ContactEntry* ContactHashTable<M>::getTable() const
{
    return m_table.getData();
}

// -------------------------------------------------------------------------------------------------
// Gets mutable pointer to the table data
template <MemType M>
ContactEntry* ContactHashTable<M>::getTable()
{
    return m_table.getData();
}

// -------------------------------------------------------------------------------------------------
// Gets the capacity of the hash table
template <MemType M>
uint ContactHashTable<M>::getCapacity() const
{
    return m_capacity;
}

// -------------------------------------------------------------------------------------------------
// Gets the current next index value
template <MemType M>
uint ContactHashTable<M>::getNextIndex() const
{
    if(m_nextIndex == nullptr)
        return 0;

    if constexpr(M == MemType::HOST)
    {
        return *m_nextIndex;
    }
    else if constexpr(M == MemType::DEVICE)
    {
        // Copy from device to host
        uint hostValue;
        cudaErrCheck(cudaMemcpy(&hostValue, m_nextIndex, sizeof(uint), cudaMemcpyDeviceToHost));
        return hostValue;
    }

    return 0;
}

// -------------------------------------------------------------------------------------------------
// Allocates memory for the hash table
template <MemType M>
void ContactHashTable<M>::allocate(uint capacity)
{
    // Deallocate existing memory if any
    if(m_table.getSize() > 0 || m_nextIndex != nullptr)
    {
        deallocate();
    }

    m_capacity = capacity;

    // Allocate and initialize the table
    m_table.initialize(capacity);
    if constexpr(M == MemType::HOST)
    {
        m_table.fill(ContactEntry());
    }
    else if constexpr(M == MemType::DEVICE)
    {
        // Initialize device memory to zero. ContactEntry is not a POD primitive,
        // so use cudaMemset to avoid instantiating device-side constructors.
        cudaErrCheck(cudaMemset(m_table.getData(), 0, capacity * sizeof(ContactEntry)));
    }

    // Allocate the index counter
    if constexpr(M == MemType::HOST)
    {
        m_nextIndex  = new uint;
        *m_nextIndex = 0;
    }
    else if constexpr(M == MemType::DEVICE)
    {
        cudaErrCheck(cudaMalloc(&m_nextIndex, sizeof(uint)));
        cudaErrCheck(cudaMemset(m_nextIndex, 0, sizeof(uint)));
    }
}

// -------------------------------------------------------------------------------------------------
// Frees memory used by the hash table
template <MemType M>
void ContactHashTable<M>::deallocate()
{
    m_table.free();
    m_capacity = 0;

    if(m_nextIndex != nullptr)
    {
        if constexpr(M == MemType::HOST)
        {
            delete m_nextIndex;
        }
        else if constexpr(M == MemType::DEVICE)
        {
            cudaFree(m_nextIndex);
        }
        m_nextIndex = nullptr;
    }
}

// -------------------------------------------------------------------------------------------------
// Clears all entries in the hash table
template <MemType M>
void ContactHashTable<M>::clear()
{
    if(m_table.getSize() > 0)
    {
        if constexpr(M == MemType::HOST)
        {
            m_table.fill(ContactEntry());
        }
        else if constexpr(M == MemType::DEVICE)
        {
            cudaErrCheck(cudaMemset(m_table.getData(), 0, m_capacity * sizeof(ContactEntry)));
        }
    }

    if(m_nextIndex != nullptr)
    {
        if constexpr(M == MemType::HOST)
        {
            *m_nextIndex = 0;
        }
        else if constexpr(M == MemType::DEVICE)
        {
            cudaErrCheck(cudaMemset(m_nextIndex, 0, sizeof(uint)));
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Resets the index counter
template <MemType M>
void ContactHashTable<M>::resetIndexCounter()
{
    if(m_nextIndex != nullptr)
    {
        if constexpr(M == MemType::HOST)
        {
            *m_nextIndex = 0;
        }
        else if constexpr(M == MemType::DEVICE)
        {
            cudaErrCheck(cudaMemset(m_nextIndex, 0, sizeof(uint)));
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Finds an existing contact index in the hash table
template <MemType M>
bool ContactHashTable<M>::find(uint2 key, uint& index) const
{
    GAssert(M == MemType::HOST, "find() is only supported for HOST memory type.");
    if(m_capacity == 0 || m_table.getData() == nullptr)
        return false;

    const ContactEntry* table = m_table.getData();
    uint                h     = primeHash(key) % m_capacity;

    for(uint i = 0; i < m_capacity; ++i)
    {
        uint                idx = (h + i) % m_capacity;
        const ContactEntry& e   = table[idx];

        // Empty slot found - key not in table
        if(e.m_valid == 0)
            return false;

        // Key found
        if(e.m_valid == 1 && e.m_key.x == key.x && e.m_key.y == key.y)
        {
            index = e.m_index;
            return true;
        }
    }

    // Table full, key not found
    return false;
}

// -------------------------------------------------------------------------------------------------
// Finds an existing contact or inserts a new one
template <MemType M>
bool ContactHashTable<M>::findOrInsert(uint2 key, uint& index)
{
    GAssert(M == MemType::HOST, "findOrInsert() is only supported for HOST memory type.");
    if(m_capacity == 0 || m_table.getData() == nullptr || m_nextIndex == nullptr)
        return false;

    ContactEntry* table = m_table.getData();
    uint          h     = primeHash(key) % m_capacity;

    for(uint i = 0; i < m_capacity; ++i)
    {
        uint          idx = (h + i) % m_capacity;
        ContactEntry& e   = table[idx];

        // Empty slot - claim it
        if(e.m_valid == 0)
        {
            e.m_valid = 1;
            e.m_key   = key;
            e.m_index = (*m_nextIndex)++;
            index     = e.m_index;
            return true;
        }
        // Slot already occupied - check if it's our key
        else if(e.m_valid == 1 && e.m_key.x == key.x && e.m_key.y == key.y)
        {
            index = e.m_index;
            return true;
        }
        // Different key, continue probing
    }

    // Hash table is full - cannot insert
    return false;
}

// -------------------------------------------------------------------------------------------------
// Removes a contact from the hash table
template <MemType M>
bool ContactHashTable<M>::remove(uint2 key)
{
    GAssert(M == MemType::HOST, "remove() is only supported for HOST memory type.");

    if(m_capacity == 0 || m_table.getData() == nullptr)
        return false;

    ContactEntry* table = m_table.getData();
    uint          h     = primeHash(key) % m_capacity;

    for(uint i = 0; i < m_capacity; ++i)
    {
        uint          idx = (h + i) % m_capacity;
        ContactEntry& e   = table[idx];

        // Empty slot - key not found
        if(e.m_valid == 0)
            return false;

        // Key found - remove it
        if(e.m_key.x == key.x && e.m_key.y == key.y)
        {
            e.m_valid = 0;
            return true;
        }
    }

    // Key not found
    return false;
}

// -------------------------------------------------------------------------------------------------
// Explicit template instantiation
template class ContactHashTable<MemType::HOST>;
template class ContactHashTable<MemType::DEVICE>;
