#ifndef _CONTACTTABLE_HH_
#define _CONTACTTABLE_HH_

#include "Basic.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief Hash table entry for contact tracking.

    Each entry stores a key (pair of component IDs), an index pointing to the ContactForce in a
    flat array, and a flag for hash table management.

    @author A.Yazdani - 2026 - Construction */
// =================================================================================================
struct ContactEntry
{
    /** @name Parameters */
    //@{
    /** \brief Contact pair key (i,j) where i < j */
    uint2 m_key;
    /** \brief Index into the ContactForce array */
    uint m_index;
    /** \brief Validity flag for the contact entry */
    uint m_valid;
    //@}

    /** @name Constructors */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor */
    __HOSTDEVICE__
    ContactEntry()
        : m_key(make_uint2(0, 0))
        , m_index(0xFFFFFFFF)  // Invalid index marker
        , m_valid(0)
    {
    }
    //@}
};

// =================================================================================================
/** @brief Lightweight view of ContactHashTable for passing to kernels.

    This structure contains only raw pointers and primitive types, making it safe to pass to
    kernels by value. Obtain this view from ContactHashTable::getView() and pass it to kernels.

    @author A.Yazdani - 2026 - Construction */
// =================================================================================================
struct ContactHashTableView
{
    /** @name Parameters */
    //@{
    /** \brief Pointer to hash table entries */
    ContactEntry* m_table;
    /** \brief Total capacity of the hash table */
    uint m_capacity;
    /** \brief Pointer to the next index counter */
    uint* m_nextIndex;
    //@}

    /** @name Constructors */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor */
    __HOSTDEVICE__
    ContactHashTableView()
        : m_table(nullptr)
        , m_capacity(0)
        , m_nextIndex(nullptr)
    {
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param table pointer to hash table entries
        @param capacity total capacity of the hash table
        @param nextIndex pointer to the next index counter */
    __HOSTDEVICE__
    ContactHashTableView(ContactEntry* table, uint capacity, uint* nextIndex)
        : m_table(table)
        , m_capacity(capacity)
        , m_nextIndex(nextIndex)
    {
    }
    //@}

    /** @name Hash table operations */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Finds an existing contact index in the hash table
        @param key pair of component IDs (i,j) where i < j
        @param index output parameter for the contact state index
        @return true if found, false otherwise */
    __DEVICE__
    bool find(uint2 key, uint& index) const
    {
        if(m_capacity == 0 || m_table == nullptr)
            return false;

        uint h = primeHash(key) % m_capacity;

        // Linear probing with unrolling hint for better performance
#pragma unroll 4
        for(uint i = 0; i < m_capacity; ++i)
        {
            uint                idx = (h + i) % m_capacity;
            const ContactEntry& e   = m_table[idx];

            // Empty slot found - key not in table
            if(e.m_valid == 0)
                return false;

            // Key found - ensure all writes are visible before reading
            if(e.m_valid == 1 && e.m_key.x == key.x && e.m_key.y == key.y)
            {
                __threadfence();
                index = e.m_index;
                return true;
            }
        }

        // Table full, key not found
        return false;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Finds an existing contact or inserts a new one
        @param key pair of component IDs (i,j) where i < j
        @param index output parameter for the contact state index
        @return true if found or inserted, false if table is full */
    __DEVICE__
    bool findOrInsert(uint2 key, uint& index)
    {
        if(m_capacity == 0 || m_table == nullptr || m_nextIndex == nullptr)
            return false;

        uint h = primeHash(key) % m_capacity;

        for(uint i = 0; i < m_capacity; ++i)
        {
            uint          idx = (h + i) % m_capacity;
            ContactEntry& e   = m_table[idx];

            // Empty slot - try to claim it atomically
            if(e.m_valid == 0)
            {
                // Atomic compare-and-swap: if e.m_valid is 0, set it to 1
                if(atomicCAS(&e.m_valid, 0u, 1u) == 0u)
                {
                    // Successfully claimed this slot
                    e.m_key = key;

                    // Assign new index atomically
                    uint newIndex = atomicAdd(m_nextIndex, 1u);
                    e.m_index     = newIndex;

                    // Ensure writes are visible to all threads before they can read
                    __threadfence();

                    index = newIndex;
                    return true;
                }
                // Another thread claimed it, continue probing
            }
            // Slot already occupied - check if it's our key
            else if(e.m_valid == 1 && e.m_key.x == key.x && e.m_key.y == key.y)
            {
                // Ensure we read the fully written data
                __threadfence();
                index = e.m_index;
                return true;
            }
            // Different key, continue probing
        }

        // Hash table is full - cannot insert
        return false;
    }

    // ---------------------------------------------------------------------------------------------
    /** @brief Removes a contact from the hash table
        @param key pair of component IDs (i,j) where i < j
        @return true if removed, false if not found */
    __DEVICE__
    bool remove(uint2 key)
    {
        if(m_capacity == 0 || m_table == nullptr)
            return false;

        uint h = primeHash(key) % m_capacity;

        for(uint i = 0; i < m_capacity; ++i)
        {
            uint          idx = (h + i) % m_capacity;
            ContactEntry& e   = m_table[idx];

            // Empty slot - key not found
            if(e.m_valid == 0)
                return false;

            // Key found - remove it
            if(e.m_key.x == key.x && e.m_key.y == key.y)
            {
                // Mark as invalid atomically
                atomicExch(&e.m_valid, 0u);
                return true;
            }
        }

        // Key not found
        return false;
    }
    //@}
};

// =================================================================================================
/** @brief Hash table manager for contact history tracking.

    This class manages memory allocation and provides a lightweight view for kernel use. The hash
    table stores indices pointing to a flat array of ContactForce objects. This class should NOT be
    passed to kernels. Instead, use getView() to obtain a ContactHashTableView that can be safely
    passed to kernels by value.

    @author A.Yazdani - 2026 - Construction */
// =================================================================================================
template <MemType M = MemType::HOST>
class ContactHashTable
{
private:
    /** @name Parameters */
    //@{
    /** \brief Hash table entries buffer */
    GrainsMemBuffer<ContactEntry, M> m_table;
    /** \brief Total capacity of the hash table */
    uint m_capacity;
    /** \brief Pointer to the next index counter (in device/host memory) */
    uint* m_nextIndex;
    //@}

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor */
    ContactHashTable();

    /** @brief Constructor with specified capacity
        @param capacity number of entries in the hash table */
    ContactHashTable(uint capacity);

    /** @brief Destructor */
    ~ContactHashTable();
    //@}

    /** @name Get methods */
    //@{
    /** @brief Gets a lightweight view for passing to kernels */
    ContactHashTableView getView();

    /** @brief Gets the pointer to the table data */
    const ContactEntry* getTable() const;
    /** @brief Gets the pointer to the table data (mutable) */
    ContactEntry* getTable();

    /** @brief Gets the capacity of the hash table */
    uint getCapacity() const;

    /** @brief Gets the current next index value (host-side read only) */
    uint getNextIndex() const;
    //@}

    /** @name Memory management methods */
    //@{
    /** @brief Allocates memory for the hash table
        @param capacity number of entries to allocate */
    void allocate(uint capacity);

    /** @brief Frees memory used by the hash table */
    void deallocate();

    /** @brief Clears all entries in the hash table */
    void clear();

    /** @brief Resets the index counter */
    void resetIndexCounter();
    //@}

    /** @name Methods */
    //@{
    /** @brief Finds an existing contact index in the hash table (host-side)
        @param key pair of component IDs (i,j) where i < j
        @param index output parameter for the contact state index */
    bool find(uint2 key, uint& index) const;

    /** @brief Finds an existing contact or inserts a new one (host-side)
        @param key pair of component IDs (i,j) where i < j
        @param index output parameter for the contact state index */
    bool findOrInsert(uint2 key, uint& index);

    /** @brief Removes a contact from the hash table (host-side)
        @param key pair of component IDs (i,j) where i < j */
    bool remove(uint2 key);
    //@}
};

#endif
