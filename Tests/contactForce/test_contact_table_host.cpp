#include <gtest/gtest.h>

#include "ContactTable.hh"
#include "GrainsUtils.hh"

// Host-side tests extracted from combined test file

class ContactTableHostTest : public ::testing::Test
{
protected:
    ContactHashTable<MemType::HOST> table;

    void SetUp() override
    {
        table.allocate(100);
    }

    void TearDown() override
    {
        table.deallocate();
    }
};

TEST_F(ContactTableHostTest, AllocationAndDeallocation)
{
    ContactHashTable<MemType::HOST> table;
    EXPECT_EQ(table.getCapacity(), 0);
    EXPECT_EQ(table.getNextIndex(), 0);
    table.allocate(100);
    EXPECT_EQ(table.getCapacity(), 100);
    EXPECT_EQ(table.getNextIndex(), 0);
    table.deallocate();
    EXPECT_EQ(table.getCapacity(), 0);
    EXPECT_EQ(table.getNextIndex(), 0);
}

TEST_F(ContactTableHostTest, ConstructorWithCapacity)
{
    ContactHashTable<MemType::HOST> table(50);
    EXPECT_EQ(table.getCapacity(), 50);
    EXPECT_EQ(table.getNextIndex(), 0);
}

TEST_F(ContactTableHostTest, BasicInsertion)
{
    uint2 pair1 = make_uint2(5, 10);
    uint  index1;
    bool  success = table.findOrInsert(pair1, index1);
    EXPECT_TRUE(success);
    EXPECT_EQ(index1, 0);
    EXPECT_EQ(table.getNextIndex(), 1);
}

TEST_F(ContactTableHostTest, FindExisting)
{
    uint2 pair1 = make_uint2(5, 10);
    uint  index1, index2;
    table.findOrInsert(pair1, index1);
    bool found = table.find(pair1, index2);
    EXPECT_TRUE(found);
    EXPECT_EQ(index1, index2);
}

TEST_F(ContactTableHostTest, FindNonExistent)
{
    uint2 pair1 = make_uint2(5, 10);
    uint2 pair2 = make_uint2(7, 12);
    uint  index;
    table.findOrInsert(pair1, index);
    bool found = table.find(pair2, index);
    EXPECT_FALSE(found);
}

TEST_F(ContactTableHostTest, MultipleInsertions)
{
    const int numPairs = 10;
    uint      indices[numPairs];
    for(int i = 0; i < numPairs; ++i)
    {
        uint2 pair    = make_uint2(i, i + 10);
        bool  success = table.findOrInsert(pair, indices[i]);
        EXPECT_TRUE(success);
        EXPECT_EQ(indices[i], i);
    }
    EXPECT_EQ(table.getNextIndex(), numPairs);
}

TEST_F(ContactTableHostTest, DuplicateInsertion)
{
    uint2 pair = make_uint2(5, 10);
    uint  index1, index2;
    table.findOrInsert(pair, index1);
    table.findOrInsert(pair, index2);
    EXPECT_EQ(index1, index2);
    EXPECT_EQ(table.getNextIndex(), 1);
}

TEST_F(ContactTableHostTest, Removal)
{
    uint2 pair = make_uint2(5, 10);
    uint  index;
    table.findOrInsert(pair, index);
    bool removed = table.remove(pair);
    EXPECT_TRUE(removed);
    bool found = table.find(pair, index);
    EXPECT_FALSE(found);
}

TEST_F(ContactTableHostTest, RemoveNonExistent)
{
    uint2 pair    = make_uint2(5, 10);
    bool  removed = table.remove(pair);
    EXPECT_FALSE(removed);
}

TEST_F(ContactTableHostTest, ClearOperation)
{
    for(int i = 0; i < 5; ++i)
    {
        uint2 pair = make_uint2(i, i + 10);
        uint  index;
        table.findOrInsert(pair, index);
    }
    EXPECT_EQ(table.getNextIndex(), 5);
    table.clear();
    EXPECT_EQ(table.getNextIndex(), 0);
    uint index;
    for(int i = 0; i < 5; ++i)
    {
        uint2 pair  = make_uint2(i, i + 10);
        bool  found = table.find(pair, index);
        EXPECT_FALSE(found);
    }
}

TEST_F(ContactTableHostTest, ResetIndexCounter)
{
    uint2 pair = make_uint2(5, 10);
    uint  index;
    table.findOrInsert(pair, index);
    EXPECT_EQ(table.getNextIndex(), 1);
    table.resetIndexCounter();
    EXPECT_EQ(table.getNextIndex(), 0);
}

TEST_F(ContactTableHostTest, FullTable)
{
    const int                       capacity = 10;
    ContactHashTable<MemType::HOST> table(capacity);
    for(int i = 0; i < capacity; ++i)
    {
        uint2 pair = make_uint2(i, i + 100);
        uint  index;
        bool  success = table.findOrInsert(pair, index);
        EXPECT_TRUE(success);
    }
}

TEST_F(ContactTableHostTest, HashCollisions)
{
    uint indices[5];
    for(int i = 0; i < 5; ++i)
    {
        uint2 pair    = make_uint2(i * 1000, i * 1000 + 1);
        bool  success = table.findOrInsert(pair, indices[i]);
        EXPECT_TRUE(success);
    }
    for(int i = 0; i < 5; ++i)
    {
        for(int j = i + 1; j < 5; ++j)
            EXPECT_NE(indices[i], indices[j]);
    }
}

TEST_F(ContactTableHostTest, PairOrdering)
{
    ContactHashTable<MemType::HOST> table(100);
    uint2                           pair1 = make_uint2(5, 10);
    uint2                           pair2 = make_uint2(10, 5);
    uint                            index1, index2;
    table.findOrInsert(pair1, index1);
    table.findOrInsert(pair2, index2);
    EXPECT_NE(index1, index2);
}
