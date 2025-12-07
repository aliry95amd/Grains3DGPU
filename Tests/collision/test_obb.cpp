#include "BoundingBox.hh"
#include "Quaternion.hh"
#include "Transform3.hh"
#include "Vector3.hh"
#include <cmath>
#include <gtest/gtest.h>

class OBBSimpleTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create bounding boxes for testing using extents (half-lengths)
        // Box A: centered at origin, dimensions 2x2x2 (extent 1,1,1)
        Vector3<double> extentA(1.0, 1.0, 1.0);
        boundingBoxA = BoundingBox<double>(extentA);

        // Box B: smaller box, dimensions 1x1x1 (extent 0.5,0.5,0.5)
        Vector3<double> extentB(0.5, 0.5, 0.5);
        boundingBoxB = BoundingBox<double>(extentB);

        // Identity transform
        identity_transform = Transform3<double>(Quaternion<double>(0.0, 0.0, 0.0, 1.0),
                                                Vector3<double>(0.0, 0.0, 0.0));
    }

    BoundingBox<double> boundingBoxA;
    BoundingBox<double> boundingBoxB;
    Transform3<double>  identity_transform;
    const double        EPSILON = 1e-10;
};

// Test basic bounding box creation and properties
TEST_F(OBBSimpleTest, BasicBoundingBoxProperties)
{
    // Test extent values
    Vector3<double> extentA = boundingBoxA.getExtent();
    EXPECT_NEAR(extentA[0], 1.0, EPSILON);
    EXPECT_NEAR(extentA[1], 1.0, EPSILON);
    EXPECT_NEAR(extentA[2], 1.0, EPSILON);

    Vector3<double> extentB = boundingBoxB.getExtent();
    EXPECT_NEAR(extentB[0], 0.5, EPSILON);
    EXPECT_NEAR(extentB[1], 0.5, EPSILON);
    EXPECT_NEAR(extentB[2], 0.5, EPSILON);
}

// Test transform creation and properties
TEST_F(OBBSimpleTest, BasicTransformProperties)
{
    // Test identity transform
    Vector3<double> origin = identity_transform.getOrigin();
    EXPECT_NEAR(origin[0], 0.0, EPSILON);
    EXPECT_NEAR(origin[1], 0.0, EPSILON);
    EXPECT_NEAR(origin[2], 0.0, EPSILON);
}

// Test various bounding box sizes
TEST_F(OBBSimpleTest, VariousBoundingBoxSizes)
{
    // Very small box (extent 0.001)
    Vector3<double>     tinyExtent(0.001, 0.001, 0.001);
    BoundingBox<double> tinyBox(tinyExtent);

    Vector3<double> tinyResult = tinyBox.getExtent();
    EXPECT_NEAR(tinyResult[0], 0.001, EPSILON);
    EXPECT_NEAR(tinyResult[1], 0.001, EPSILON);
    EXPECT_NEAR(tinyResult[2], 0.001, EPSILON);

    // Very large box (extent 1000.0)
    Vector3<double>     hugeExtent(1000.0, 1000.0, 1000.0);
    BoundingBox<double> hugeBox(hugeExtent);

    Vector3<double> hugeResult = hugeBox.getExtent();
    EXPECT_NEAR(hugeResult[0], 1000.0, EPSILON);
    EXPECT_NEAR(hugeResult[1], 1000.0, EPSILON);
    EXPECT_NEAR(hugeResult[2], 1000.0, EPSILON);
}

// Test different constructors
TEST_F(OBBSimpleTest, DifferentConstructors)
{
    // Constructor with individual components
    BoundingBox<double> box1(2.0, 3.0, 4.0);
    Vector3<double>     extent1 = box1.getExtent();
    EXPECT_NEAR(extent1[0], 2.0, EPSILON);
    EXPECT_NEAR(extent1[1], 3.0, EPSILON);
    EXPECT_NEAR(extent1[2], 4.0, EPSILON);

    // Constructor with vector
    Vector3<double>     extent_vector(5.0, 6.0, 7.0);
    BoundingBox<double> box2(extent_vector);
    Vector3<double>     extent2 = box2.getExtent();
    EXPECT_NEAR(extent2[0], 5.0, EPSILON);
    EXPECT_NEAR(extent2[1], 6.0, EPSILON);
    EXPECT_NEAR(extent2[2], 7.0, EPSILON);

    // Default constructor
    BoundingBox<double> box3;
    Vector3<double>     extent3 = box3.getExtent();
    // Default extents should be defined values
    EXPECT_TRUE(extent3[0] >= 0.0);
    EXPECT_TRUE(extent3[1] >= 0.0);
    EXPECT_TRUE(extent3[2] >= 0.0);
}
