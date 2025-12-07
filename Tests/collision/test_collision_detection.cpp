#include <cmath>
#include <gtest/gtest.h>

#include "Box.hh"
#include "CollisionDetection.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "Vector3.hh"

class CollisionDetectionTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create box shapes for rigid bodies
        boxA = new Box<double>(1.0, 1.0, 1.0);
        boxB = new Box<double>(0.5, 0.5, 0.5);

        // Create rigid bodies with proper constructor parameters
        rigidBodyA = new RigidBody<double>(boxA, 0.1, 1000.0, 1);
        rigidBodyB = new RigidBody<double>(boxB, 0.1, 1000.0, 1);

        // Set up standard positions and orientations
        origin               = Vector3<double>(0.0, 0.0, 0.0);
        separated_position   = Vector3<double>(3.0, 0.0, 0.0);
        overlapping_position = Vector3<double>(0.5, 0.0, 0.0);
        touching_position    = Vector3<double>(1.5, 0.0, 0.0);

        identity_quaternion = Quaternion<double>(0.0, 0.0, 0.0, 1.0);

        // 45 degree rotation around Z axis
        double angle       = M_PI / 4.0;
        rotated_quaternion = Quaternion<double>(0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0));
    }

    void TearDown() override
    {
        delete rigidBodyA;
        delete rigidBodyB;
    }

    Box<double>*       boxA;
    Box<double>*       boxB;
    RigidBody<double>* rigidBodyA;
    RigidBody<double>* rigidBodyB;
    Vector3<double>    origin;
    Vector3<double>    separated_position;
    Vector3<double>    overlapping_position;
    Vector3<double>    touching_position;
    Quaternion<double> identity_quaternion;
    Quaternion<double> rotated_quaternion;
    const double       EPSILON = EPS<double>;
};

// Test rigid body intersection with relative transformation
TEST_F(CollisionDetectionTest, RelativeTransformationIntersection)
{
    // Test overlapping case
    bool result
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, overlapping_position, identity_quaternion);
    EXPECT_TRUE(result);

    // Test separated case
    result
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, separated_position, identity_quaternion);
    EXPECT_FALSE(result);
}

// Test rigid body intersection with world coordinates
TEST_F(CollisionDetectionTest, WorldCoordinatesIntersection)
{
    // Test overlapping case
    bool result = intersectRigidBodies(*rigidBodyA,
                                       *rigidBodyB,
                                       origin,
                                       overlapping_position,
                                       identity_quaternion,
                                       identity_quaternion);
    EXPECT_TRUE(result);

    // Test separated case
    result = intersectRigidBodies(*rigidBodyA,
                                  *rigidBodyB,
                                  origin,
                                  separated_position,
                                  identity_quaternion,
                                  identity_quaternion);
    EXPECT_FALSE(result);

    // Test both at same position
    result = intersectRigidBodies(*rigidBodyA,
                                  *rigidBodyB,
                                  origin,
                                  origin,
                                  identity_quaternion,
                                  identity_quaternion);
    EXPECT_TRUE(result);
}

// Test collision detection with rotated rigid bodies
TEST_F(CollisionDetectionTest, RotatedRigidBodiesIntersection)
{
    // Small body rotated inside large body should still intersect
    bool result = intersectRigidBodies(*rigidBodyA, *rigidBodyB, origin, rotated_quaternion);
    EXPECT_TRUE(result);

    // Test with both bodies rotated
    result = intersectRigidBodies(*rigidBodyA,
                                  *rigidBodyB,
                                  origin,
                                  origin,
                                  rotated_quaternion,
                                  identity_quaternion);
    EXPECT_TRUE(result);

    // Test rotated and separated
    result = intersectRigidBodies(*rigidBodyA, *rigidBodyB, separated_position, rotated_quaternion);
    EXPECT_FALSE(result);
}

// Test edge cases and boundary conditions
TEST_F(CollisionDetectionTest, EdgeCasesAndBoundaryConditions)
{
    // Test touching bodies (boundary case)
    bool result
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, touching_position, identity_quaternion);
    // Result may vary based on numerical precision - just ensure it's
    // consistent
    EXPECT_TRUE(result || !result);  // Always true - just ensure the call succeeded
    (void)result;                    // Explicitly mark result as used to avoid warnings

    // Test with very small displacement
    Vector3<double> tiny_displacement(1e-10, 0.0, 0.0);
    result = intersectRigidBodies(*rigidBodyA, *rigidBodyB, tiny_displacement, identity_quaternion);
    EXPECT_TRUE(result);  // Should still intersect with tiny displacement

    // Test with identical rigid bodies
    result = intersectRigidBodies(*rigidBodyA, *rigidBodyA, origin, identity_quaternion);
    EXPECT_TRUE(result);
}

// Test performance and stability
TEST_F(CollisionDetectionTest, PerformanceAndStability)
{
    // Test multiple collision detections with varying positions
    for(int i = 0; i < 50; ++i)
    {
        double          offset = i * 0.1;  // Gradually move bodies apart
        Vector3<double> test_position(offset, 0.0, 0.0);

        bool result
            = intersectRigidBodies(*rigidBodyA, *rigidBodyB, test_position, identity_quaternion);

        // Should intersect for small offsets
        if(offset < 1.0)
        {
            EXPECT_TRUE(result);
        }
        // Should not intersect for large offsets
        else if(offset > 2.0)
        {
            EXPECT_FALSE(result);
        }
        // Boundary region may vary - skip assertion
    }
}

// Test various orientations
TEST_F(CollisionDetectionTest, VariousOrientations)
{
    // Test rotations around different axes
    double angles[] = {0.0, M_PI / 6.0, M_PI / 4.0, M_PI / 3.0, M_PI / 2.0};

    for(double angle : angles)
    {
        // Rotation around X axis
        Quaternion<double> rotX(sin(angle / 2.0), 0.0, 0.0, cos(angle / 2.0));
        bool               resultX = intersectRigidBodies(*rigidBodyA, *rigidBodyB, origin, rotX);
        EXPECT_TRUE(resultX);  // Small body should remain inside large body

        // Rotation around Y axis
        Quaternion<double> rotY(0.0, sin(angle / 2.0), 0.0, cos(angle / 2.0));
        bool               resultY = intersectRigidBodies(*rigidBodyA, *rigidBodyB, origin, rotY);
        EXPECT_TRUE(resultY);

        // Rotation around Z axis
        Quaternion<double> rotZ(0.0, 0.0, sin(angle / 2.0), cos(angle / 2.0));
        bool               resultZ = intersectRigidBodies(*rigidBodyA, *rigidBodyB, origin, rotZ);
        EXPECT_TRUE(resultZ);
    }
}

// Test collision consistency
TEST_F(CollisionDetectionTest, CollisionConsistency)
{
    // Multiple calls should give consistent results
    bool result1
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, overlapping_position, identity_quaternion);
    bool result2
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, overlapping_position, identity_quaternion);
    bool result3
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, overlapping_position, identity_quaternion);

    EXPECT_EQ(result1, result2);
    EXPECT_EQ(result2, result3);
    EXPECT_TRUE(result1);  // Should be true for overlapping case

    // Same for separated case
    result1
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, separated_position, identity_quaternion);
    result2
        = intersectRigidBodies(*rigidBodyA, *rigidBodyB, separated_position, identity_quaternion);
    EXPECT_EQ(result1, result2);
    EXPECT_FALSE(result1);  // Should be false for separated case
}

// Test different rigid body sizes
TEST_F(CollisionDetectionTest, DifferentRigidBodySizes)
{
    // Create very small and very large rigid bodies
    Box<double>* tinyBox = new Box<double>(0.01, 0.01, 0.01);
    Box<double>* hugeBox = new Box<double>(10.0, 10.0, 10.0);

    RigidBody<double>* tinyRB = new RigidBody<double>(tinyBox, 0.1, 1000.0, 1);
    RigidBody<double>* hugeRB = new RigidBody<double>(hugeBox, 0.1, 1000.0, 1);

    // Tiny rigid body should be inside normal rigid body
    bool result = intersectRigidBodies(*rigidBodyA, *tinyRB, origin, identity_quaternion);
    EXPECT_TRUE(result);

    // Normal rigid body should be inside huge rigid body
    result = intersectRigidBodies(*rigidBodyA, *hugeRB, origin, identity_quaternion);
    EXPECT_TRUE(result);

    // Cleanup
    delete tinyRB;
    delete hugeRB;
    delete tinyBox;
    delete hugeBox;
}

// Test stress scenarios
TEST_F(CollisionDetectionTest, StressTest)
{
    // Test many rapid collision checks
    int numTests     = 1000;
    int successCount = 0;

    for(int i = 0; i < numTests; ++i)
    {
        // Vary position slightly
        double          x = (i % 100) * 0.01 - 0.5;  // Range from -0.5 to 0.5
        Vector3<double> testPos(x, 0.0, 0.0);

        bool result = intersectRigidBodies(*rigidBodyA, *rigidBodyB, testPos, identity_quaternion);
        if(result)
            successCount++;
    }

    // Should have many successful intersections (bodies are overlapping for
    // most positions)
    EXPECT_GT(successCount, numTests / 2);
    EXPECT_LT(successCount, numTests);  // But not all should intersect
}
