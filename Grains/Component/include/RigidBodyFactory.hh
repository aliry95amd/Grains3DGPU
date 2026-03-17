#ifndef _RIGIDBODYFACTORY_HH_
#define _RIGIDBODYFACTORY_HH_

#include "GrainsMemBuffer.hh"
#include "RigidBody.hh"

// =================================================================================================
/** @brief The class RigidBodyFactory.

    Creates the rigid body for each particle and obstacle.

    @author A.YAZDANI - 2025 - Construction */
// =================================================================================================
template <typename T>
class RigidBodyFactory
{
private:
    /**@name Contructors & Destructor */
    //@{
    /** @brief Default constructor (forbidden) */
    RigidBodyFactory() = default;

    /** @brief Destructor (forbidden) */
    ~RigidBodyFactory() = default;
    //@}

public:
    /**@name Methods */
    //@{
    /** @brief Creates and returns a buffer of reference rigid bodies given an XML node
        @param root XML node
        @param refObstacleRB Memory buffer for storing the reference obstacles
        @param refParticleRB Memory buffer for storing the reference particles
        @param refObstacleInitialPosition Memory buffer for the initial positions of obstacles
        @param refParticleInitialPosition Memory buffer for the initial positions of particles
        @param refObstacleInitialOrientation Memory buffer for the initial orientations of obstacles
        @param refParticleInitialOrientation Memory buffer for the initial orientations of particles
        @param numEachRefObstacle number of each reference obstacle
        @param numEachRefParticle number of each reference particle
        @param numObstacles Total number of obstacles in the simulation
        @param numParticles Total number of particles in the simulation */
    static void create(DOMNode*                        obstacles,
                       DOMNode*                        particles,
                       GrainsMemBuffer<RigidBody<T>*>& refObstacleRB,
                       GrainsMemBuffer<RigidBody<T>*>& refParticleRB,
                       GrainsMemBuffer<Vector3<T>>&    refObstacleInitialPosition,
                       GrainsMemBuffer<Vector3<T>>&    refParticleInitialPosition,
                       GrainsMemBuffer<Quaternion<T>>& refObstacleInitialOrientation,
                       GrainsMemBuffer<Quaternion<T>>& refParticleInitialOrientation,
                       GrainsMemBuffer<uint>&          numEachRefObstacle,
                       GrainsMemBuffer<uint>&          numEachRefParticle,
                       uint&                           numObstacles,
                       uint&                           numParticles);

    /** @brief RigidBody objects must be instantiated on device, if we want to use them on device.
        Copying from host is not supported due to runtime polymorphism for this class. This
        function reads a host-side RigidBody object, and mimics it in a given device buffer. It
        calls a device kernel that is implemented in the source file.
        @param h_RB Host-side RigidBody object
        @param d_RB Device-side RigidBody object */
    static void copyHostToDevice(GrainsMemBuffer<RigidBody<T>*, MemType::HOST>&   h_RB,
                                 GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>& d_RB);

    /** @brief Frees RigidBody objects that were created on device via copyHostToDevice.
        Launches a kernel that calls device-side delete on every pointer, then releases
        the pointer array.  Safe to call even if d_RB is empty.
        @param d_RB Device-side RigidBody pointer buffer */
    static void freeDevice(GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>& d_RB);
    //@}
};

#endif
