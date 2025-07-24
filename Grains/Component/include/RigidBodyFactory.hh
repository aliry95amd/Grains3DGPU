#ifndef _RIGIDBODYFACTORY_HH_
#define _RIGIDBODYFACTORY_HH_

#include "GrainsMemBuffer.hh"
#include "RigidBody.hh"

// =============================================================================
/** @brief The class RigidBodyFactory.

	Creates the rigid body for each particle and obstacle.

    @author A.YAZDANI - 2025 - Construction */
// =============================================================================
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
    /** @brief Creates and returns a buffer of reference rigid bodies given an 
		XML node
        @param root XML node
        @param refRB Memory buffer for storing the reference rigid bodies
		@param initPositions Memory buffer for storing the initial positions
		@param initOrientations Memory buffer for storing the initial orientations
		@param numEachRefParticle Memory buffer for storing the number of each 
		reference particle
		@param numParticles Total number of particles in the simulation */
    static void
        create(DOMNode*                                       root,
               GrainsMemBuffer<RigidBody<T>*, MemType::HOST>& refRB,
               GrainsMemBuffer<Vector3<T>, MemType::HOST>&    initPositions,
               GrainsMemBuffer<Quaternion<T>, MemType::HOST>& initOrientations,
               GrainsMemBuffer<uint, MemType::HOST>& numEachRefParticle,
               uint&                                 numParticles);

    /** @brief RigidBody objects must be instantiated on device, if
		we want to use them on device. Copying from host is not supported due to
		runtime polymorphism for this class.
		This function reads a host-side RigidBody object, and mimics it
		in a given device buffer.
		It calls a device kernel that is implemented in the source file.
		@param h_RB Host-side RigidBody object
		@param d_RB Device-side RigidBody object */
    static void
        copyHostToDevice(GrainsMemBuffer<RigidBody<T>*, MemType::HOST>&   h_RB,
                         GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>& d_RB);
    //@}
};

#endif
