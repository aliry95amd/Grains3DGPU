#include "RigidBodyFactory.hh"
#include "Box.hh"
#include "Cone.hh"
#include "Convex.hh"
#include "Cylinder.hh"
#include "Rectangle.hh"
#include "RigidBody.hh"
#include "Sphere.hh"
#include "Superquadric.hh"

/* ============================================================================================== */
/* Low-Level Methods                                                                              */
/* ============================================================================================== */
// GPU kernel to construct the rigidbody on device.
// This is mandatory as we cannot access device memory addresses on the host
// So, we pass a device memory address to a kernel.
// Memory address is then populated within the kernel.
// This kernel is not declared in any header file since we directly use it below
// It helps to NOT explicitly instantiate it.
template <typename T, typename... Arguments>
__GLOBAL__ void createRigidBodyKernel(RigidBody<T>** rb,
                                      uint           index,
                                      T              crustThickness,
                                      uint           material,
                                      T              density,
                                      ConvexType     convexType,
                                      Arguments... args)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID > 0)
        return;

    Convex<T>* convex = nullptr;
    if constexpr(sizeof...(args) == 1)
    {
        if(convexType == SPHERE)
            convex = new Sphere<T>(args...);
    }
    else if constexpr(sizeof...(args) == 2)
    {
        if(convexType == CYLINDER)
            convex = new Cylinder<T>(args...);
        if(convexType == CONE)
            convex = new Cone<T>(args...);
        if(convexType == RECTANGLE)
            convex = new Rectangle<T>(args...);
    }
    else if constexpr(sizeof...(args) == 3)
    {
        if(convexType == BOX)
            convex = new Box<T>(args...);
    }
    else if constexpr(sizeof...(args) == 5)
    {
        if(convexType == SUPERQUADRIC)
            convex = new Superquadric<T>(args...);
    }
    GAssert(convex, "Convex is not created! Aborting Grains!");
    rb[index] = new RigidBody<T>(convex, crustThickness, density, material);
}

/* ============================================================================================== */
/* High-Level Methods                                                                             */
/* ============================================================================================== */
// Creates and stores a RigidBody object in the host memory.
template <typename T>
__HOST__ void
    RigidBodyFactory<T>::create(DOMNode*                        obstacles,
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
                                uint&                           numParticles)
{
    // Obstacles
    numObstacles              = 0;
    DOMNodeList* allObstacles = ReaderXML::getNodes(obstacles);
    // Number of unique shapes (rigid bodies) in the simulation
    uint numRefObstacles = allObstacles->getLength();
    refObstacleRB.initialize(numRefObstacles);
    refObstacleInitialPosition.initialize(numRefObstacles);
    refObstacleInitialOrientation.initialize(numRefObstacles);
    numEachRefObstacle.initialize(numRefObstacles);
    for(uint i = 0; i < numRefObstacles; ++i)
    {
        DOMNode* nObstacle       = allObstacles->item(i);
        numEachRefObstacle[i]    = 1;
        refObstacleRB[i]         = new RigidBody<T>(nObstacle);
        DOMNode*      nTransform = ReaderXML::getNode(nObstacle, "Transformation");
        Vector3<T>    centre(T(0), T(0), T(0));
        Quaternion<T> rotation(T(0), T(0), T(0), T(1));
        if(nTransform)
        {
            DOMNode* nCentre = ReaderXML::getNode(nTransform, "Centre");
            if(nCentre)
                centre = Vector3<T>(nCentre);

            DOMNode* nRotation = ReaderXML::getNode(nTransform, "AngularPosition");
            if(nRotation)
                rotation = Quaternion<T>(nRotation);
        }
        refObstacleInitialPosition[i]    = centre;
        refObstacleInitialOrientation[i] = rotation;
        numObstacles += numEachRefObstacle[i];
    }

    // Particles
    numParticles              = 0;
    DOMNodeList* allParticles = ReaderXML::getNodes(particles);
    // Number of unique shapes (rigid bodies) in the simulation
    uint numRefParticles = allParticles->getLength();
    refParticleRB.initialize(numRefParticles);
    refParticleInitialPosition.initialize(numRefParticles);
    refParticleInitialOrientation.initialize(numRefParticles);
    numEachRefParticle.initialize(numRefParticles);
    for(uint i = 0; i < numRefParticles; ++i)
    {
        DOMNode* nParticle    = allParticles->item(i);
        numEachRefParticle[i] = static_cast<uint>(ReaderXML::getNodeAttr_Int(nParticle, "Number"));
        refParticleRB[i]      = new RigidBody<T>(nParticle);
        DOMNode*      nTransform = ReaderXML::getNode(nParticle, "Transformation");
        Vector3<T>    centre(T(0), T(0), T(0));
        Quaternion<T> rotation(T(0), T(0), T(0), T(1));
        if(nTransform)
        {
            DOMNode* nCentre = ReaderXML::getNode(nTransform, "Centre");
            if(nCentre)
                centre = Vector3<T>(nCentre);

            DOMNode* nRotation = ReaderXML::getNode(nTransform, "AngularPosition");
            if(nRotation)
                rotation = Quaternion<T>(nRotation);
        }
        refParticleInitialPosition[i]    = centre;
        refParticleInitialOrientation[i] = rotation;
        numParticles += numEachRefParticle[i];
    }
}

// -------------------------------------------------------------------------------------------------
// Constructs a ContactForceModel object on device.
template <typename T>
__HOST__ void
    RigidBodyFactory<T>::copyHostToDevice(GrainsMemBuffer<RigidBody<T>*, MemType::HOST>&   h_RB,
                                          GrainsMemBuffer<RigidBody<T>*, MemType::DEVICE>& d_RB)
{
    d_RB.initialize(h_RB.getSize());
    for(uint i = 0; i < h_RB.getSize(); ++i)
    {
        // Extracting info from the host side object
        Convex<T>* convex   = h_RB[i]->getConvex();
        ConvexType cvxType  = convex->getConvexType();
        T          ct       = h_RB[i]->getCrustThickness();
        uint       material = h_RB[i]->getMaterial();
        // We also need the density to calculate the mass of the rigid body.
        // However, it is not available here. So, we manually compute it:
        T density = h_RB[i]->getMass() / h_RB[i]->getVolume();

        if(cvxType == SPHERE)
        {
            Sphere<T>* c = dynamic_cast<Sphere<T>*>(convex);
            T          r = c->getRadius();
            createRigidBodyKernel<<<1, 1>>>(d_RB.getData(), i, ct, material, density, SPHERE, r);
        }
        else if(cvxType == BOX)
        {
            Box<T>*    c = dynamic_cast<Box<T>*>(convex);
            Vector3<T> L = c->getExtent();
            createRigidBodyKernel<<<1, 1>>>(d_RB.getData(),
                                            i,
                                            ct,
                                            material,
                                            density,
                                            BOX,
                                            L[X],
                                            L[Y],
                                            L[Z]);
        }
        else if(cvxType == CYLINDER)
        {
            Cylinder<T>* c = dynamic_cast<Cylinder<T>*>(convex);
            T            r = c->getRadius();
            T            h = c->getHeight();
            createRigidBodyKernel<<<1, 1>>>(d_RB.getData(),
                                            i,
                                            ct,
                                            material,
                                            density,
                                            CYLINDER,
                                            r,
                                            h);
        }
        else if(cvxType == CONE)
        {
            Cone<T>* c = dynamic_cast<Cone<T>*>(convex);
            T        r = c->getRadius();
            T        h = c->getHeight();
            createRigidBodyKernel<<<1, 1>>>(d_RB.getData(), i, ct, material, density, CONE, r, h);
        }
        else if(cvxType == SUPERQUADRIC)
        {
            Superquadric<T>* c = dynamic_cast<Superquadric<T>*>(convex);
            Vector3<T>       L = c->getExtent();
            Vector3<T>       N = c->getExponent();
            createRigidBodyKernel<<<1, 1>>>(d_RB.getData(),
                                            i,
                                            ct,
                                            material,
                                            density,
                                            SUPERQUADRIC,
                                            L[X],
                                            L[Y],
                                            L[Z],
                                            N[X],
                                            N[Y]);
        }
        else if(cvxType == RECTANGLE)
        {
            Rectangle<T>* c = dynamic_cast<Rectangle<T>*>(convex);
            Vector3<T>    L = c->getExtent();
            createRigidBodyKernel<<<1, 1>>>(d_RB.getData(),
                                            i,
                                            ct,
                                            material,
                                            density,
                                            RECTANGLE,
                                            L[X],
                                            L[Y]);
        }
        else
            GAbort("Convex type is not implemented for GPU! Aborting Grains!");
    }
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class RigidBodyFactory<float>;
template class RigidBodyFactory<double>;
