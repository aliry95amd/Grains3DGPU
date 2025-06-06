#ifndef _GRAINS_HH_
#define _GRAINS_HH_

#include "Basic.hh"
#include "ComponentManager.hh"
#include "ContactForceModel.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "Insertion.hh"
#include "PostProcessingWriter.hh"
#include "RigidBody.hh"
#include "TimeIntegrator.hh"

// =============================================================================
/** @brief The class Grains.

    Standard Grains3D application as an abstract class.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
template <typename T>
class Grains
{
protected:
    /** @name Parameters */
    //@{
    /** \brief Parameters used in the simulation on the host memory. */
    GrainsParameters<T> m_parameters;
    /** \brief Buffer of particles rigid bodies. The pointer is used because we
    want to use runtime polymorphism for switching between different particle
    types. */
    GrainsMemBuffer<RigidBody<T, T>*, MemType::HOST> m_particleRigidBodyList;
    /** \brief Buffer of obstacles rigid bodies. The pointer is used because we 
    want to use runtime polymorphism for switching between different obstacle 
    types. */
    GrainsMemBuffer<RigidBody<T, T>*, MemType::HOST> m_obstacleRigidBodyList;
    /** \brief Insertion object. */
    std::unique_ptr<Insertion<T>> m_insertion;
    /** \brief Manager of the components in the simulation on the host memory. 
    We use a pointer here as we want to use runtime polymorphism for switching 
    between ComponentManagerCPU and ComponentManagerGPU. */
    std::unique_ptr<ComponentManager<T, MemType::HOST>> m_components;
    /** \brief Buffer of Linked cells. */
    GrainsMemBuffer<LinkedCell<T>*, MemType::HOST> m_linkedCell;
    /** \brief Buffer of contact forces. */
    GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST> m_contactForce;
    /** \brief Buffer of time integrators. */
    GrainsMemBuffer<TimeIntegrator<T>*, MemType::HOST> m_timeIntegrator;
    /** \brief List of post-processing writers. */
    std::list<std::unique_ptr<PostProcessingWriter<T>>> m_postProcessor;
    //@}

public:
    /** @name Contructors & Destructor */
    //@{
    /** @brief Default constructor */
    Grains();

    /** @brief Destructor */
    virtual ~Grains();
    //@}

    /** @name High-level methods */
    //@{
    /** @brief Tasks to perform before time-stepping, mostly reading setting
        variables in GrainsParameters.
        @param rootElement XML root */
    virtual void initialize(DOMElement* rootElement);

    /** @brief Runs the simulation over the prescribed time interval */
    virtual void simulate() = 0;

    /** @brief Performs post-processing
        @param cm ComponentManager object, either host or device */
    void postProcess(
        const std::unique_ptr<ComponentManager<T, MemType::HOST>>& cm) const;

    /** @brief Tasks to perform after time-stepping */
    virtual void finalize();
    //@}

    /**@name Low-level methods */
    //@{
    /** @brief Construction of the simulation: linked cell, particles &
        obstacles, domain decomposition 
        @param rootElement XML root */
    void Construction(DOMElement* rootElement);

    /** @brief External force definition
        @param rootElement XML root */
    void Forces(DOMElement* rootElement);

    /** @brief Additional features of the simulation: insertion, 
        post-processing
        @param rootElement XML root */
    void AdditionalFeatures(DOMElement* rootElement);
    //@}
};

#endif
