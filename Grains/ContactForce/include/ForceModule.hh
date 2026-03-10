#ifndef _FORCEMODULE_HH_
#define _FORCEMODULE_HH_

#include <memory>

#include "ContactForceModel.hh"
#include "ContactInfo.hh"
#include "ContactTable.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "Kinematics.hh"
#include "RigidBody.hh"
#include "Torce.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief The class ForceModule.

    Encapsulates the full force computation pipeline:
      1. Periodic contact-table cleanup (mark-and-sweep)
      2. [GPU only] Build compact list of active pair indices
      3. Contact force computation (parallel per-pair on GPU, sequential loop on CPU)
      4. [GPU only] Atomic reduction of per-pair intermediate torces to per-particle torces
      5. External forces (gravity applied to each moving particle)

    ComponentManager owns a ForceModule by unique_ptr and calls run() once per timestep after
    detectCollisions().  All particle-indexed arrays are passed in as non-owning references; the
    module reads/writes directly into ComponentManager's buffers.

    Owned resources:
      - ContactHashTable<T,M>     m_contactTable   (contact history for memory-enabled models)
      - GrainsMemBuffer<uint,M>   m_prefixScan     (exclusive-scan workspace; DEVICE path only)
      - GrainsMemBuffer<uint,M>   m_activeIndex    (compact active-pair indices; DEVICE path only)
      - GrainsMemBuffer<Torce,M>  m_intermediateTorceA/B (per-pair staging; DEVICE path only)

    @author A.Yazdani - 2025 - Construction */
// =================================================================================================
template <typename T, MemType M = MemType::HOST>
class ForceModule
{
private:
    /** @name Owned resources */
    //@{
    /** \brief Exclusive-scan workspace for active-pair compaction (DEVICE path only) */
    GrainsMemBuffer<uint, M> m_prefixScan;
    /** \brief Compact array of active pair indices (DEVICE path only) */
    GrainsMemBuffer<uint, M> m_activeIndex;
    /** \brief Per-pair intermediate torce for particle A (DEVICE path only; lazy resize) */
    GrainsMemBuffer<Torce<T>, M> m_intermediateTorceA;
    /** \brief Per-pair intermediate torce for particle B (DEVICE path only; lazy resize) */
    GrainsMemBuffer<Torce<T>, M> m_intermediateTorceB;
    /** \brief Contact history hash table (allocated only when isContactWithMemory == true) */
    ContactHashTable<T, M> m_contactTable;
    //@}

public:
    /** @name Constructors */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Default constructor (forbidden; use ForceModuleFactory) */
    ForceModule() = delete;

    // ---------------------------------------------------------------------------------------------
    /** @brief Constructor.
        @param pairCapacity      Initial pair buffer capacity (from CDModule::getPairBufferSize())
        @param isContactWithMemory Whether contact history tracking is needed */
    ForceModule(size_t pairCapacity, bool isContactWithMemory);

    // ---------------------------------------------------------------------------------------------
    /** @brief Destructor */
    ~ForceModule() = default;

    // Deleted copy; allow move
    ForceModule(const ForceModule&)            = delete;
    ForceModule& operator=(const ForceModule&) = delete;
    ForceModule(ForceModule&&)                 = default;
    ForceModule& operator=(ForceModule&&)      = default;
    //@}

    /** @name Primary interface */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Resizes internal GPU compaction buffers when the pair buffer capacity grows.
        @param newPairCapacity New pair buffer capacity */
    void resizeBuffers(size_t newPairCapacity);

    // ---------------------------------------------------------------------------------------------
    /** @brief Runs the complete force computation pipeline.

        Steps:
          1. cleanupContactTable (periodic mark-and-sweep every 1000 NL updates)
          2. [DEVICE] resize m_intermediateTorceA/B to numPairs
          3. [DEVICE] computeContactForces_Kernel → reduceTorces_Kernel
             [HOST]   sequential computeContactForces_common loop
          4. addExternalForces (gravity)

        All particle-indexed arrays are passed by reference (non-owning); the module reads/writes
        directly.  torce is modified in-place (contact contributions accumulated atomically on GPU,
        sequentially on CPU; gravity then appended for each moving particle).

        @param CF            Array of contact force models
        @param contactInfo   Per-pair contact information in world frame (from CDModule)
        @param pairList      Per-pair component index pairs (from CDModule)
        @param numPairs      Number of active pairs (from CDModule)
        @param position      Per-component position array
        @param velocity      Per-component kinematics array
        @param torce         Per-component torce array (modified in-place)
        @param rigidBody     Per-component rigid body pointer array
        @param numObstacles  Number of obstacle components
        @param numParticles  Number of moving particle components */
    void run(const GrainsMemBuffer<ContactForceModel<T>*, M>& CF,
             const GrainsMemBuffer<ContactInfo<T>, M>&        contactInfo,
             const GrainsMemBuffer<uint2, M>&                 pairList,
             uint                                             numPairs,
             const GrainsMemBuffer<Vector3<T>, M>&            position,
             const GrainsMemBuffer<Kinematics<T>, M>&         velocity,
             GrainsMemBuffer<Torce<T>, M>&                    torce,
             const GrainsMemBuffer<RigidBody<T>*, M>*         rigidBody,
             uint                                             numObstacles,
             uint                                             numParticles);
    //@}

private:
    /** @name Internal helpers */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Performs periodic mark-and-sweep cleanup of the contact hash table.
        No-op when isContactWithMemory == false or the cleanup interval has not elapsed. */
    void cleanupContactTable();
    //@}
};

#endif
