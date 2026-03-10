#ifndef _COLLISIONDETECTIONMODULE_HH_
#define _COLLISIONDETECTIONMODULE_HH_

#include <cstdint>
#include <memory>

#include "CollisionDetectionCommon.hh"
#include "ContactInfo.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "Kinematics.hh"
#include "NeighborList.hh"
#include "NeighborListFactory.hh"
#include "ParticleSorter.hh"
#include "Quaternion.hh"
#include "RigidBody.hh"
#include "Torce.hh"
#include "Vector3.hh"

// =================================================================================================
/** @brief The class CollisionDetectionModule.

    Encapsulates the full collision detection pipeline:
      1. Neighbor list update (broad phase)
      2. Relative transformation computation
      3. Optional bounding volume pre-filter (Sphere/OBB/OBC)
      4. Narrow phase GJK (closestPointsRigidBodies)
      5. Contact information world-frame transform

    ComponentManager owns a CollisionDetectionModule by unique_ptr and delegates all detection
    work to it by passing raw non-owning pointers to the particle data arrays.  No particle data
    is copied, and the module reads/writes directly into ComponentManager's buffers.

    @author A.Yazdani - 2026 - Construction */
// =================================================================================================
template <typename T, MemType M = MemType::HOST>
class CollisionDetectionModule
{
private:
    /** @name Parameters */
    //@{
    /** \brief Particle sorter for Morton code-based reordering of particle arrays */
    ParticleSorter<T, M> m_particleSorter;
    /** \brief Neighbor list (broad phase) */
    std::unique_ptr<NeighborList<T, M>> m_neighborList;
    /** \brief Per-pair relative position of B in A-local frame */
    GrainsMemBuffer<Vector3<T>, M> m_relPosition;
    /** \brief Per-pair relative quaternion of B in A-local frame */
    GrainsMemBuffer<Quaternion<T>, M> m_relQuaternion;
    /** \brief Per-pair contact information in A-local frame (output of GJK) */
    GrainsMemBuffer<ContactInfo<T>, M> m_contactInfoLocal;
    /** \brief Per-pair BV pass/fail flags written by filterPairsBV_Kernel */
    GrainsMemBuffer<uint8_t, M> m_bvPassFlags;
    /** \brief Compacted list of original pair indices that passed the BV filter */
    GrainsMemBuffer<uint, M> m_bvPassPairIndices;
    /** \brief Size-1 device buffer; CUB writes the passing-pair count here */
    GrainsMemBuffer<int, M> m_bvPassPairCountDevice;
    /** \brief Host-side copy of the passing-pair count after CUB compaction */
    int m_bvPassPairCount = 0;
    /** \brief Scratch space for CUB DeviceSelect::Flagged */
    GrainsMemBuffer<uint8_t, M> m_cubTempStorage;
    /** \brief Byte size of the CUB scratch buffer */
    size_t m_cubTempStorageBytes = 0;
    //@}

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor (forbidden) */
    CollisionDetectionModule() = delete;

    /** @brief Constructor — builds the NeighborList internally.
        @param rigidBody    Pointer to the rigid body buffer
        @param positions    Position buffer (used for initial cell assignment)
        @param orientations Quaternion buffer
        @param CD           Collision detection parameters
        @param nObstacles   Number of obstacles
        @param nParticles   Number of moving particles */
    CollisionDetectionModule(const GrainsMemBuffer<RigidBody<T>*, M>* rigidBody,
                             const GrainsMemBuffer<Vector3<T>, M>&    positions,
                             const GrainsMemBuffer<Quaternion<T>, M>& orientations,
                             const CollisionDetectionParameters<T>&   CD,
                             uint                                     nObstacles,
                             uint                                     nParticles);

    /** @brief Destructor */
    ~CollisionDetectionModule() = default;

    /** @brief Deleted copy constructor */
    CollisionDetectionModule(const CollisionDetectionModule&) = delete;

    /** @brief Deleted copy assignment operator */
    CollisionDetectionModule& operator=(const CollisionDetectionModule&) = delete;

    /** @brief Defaulted move constructor */
    CollisionDetectionModule(CollisionDetectionModule&&) = default;

    /** @brief Defaulted move assignment operator */
    CollisionDetectionModule& operator=(CollisionDetectionModule&&) = default;
    //@}

    /** @name Get methods */
    //@{
    /** @brief Returns raw pointer to the pair list (size = getPairCount()) */
    const uint2* getPairList() const;

    /** @brief Returns the current number of active pairs in the neighbor list */
    uint getPairCount() const;

    /** @brief Returns the allocated size of the pair buffers (may exceed getPairCount()) */
    size_t getPairBufferSize() const;

    /** @brief Returns the neighbor list (read-only) */
    const NeighborList<T, M>* getNeighborList() const;
    //@}

    /** @name Methods */
    //@{
    /** @brief Runs the complete collision detection pipeline.
        Pipeline: 1. sortParticles
                  2. updateNeighborList
                  3. computeRelativeTransformations
                  4. filterPairsBV (DEVICE and BV only)
                  5. detectCollisionsComponents
                  6. transformContactInfo
        @param rigidBodies  Raw pointer array of RigidBody*
        @param positions    Position buffer (modified in-place by sorter)
        @param orientations Quaternion buffer (modified in-place by sorter)
        @param velocities   Kinematics buffer (modified in-place by sorter)
        @param torces       Torce buffer (modified in-place by sorter)
        @param rigidBodyIds RigidBodyId buffer (modified in-place by sorter)
        @param componentIds ComponentId buffer (modified in-place by sorter)
        @param contactInfo  ContactInfo world-frame buffer (owned by ComponentManager)
        @param pairList     Pair list buffer (owned by ComponentManager)
        @param numPairs     Current active pair count (updated on NL rebuild)
        @param nObstacles   Number of obstacles
        @param nParticles   Number of moving particles */
    void run(const RigidBody<T>* const*          rigidBodies,
             GrainsMemBuffer<Vector3<T>, M>&     positions,
             GrainsMemBuffer<Quaternion<T>, M>&  orientations,
             GrainsMemBuffer<Kinematics<T>, M>&  velocities,
             GrainsMemBuffer<Torce<T>, M>&       torces,
             GrainsMemBuffer<uint, M>&           rigidBodyIds,
             GrainsMemBuffer<uint, M>&           componentIds,
             GrainsMemBuffer<ContactInfo<T>, M>& contactInfo,
             GrainsMemBuffer<uint2, M>&          pairList,
             uint&                               numPairs,
             uint                                nObstacles,
             uint                                nParticles);
    //@}

private:
    /** @name Pipeline steps */
    //@{
    /** @brief Resizes per-pair buffers to at least `size` pairs.
        @param pairList    ComponentManager-owned buffer resized in lock-step
        @param size        New minimum capacity
        @param contactInfo ComponentManager-owned buffer resized in lock-step */
    void resizePairBuffers(GrainsMemBuffer<uint2, M>&          pairList,
                           GrainsMemBuffer<ContactInfo<T>, M>& contactInfo,
                           uint                                size);

    /** @brief Calls the neighbor list update; resizes buffers and increments the global update
        counter if the list was rebuilt.
        @param positions   Position buffer required by the NL algorithm
        @param pairList    ComponentManager-owned buffer resized if NL grew
        @param contactInfo ComponentManager-owned buffer resized if NL grew
        @param numPairs    Updated with the new pair count on rebuild
        @param nObstacles  Number of obstacles
        @param nParticles  Number of moving particles */
    void updateNeighborList(GrainsMemBuffer<Vector3<T>, M>&     positions,
                            GrainsMemBuffer<uint2, M>&          pairList,
                            GrainsMemBuffer<ContactInfo<T>, M>& contactInfo,
                            uint&                               numPairs,
                            uint                                nObstacles,
                            uint                                nParticles);

    /** @brief Computes per-pair relative position/quaternion (B expressed in A-local frame).
        @param positions    World position buffer
        @param orientations World quaternion buffer
        @param pairList     Pair list buffer */
    void computeRelativeTransformations(const GrainsMemBuffer<Vector3<T>, M>&    positions,
                                        const GrainsMemBuffer<Quaternion<T>, M>& orientations,
                                        const GrainsMemBuffer<uint2, M>&         pairList);

    /** @brief Runs the BV pre-filter (sphere + OBB SAT) as a dedicated DEVICE kernel.
        Writes per-pair pass/fail flags, writes no-contact sentinels for rejected pairs, then
        uses CUB DeviceSelect::Flagged to compact the passing pair indices into @p
        m_bvPassPairIndices and copies the result count to @p m_bvPassPairCount.  Must be called
        after computeRelativeTransformations.  HOST+OBB uses the fused path inside
        detectCollisionsComponents_common instead.
        @param rigidBodies      Raw pointer array of RigidBody*
        @param pairList         Pair list buffer
        @param contactInfoLocal Per-pair local ContactInfo buffer */
    void filterPairsBV(const RigidBody<T>* const*          rigidBodies,
                       const GrainsMemBuffer<uint2, M>&    pairList,
                       GrainsMemBuffer<ContactInfo<T>, M>& contactInfoLocal);

    /** @brief Runs narrow-phase GJK and writes ContactInfo in A-local frame.
        @param rigidBodies Raw pointer array of RigidBody*
        @param pairList    Pair list buffer */
    void detectCollisionsComponents(const RigidBody<T>* const*       rigidBodies,
                                    const GrainsMemBuffer<uint2, M>& pairList);

    /** @brief Transforms per-pair ContactInfo from A-local frame to world frame.
        @param positions    World position buffer
        @param orientations World quaternion buffer
        @param pairList     Pair list buffer
        @param contactInfo  ComponentManager's world-frame ContactInfo buffer */
    void transformContactInfo(const GrainsMemBuffer<Vector3<T>, M>&    positions,
                              const GrainsMemBuffer<Quaternion<T>, M>& orientations,
                              const GrainsMemBuffer<uint2, M>&         pairList,
                              GrainsMemBuffer<ContactInfo<T>, M>&      contactInfo);

    /** @brief Sorts particles by Morton codes for improved cache efficiency.
        @param positions    Position buffer (reordered in-place)
        @param orientations Quaternion buffer (reordered in-place)
        @param velocities   Kinematics buffer (reordered in-place)
        @param torces       Torce buffer (reordered in-place)
        @param rigidBodyIds RigidBodyId buffer (reordered in-place)
        @param componentIds ComponentId buffer (reordered in-place)
        @param nObstacles   Number of obstacles
        @param nParticles   Number of moving particles */
    void sortParticles(GrainsMemBuffer<Vector3<T>, M>&    positions,
                       GrainsMemBuffer<Quaternion<T>, M>& orientations,
                       GrainsMemBuffer<Kinematics<T>, M>& velocities,
                       GrainsMemBuffer<Torce<T>, M>&      torces,
                       GrainsMemBuffer<uint, M>&          rigidBodyIds,
                       GrainsMemBuffer<uint, M>&          componentIds,
                       uint                               nObstacles,
                       uint                               nParticles);
    //@}
