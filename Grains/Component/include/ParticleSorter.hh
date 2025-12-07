#ifndef _PARTICLESORTER_HH_
#define _PARTICLESORTER_HH_

#include <algorithm>
#include <cuda_runtime.h>

#include "Basic.hh"
#include "Cells.hh"
#include "CellsFactory.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsParameters.hh"
#include "GrainsUtils.hh"
#include "Kinematics.hh"
#include "ParticleSorter_Kernels.hh"
#include "Torce.hh"

// =============================================================================
/** @brief The class ParticleSorter.

    This class sorts particles based on their Morton codes (Z-order curve)
    to improve memory access patterns and cache efficiency during collision
    detection. It maintains a separate Cells object with Morton ordering and
    provides methods to reorder particle data arrays accordingly.

    @author A.Yazdani - 2025 - Construction */
// =============================================================================
template <typename T, MemType M>
class ParticleSorter
{
protected:
    /** @name Parameters */
    //@{
    /** \brief Cells object with Morton ordering */
    GrainsMemBuffer<Cells<T, CellOrdering::MORTON>*, M> m_cells;
    /** \brief Morton codes for each particle (Z-order curve) */
    GrainsMemBuffer<uint64_t, M> m_mortonCodes;
    /** \brief Sorted indices array (maps sorted position to original) */
    GrainsMemBuffer<uint, M> m_sortedIndices;
    /** \brief Temporary buffers for gathering sorted data */
    GrainsMemBuffer<Vector3<T>, M>    m_tempPosition;
    GrainsMemBuffer<Kinematics<T>, M> m_tempVelocity;
    GrainsMemBuffer<Quaternion<T>, M> m_tempQuaternion;
    GrainsMemBuffer<Torce<T>, M>      m_tempTorce;
    GrainsMemBuffer<uint, M>          m_tempRigidBodyId;
    GrainsMemBuffer<uint, M>          m_tempComponentId;
    /** \brief CUDA streams for parallel gather operations (device only) */
    cudaStream_t m_streams[6];
    //@}

public:
    /** @name Constructors */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Default constructor */
    ParticleSorter() = default;

    // -------------------------------------------------------------------------
    /** @brief Constructor with parameters
        @param numObstacles total number of obstacles
        @param numParticles total number of particles */
    ParticleSorter(uint numObstacles, uint numParticles)
    {
        using GP             = GrainsParameters<T>;
        const auto& CD       = GP::m_collisionDetection;
        const auto& LCParams = CD.linkedCellParameters;
        // Create Cells object with Morton ordering
        CellsFactory<T, CellOrdering::MORTON>::template create<M>(
            LCParams.minCorner,
            LCParams.maxCorner,
            LCParams.cellSizeFactor,
            m_cells);

        // Allocate buffers
        m_mortonCodes.reserve(numParticles);
        m_sortedIndices.reserve(numParticles);
        m_tempPosition.reserve(numParticles);
        m_tempVelocity.reserve(numParticles);
        m_tempQuaternion.reserve(numParticles);
        m_tempTorce.reserve(numParticles);
        m_tempRigidBodyId.reserve(numParticles);
        m_tempComponentId.reserve(numParticles);

        if constexpr(M == MemType::DEVICE)
        {
            // Create CUDA streams for parallel gather operations
            for(int i = 0; i < 6; ++i)
                cudaStreamCreate(&m_streams[i]);
        }
    }

    // -------------------------------------------------------------------------
    /** @brief Destructor */
    ~ParticleSorter()
    {
        // Clean up the Cells object
        if constexpr(M == MemType::HOST)
        {
            if(m_cells.getSize() > 0 && m_cells[0] != nullptr)
                delete m_cells[0];
        }
        else if constexpr(M == MemType::DEVICE)
        {
            // Destroy CUDA streams first
            for(int i = 0; i < 6; ++i)
                cudaStreamDestroy(m_streams[i]);

            // Free the Cells object on device
            if(m_cells.getSize() > 0)
            {
                Cells<T, CellOrdering::MORTON>** d_cells = m_cells.getData();
                if(d_cells != nullptr)
                {
                    Cells<T, CellOrdering::MORTON>* h_cellPtr = nullptr;
                    cudaMemcpy(&h_cellPtr,
                               d_cells,
                               sizeof(Cells<T, CellOrdering::MORTON>*),
                               cudaMemcpyDeviceToHost);
                    if(h_cellPtr != nullptr)
                        cudaFree(h_cellPtr);
                }
            }
        }
    }
    //@}

    /** @name Methods */
    //@{
    // -------------------------------------------------------------------------
    /** @brief Sorts particle data arrays based on Morton codes
        @param position particle positions (input/output)
        @param velocity particle velocities (input/output)
        @param quaternion particle orientations (input/output)
        @param torce particle forces and torques (input/output)
        @param rigidBodyId rigid body IDs (input/output)
        @param componentId component IDs (input/output)
        @param numObstacles number of obstacles
        @param numParticles number of particles to sort */
    void sortParticles(GrainsMemBuffer<Vector3<T>, M>&    position,
                       GrainsMemBuffer<Kinematics<T>, M>& velocity,
                       GrainsMemBuffer<Quaternion<T>, M>& quaternion,
                       GrainsMemBuffer<Torce<T>, M>&      torce,
                       GrainsMemBuffer<uint, M>&          rigidBodyId,
                       GrainsMemBuffer<uint, M>&          componentId,
                       uint                               numObstacles,
                       uint                               numParticles)
    {
        if constexpr(M == MemType::HOST)
        {
            // Step 1: Compute Morton codes for particles only (skip obstacles)
            uint64_t*         mortonCodes = m_mortonCodes.getData();
            const Vector3<T>* pos         = position.getData();

            for(uint i = 0; i < numParticles; ++i)
                mortonCodes[i]
                    = m_cells[0]->computeCellHash(pos[numObstacles + i]);

            // Step 2: Initialize indices [0, 1, 2, ..., numParticles-1]
            uint* indices = m_sortedIndices.getData();
            for(uint i = 0; i < numParticles; ++i)
                indices[i] = i;

            // Step 3: Sort indices based on Morton codes
            std::sort(indices,
                      indices + numParticles,
                      [mortonCodes](uint a, uint b) {
                          return mortonCodes[a] < mortonCodes[b];
                      });

            // Step 4: Gather particle arrays according to sorted indices
            Vector3<T>*    tempPos    = m_tempPosition.getData();
            Kinematics<T>* tempVel    = m_tempVelocity.getData();
            Quaternion<T>* tempQuat   = m_tempQuaternion.getData();
            Torce<T>*      tempTorce  = m_tempTorce.getData();
            uint*          tempRBID   = m_tempRigidBodyId.getData();
            uint*          tempCompID = m_tempComponentId.getData();

            const Vector3<T>*    srcPos   = position.getData() + numObstacles;
            const Kinematics<T>* srcVel   = velocity.getData() + numObstacles;
            const Quaternion<T>* srcQuat  = quaternion.getData() + numObstacles;
            const Torce<T>*      srcTorce = torce.getData() + numObstacles;
            const uint*          srcRBID = rigidBodyId.getData() + numObstacles;
            const uint* srcCompID        = componentId.getData() + numObstacles;

            for(uint i = 0; i < numParticles; ++i)
            {
                uint idx      = indices[i];
                tempPos[i]    = srcPos[idx];
                tempVel[i]    = srcVel[idx];
                tempQuat[i]   = srcQuat[idx];
                tempTorce[i]  = srcTorce[idx];
                tempRBID[i]   = srcRBID[idx];
                tempCompID[i] = srcCompID[idx];
            }

            // Step 5: Copy sorted data back to particle arrays (skip obstacles)
            Vector3<T>*    dstPos    = position.getData() + numObstacles;
            Kinematics<T>* dstVel    = velocity.getData() + numObstacles;
            Quaternion<T>* dstQuat   = quaternion.getData() + numObstacles;
            Torce<T>*      dstTorce  = torce.getData() + numObstacles;
            uint*          dstRBID   = rigidBodyId.getData() + numObstacles;
            uint*          dstCompID = componentId.getData() + numObstacles;

            for(uint i = 0; i < numParticles; ++i)
            {
                dstPos[i]    = tempPos[i];
                dstVel[i]    = tempVel[i];
                dstQuat[i]   = tempQuat[i];
                dstTorce[i]  = tempTorce[i];
                dstRBID[i]   = tempRBID[i];
                dstCompID[i] = tempCompID[i];
            }
        }
        else if constexpr(M == MemType::DEVICE)
        {
            using GP = GrainsParameters<T>;
            // Compute grid dimensions
            uint numBlocks, threadsPerBlock;
            computeOptimalThreadsAndBlocks(numParticles,
                                           GP::m_GPU,
                                           threadsPerBlock,
                                           numBlocks);

            // Step 1: Compute Morton codes for particles only (skip obstacles)
            computeMortonCodes_Kernel<<<numBlocks, threadsPerBlock>>>(
                m_cells.getData(),
                position.getData() + numObstacles,
                m_mortonCodes.getData(),
                numParticles);
            cudaDeviceSynchronize();

            // Step 2: Initialize indices [0, 1, 2, ..., numParticles-1]
            thrust::device_ptr<uint> indicesPtr(m_sortedIndices.getData());
            thrust::sequence(indicesPtr, indicesPtr + numParticles);

            // Step 3: Sort indices based on Morton codes using thrust
            thrust::device_ptr<uint64_t> mortonPtr(m_mortonCodes.getData());
            thrust::sort_by_key(mortonPtr,
                                mortonPtr + numParticles,
                                indicesPtr);

            // Step 4: Gather particle arrays according to sorted indices using
            // multiple streams (skip obstacles)
            // Stream 0: Position
            gather_Kernel<<<numBlocks, threadsPerBlock, 0, m_streams[0]>>>(
                position.getData() + numObstacles,
                m_tempPosition.getData(),
                m_sortedIndices.getData(),
                numParticles);

            // Stream 1: Velocity
            gather_Kernel<<<numBlocks, threadsPerBlock, 0, m_streams[1]>>>(
                velocity.getData() + numObstacles,
                m_tempVelocity.getData(),
                m_sortedIndices.getData(),
                numParticles);

            // Stream 2: Quaternion
            gather_Kernel<<<numBlocks, threadsPerBlock, 0, m_streams[2]>>>(
                quaternion.getData() + numObstacles,
                m_tempQuaternion.getData(),
                m_sortedIndices.getData(),
                numParticles);

            // Stream 3: Torce
            gather_Kernel<<<numBlocks, threadsPerBlock, 0, m_streams[3]>>>(
                torce.getData() + numObstacles,
                m_tempTorce.getData(),
                m_sortedIndices.getData(),
                numParticles);

            // Stream 4: RigidBodyId
            gather_Kernel<<<numBlocks, threadsPerBlock, 0, m_streams[4]>>>(
                rigidBodyId.getData() + numObstacles,
                m_tempRigidBodyId.getData(),
                m_sortedIndices.getData(),
                numParticles);

            // Stream 5: ComponentId
            gather_Kernel<<<numBlocks, threadsPerBlock, 0, m_streams[5]>>>(
                componentId.getData() + numObstacles,
                m_tempComponentId.getData(),
                m_sortedIndices.getData(),
                numParticles);

            // Wait for all gather operations to complete
            for(int i = 0; i < 6; ++i)
                cudaStreamSynchronize(m_streams[i]);

            // Step 5: Copy sorted data back to particle arrays using multiple
            // streams (skip obstacles)
            // Stream 0: Position
            cudaMemcpyAsync(position.getData() + numObstacles,
                            m_tempPosition.getData(),
                            numParticles * sizeof(Vector3<T>),
                            cudaMemcpyDeviceToDevice,
                            m_streams[0]);

            // Stream 1: Velocity
            cudaMemcpyAsync(velocity.getData() + numObstacles,
                            m_tempVelocity.getData(),
                            numParticles * sizeof(Kinematics<T>),
                            cudaMemcpyDeviceToDevice,
                            m_streams[1]);

            // Stream 2: Quaternion
            cudaMemcpyAsync(quaternion.getData() + numObstacles,
                            m_tempQuaternion.getData(),
                            numParticles * sizeof(Quaternion<T>),
                            cudaMemcpyDeviceToDevice,
                            m_streams[2]);

            // Stream 3: Torce
            cudaMemcpyAsync(torce.getData() + numObstacles,
                            m_tempTorce.getData(),
                            numParticles * sizeof(Torce<T>),
                            cudaMemcpyDeviceToDevice,
                            m_streams[3]);

            // Stream 4: RigidBodyId
            cudaMemcpyAsync(rigidBodyId.getData() + numObstacles,
                            m_tempRigidBodyId.getData(),
                            numParticles * sizeof(uint),
                            cudaMemcpyDeviceToDevice,
                            m_streams[4]);

            // Stream 5: ComponentId
            cudaMemcpyAsync(componentId.getData() + numObstacles,
                            m_tempComponentId.getData(),
                            numParticles * sizeof(uint),
                            cudaMemcpyDeviceToDevice,
                            m_streams[5]);

            // Wait for all copy operations to complete
            for(int i = 0; i < 6; ++i)
                cudaStreamSynchronize(m_streams[i]);
        }
    }
    //@}
};

#endif
