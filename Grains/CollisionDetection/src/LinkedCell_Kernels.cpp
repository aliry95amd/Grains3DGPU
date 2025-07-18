#include <cooperative_groups.h>

#include "Basic.hh"

// -----------------------------------------------------------------------------
// Kernel to find the start of each cell
__GLOBAL__ void computeCellStart_Kernel(const uint* particleHash,
                                        uint        numParticles,
                                        uint*       cellStart)
{
    using namespace cooperative_groups;
    // Handle to thread block group
    thread_block           cta = this_thread_block();
    extern __shared__ uint sharedHash[]; // blockSize + 1 elements
    uint                   tid = blockIdx.x * blockDim.x + threadIdx.x;

    uint hash;
    if(tid < numParticles)
    {
        hash = particleHash[tid];
        // Load hash data into shared memory so that we can look at neighboring
        // particle's hash value without loading two hash values per thread
        sharedHash[threadIdx.x + 1] = hash;
        // first thread in block must load neighboring particle hash as well
        if(tid > 0 && threadIdx.x == 0)
            sharedHash[0] = particleHash[tid - 1];
    }
    sync(cta);

    if(tid < numParticles)
    {
        // If this particle has a different cell hash value to the previous
        // particle then it must be the first particle in the cell.
        // As it isn't the first particle, it must also be the end of the
        // previous particle's cell.
        if(tid == 0 || hash != sharedHash[threadIdx.x])
        {
            cellStart[hash] = tid;
        }
    }
}