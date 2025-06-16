#include "LinkedCellGPUWrapper.hh"
#include "LinkedCell.hh"
#include "Vector3.hh"

// -----------------------------------------------------------------------------
// Kernel for computing the linear linked cell hash values for all components
template <typename T>
__GLOBAL__ void
    computeLinearLinkedCellHashGPU_Kernel(LinkedCell<T> const* const* LC,
                                          Transform3<T> const*        tr,
                                          uint  numComponents,
                                          uint* componentCellHash)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID >= numComponents)
        return;

    // componentCellHash[tID] =
    // (*LC)->computeLinearCellHash( (*LC)->computeCellId( tr[tID].getOrigin() ) );
    componentCellHash[tID] = (*LC)->computeLinearCellHash(tr[tID].getOrigin());
}

// -----------------------------------------------------------------------------
// Explicit instantiation
#define X(T)                                                        \
    template __GLOBAL__ void computeLinearLinkedCellHashGPU_Kernel( \
        LinkedCell<T> const* const* LC,                             \
        Transform3<T> const*        tr,                             \
        uint                        numComponents,                  \
        uint*                       componentCellHash);
X(float)
X(double)
#undef X