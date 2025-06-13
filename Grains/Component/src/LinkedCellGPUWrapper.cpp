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
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= numComponents)
        return;

    // componentCellHash[tid] =
    // (*LC)->computeLinearCellHash( (*LC)->computeCellId( tr[tid].getOrigin() ) );
    componentCellHash[tid] = (*LC)->computeLinearCellHash(tr[tid].getOrigin());
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