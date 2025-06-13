#ifndef _LINKEDCELLGPUWRAPPER_HH_
#define _LINKEDCELLGPUWRAPPER_HH_

#include "LinkedCell.hh"
#include "Transform3.hh"
#include "Vector3.hh"

// =============================================================================
/** @brief Header for LinkedCellGPUWrapper.

    Various GPU kernels used in the LinkedCell class.

    @author A.Yazdani - 2024 - Construction */
// =============================================================================
/** @name LinkedCellGPUWrapper : External methods */
//@{
/** @brief Computes the linear linked cell hash values for all components
@param LC linked cell
@param tr list of all transformations
@param numComponents number of all components in the simulation */
template <typename T>
__GLOBAL__ void
    computeLinearLinkedCellHashGPU_Kernel(LinkedCell<T> const* const* LC,
                                          Transform3<T> const*        tr,
                                          uint  numComponents,
                                          uint* componentCellHash);
//@}

#endif