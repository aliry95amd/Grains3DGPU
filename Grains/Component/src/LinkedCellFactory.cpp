#include "LinkedCellFactory.hh"
#include "GrainsParameters.hh"
#include "LinkedCell.hh"

/* ========================================================================== */
/*                             Low-Level Methods                              */
/* ========================================================================== */
// GPU kernel to construct the LinkedCell on device.
// This is mandatory as we cannot access device memory addresses on the host
// So, we pass a device memory address to a kernel.
// Memory address is then populated within the kernel.
template <typename T, typename... Arguments>
__GLOBAL__ void createLinkedCellKernel(LinkedCell<T>** LC,
                                       uint            index,
                                       T               minX,
                                       T               minY,
                                       T               minZ,
                                       T               maxX,
                                       T               maxY,
                                       T               maxZ,
                                       T               size,
                                       uint*           numCells)
{
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid > 0)
        return;

    LC[index] = new LinkedCell<T>(Vector3<T>(minX, minY, minZ),
                                  Vector3<T>(maxX, maxY, maxZ),
                                  size);
    *numCells = LC[index]->getNumCells();
}

/* ========================================================================== */
/*                             High-Level Methods                             */
/* ========================================================================== */
// Creates and stores a LinkedCell object in the host memory.
template <typename T>
__HOST__ void LinkedCellFactory<T>::create(
    DOMNode*                                        root,
    GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>& LC,
    uint&                                           numCells)
{
    using GP = GrainsParameters<T>;

    LC.reserve(1);
    T LC_coeff = T(1);
    if(ReaderXML::hasNodeAttr(root, "CellSizeFactor"))
        LC_coeff = T(ReaderXML::getNodeAttr_Double(root, "CellSizeFactor"));
    if(LC_coeff < T(1))
        LC_coeff = T(1);
    T size = LC_coeff * T(2) * GP::m_maxRadius;

    LC[0]    = new LinkedCell<T>(GP::m_origin, GP::m_maxCoordinate, size);
    numCells = LC[0]->getNumCells();
    GoutWI(9, "LinkedCell with", numCells, "cells is created on host.");
}

// -----------------------------------------------------------------------------
// Constructs a LinkedCell object on device.
template <typename T>
__HOST__ void LinkedCellFactory<T>::copyHostToDevice(
    GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>&   h_LC,
    GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>& d_LC)
{
    // Allocate the device memory for the linked cells
    d_LC.allocate(h_LC.getSize());
    uint  h_numCells = 0;
    uint* d_numCells;
    cudaMalloc(&d_numCells, sizeof(uint));
    for(uint i = 0; i < h_LC.getSize(); ++i)
    {
        if(h_LC[i] == nullptr)
            continue;

        // Extracting info from the host side object
        Vector3<T> origin        = h_LC[i]->getMinCorner();
        Vector3<T> maxCoordinate = h_LC[i]->getMaxCorner();
        T          size          = h_LC[i]->getCellExtents();
        createLinkedCellKernel<<<1, 1>>>(d_LC.getData(),
                                         i,
                                         origin[X],
                                         origin[Y],
                                         origin[Z],
                                         maxCoordinate[X],
                                         maxCoordinate[Y],
                                         maxCoordinate[Z],
                                         size,
                                         d_numCells);
        cudaMemcpy(&h_numCells,
                   d_numCells,
                   sizeof(uint),
                   cudaMemcpyDeviceToHost);
        GoutWI(9, "LinkedCell with", h_numCells, "cells is created on device.");
    }
    cudaDeviceSynchronize();
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class LinkedCellFactory<float>;
template class LinkedCellFactory<double>;
