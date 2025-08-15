#include "CellsFactory.hh"
#include "Cells.hh"

/* ========================================================================== */
/*                             Low-Level Methods                              */
/* ========================================================================== */
// GPU kernel to construct the Cells on device.
// This is mandatory as we cannot access device memory addresses on the host
// So, we pass a device memory address to a kernel.
// Memory address is then populated within the kernel.
template <typename T, typename... Arguments>
__GLOBAL__ void createCellsKernel(Cells<T>** cells,
                                  uint       index,
                                  T          minX,
                                  T          minY,
                                  T          minZ,
                                  T          maxX,
                                  T          maxY,
                                  T          maxZ,
                                  T          size,
                                  uint*      numCells)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID > 0)
        return;

    cells[index] = new Cells<T>(Vector3<T>(minX, minY, minZ),
                                Vector3<T>(maxX, maxY, maxZ),
                                size);
    *numCells    = cells[index]->getNumCells();
}

/* ========================================================================== */
/*                             High-Level Methods                             */
/* ========================================================================== */
// Creates and stores a LinkedCell object in the host memory.
template <typename T>
__HOST__ void
    CellsFactory<T>::create(const Vector3<T>& minCorner,
                            const Vector3<T>& maxCorner,
                            const T           cellSize,
                            GrainsMemBuffer<Cells<T>*, MemType::HOST>& cells,
                            uint*                                      numCells)
{
    cells.reserve(1);
    cells[0]  = new Cells<T>(minCorner, maxCorner, cellSize);
    *numCells = cells[0]->getNumCells();
    GoutWI(9, "LinkedCell with", *numCells, "cells is created on host.");
}

// -----------------------------------------------------------------------------
// Constructs a LinkedCell object on device.
template <typename T>
__HOST__ void CellsFactory<T>::copyHostToDevice(
    GrainsMemBuffer<Cells<T>*, MemType::HOST>&   h_cells,
    GrainsMemBuffer<Cells<T>*, MemType::DEVICE>& d_cells)
{
    // Allocate the device memory for the linked cells
    d_cells.allocate(h_cells.getSize());
    uint  h_numCells = 0;
    uint* d_numCells;
    cudaMalloc(&d_numCells, sizeof(uint));
    for(uint i = 0; i < h_cells.getSize(); ++i)
    {
        if(h_cells[i] == nullptr)
            continue;

        // Extracting info from the host side object
        Vector3<T> origin        = h_cells[i]->getMinCorner();
        Vector3<T> maxCoordinate = h_cells[i]->getMaxCorner();
        T          size          = h_cells[i]->getCellSize();
        createCellsKernel<<<1, 1>>>(d_cells.getData(),
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
template class CellsFactory<float>;
template class CellsFactory<double>;
