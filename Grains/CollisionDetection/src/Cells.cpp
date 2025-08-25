#include "Cells.hh"
#include "Transform3.hh"
#include "VectorMath.hh"

// -----------------------------------------------------------------------------
// Default constructor
template <typename T>
__HOSTDEVICE__ Cells<T>::Cells()
{
}

// -----------------------------------------------------------------------------
// Constructor with min and max points along with extent of each cell
template <typename T>
__HOSTDEVICE__
    Cells<T>::Cells(const Vector3<T>& min, const Vector3<T>& max, T cellSize)
    : m_minCorner(min)
    , m_maxCorner(max)
{
    resize(cellSize);
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
__HOSTDEVICE__ Cells<T>::~Cells()
{
}

// -----------------------------------------------------------------------------
// Gets the min corner point of the linked cell
template <typename T>
__HOSTDEVICE__ const Vector3<T>& Cells<T>::getMinCorner() const
{
    return (m_minCorner);
}

// -----------------------------------------------------------------------------
// Gets the max corner point of the linked cell
template <typename T>
__HOSTDEVICE__ const Vector3<T>& Cells<T>::getMaxCorner() const
{
    return (m_maxCorner);
}

// -----------------------------------------------------------------------------
// Gets the extent of each cell
template <typename T>
__HOSTDEVICE__ T Cells<T>::getCellSize() const
{
    return (m_cellSize);
}

// -----------------------------------------------------------------------------
// Gets the number of cells
template <typename T>
__HOSTDEVICE__ uint Cells<T>::getNumCells() const
{
    return (m_numCells.w);
}

// -----------------------------------------------------------------------------
// Gets the required size for neighbor cells
template <typename T>
__HOSTDEVICE__ uint Cells<T>::getSizeOfNeighborCells() const
{
    return (27 * m_numCells.w);
}

// -----------------------------------------------------------------------------
// Resizes the linked cells
template <typename T>
__HOSTDEVICE__ void Cells<T>::resize(const T cellSize)
{
    T DX = m_maxCorner[X] - m_minCorner[X];
    T DY = m_maxCorner[Y] - m_minCorner[Y];
    T DZ = m_maxCorner[Z] - m_minCorner[Z];

    m_cellSize     = cellSize;
    m_cellSize_inv = T(1) / cellSize;
    m_numCells.x   = uint(DX * m_cellSize_inv);
    m_numCells.y   = uint(DY * m_cellSize_inv);
    m_numCells.z   = uint(DZ * m_cellSize_inv);
    m_numCells.w   = m_numCells.x * m_numCells.y * m_numCells.z;

    m_minCornerLinkedCell[X] = m_minCorner[X] - (m_numCells.x * cellSize - DX);
    m_minCornerLinkedCell[Y] = m_minCorner[Y] - (m_numCells.y * cellSize - DY);
    m_minCornerLinkedCell[Z] = m_minCorner[Z] - (m_numCells.z * cellSize - DZ);
}

// -----------------------------------------------------------------------------
// Generates neighbor list for cells
// TODO: We can also improve this so each thread can take more than one cell,
// but that would only be useful for cases where this is a bottleneck.
// TODO: The length of the array is fixed as 27 * m_numCells.w. We can implement
// a more dynamic approach where we first compute the maximum possible
// number of neighbors for each cell, and then use that to allocate the neighbor
// list. This would be more efficient in terms of memory usage.
template <typename T>
__HOSTDEVICE__ void Cells<T>::generateNeighborCells(uint* neighborCells,
                                                    uint  start,
                                                    uint  end) const
{
    if(end == 0)
        end = m_numCells.w;
    constexpr uint numNeighbors = 27;
    // Allocate memory for the flat array
    uint offset;
    // Precompute neighbors for each cell
    // clang-format off
    for(uint cellHash = start; cellHash < end; ++cellHash)
    {
        offset       = numNeighbors * cellHash;
        uint3 cellId = { cellHash % m_numCells.x,
                        (cellHash / m_numCells.x) % m_numCells.y,
                         cellHash / (m_numCells.x * m_numCells.y)};
        for(int k = -1; k < 2; ++k) {
        for(int j = -1; j < 2; ++j) {
        for(int i = -1; i < 2; ++i) {
            int nx = cellId.x + i;
            int ny = cellId.y + j;
            int nz = cellId.z + k;

            // Check if the neighbor is within bounds
            if(nx >= 0 && nx < m_numCells.x && 
               ny >= 0 && ny < m_numCells.y && 
               nz >= 0 && nz < m_numCells.z)
            {
                uint neighborHash = nx + 
                                    ny * m_numCells.x +
                                    nz * m_numCells.x * m_numCells.y;
                neighborCells[offset++] = neighborHash;
            }
            else
            {
                // Invalid neighbor
                neighborCells[offset++] = UINT_MAX;
            }
        } } }
    }
    // clang-format on
}

// -----------------------------------------------------------------------------
// Checks if a cell Id is in range
template <typename T>
__HOSTDEVICE__ void Cells<T>::checkBound(const uint3& id) const
{
    GAssert(id.x < m_numCells.x || id.y < m_numCells.y || id.z < m_numCells.z,
            "Linked cell range exceeded!");
}

// -----------------------------------------------------------------------------
// Returns the 3d Id of the cell which the point belongs to
template <typename T>
__HOSTDEVICE__ uint3 Cells<T>::computeCellID(const Vector3<T>& p) const
{
    uint3 cellId;
    // static_cast is faster than floor, though it comes with a cost ...
    // if the operand is -0.7, it gives 0.
    // cellId.x = static_cast<int>((p[X] - m_minCorner[X]) * m_cellSize_inv);
    // cellId.y = static_cast<int>((p[Y] - m_minCorner[Y]) * m_cellSize_inv);
    // cellId.z = static_cast<int>((p[Z] - m_minCorner[Z]) * m_cellSize_inv);
    cellId.x = floor((p[X] - m_minCornerLinkedCell[X]) * m_cellSize_inv);
    cellId.y = floor((p[Y] - m_minCornerLinkedCell[Y]) * m_cellSize_inv);
    cellId.z = floor((p[Z] - m_minCornerLinkedCell[Z]) * m_cellSize_inv);
    checkBound(cellId);
    return (cellId);
}

// -----------------------------------------------------------------------------
// Returns the cell hash value of a given point
template <typename T>
__HOSTDEVICE__ uint Cells<T>::computeCellHash(const Vector3<T>& p) const
{
    return (computeCellHash(computeCellID(p)));
}

// -----------------------------------------------------------------------------
// Returns the cell hash value from the 3d Id of the cell
template <typename T>
__HOSTDEVICE__ uint Cells<T>::computeCellHash(const uint3& cellID) const
{
    return ((cellID.z * m_numCells.y + cellID.y) * m_numCells.x + cellID.x);
}

// -----------------------------------------------------------------------------
// Returns the cell hash value from the Id along each axis
template <typename T>
__HOSTDEVICE__ uint Cells<T>::computeCellHash(const uint i,
                                              const uint j,
                                              const uint k) const
{
    return ((k * m_numCells.y + j) * m_numCells.x + i);
}

// -----------------------------------------------------------------------------
// Returns the cell hash value for a neighboring cell in the direction given by
// (i, j, k)
template <typename T>
__HOSTDEVICE__ uint Cells<T>::computeNeighborCellHash(uint cellHash,
                                                      uint i,
                                                      uint j,
                                                      uint k) const
{
    uint z = cellHash / (m_numCells.x * m_numCells.y);
    uint y = (cellHash / m_numCells.x) % m_numCells.y;
    uint x = cellHash % m_numCells.x;
    return (computeCellHash(x + i, y + j, z + k));
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class Cells<float>;
template class Cells<double>;