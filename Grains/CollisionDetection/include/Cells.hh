#ifndef _CELLS_HH_
#define _CELLS_HH_

#include "GrainsMemBuffer.hh"
#include "Vector3.hh"

// =============================================================================
/** @brief The class Cells.

    The broad-phase detection is done through Cells class. It limits the number 
    of potential collisions for collision to the neighboring components. 
    Neighboring components are those who belong to adjacent cells given a 
    uniform Cartesian grid for cells.

    @author A.Yazdani - 2024 - Construction 
    @author A.Yazdani - 2025 - Modification for NeighborList */
// =============================================================================
template <typename T>
class Cells
{
protected:
    /** @name Parameters */
    //@{
    /** \brief Min corner point of the domain */
    Vector3<T> m_minCorner;
    /** \brief Max corner point of the domain */
    Vector3<T> m_maxCorner;
    /** \brief Min corner point of the linked cell. This might be different from
        m_minCorner since some offsets might have been applied to make the
        domain fit symmetrically within the linked cell. */
    Vector3<T> m_minCornerLinkedCell;
    /** \brief Number of cells per each direction + total number of cells */
    uint4 m_numCells;
    /** \brief Size of each cell */
    T m_cellSize;
    /** \brief Inverse of size of each cell */
    T m_cellSize_inv;
    //@}

public:
    /** @name Constructors */
    //@{
    /** @brief Default constructor (forbidden except in derived classes) */
    __HOSTDEVICE__ Cells<T>();

    /** @brief Constructor with parameters
        @param minCorner minimum corner of the domain
        @param maxCorner maximum corner of the domain
        @param cellSize size of the cell */
    __HOSTDEVICE__
    Cells(const Vector3<T>& min, const Vector3<T>& max, T cellSize);

    /** @brief Destructor */
    __HOSTDEVICE__ ~Cells<T>();
    //@}

    /** @name Get methods */
    //@{
    /** @brief Gets the min corner point of the linked cell */
    __HOSTDEVICE__
    const Vector3<T>& getMinCorner() const;

    /** @brief Gets the max corner point of the linked cell */
    __HOSTDEVICE__
    const Vector3<T>& getMaxCorner() const;

    /** @brief Gets the min corner point of the linked cell */
    __HOSTDEVICE__
    const Vector3<T>& getMinCornerLinkedCell() const;

    /** @brief Gets the size of each cell */
    __HOSTDEVICE__
    T getCellSize() const;

    /** @brief Gets the number of cells along each direction and total */
    __HOSTDEVICE__
    uint4 getNumCellsPerDirection() const;

    /** @brief Gets the number of cells */
    __HOSTDEVICE__
    uint getNumCells() const;

    /** @brief Returns the size required for neighbor list */
    __HOSTDEVICE__
    uint getSizeOfNeighborCells() const;
    //@}

    /** @name Methods */
    //@{
    /** @brief Resizes the linked cells
        @param cellSize new size of the cell */
    __HOSTDEVICE__
    void resize(const T cellSize);

    /** @brief Generates neighbor list for cells
        @note On device side, the list is generated using a threadPerCell
        strategy. Starting position for each cell is given by the `start`
        parameter. On host side, there is no need for this parameter, as the
        list is generated with a single thread.
        @param neighborCells output array for neighbor cells
        @param start starting index for neighbor cells
        @param end ending index for neighbor cells */
    __HOSTDEVICE__
    void generateNeighborCells(uint* neighborCells,
                               uint  start = 0,
                               uint  end   = 0) const;

    /** @brief Checks if a cell Id is in range
        @param id 3D Id */
    __HOSTDEVICE__
    bool isValid(const uint3& id) const;

    /** @brief Returns the 3d Id of the cell which the point belongs to
        @param p point
        @param checkIfValid flag to check if the cell ID is valid */
    __HOSTDEVICE__
    uint3 computeCellID(const Vector3<T>& p, bool checkIfValid = true) const;

    /** @brief Returns the 3d Id of the cell given its hash
        @param cellHash cell hash */
    __HOSTDEVICE__
    uint3 computeCellID(const uint cellHash) const;

    /** @brief Returns the cell hash of a given point
        @param p point */
    __HOSTDEVICE__
    uint computeCellHash(const Vector3<T>& p) const;

    /** @brief Returns the cell hash from the 3d Id of the cell
        @param cellId 3d cell Id */
    __HOSTDEVICE__
    uint computeCellHash(const uint3& cellId) const;

    /** @brief Returns the cell hash from the 3d Id of the cell
        @param i position of the cell in the x-direction
        @param j position of the cell in the y-direction
        @param k position of the cell in the z-direction */
    __HOSTDEVICE__
    uint computeCellHash(uint i, uint j, uint k) const;
    //@}
};

#endif