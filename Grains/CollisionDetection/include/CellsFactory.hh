#ifndef _CELLSFACTORY_HH_
#define _CELLSFACTORY_HH_

#include "Cells.hh"
#include "GrainsMemBuffer.hh"

// =============================================================================
/** @brief The class CellsFactory.

	Creates cells object for the simulation.

    @author A.YAZDANI - 2025 - Construction */
// =============================================================================
template <typename T>
class CellsFactory
{
private:
    /**@name Contructors & Destructor */
    //@{
    /** @brief Default constructor (forbidden) */
    CellsFactory() = default;

    /** @brief Destructor (forbidden) */
    ~CellsFactory() = default;
    //@}

public:
    /**@name Methods */
    //@{
    /** @brief Creates and returns a buffer of linked cells
        @param LC Memory buffer for storing the linked cell object(s)
        @param numCells Total number of cells in the simulation */
    static void create(GrainsMemBuffer<Cells<T>*, MemType::HOST>& LC,
                       uint*                                      numCells);

    /** @brief Cells objects must be instantiated on device, if
		we want to use them on device. Copying from host is not supported due to
		runtime polymorphism for this class.
		This function reads a host-side Cells object, and mimics it
		in a given device buffer.
		It calls a device kernel that is implemented in the source file.
		@param h_LC Host-side Cells object
		@param d_LC Device-side Cells object */
    static void
        copyHostToDevice(GrainsMemBuffer<Cells<T>*, MemType::HOST>&   h_LC,
                         GrainsMemBuffer<Cells<T>*, MemType::DEVICE>& d_LC);
    //@}
};

#endif
