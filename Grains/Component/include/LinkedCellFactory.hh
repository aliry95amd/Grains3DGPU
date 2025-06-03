#ifndef _LINKEDCELLFACTORY_HH_
#define _LINKEDCELLFACTORY_HH_

#include "GrainsMemBuffer.hh"
#include "LinkedCell.hh"

// =============================================================================
/** @brief The class LinkedCellFactory.

	Creates the linked cell for the simulation.

    @author A.YAZDANI - 2025 - Construction */
// =============================================================================
template <typename T>
class LinkedCellFactory
{
private:
    /**@name Contructors & Destructor */
    //@{
    /** @brief Default constructor (forbidden) */
    LinkedCellFactory() = default;

    /** @brief Destructor (forbidden) */
    ~LinkedCellFactory() = default;
    //@}

public:
    /**@name Methods */
    //@{
    /** @brief Creates and returns a buffer of reference rigid bodies given an 
		XML node
        @param root XML node
        @param LC Memory buffer for storing the linked cell object(s)
        @param numCells Total number of cells in the simulation */
    static void create(DOMNode*                                        root,
                       GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>& LC,
                       uint& numCells);

    /** @brief LinkedCell objects must be instantiated on device, if
		we want to use them on device. Copying from host is not supported due to
		runtime polymorphism for this class.
		This function reads a host-side LinkedCell object, and mimics it
		in a given device buffer.
		It calls a device kernel that is implemented in the source file.
		@param h_LC Host-side LinkedCell object
		@param d_LC Device-side LinkedCell object */
    static void copyHostToDevice(
        GrainsMemBuffer<LinkedCell<T>*, MemType::HOST>&   h_LC,
        GrainsMemBuffer<LinkedCell<T>*, MemType::DEVICE>& d_LC);
    //@}
};

#endif
