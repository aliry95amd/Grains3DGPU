#ifndef _FORCEMODULEFACTORY_HH_
#define _FORCEMODULEFACTORY_HH_

#include <memory>

#include "ForceModule.hh"
#include "GrainsParameters.hh"

// =================================================================================================
/** @brief The class ForceModuleFactory.

    Creates a ForceModule<T,M> from the pair buffer capacity and the global
    GrainsParameters flag isContactWithMemory.  Mirrors the factory pattern used by
    CollisionDetectionModuleFactory.

    @author A.Yazdani - 2025 - Construction */
// =================================================================================================
template <typename T, MemType M>
class ForceModuleFactory
{
    static_assert(M == MemType::HOST || M == MemType::DEVICE,
                  "ForceModuleFactory only supports MemType::HOST or MemType::DEVICE");

private:
    /** @name Constructors */
    //@{
    ForceModuleFactory()  = default;
    ~ForceModuleFactory() = default;
    //@}

public:
    /** @name Methods */
    //@{
    // ---------------------------------------------------------------------------------------------
    /** @brief Creates and returns a fully initialised ForceModule.
        @param pairCapacity Initial pair buffer capacity (obtained from
                            CollisionDetectionModule::getPairBufferSize())
        @return unique_ptr to the created ForceModule */
    static std::unique_ptr<ForceModule<T, M>> create(size_t pairCapacity)
    {
        bool isContactWithMemory = GrainsParameters<T>::m_isContactWithMemory;
        return std::make_unique<ForceModule<T, M>>(pairCapacity, isContactWithMemory);
    }
    //@}
};

#endif
