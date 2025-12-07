#include "ContactForceModelFactory.hh"
#include "GrainsParameters.hh"
#include "HookeContactForceModel.hh"

/* ============================================================================================== */
/* Low-Level Methods                                                                              */
/* ============================================================================================== */
// GPU kernel to construct the ContactForceModel on device.
// This is mandatory as we cannot access device memory addresses on the host
// So, we pass a device memory address to a kernel.
// Memory address is then populated within the kernel.
template <typename T, typename... Arguments>
__GLOBAL__ void createContactForceModelKernel(ContactForceModel<T>** CF,
                                              uint                   index,
                                              ContactForceModelType  contactForceModelType,
                                              Arguments... args)
{
    uint tID = blockIdx.x * blockDim.x + threadIdx.x;
    if(tID > 0)
        return;

    if(contactForceModelType == HOOKE)
    {
        static_assert(sizeof...(args) == 5, "Hooke contact force model requires 5 parameters!");
        CF[index] = new HookeContactForceModel<T>(args...);
    }
}

/* ============================================================================================== */
/* High-Level Methods                                                                             */
/* ============================================================================================== */
// Creates and stores a ContactForceModel object in the host memory.
template <typename T>
__HOST__ void
    ContactForceModelFactory<T>::create(DOMElement*                                            root,
                                        GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>& CF)
{
    uint numContactPairs = GrainsParameters<T>::m_numContactPairs;
    CF.initialize(numContactPairs);
    DOMNodeList* allContacts = ReaderXML::getNodes(root, "ContactForceModel");
    for(XMLSize_t i = 0; i < allContacts->getLength(); i++)
    {
        DOMNode*    contact     = allContacts->item(i);
        std::string contactType = ReaderXML::getNodeAttr_String(contact, "Type");
        // Contact Pair
        DOMNode*    material = ReaderXML::getNode(contact, "Material");
        std::string matA     = ReaderXML::getNodeAttr_String(material, "materialA");
        std::string matB     = ReaderXML::getNodeAttr_String(material, "materialB");
        uint        matA_id  = GrainsParameters<T>::m_materialMap[matA];
        uint        matB_id  = GrainsParameters<T>::m_materialMap[matB];
        uint        index    = computeHash(matA_id, matB_id);
        // Contact Parameters
        DOMNode* parameters = ReaderXML::getNode(contact, "Parameters");
        if(contactType == "Hooke")
            CF[index] = new HookeContactForceModel<T>(parameters);
        else
            GAbort("Unknown contact force model! Aborting Grains!");
    }
}

// -------------------------------------------------------------------------------------------------
// Hash function to map a pair of material IDs x and y to a single ID to access
// the proper contact force model between them.
// We DO NOT CHECK the inputs. We TRUST the user on this. However, inproper
// inputs would persumably crash the code by accessing memory accesses that are
// not valid.
template <typename T>
__HOSTDEVICE__ uint ContactForceModelFactory<T>::computeHash(uint x, uint y)
{
    uint s = min(x, y);  // smaller one
    uint l = max(x, y);  // larger one
    return (l * (l + 1) / 2 + s);
}

// -------------------------------------------------------------------------------------------------
// Constructs a ContactForceModel object on device.
template <typename T>
__HOST__ void ContactForceModelFactory<T>::copyHostToDevice(
    GrainsMemBuffer<ContactForceModel<T>*, MemType::HOST>&   h_CF,
    GrainsMemBuffer<ContactForceModel<T>*, MemType::DEVICE>& d_CF)
{
    // Allocate the device memory for the contact force models
    d_CF.initialize(h_CF.getSize());
    for(uint i = 0; i < h_CF.getSize(); ++i)
    {
        if(h_CF[i] == nullptr)
            continue;

        // Extracting info from the host side object
        ContactForceModel<T>* contact     = h_CF[i];
        ContactForceModelType contactType = contact->getContactForceModelType();

        if(contactType == HOOKE)
        {
            HookeContactForceModel<T>* c = dynamic_cast<HookeContactForceModel<T>*>(contact);
            T                          kn, en, etat, muc, kr;
            c->getContactForceModelParameters(kn, en, etat, muc, kr);
            createContactForceModelKernel<<<1, 1>>>(d_CF.getData(),
                                                    i,
                                                    HOOKE,
                                                    kn,
                                                    en,
                                                    etat,
                                                    muc,
                                                    kr);
        }
        else
            GAbort("Contact force model is not implemented for GPU!", "Aborting Grains!");
    }
    cudaDeviceSynchronize();
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class ContactForceModelFactory<float>;
template class ContactForceModelFactory<double>;
