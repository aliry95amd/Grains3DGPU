#include "Insertion.hh"
#include "GrainsUtils.hh"
#include <cstdlib>
#include <ctime>

/* ========================================================================== */
/*                             Low-Level Methods                              */
/* ========================================================================== */
// Reads if the root is of type Random
template <typename T>
__HOST__ static INLINE InsertionInfo<T> readDataRand(DOMNode* root)
{
    // Random generator seed. We pass it to InsertionWindow directly.
    // We also set the seed with srand. We use it for to randomly pick an
    // insertion window
    RandomGeneratorSeed rgs;
    unsigned            cSeed = 1; // default C RNG seed for window selection
    std::string seedString    = ReaderXML::getNodeAttr_String(root, "Seed");
    if(seedString == "UserDefined")
    {
        uint val = ReaderXML::getNodeAttr_Int(root, "Value");
        GAssert(val, "Seed value is not provided. Aborting Grains!");
        rgs   = RGS_UDEF;
        cSeed = static_cast<unsigned>(val);
        GoutWI(12,
               "Random initialization with seed",
               std::to_string(val) + ".");
    }
    else if(seedString == "Random")
    {
        rgs   = RGS_RANDOM;
        cSeed = static_cast<unsigned>(time(NULL));
        GoutWI(12, "Random initialization with random seed.");
    }
    // if ( seedString == "Default" )
    else
    {
        rgs   = RGS_DEFAULT;
        cSeed = 1u;
        GoutWI(12, "Random initialization with default seed.");
    }
    // Seed C RNG for selecting among multiple insertion windows
    srand(cSeed);

    // Insertion window
    DOMNode* nWindows = ReaderXML::getNode(root, "Windows");
    std::vector<InsertionWindow<T>> insertionWindows;
    if(nWindows)
    {
        // DOMNodeList* allWindows = ReaderXML::getNodes( nWindows, "Window" );
        DOMNodeList* allWindows = ReaderXML::getNodes(nWindows);
        for(int i = 0; i < allWindows->getLength(); i++)
        {
            DOMNode* nWindow = allWindows->item(i);
            insertionWindows.push_back(InsertionWindow<T>(nWindow, rgs));
        }
    }

    return (insertionWindows);
}

// -----------------------------------------------------------------------------
// Reads if the root is of type File
template <typename T>
__HOST__ static INLINE InsertionInfo<T> readDataFile(DOMNode* root)
{
    std::string   fileName = ReaderXML::getNodeAttr_String(root, "Name");
    std::ifstream file(fileName);
    GAssert(file.good(), "File initialization failed! Aborting Grains!");
    GoutWI(12, "File initialization with path" + fileName + ".");

    return (file);
}

// -----------------------------------------------------------------------------
// Reads if the root is of type Constant
template <typename T>
__HOST__ static INLINE InsertionInfo<T> readDataCons(DOMNode* root)
{
    T          xVal = T(ReaderXML::getNodeAttr_Double(root, "X"));
    T          yVal = T(ReaderXML::getNodeAttr_Double(root, "Y"));
    T          zVal = T(ReaderXML::getNodeAttr_Double(root, "Z"));
    Vector3<T> vec(xVal, yVal, zVal);
    GoutWI(12, "Constant initialization with", Vector3ToString(vec), ".");

    return (vec);
}

// -----------------------------------------------------------------------------
// Reads if the root is of type Zero
template <typename T>
__HOST__ static INLINE InsertionInfo<T> readDataZero(DOMNode* root)
{
    Vector3<T> vec(T(0), T(0), T(0));
    GoutWI(12, "Zero initialization.");

    return (vec);
}

/* ========================================================================== */
/*                             High-Level Methods                             */
/* ========================================================================== */
// Default constructor
template <typename T>
__HOST__ Insertion<T>::Insertion()
    : m_positionType(DEFAULTINSERTION)
    , m_orientationType(DEFAULTINSERTION)
    , m_translationalVelType(DEFAULTINSERTION)
    , m_angularVelType(DEFAULTINSERTION)
    , m_positionInsertionInfo(Vector3<T>())
    , m_orientationInsertionInfo(Vector3<T>())
    , m_translationalVelInsertionInfo(Vector3<T>())
    , m_angularVelInsertionInfo(Vector3<T>())
    , m_forceInsertion(false)
{
}

// -----------------------------------------------------------------------------
// Constructor with XML node
template <typename T>
__HOST__ Insertion<T>::Insertion(DOMNode* dn)
{
    // We define a lambda function to read the XML node
    auto read = [](DOMNode* root, InsertionType& type, InsertionInfo<T>& data) {
        std::string nType = ReaderXML::getNodeAttr_String(root, "Type");
        if(nType == "Random")
        {
            type = RANDOMINSERTION;
            data = readDataRand<T>(root);
        }
        else if(nType == "File")
        {
            type = FILEINSERTION;
            data = readDataFile<T>(root);
        }
        else if(nType == "Constant")
        {
            type = CONSTANTINSERTION;
            data = readDataCons<T>(root);
        }
        else if(nType == "Zero")
        {
            type = DEFAULTINSERTION;
            data = readDataZero<T>(root);
        }
        else
            GAbort("Unknown Type in ParticleInsertion! Aborting Grains!");
    };

    GAssert(dn, "ParticleInsertion node is missing! Aborting Grains!");

    GoutWI(9, "Reading PositionInsertion Policy ...");
    if(ReaderXML::getNode(dn, "InitialPosition"))
    {
        DOMNode* nIP = ReaderXML::getNode(dn, "InitialPosition");
        read(nIP, m_positionType, m_positionInsertionInfo);
    }
    else
    {
        GAbort("InitialPosition node is missing in ParticleInsertion! Aborting "
               "Grains!");
    }

    GoutWI(9, "Reading OrientationInsertion Policy ...");
    if(ReaderXML::getNode(dn, "InitialOrientation"))
    {
        DOMNode* nIO = ReaderXML::getNode(dn, "InitialOrientation");
        read(nIO, m_orientationType, m_orientationInsertionInfo);
    }
    else
    {
        m_orientationType          = DEFAULTINSERTION;
        m_orientationInsertionInfo = Vector3<T>(T(0), T(0), T(0));
        GoutWI(12, "No InitialOrientation node found. Using default.");
    }

    GoutWI(9, "Reading VeclocityInsertion Policy ...");
    if(ReaderXML::getNode(dn, "InitialVelocity"))
    {
        DOMNode* nIV = ReaderXML::getNode(dn, "InitialVelocity");
        read(nIV, m_translationalVelType, m_translationalVelInsertionInfo);
    }
    else
    {
        m_translationalVelType          = DEFAULTINSERTION;
        m_translationalVelInsertionInfo = Vector3<T>(T(0), T(0), T(0));
        GoutWI(12, "No InitialVelocity node found. Using default.");
    }

    GoutWI(9, "Reading AngularVeclocityInsertion Policy ...");
    if(ReaderXML::getNode(dn, "InitialAngularVelocity"))
    {
        DOMNode* nIA = ReaderXML::getNode(dn, "InitialAngularVelocity");
        read(nIA, m_angularVelType, m_angularVelInsertionInfo);
    }
    else
    {
        m_angularVelType          = DEFAULTINSERTION;
        m_angularVelInsertionInfo = Vector3<T>(T(0), T(0), T(0));
        GoutWI(12, "No InitialAngularVelocity node found. Using default.");
    }

    if(ReaderXML::hasNodeAttr(dn, "ForceInsertion"))
        m_forceInsertion = static_cast<bool>(
            ReaderXML::getNodeAttr_Int(dn, "ForceInsertion"));
    else
        m_forceInsertion = false;
}

// -----------------------------------------------------------------------------
// Destructor
template <typename T>
__HOST__ Insertion<T>::~Insertion()
{
    if(std::holds_alternative<std::ifstream>(m_positionInsertionInfo))
        (std::get<std::ifstream>(m_positionInsertionInfo)).close();
    if(std::holds_alternative<std::ifstream>(m_orientationInsertionInfo))
        (std::get<std::ifstream>(m_orientationInsertionInfo)).close();
    if(std::holds_alternative<std::ifstream>(m_translationalVelInsertionInfo))
        (std::get<std::ifstream>(m_translationalVelInsertionInfo)).close();
    if(std::holds_alternative<std::ifstream>(m_angularVelInsertionInfo))
        (std::get<std::ifstream>(m_angularVelInsertionInfo)).close();
}

// -----------------------------------------------------------------------------
// Returns a vector of Vector3 accroding to type and data
template <typename T>
__HOST__ Vector3<T>
         Insertion<T>::fetchInsertionDataForEach(InsertionType const type,
                                            InsertionInfo<T>&   data)
{
    // We only return a vector3. It is clear how it works for position, and
    // kinematics. However, for orientation, it returns the vector3 of rotation
    // angles. We later construct a quaternion.
    if(type == RANDOMINSERTION)
    {
        auto& IWs = std::get<std::vector<InsertionWindow<T>>>(data);
        GAssert(!IWs.empty(),
                "Random insertion selected but no InsertionWindow defined!");
        if(IWs.size() == 1)
            return IWs[0].generateRandomPoint();

        // Randomly choose between the available insertion windows
        int random_IW = static_cast<int>(rand() % IWs.size());
        return IWs[random_IW].generateRandomPoint();
    }
    else if(type == FILEINSERTION)
    {
        Vector3<T> output;
        std::get<std::ifstream>(data) >> output;
        return (output);
    }
    else if(type == CONSTANTINSERTION)
        return (std::get<Vector3<T>>(data));
    else
        return (Vector3<T>());
}

// -----------------------------------------------------------------------------
// Returns all required data members to insert components as a vector
template <typename T>
__HOST__ std::tuple<Vector3<T>, Quaternion<T>, Kinematics<T>, bool>
         Insertion<T>::fetchInsertionData()
{
    // Position
    Vector3<T> pos
        = fetchInsertionDataForEach(m_positionType, m_positionInsertionInfo);

    // Orientation angles. These are not matrices, so we have to compute the
    // rotation matrices.
    Vector3<T>    ori = fetchInsertionDataForEach(m_orientationType,
                                               m_orientationInsertionInfo);
    Quaternion<T> quat(ori[X], ori[Y], ori[Z]);

    // Kinematics
    Vector3<T>    vel = fetchInsertionDataForEach(m_translationalVelType,
                                               m_translationalVelInsertionInfo);
    Vector3<T>    ang = fetchInsertionDataForEach(m_angularVelType,
                                               m_angularVelInsertionInfo);
    Kinematics<T> k(vel, ang);

    return (std::make_tuple(pos, quat, k, m_forceInsertion));
}

// -----------------------------------------------------------------------------
// Explicit instantiation
template class Insertion<float>;
template class Insertion<double>;