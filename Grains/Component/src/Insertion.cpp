#include <cstdlib>
#include <ctime>

#include "GJK.hh"
#include "GrainsMemBuffer.hh"
#include "GrainsUtils.hh"
#include "Insertion.hh"
#include "LinkedCell_Host.hh"
#include "OBB.hh"
#include "QuaternionMath.hh"

/* ============================================================================================== */
/* Low-Level Methods                                                                              */
/* ============================================================================================== */
// Reads if the root is of type Random
template <typename T>
__HOST__ static INLINE InsertionInfo<T> readDataRand(DOMNode* root)
{
    // Random generator seed. We pass it to InsertionWindow directly.
    // We also set the seed with srand. We use it for to randomly pick an
    // insertion window
    RandomGeneratorSeed rgs;
    unsigned            cSeed      = 1;  // default C RNG seed for window selection
    std::string         seedString = ReaderXML::getNodeAttr_String(root, "Seed");
    if(seedString == "UserDefined")
    {
        uint val = ReaderXML::getNodeAttr_Int(root, "Value");
        GAssert(val, "Seed value is not provided. Aborting Grains!");
        rgs   = RGS_UDEF;
        cSeed = static_cast<unsigned>(val);
        GoutWI(12, "Random initialization with seed", std::to_string(val) + ".");
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
    DOMNode*                        nWindows = ReaderXML::getNode(root, "Windows");
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

// -------------------------------------------------------------------------------------------------
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

// -------------------------------------------------------------------------------------------------
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

// -------------------------------------------------------------------------------------------------
// Reads if the root is of type Zero
template <typename T>
__HOST__ static INLINE InsertionInfo<T> readDataZero(DOMNode* root)
{
    Vector3<T> vec(T(0), T(0), T(0));
    GoutWI(12, "Zero initialization.");

    return (vec);
}

/* ============================================================================================== */
/* High-Level Methods                                                                             */
/* ============================================================================================== */
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

// -------------------------------------------------------------------------------------------------
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
        m_forceInsertion = static_cast<bool>(ReaderXML::getNodeAttr_Int(dn, "ForceInsertion"));
    else
        m_forceInsertion = false;
}

// -------------------------------------------------------------------------------------------------
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

// -------------------------------------------------------------------------------------------------
// Returns a vector of Vector3 accroding to type and data
template <typename T>
__HOST__ Vector3<T> Insertion<T>::fetchInsertionData(InsertionType const type,
                                                     InsertionInfo<T>&   data)
{
    // We only return a vector3. It is clear how it works for position, and
    // kinematics. However, for orientation, it returns the vector3 of rotation
    // angles. We later construct a quaternion.
    if(type == RANDOMINSERTION)
    {
        auto& IWs = std::get<std::vector<InsertionWindow<T>>>(data);
        GAssert(!IWs.empty(), "Random insertion selected but no InsertionWindow defined!");
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

// -------------------------------------------------------------------------------------------------
// Populates position, orientation, and kinematics according to the insertion
// policy
template <typename T>
__HOST__ void Insertion<T>::insert(const GrainsMemBuffer<RigidBody<T>*>* rigidBody,
                                   GrainsMemBuffer<Vector3<T>>&          position,
                                   GrainsMemBuffer<Quaternion<T>>&       orientation,
                                   GrainsMemBuffer<Kinematics<T>>&       kinematics,
                                   const uint                            numObstacles,
                                   const uint                            numParticles)
{
    using GP = GrainsParameters<T>;
    GoutWI(3, "Inserting", std::to_string(numParticles), "particles ...");

    if(m_forceInsertion)
    {
        for(uint i = 0; i < numParticles; ++i)
        {
            const uint insertID = i + numObstacles;
            position[insertID]  = fetchInsertionData(m_positionType, m_positionInsertionInfo);

            // Orientation angles. These are not matrices, so we have to
            // compute the quaternions later.
            Vector3<T> ori = fetchInsertionData(m_orientationType, m_orientationInsertionInfo);
            orientation[insertID] = Quaternion<T>(ori[X], ori[Y], ori[Z]) * orientation[insertID];

            Vector3<T> vel
                = fetchInsertionData(m_translationalVelType, m_translationalVelInsertionInfo);

            Vector3<T> ang       = fetchInsertionData(m_angularVelType, m_angularVelInsertionInfo);
            kinematics[insertID] = Kinematics<T>(vel, ang);
        }
    }
    else
    {
        // Max attempts to place a particle
        const uint maxAttempts = 1000;

        // Build a temporary linked-cell structure for strict insertion checks
        auto LCParameters = GP::m_collisionDetection.linkedCellParameters;
        LCParameters.type = LinkedCellType::HOST;  // Ensure HOST type
        LinkedCell_Host<T> LC(rigidBody,
                              position,
                              orientation,
                              LCParameters,
                              numObstacles,
                              numParticles);

        // Overlap test lambda function
        auto canInsert = [&](const uint           insertID,
                             const Vector3<T>&    insertPosition,
                             const Quaternion<T>& insertQuaternion) {
            const Convex<T>& convexNew = *(*rigidBody)[insertID]->getConvex();

            // Check against already-inserted components via LC
            std::vector<uint> neighborList;
            LC.collectPotentialNeighbors(insertPosition, insertID, neighborList);
            for(uint j : neighborList)
            {
                const Convex<T>& convexJ = *(*rigidBody)[j]->getConvex();
                bool BVintersect = intersectOrientedBoundingBox(convexJ.computeBoundingBox(),
                                                                convexNew.computeBoundingBox(),
                                                                position[j],
                                                                insertPosition,
                                                                orientation[j],
                                                                insertQuaternion);
                // if(BVintersect
                //    && intersectGJK<T>(convexJ,
                //                       convexNew,
                //                       position[j],
                //                       insertPosition,
                //                       orientation[j],
                //                       insertQuaternion))
                if(BVintersect)
                    return false;
            }
            return true;
        };

        // Inserting particles
        for(uint i = 0; i < numParticles; ++i)
        {
            const uint insertID = i + numObstacles;
            bool       placed   = false;
            for(uint attempt = 0; attempt < maxAttempts && !placed; ++attempt)
            {
                const Vector3<T>& pCand
                    = fetchInsertionData(m_positionType, m_positionInsertionInfo);
                // Orientation angles. These are not matrices, so we have to
                // compute the quaternions later.
                Vector3<T> ori = fetchInsertionData(m_orientationType, m_orientationInsertionInfo);
                Quaternion<T>        quat(ori[X], ori[Y], ori[Z]);
                const Quaternion<T>& qCand = quat * orientation[insertID];
                // Check if candidate position is within domain bounds
                // clang-format off
                bool withinBounds = (pCand[0] >= GP::m_origin[0] && 
                                     pCand[0] <= GP::m_maxCoordinate[0] && 
                                     pCand[1] >= GP::m_origin[1] && 
                                     pCand[1] <= GP::m_maxCoordinate[1] &&
                                     pCand[2] >= GP::m_origin[2] &&
                                     pCand[2] <= GP::m_maxCoordinate[2]);
                // clang-format on
                if(withinBounds && canInsert(insertID, pCand, qCand))
                {
                    position[insertID]    = pCand;
                    orientation[insertID] = qCand;
                    Vector3<T> vel        = fetchInsertionData(m_translationalVelType,
                                                        m_translationalVelInsertionInfo);
                    Vector3<T> ang
                        = fetchInsertionData(m_angularVelType, m_angularVelInsertionInfo);
                    kinematics[insertID] = Kinematics<T>(vel, ang);
                    placed               = true;
                    // Add new particle to linked cells for the next insert
                    const Cells<T>* cells  = LC.getLinkedCell()[0];
                    const uint      cellID = cells->computeCellHash(pCand);
                    LC.addParticleToCell(insertID, cellID);
                }
            }

            GAssert(placed,
                    "Failed to place a particle without overlap after too many "
                    "attempts.");
        }
    }
    GoutWI(3, "Inserted", std::to_string(numParticles), "particles.");
}

// -------------------------------------------------------------------------------------------------
// Explicit instantiation
template class Insertion<float>;
template class Insertion<double>;