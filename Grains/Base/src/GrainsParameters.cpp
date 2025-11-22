#include "GrainsParameters.hh"

// -----------------------------------------------------------------------------
// Static variables
/* Spatial */
template <typename T>
Vector3<T> GrainsParameters<T>::m_origin = zeroVector3T;
template <typename T>
Vector3<T> GrainsParameters<T>::m_maxCoordinate = zeroVector3T;
template <typename T>
bool GrainsParameters<T>::m_isPeriodic = false;

/* Temporal */
template <typename T>
T GrainsParameters<T>::m_tStart;
template <typename T>
T GrainsParameters<T>::m_tEnd;
template <typename T>
T GrainsParameters<T>::m_dt;
template <typename T>
T GrainsParameters<T>::m_time;

/* Numbers */
template <typename T>
uint GrainsParameters<T>::m_numParticles = 0;
template <typename T>
uint GrainsParameters<T>::m_numObstacles = 0;

/* Physical */
template <typename T>
Vector3<T> GrainsParameters<T>::m_gravity = zeroVector3T;

/* Material */
template <typename T>
std::unordered_map<std::string, uint> GrainsParameters<T>::m_materialMap;
template <typename T>
uint GrainsParameters<T>::m_numContactPairs = 0;

/* Post-Processing */
template <typename T>
std::queue<T> GrainsParameters<T>::m_tSave;

/* GPU */
template <typename T>
bool GrainsParameters<T>::m_isGPU = false;
template <typename T>
cudaDeviceProp GrainsParameters<T>::m_GPU = {};

/* Collision Detection */
template <typename T>
typename GrainsParameters<T>::CollisionDetectionParameters
    GrainsParameters<T>::m_collisionDetection;

// -----------------------------------------------------------------------------
// Explicit instantiation
template class GrainsParameters<float>;
template class GrainsParameters<double>;