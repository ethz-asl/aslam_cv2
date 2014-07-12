#include <aslam/cameras/camera-rig.h>
#include <glog/logging.h>
#include <sm/PropertyTree.hpp>

namespace aslam {

/// \brief default constructor builds an empty camera rig
CameraRig::CameraRig() { }

CameraRig::CameraRig(const CameraRigId& id,
                     const TransformationVector& T_C_B, 
                     const CameraVector& cameras,
                     const std::string& label) :
    id_(id), T_C_B_(T_C_B), 
    cameras_(cameras), label_(label) { 
  initInternal(); 
}

/// \brief initialize from a property tree
CameraRig::CameraRig(const sm::PropertyTree& /* propertyTree */) {
  // \todo(PTF) fill in
}
  
CameraRig::~CameraRig() {}

void CameraRig::initInternal(){
  CHECK_EQ(cameras_.size(), T_C_B_.size());
  for(size_t i = 0; i < cameras_.size(); ++i) {
    CHECK_NOTNULL(cameras_[i].get());
    CHECK_NOTNULL(T_C_B_[i].get());
    idToIndex_[cameras_[i]->getId()] = i;
  }
}

/// \brief get the number of cameras
size_t CameraRig::getNumCameras() const {
  return cameras_.size();
}

/// \brief get the pose of body frame with respect to the camera i
const Transformation& CameraRig::get_T_C_B(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  return *T_C_B_[cameraIndex];
}

/// \brief get the pose of body frame with respect to the camera i
TransformationPtr CameraRig::get_T_C_B_mutable(size_t cameraIndex) {
  CHECK_LT(cameraIndex, cameras_.size());
  return T_C_B_[cameraIndex];
}

/// \brief set the pose of body frame with respect to the camera i
void CameraRig::set_T_C_B(size_t cameraIndex, const Transformation& T_Ci_B) {
  CHECK_LT(cameraIndex, cameras_.size());
  *T_C_B_[cameraIndex] = T_Ci_B;
}

/// \brief set the pose of body frame with respect to the camera i
void CameraRig::set_T_C_B(size_t cameraIndex, TransformationPtr T_Ci_B) {
  CHECK_LT(cameraIndex, cameras_.size());
  T_C_B_[cameraIndex] = T_Ci_B;
}

/// \brief get all transformations
const CameraRig::TransformationVector& CameraRig::getTransformationVector() {
  return T_C_B_;
}

/// \brief get the geometry object for camera i
const Camera& CameraRig::getCamera(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  return *cameras_[cameraIndex];
}

/// \brief get the geometry object for camera i
Camera::Ptr CameraRig::getCameraMutable(size_t cameraIndex) {
  CHECK_LT(cameraIndex, cameras_.size());
  return cameras_[cameraIndex];
}

/// \brief get the geometry object for camera i
void CameraRig::setCamera(size_t cameraIndex, Camera::Ptr camera) {
  CHECK_LT(cameraIndex, cameras_.size());
  cameras_[cameraIndex] = camera;
}
  
/// \brief how many cameras does this system have?
size_t CameraRig::numCameras() const {
  return cameras_.size();
}

/// \brief get all cameras
const CameraRig::CameraVector& CameraRig::getCameraVector() const {
  return cameras_;
}

/// \brief gets the id for the camera at index i
CameraId CameraRig::getCameraId(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  return cameras_[cameraIndex]->getId();
}
  
/// \brief does this rig have a camera with this id
bool CameraRig::hasCameraWithId(const CameraId& id) const {
  return idToIndex_.find(id) != idToIndex_.end();
}
  
/// \brief get the index of the camera with the id
/// @returns -1 if the rig doesn't have a camera with this id
size_t CameraRig::getCameraIndex(const CameraId& id) const {
  auto it = idToIndex_.find(id);
  if(it == idToIndex_.end()) {
    return -1;
  } else {
    return it->second;
  }
}

} // namespace aslam
