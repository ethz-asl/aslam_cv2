#include <glog/logging.h>

#include <aslam/cameras/ncamera.h>
#include <aslam/common/predicates.h>
#include <sm/PropertyTree.hpp>

namespace aslam {

/// \brief default constructor builds an empty camera rig
NCamera::NCamera() { }

NCamera::NCamera(const NCameraId& id,
                 const TransformationVector& T_C_B,
                 const std::vector<Camera::Ptr>& cameras,
                 const std::string& label) :
        id_(id), T_C_B_(T_C_B),
        cameras_(cameras), label_(label) {
  initInternal(); 
}

/// \brief initialize from a property tree
NCamera::NCamera(const sm::PropertyTree& /* propertyTree */) {
  // \todo(PTF) fill in
  CHECK(false) << "Not implemented";
}

void NCamera::initInternal(){
  CHECK_EQ(cameras_.size(), T_C_B_.size());
  idToIndex_.clear();
  for(size_t i = 0; i < cameras_.size(); ++i) {
    CHECK_NOTNULL(cameras_[i].get());
    idToIndex_[cameras_[i]->getId()] = i;
  }
}

/// \brief get the number of cameras
size_t NCamera::getNumCameras() const {
  return cameras_.size();
}

/// \brief get the pose of body frame with respect to the camera i
const Transformation& NCamera::get_T_C_B(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  return T_C_B_[cameraIndex];
}

/// \brief get the pose of body frame with respect to the camera i
Transformation& NCamera::get_T_C_B_Mutable(size_t cameraIndex) {
  CHECK_LT(cameraIndex, cameras_.size());
  return T_C_B_[cameraIndex];
}

/// \brief set the pose of body frame with respect to the camera i
void NCamera::set_T_C_B(size_t cameraIndex, const Transformation& T_Ci_B) {
  CHECK_LT(cameraIndex, T_C_B_.size());
  T_C_B_[cameraIndex] = T_Ci_B;
}

/// \brief get all transformations
const NCamera::TransformationVector& NCamera::getTransformationVector() const {
  return T_C_B_;
}

/// \brief get the geometry object for camera i
const Camera& NCamera::getCamera(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  CHECK_NOTNULL(cameras_[cameraIndex].get());
  return *cameras_[cameraIndex];
}

/// \brief get the geometry object for camera i
Camera& NCamera::getCameraMutable(size_t cameraIndex) {
  CHECK_LT(cameraIndex, cameras_.size());
  CHECK_NOTNULL(cameras_[cameraIndex].get());
  return *cameras_[cameraIndex];
}

/// \brief get the geometry object for camera i
Camera::Ptr NCamera::getCameraShared(size_t cameraIndex) {
  CHECK_LT(cameraIndex, cameras_.size());
  return cameras_[cameraIndex];
}

/// \brief get the geometry object for camera i
Camera::ConstPtr NCamera::getCameraShared(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  return cameras_[cameraIndex];
}

/// \brief get the geometry object for camera i
void NCamera::setCamera(size_t cameraIndex, Camera::Ptr camera) {
  CHECK_LT(cameraIndex, cameras_.size());
  cameras_[cameraIndex] = camera;
}

/// \brief how many cameras does this system have?
size_t NCamera::numCameras() const {
  return cameras_.size();
}

/// \brief get all cameras
const std::vector<Camera::Ptr>& NCamera::getCameraVector() const {
  return cameras_;
}

/// \brief gets the id for the camera at index i
const CameraId& NCamera::getCameraId(size_t cameraIndex) const {
  CHECK_LT(cameraIndex, cameras_.size());
  return cameras_[cameraIndex]->getId();
}

/// \brief does this rig have a camera with this id
bool NCamera::hasCameraWithId(const CameraId& id) const {
  return idToIndex_.find(id) != idToIndex_.end();
}

/// \brief get the index of the camera with the id
/// @returns -1 if the rig doesn't have a camera with this id
size_t NCamera::getCameraIndex(const CameraId& id) const {
  std::unordered_map<CameraId, size_t>::const_iterator it = idToIndex_.find(id);
  if(it == idToIndex_.end()) {
    return -1;
  } else {
    return it->second;
  }
}

bool NCamera::operator==(const NCamera& other) const {
  bool same = true;
  same &= getNumCameras() == other.getNumCameras();
  same &= label_ == other.label_;
  same &= id_ == other.id_;
  if(same) {
    for(size_t i = 0; i < getNumCameras(); ++i) {
      same &= aslam::checkSharedEqual(cameras_[i], other.cameras_[i]);
      same &= T_C_B_[i] == other.T_C_B_[i];
    }
  }
  return same;
}

} // namespace aslam
