#include <string>
#include <utility>

#include <glog/logging.h>
#include <sm/PropertyTree.hpp>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/predicates.h>
#include <aslam/common/unique-id.h>

namespace aslam {

NCamera::NCamera(const NCameraId& id, const TransformationVector& T_C_B,
                 const std::vector<Camera::Ptr>& cameras, const std::string& label)
    : id_(id),
      T_C_B_(T_C_B),
      cameras_(cameras),
      label_(label) {
  CHECK(id.isValid());
  initInternal();
}

NCamera::NCamera(const sm::PropertyTree& /* propertyTree */) {
  // TODO(PTF): fill in
  CHECK(false) << "Not implemented";
}

void NCamera::initInternal() {
  CHECK_EQ(cameras_.size(), T_C_B_.size());
  id_to_index_.clear();
  for (size_t i = 0; i < cameras_.size(); ++i) {
    CHECK_NOTNULL(cameras_[i].get());
    CHECK(cameras_[i]->getId().isValid());
    id_to_index_[cameras_[i]->getId()] = i;
  }
}

size_t NCamera::getNumCameras() const {
  return cameras_.size();
}

const Transformation& NCamera::get_T_C_B(size_t camera_index) const {
  CHECK_LT(camera_index, cameras_.size());
  return T_C_B_[camera_index];
}

Transformation& NCamera::get_T_C_B_Mutable(size_t camera_index) {
  CHECK_LT(camera_index, cameras_.size());
  return T_C_B_[camera_index];
}

const Transformation& NCamera::get_T_C_B(const CameraId& camera_id) const {
  CHECK(camera_id.isValid());
  int camera_idx = getCameraIndex(camera_id);
  CHECK_GE(camera_idx, 0);
  return get_T_C_B(camera_idx);
}

Transformation& NCamera::get_T_C_B_Mutable(const CameraId& camera_id) {
  CHECK(camera_id.isValid());
  int camera_idx = getCameraIndex(camera_id);
  CHECK_GE(camera_idx, 0);
  return get_T_C_B_Mutable(camera_idx);
}

void NCamera::set_T_C_B(size_t camera_index, const Transformation& T_Ci_B) {
  CHECK_LT(camera_index, T_C_B_.size());
  T_C_B_[camera_index] = T_Ci_B;
}

const NCamera::TransformationVector& NCamera::getTransformationVector() const {
  return T_C_B_;
}

const Camera& NCamera::getCamera(size_t camera_index) const {
  CHECK_LT(camera_index, cameras_.size());
  CHECK_NOTNULL(cameras_[camera_index].get());
  return *cameras_[camera_index];
}

Camera& NCamera::getCameraMutable(size_t camera_index) {
  CHECK_LT(camera_index, cameras_.size());
  CHECK_NOTNULL(cameras_[camera_index].get());
  return *cameras_[camera_index];
}

Camera::Ptr NCamera::getCameraShared(size_t camera_index) {
  CHECK_LT(camera_index, cameras_.size());
  return cameras_[camera_index];
}

Camera::ConstPtr NCamera::getCameraShared(size_t camera_index) const {
  CHECK_LT(camera_index, cameras_.size());
  return cameras_[camera_index];
}

void NCamera::setCamera(size_t camera_index, Camera::Ptr camera) {
  CHECK_LT(camera_index, cameras_.size());
  cameras_[camera_index] = camera;
}

size_t NCamera::numCameras() const {
  return cameras_.size();
}

const std::vector<Camera::Ptr>& NCamera::getCameraVector() const {
  return cameras_;
}

const CameraId& NCamera::getCameraId(size_t camera_index) const {
  CHECK_LT(camera_index, cameras_.size());
  return cameras_[camera_index]->getId();
}

bool NCamera::hasCameraWithId(const CameraId& id) const {
  CHECK(id.isValid());
  return id_to_index_.find(id) != id_to_index_.end();
}

int NCamera::getCameraIndex(const CameraId& id) const {
  CHECK(id.isValid());
  std::unordered_map<CameraId, size_t>::const_iterator it = id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    return -1;
  } else {
    return it->second;
  }
}

NCamera::Ptr NCamera::createTestNCamera(size_t num_cameras) {
  std::vector<aslam::Camera::Ptr> cameras;
  aslam::Aligned<std::vector, aslam::Transformation>::type T_C_B_vector;

  for(size_t camera_idx = 0; camera_idx < num_cameras; ++num_cameras) {
    cameras.push_back(aslam::PinholeCamera::createTestCamera());

    // Offset each camera 0.1 m in x direction.
    aslam::Transformation T_C_B;
    T_C_B.getPosition()(0) = 0.1 * num_cameras;
    T_C_B_vector.push_back(T_C_B);
  }

  aslam::NCameraId rig_id;
  rig_id.randomize();
  std::string label("Test camera rig");
  return aslam::NCamera::Ptr(new aslam::NCamera(rig_id, T_C_B_vector, cameras, label));
}

bool NCamera::operator==(const NCamera& other) const {
  bool same = true;
  same &= getNumCameras() == other.getNumCameras();
  same &= label_ == other.label_;
  same &= id_ == other.id_;
  if (same) {
    for (size_t i = 0; i < getNumCameras(); ++i) {
      same &= aslam::checkSharedEqual(cameras_[i], other.cameras_[i]);
      same &= T_C_B_[i] == other.T_C_B_[i];
    }
  }
  return same;
}

}  // namespace aslam
