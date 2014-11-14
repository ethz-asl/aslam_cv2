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
  CHECK_GE(camera_idx, 0) << "Camera with ID " << camera_id
                          << " not in NCamera container!";
  return get_T_C_B(camera_idx);
}

Transformation& NCamera::get_T_C_B_Mutable(const CameraId& camera_id) {
  CHECK(camera_id.isValid());
  int camera_idx = getCameraIndex(camera_id);
  CHECK_GE(camera_idx, 0) << "Camera with ID " << camera_id 
                          << " not in NCamera! container";
  return get_T_C_B_Mutable(camera_idx);
}

void NCamera::set_T_C_B(size_t camera_index, const Transformation& T_Ci_B) {
  CHECK_LT(camera_index, T_C_B_.size());
  T_C_B_[camera_index] = T_Ci_B;
}

const TransformationVector& NCamera::getTransformationVector() const {
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
  CHECK(camera);
  CHECK_LT(camera_index, cameras_.size());
  id_to_index_.erase(cameras_[camera_index]->getId());
  cameras_[camera_index] = camera;
  id_to_index_[camera->getId()] = camera_index;
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

  for(size_t camera_idx = 0; camera_idx < num_cameras; ++camera_idx) {
    cameras.push_back(aslam::PinholeCamera::createTestCamera());

    // Offset each camera 0.1 m in x direction.
    aslam::Transformation T_C_B;
    T_C_B.getPosition()(0) = 0.1 * camera_idx;
    T_C_B_vector.push_back(T_C_B);
  }

  aslam::NCameraId rig_id;
  rig_id.randomize();
  std::string label("Test camera rig");
  return aslam::NCamera::Ptr(new aslam::NCamera(rig_id, T_C_B_vector, cameras, label));
}

NCamera::Ptr NCamera::createVChargeTestNCamera() {
  std::vector<aslam::Camera::Ptr> cameras;
  cameras.push_back(aslam::PinholeCamera::createTestCamera());
  cameras.push_back(aslam::PinholeCamera::createTestCamera());
  cameras.push_back(aslam::PinholeCamera::createTestCamera());
  cameras.push_back(aslam::PinholeCamera::createTestCamera());
  aslam::NCameraId rig_id;
  rig_id.randomize();
  // This defines an artificial camera system similar to the one on the V-Charge or JanETH car.
  aslam::Position3D t_B_C0(2.0, 0.0, 0.0);
  Eigen::Matrix3d R_B_C0 = Eigen::Matrix3d::Zero();
  R_B_C0(1, 0) = -1.0;
  R_B_C0(2, 1) = -1.0;
  R_B_C0(0, 2) = 1.0;
  aslam::Quaternion q_B_C0(R_B_C0);
  aslam::Position3D t_B_C1(0.0, 1.0, 0.0);
  Eigen::Matrix3d R_B_C1 = Eigen::Matrix3d::Zero();
  R_B_C1(0, 0) = 1.0;
  R_B_C1(2, 1) = -1.0;
  R_B_C1(1, 2) = 1.0;
  aslam::Quaternion q_B_C1(R_B_C1);
  aslam::Position3D t_B_C2(-1.0, 0.0, 0.0);
  Eigen::Matrix3d R_B_C2 = Eigen::Matrix3d::Zero();
  R_B_C2(1, 0) = 1.0;
  R_B_C2(2, 1) = -1.0;
  R_B_C2(0, 2) = -1.0;
  aslam::Quaternion q_B_C2(R_B_C2);
  aslam::Position3D t_B_C3(0.0, -1.0, 0.0);
  Eigen::Matrix3d R_B_C3 = Eigen::Matrix3d::Zero();
  R_B_C3(0, 0) = -1.0;
  R_B_C3(2, 1) = -1.0;
  R_B_C3(1, 2) = -1.0;
  aslam::Quaternion q_B_C3(R_B_C3);
  aslam::TransformationVector rig_transformations;
  rig_transformations.emplace_back(q_B_C0.inverted(), -t_B_C0);
  rig_transformations.emplace_back(q_B_C1.inverted(), -t_B_C1);
  rig_transformations.emplace_back(q_B_C2.inverted(), -t_B_C2);
  rig_transformations.emplace_back(q_B_C3.inverted(), -t_B_C3);
  std::string label = "Artificial Planar 4-Pinhole-Camera-Rig";
  return aslam::aligned_shared<aslam::NCamera>(rig_id, rig_transformations, cameras, label);
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
