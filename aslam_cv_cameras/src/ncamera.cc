#include <string>
#include <utility>

#include <glog/logging.h>

#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/cameras/random-camera-generator.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/predicates.h>
#include <aslam/common/unique-id.h>
#include <aslam/common/yaml-serialization.h>

namespace aslam {

NCamera::NCamera() {}

NCamera::NCamera(
    const NCameraId& id, const TransformationVector& T_C_B,
    const std::vector<Camera::Ptr>& cameras, const std::string& description)
    : Sensor(id), T_C_B_(T_C_B), cameras_(cameras) {
  CHECK(id.isValid());
  description_ = description;
  initInternal();
}

NCamera::NCamera(const sm::PropertyTree& /* propertyTree */) {
  // TODO(PTF): fill in
  CHECK(false) << "Not implemented";
}

NCamera::NCamera(const NCamera& other) :
    Sensor(other),
    T_C_B_(other.T_C_B_) {
  // Clone all contained cameras.
  for (size_t idx = 0u; idx < other.getNumCameras(); ++idx) {
    cameras_.emplace_back(other.getCamera(idx).clone());
  }
  initInternal();
}

bool NCamera::loadFromYamlNodeImpl(const YAML::Node& yaml_node) {
  CHECK(yaml_node.IsMap());

  // Parse the cameras.
  const YAML::Node& cameras_node = yaml_node["cameras"];
  if (!cameras_node.IsDefined() || cameras_node.IsNull()) {
    LOG(ERROR)
        << "Invalid cameras YAML node in NCamera. Can not parse NCamera with"
        << " no cameras defined.";
    return false;
  }

  if (!cameras_node.IsSequence()) {
    LOG(ERROR)
        << "Unable to parse the cameras because the camera node is not a sequence.";
    return false;
  }

  size_t num_cameras = cameras_node.size();
  if (num_cameras == 0) {
    LOG(ERROR) << "Number of cameras is 0.";
    return false;
  }

  for (size_t camera_index = 0; camera_index < num_cameras; ++camera_index) {
    // Decode the camera
    const YAML::Node& camera_node = cameras_node[camera_index];
    if (!camera_node) {
      LOG(ERROR) << "Unable to get camera node for camera " << camera_index;
      return false;
    }

    if (!camera_node.IsMap()) {
      LOG(ERROR)
          << "Camera node for camera " << camera_index << " is not a map.";
      return false;
    }

    // Retrieve the type of the camera
    const YAML::Node& intrinsics_node = camera_node["camera"];
    Camera::Ptr camera = createCamera(intrinsics_node);
    if (!camera) {
      LOG(ERROR) << "Failed to deserialize camera " << camera_index;
      return false;
    }

    // Get the transformation matrix T_B_C (takes points from the frame C to frame B).
    Eigen::Matrix4d T_B_C_raw;
    if (!YAML::safeGet(camera_node, "T_B_C", &T_B_C_raw)) {
      LOG(ERROR)
          << "Unable to get extrinsic transformation T_B_C for camera "
          << camera_index;
      return false;
    }
    // This call will fail hard if the matrix is not a rotation matrix.
    aslam::Quaternion q_B_C = aslam::Quaternion(
        static_cast<Eigen::Matrix3d>(T_B_C_raw.block<3,3>(0,0)));
    aslam::Transformation T_B_C(q_B_C, T_B_C_raw.block<3,1>(0,3));

    // Fill in the data in the ncamera.
    cameras_.push_back(camera);
    T_C_B_.push_back(T_B_C.inverse());
  }

  initInternal();

  return true;
}

void NCamera::saveToYamlNodeImpl(YAML::Node* yaml_node) const {
  CHECK_NOTNULL(yaml_node);
  YAML::Node& node = *yaml_node;

  YAML::Node cameras_node;
  size_t num_cameras = numCameras();
  for (size_t camera_index = 0; camera_index < num_cameras; ++camera_index) {
    YAML::Node intrinsics_node;
    getCamera(camera_index).serialize(&intrinsics_node);

    YAML::Node camera_node;
    camera_node["camera"] = intrinsics_node;
    camera_node["T_B_C"] =
        get_T_C_B(camera_index).inverse().getTransformationMatrix();
    cameras_node.push_back(camera_node);
  }

  node["cameras"] = cameras_node;
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

aslam::NCamera::Ptr NCamera::cloneRigWithoutDistortion() const {
  aslam::NCamera::Ptr rig_without_distortion(this->clone());
  // Remove distortion and assign new IDs to the rig and all cameras.
  for (Camera::Ptr& camera : rig_without_distortion->cameras_) {
    camera->removeDistortion();
    aslam::CameraId cam_id;
    generateId(&cam_id);
    camera->setId(cam_id);
  }

  aslam::NCameraId ncam_id;
  generateId(&ncam_id);
  rig_without_distortion->setId(ncam_id);
  return rig_without_distortion;
}

// TODO(smauq): Fix this with respect to base class
std::string NCamera::getComparisonString(const NCamera& other) const {
  if (operator==(other)) {
    return "There is no difference between the given ncameras.\n";
  }

  std::ostringstream ss;

  if (id_ != other.id_) {
    ss << "The id is " << id_ << ", the other id is " << other.id_ << std::endl;
  }

  if (description_ != other.description_) {
    ss << "The description is " << description_ << ", the other description is "
        << other.description_ << std::endl;
  }

  if (getNumCameras() != other.getNumCameras()) {
    ss << "The number of cameras is " << getNumCameras()
        << ", the other number of cameras is " << other.getNumCameras()
        << std::endl;
  } else {
    for (size_t i = 0; i < getNumCameras(); ++i) {
      double max_coeff_diff = (T_C_B_[i].getTransformationMatrix() -
            other.T_C_B_[i].getTransformationMatrix()).cwiseAbs().maxCoeff();
      if (max_coeff_diff >= common::macros::kEpsilon) {
        ss << "The maximum coefficient of camera transformation " << i
            << " differs by " << max_coeff_diff << std::endl;
        ss << "The transformation matrices are:\n" << T_C_B_[i] << "\nand\n"
            << other.T_C_B_[i] << std::endl;
      }
      if (!aslam::checkSharedEqual(cameras_[i], other.cameras_[i])) {
        ss << "Camera " << i << " differs" << std::endl;
      }
    }
  }

  if (id_to_index_ != other.id_to_index_) {
    ss << "The id to index map differs!" << std::endl;
  }

  return ss.str();
}

bool NCamera::isValidImpl() const {
  for (const aslam::Camera::Ptr& camera : cameras_) {
    CHECK(camera);
    if (!camera->isValid()) {
      return false;
    }
  }
  return true;
}

void NCamera::setRandomImpl() {
  id_to_index_.clear();
  T_C_B_.clear();
  cameras_.clear();

  // TODO(all): use inified random number generator with fixable seed.
  Eigen::Vector2f random_numbers;
  random_numbers.setRandom();
  (random_numbers + Eigen::Vector2f::Ones()) *
      8.f;  // random_numbers elements are in [1,16]
  TransformationVector T_C_B_vector;

  std::vector<aslam::Camera::Ptr> cameras;
  const size_t num_cameras = random_numbers(0);  // in [1,16]
  CHECK_LE(num_cameras, 16u);
  CHECK_GE(num_cameras, 1u);
  for (size_t camera_idx = 0u; camera_idx < num_cameras; ++camera_idx) {
    aslam::Camera::Ptr random_camera;
    if (random_numbers(1) > 8) {
      random_camera.reset(new aslam::PinholeCamera());
      random_camera->setRandom();
      cameras.push_back(random_camera);
    } else {
      random_camera.reset(new aslam::UnifiedProjectionCamera());
      random_camera->setRandom();
      cameras.push_back(random_camera);
    }

    // Offset each camera 0.1 m in x direction and rotate it to face forward.
    Eigen::Vector3d position(0.1 * (camera_idx + 1), 0.0, 0.0);
    aslam::Quaternion q_C_B(0.5, 0.5, -0.5, 0.5);
    aslam::Transformation T_C_B(q_C_B, position);
    T_C_B_vector.push_back(T_C_B);
    id_to_index_[random_camera->getId()] = camera_idx;
  }
  T_C_B_ = T_C_B_vector;
  cameras_ = cameras;
}

bool NCamera::isEqualImpl(const Sensor& other) const {
  const NCamera* other_ncamera = dynamic_cast<const NCamera*>(&other);
  if (other_ncamera == nullptr) {
    return false;
  }

  if (getNumCameras() != other_ncamera->getNumCameras()) {
    return false;
  }
  bool is_equal = true;
  for (size_t i = 0; i < getNumCameras(); ++i) {
    is_equal &= aslam::checkSharedEqual(cameras_[i], other_ncamera->cameras_[i]);
    is_equal &= ((T_C_B_[i].getTransformationMatrix() - other_ncamera->T_C_B_[i].getTransformationMatrix())
        .cwiseAbs().maxCoeff() < common::macros::kEpsilon);
  }
  return is_equal;
}

}  // namespace aslam
