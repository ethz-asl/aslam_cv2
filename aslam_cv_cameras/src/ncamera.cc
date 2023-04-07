#include <aslam/common/pose-types.h>
#include <aslam/common/predicates.h>
#include <aslam/common/unique-id.h>
#include <aslam/common/yaml-serialization.h>
#include <chrono>
#include <glog/logging.h>
#include <random>
#include <string>
#include <utility>

#include "aslam/cameras/camera-factory.h"
#include "aslam/cameras/camera-pinhole.h"
#include "aslam/cameras/camera-unified-projection.h"
#include "aslam/cameras/camera.h"
#include "aslam/cameras/distortion-equidistant.h"
#include "aslam/cameras/distortion-fisheye.h"
#include "aslam/cameras/distortion-radtan.h"
#include "aslam/cameras/ncamera.h"

namespace aslam {

/// Methods to clone this instance. All contained camera objects are cloned.
NCamera* NCamera::clone() const {
  return new NCamera(static_cast<NCamera const&>(*this));
}

NCamera* NCamera::cloneWithNewIds() const {
  NCamera* new_ncamera = new NCamera(static_cast<NCamera const&>(*this));
  aslam::SensorId ncamera_id;
  generateId(&ncamera_id);
  new_ncamera->setId(ncamera_id);

  // Recurse into cameras and change the cloned camera ids
  for (size_t camera_index = 0u; camera_index < new_ncamera->numCameras();
       ++camera_index) {
    SensorId camera_id;
    generateId(&camera_id);
    aslam::Camera::Ptr camera = new_ncamera->getCameraShared(camera_index);
    camera->setId(camera_id);
  }

  new_ncamera->initInternal();
  CHECK(new_ncamera->isValid());
  return new_ncamera;
}

NCamera::NCamera() {}

NCamera::NCamera(
    const NCameraId& id, const std::string& description,
    const std::vector<Camera::Ptr>& cameras)
    : Sensor(id, std::string(), description), cameras_(cameras) {
  CHECK(id.isValid());
  initInternal();
}

NCamera::NCamera(const NCamera& other) : Sensor(other) {
  // Clone all contained cameras.
  for (size_t idx = 0u; idx < other.getNumCameras(); ++idx) {
    cameras_.emplace_back(other.getCamera(idx).clone());
  }
  initInternal();
  CHECK(isValid());
}

bool NCamera::loadFromYamlNodeImpl(const YAML::Node& yaml_node) {
  if (!yaml_node.IsMap()) {
    LOG(ERROR) << "Unable to parse the NCamera because the node is not a map.";
    return false;
  }

  // Parse the cameras.
  const YAML::Node& cameras_node = yaml_node["cameras"];
  if (!cameras_node.IsDefined() || cameras_node.IsNull()) {
    LOG(ERROR)
        << "Invalid cameras YAML node in NCamera. Can not parse NCamera with"
        << " no cameras defined.";
    return false;
  }

  if (!cameras_node.IsSequence()) {
    LOG(ERROR) << "Unable to parse the cameras because the camera node is not "
                  "a sequence.";
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
      LOG(ERROR) << "Camera node for camera " << camera_index
                 << " is not a map.";
      return false;
    }

    // Warn against old yaml format
    if (camera_node["T_B_C"] || camera_node["T_C_B"]) {
      LOG(ERROR) << "Old sensor format! Please use T_B_S or T_S_B instead of "
                 << "T_B_C and T_C_B, and add it as a property of the cameras "
                 << "(i.e. indent by one). Otherwise assumes unity transform.";
      return false;
    }

    // Retrieve the type of the camera
    const YAML::Node& intrinsics_node = camera_node["camera"];
    Camera::Ptr camera = createCamera(intrinsics_node);
    if (!camera) {
      LOG(ERROR) << "Failed to deserialize camera " << camera_index;
      return false;
    }

    cameras_.emplace_back(camera);
  }

  initInternal();

  return true;
}

void NCamera::saveToYamlNodeImpl(YAML::Node* yaml_node) const {
  CHECK_NOTNULL(yaml_node);
  YAML::Node& node = *yaml_node;

  YAML::Node cameras_node;

  size_t num_cameras = numCameras();
  for (size_t camera_index = 0u; camera_index < num_cameras; ++camera_index) {
    YAML::Node intrinsics_node;
    getCamera(camera_index).serialize(&intrinsics_node);

    YAML::Node camera_node;
    camera_node["camera"] = intrinsics_node;
    cameras_node.push_back(camera_node);
  }

  node["cameras"] = cameras_node;
}

void NCamera::initInternal() {
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

const Transformation& NCamera::get_T_B_C(size_t camera_index) const {
  CHECK_LT(camera_index, cameras_.size());
  return cameras_[camera_index]->get_T_B_S();
}

const Transformation& NCamera::get_T_B_C(const CameraId& camera_id) const {
  CHECK(camera_id.isValid());
  int camera_index = getCameraIndex(camera_id);
  CHECK_GE(camera_index, 0)
      << "Camera with ID " << camera_id << " not in NCamera container!";
  return cameras_[camera_index]->get_T_B_S();
}

void NCamera::set_T_B_C(size_t camera_index, const Transformation& T_B_Ci) {
  CHECK_LT(camera_index, cameras_.size());
  cameras_[camera_index]->set_T_B_S(T_B_Ci);
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
  std::unordered_map<CameraId, size_t>::const_iterator it =
      id_to_index_.find(id);
  if (it == id_to_index_.end()) {
    return -1;
  } else {
    return it->second;
  }
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

bool NCamera::isEqualImpl(const Sensor& other, const bool verbose) const {
  const NCamera* other_ncamera = dynamic_cast<const NCamera*>(&other);
  if (other_ncamera == nullptr) {
    if (verbose) {
      LOG(ERROR) << "Other ncamera is a nullptr!";
    }
    return false;
  }

  const size_t num_cameras = cameras_.size();
  if (num_cameras != other_ncamera->cameras_.size()) {
    if (verbose) {
      LOG(ERROR)
          << "The two NCameras have different number of cameras, ncamera A: "
          << num_cameras << " ncamera B: " << other_ncamera->cameras_.size();
    }
    return false;
  }

  bool is_equal = true;
  for (size_t i = 0u; i < num_cameras && is_equal; ++i) {
    const bool is_same_camera =
        aslam::checkSharedEqual(cameras_[i], other_ncamera->cameras_[i]);
    is_equal &= is_same_camera;

    if (verbose && !is_same_camera) {
      LOG(ERROR) << "Camera at idx " << i << " with id " << cameras_[i]->getId()
                 << " is not the same.";
    }
  }

  return is_equal;
}

NCamera::Ptr NCamera::createTestNCamera(
    size_t num_cameras, const SensorId& base_sensor_id) {
  return aligned_shared<aslam::NCamera>(
      *createUniqueTestNCamera(num_cameras, base_sensor_id));
}

NCamera::UniquePtr NCamera::createUniqueTestNCamera(
    size_t num_cameras, const SensorId& base_sensor_id) {
  CHECK(base_sensor_id.isValid());
  CHECK_GE(num_cameras, 1u);

  const unsigned kSeed =
      std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine random_number_engine(kSeed);

  std::vector<Camera::Ptr> cameras;
  for (size_t camera_index = 0u; camera_index < num_cameras; ++camera_index) {
    Camera::Ptr camera;
    const unsigned random_type = random_number_engine() % 4;
    switch (random_type) {
      case 0:
        camera = PinholeCamera::createTestCamera<RadTanDistortion>();
        break;
      case 1:
        camera = PinholeCamera::createTestCamera<EquidistantDistortion>();
        break;
      case 2:
        camera = UnifiedProjectionCamera::createTestCamera<RadTanDistortion>();
        break;
      case 3:
        camera = UnifiedProjectionCamera::createTestCamera<FisheyeDistortion>();
        break;
      default:
        LOG(FATAL) << "Undefined random camera type.";
    }

    // Offset each camera 0.1 m in x direction and rotate it to face forward.
    Eigen::Vector3d position(0.1 * (camera_index + 1), 0.0, 0.0);
    aslam::Quaternion q_S_B(0.5, 0.5, -0.5, 0.5);
    aslam::Transformation T_S_B(q_S_B, position);
    camera->set_T_B_S(T_S_B.inverse(), base_sensor_id);

    cameras.emplace_back(camera);
  }

  NCameraId ncamera_id;
  generateId(&ncamera_id);
  NCamera::UniquePtr ncamera(
      new NCamera(ncamera_id, "Unit test ncamera", cameras));

  return std::move(ncamera);
}

}  // namespace aslam
