#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/cameras/yaml/camera-yaml-serialization.h>
#include <aslam/cameras/yaml/ncamera-yaml-serialization.h>
#include <aslam/common/memory.h>
#include <aslam/common/yaml-serialization.h>

namespace YAML {

bool convert<std::shared_ptr<aslam::NCamera> >::decode(const Node& node,
                                                       aslam::NCamera::Ptr& ncamera) {
  ncamera.reset();
  try {
    if (!node.IsMap()) {
      LOG(ERROR) << "Unable to parse the ncamera because the node is not a map.";
      return true;
    }

    // Parse the label.
    std::string label = "";
    if (!YAML::safeGet<std::string>(node, "label", &label)) {
      LOG(ERROR) << "Unable to get the label for the ncamera.";
      return true;
    }

    // Parse the id.
    aslam::NCameraId ncam_id;
    std::string id_string;
    if (!YAML::safeGet<std::string>(node, "id", &id_string)) {
      LOG(WARNING) << "Unable to get the id for the ncamera. Generating new random id.";
      ncam_id.randomize();
    } else {
      ncam_id.fromHexString(id_string);
    }

    // Parse the cameras.
    const Node& cameras_node = node["cameras"];
    if (!cameras_node.IsSequence()) {
      LOG(ERROR) << "Unable to parse the cameras because the camera node is not a sequence.";
      return true;
    }

    size_t num_cameras = cameras_node.size();
    if (num_cameras == 0) {
      LOG(ERROR) << "Number of cameras is 0.";
      return true;
    }

    aslam::TransformationVector T_C_Bs;
    std::vector<aslam::Camera::Ptr> cameras;
    for (size_t camera_index = 0; camera_index < num_cameras; ++camera_index) {
      // Decode the camera
      const Node& camera_node = cameras_node[camera_index];
      if (!camera_node) {
        LOG(ERROR) << "Unable to get camera node for camera " << camera_index;
        return true;
      }
      if (!camera_node.IsMap()) {
        LOG(ERROR) << "Camera node for camera " << camera_index << " is not a map.";
        return true;
      }

      aslam::Camera::Ptr camera;
      if (!YAML::safeGet(camera_node, "camera", &camera)) {
        LOG(ERROR) << "Unable to retrieve camera " << camera_index;
        return true;
      }

      const Node& extrinsics_node = camera_node["extrinsics"];
      if (!extrinsics_node) {
        LOG(ERROR) << "No extrinsics node for camera " << camera_index;
        return true;
      }
      if (!extrinsics_node.IsMap()) {
        LOG(ERROR) << "Extrinsics node for camera " << camera_index << " is not a map.";
        return true;
      }

      Eigen::Vector3d t_B_C;
      if (!YAML::safeGet(extrinsics_node, "t_B_C", &t_B_C)) {
        LOG(ERROR) << "Unable to get extrinsic position t_B_C for camera " << camera_index;
        return true;
      }

      Eigen::Vector4d q_B_C_raw;
      if (!YAML::safeGet(extrinsics_node, "q_B_C", &q_B_C_raw)) {
        LOG(ERROR) << "Unable to get extrinsic rotation q_B_C for camera " << camera_index;
        return true;
      }
      aslam::Quaternion q_B_C(q_B_C_raw(0), q_B_C_raw(1), q_B_C_raw(2), q_B_C_raw(3));

      aslam::Transformation T_B_C(q_B_C, t_B_C);

      // Fill in the data in the ncamera.
      cameras.push_back(camera);
      T_C_Bs.push_back(T_B_C.inverted());
    }

    // Create the ncamera and fill in all the data.

    ncamera.reset(new aslam::NCamera(ncam_id, T_C_Bs, cameras, label));
  } catch(const std::exception& e) {
    ncamera = nullptr;
    LOG(ERROR) << "Yaml exception during parsing: " << e.what();
    ncamera.reset();
    return true;
  }
  return true;
}

Node convert<std::shared_ptr<aslam::NCamera> >::encode(
    const std::shared_ptr<aslam::NCamera>& ncamera) {
  CHECK_NOTNULL(ncamera.get());
  Node ncamera_node;

  ncamera_node["label"] = ncamera->getLabel();
  if(ncamera->getId().isValid()) {
    ncamera_node["id"] = ncamera->getId().hexString();
  }

  Node cameras_node;
  size_t num_cameras = ncamera->numCameras();
  for (size_t camera_index = 0; camera_index < num_cameras; ++camera_index) {
    Node camera_node;

    camera_node["camera"]  = ncamera->getCameraShared(camera_index);

    Eigen::Vector3d t_B_C = ncamera->get_T_C_B(camera_index).inverted().getPosition();
    Eigen::Vector4d q_B_C = ncamera->get_T_C_B(camera_index).inverted().getRotation().vector();

    Node extrinsics;
    extrinsics["t_B_C"] = t_B_C;
    extrinsics["q_B_C"] = q_B_C;

    camera_node["extrinsics"] = extrinsics;

    cameras_node.push_back(camera_node);
  }

  ncamera_node["cameras"] = cameras_node;

  return ncamera_node;
}

}  // namespace YAML
