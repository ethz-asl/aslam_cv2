#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/cameras/yaml/camera-yaml-serialization.h>
#include <aslam/cameras/distortion-equidistant.h>
#include <aslam/cameras/distortion-fisheye.h>
#include <aslam/cameras/distortion-null.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/common/yaml-serialization.h>

namespace YAML {

bool convert<std::shared_ptr<aslam::Camera> >::decode(const Node& node,
                                                      aslam::Camera::Ptr& camera) {
  camera.reset();
  try {
    if(!node.IsMap()) {
      LOG(ERROR) << "Unable to get parse the camera because the node is not a map.";
      return true;
    }
    // Determine the distortion type. Start with no distortion.
    aslam::Distortion::UniquePtr distortion;
    const YAML::Node distortion_config = node["distortion"];

    if(distortion_config) {
      std::string distortion_type;
      Eigen::VectorXd distortion_parameters;

      if(YAML::safeGet(distortion_config, "type", &distortion_type) &&
         YAML::safeGet(distortion_config, "parameters", &distortion_parameters)) {
        if(distortion_type == "none") {
            distortion.reset(new aslam::NullDistortion());
        } else if(distortion_type == "equidistant") {
          if (aslam::EquidistantDistortion::areParametersValid(distortion_parameters)) {
            distortion.reset(new aslam::EquidistantDistortion(distortion_parameters));
          } else {
            LOG(ERROR) << "Invalid distortion parameters for the Equidistant distortion model: "
                << distortion_parameters.transpose() << std::endl <<
                "See aslam::EquidistantDistortion::areParametersValid(...) for conditions on what "
                "valid Equidistant distortion parameters look like.";
            return true;
          }
        } else if(distortion_type == "fisheye") {
          if (aslam::FisheyeDistortion::areParametersValid(distortion_parameters)) {
            distortion.reset(new aslam::FisheyeDistortion(distortion_parameters));
          } else {
            LOG(ERROR) << "Invalid distortion parameters for the Fisheye distortion model: "
                << distortion_parameters.transpose() << std::endl <<
                "See aslam::FisheyeDistortion::areParametersValid(...) for conditions on what "
                "valid Fisheye distortion parameters look like.";
            return true;
          }
        } else if(distortion_type == "radial-tangential") {
          if (aslam::RadTanDistortion::areParametersValid(distortion_parameters)) {
            distortion.reset(new aslam::RadTanDistortion(distortion_parameters));
          } else {
            LOG(ERROR) << "Invalid distortion parameters for the RadTan distortion model: "
                << distortion_parameters.transpose() << std::endl <<
                "See aslam::RadTanDistortion::areParametersValid(...) for conditions on what "
                "valid RadTan distortion parameters look like.";
            return true;
          }
        } else {
            LOG(ERROR) << "Unknown distortion model: \"" << distortion_type << "\". "
                << "Valid values are {none, equidistant, fisheye, radial-tangential}.";
            return true;
        }
        if (!distortion->distortionParametersValid(distortion_parameters)) {
          LOG(ERROR) << "Invalid distortion parameters: " << distortion_parameters.transpose();
          return true;
        }
      } else {
        LOG(ERROR) << "Unable to get the required parameters from the distortion. "
            << "Required: string type, VectorXd parameters.";
        return true;
      }
    } else {
      distortion.reset(new aslam::NullDistortion());
    }

    std::string camera_type;
    unsigned image_width;
    unsigned image_height;
    Eigen::VectorXd intrinsics;
    if(YAML::safeGet(node, "type", &camera_type) &&
       YAML::safeGet(node, "image_width", &image_width) &&
       YAML::safeGet(node, "image_height", &image_height) &&
       YAML::safeGet(node, "intrinsics", &intrinsics)){
      if(camera_type == "pinhole") {
        if (aslam::PinholeCamera::areParametersValid(intrinsics)) {
          camera.reset(new aslam::PinholeCamera(intrinsics, image_width, image_height,
                                         distortion));
        } else {
          LOG(ERROR) << "Invalid intrinsics parameters for the Pinhole camera model: "
              << intrinsics.transpose() << std::endl <<
              "See aslam::PinholeCamera::areParametersValid(...) for conditions on what "
              "valid Pinhole camera intrinsics look like.";
          return true;
        }
      } else if(camera_type == "unified-projection") {
        if (aslam::UnifiedProjectionCamera::areParametersValid(intrinsics)) {
          camera.reset(new aslam::UnifiedProjectionCamera(intrinsics, image_width,
                                                   image_height, distortion));
        } else {
          LOG(ERROR) << "Invalid intrinsics parameters for the UnifiedProjection camera model: "
              << intrinsics.transpose() << std::endl <<
              "See aslam::UnifiedProjectionCamera::areParametersValid(...) for conditions on what "
              "valid UnifiedProjection camera intrinsics look like.";
          return true;
        }
      } else {
        LOG(ERROR) << "Unknown camera model: \"" << camera_type << "\". "
            << "Valid values are {pinhole, unified-projection}.";
        return true;
      }
    } else {
      LOG(ERROR) << "Unable to get the required parameters from the camera. "
          << "Required: string type, int image_height, int image_width, VectorXd intrinsics.";
      return true;
    }
    // ID
    aslam::CameraId id;
    if(node["id"]) {
      std::string id_string = node["id"].as<std::string>();
      if(!id.fromHexString(id_string)) {
        LOG(ERROR) << "Unable to parse \"" << id_string << "\" as a hex string.";
        camera.reset();
        return true;
      }
    } else {
      LOG(WARNING) << "Unable to get the id for the camera. Generating new random id.";
      id.randomize();
    }
    camera->setId(id);
    if(node["line-delay-nanoseconds"]) {
      uint64_t line_delay_nanoseconds = camera->getLineDelayNanoSeconds();
      if(YAML::safeGet(node, "line-delay-nanoseconds", &line_delay_nanoseconds)){
        camera->setLineDelayNanoSeconds(line_delay_nanoseconds);
      } else {
        LOG(ERROR) << "Unable to parse the parameter line-delay-nanoseconds.";
        camera.reset();
        return true;
      }
    }

    if(node["label"]) {
      camera->setLabel(node["label"].as<std::string>());
    }
  } catch(const std::exception& e) {
    LOG(ERROR) << "Yaml exception during parsing: " << e.what();
    camera.reset();
    return true;
  }
  return true;
}

Node convert<aslam::Camera::Ptr>::encode(const aslam::Camera::Ptr& camera) {
  return convert<aslam::Camera>::encode(*CHECK_NOTNULL(camera.get()));
}

bool convert<aslam::Camera>::decode(const Node& /*node*/, aslam::Camera& /*camera*/) {
  LOG(FATAL) << "Not implemented!";
  return false;
}

Node convert<aslam::Camera>::encode(const aslam::Camera& camera) {
  Node camera_node;

  camera_node["label"] = camera.getLabel();
  if(camera.getId().isValid()) {
    camera_node["id"] = camera.getId().hexString();
  }
  camera_node["line-delay-nanoseconds"] = camera.getLineDelayNanoSeconds();
  camera_node["image_height"] = camera.imageHeight();
  camera_node["image_width"] = camera.imageWidth();
  switch(camera.getType()) {
    case aslam::Camera::Type::kPinhole:
      camera_node["type"] = "pinhole";
      break;
    case aslam::Camera::Type::kUnifiedProjection:
      camera_node["type"] = "unified-projection";
      break;
    default:
      LOG(ERROR) << "Unknown camera model: "
        << static_cast<std::underlying_type<aslam::Camera::Type>::type>(camera.getType());
  }
  camera_node["intrinsics"] = camera.getParameters();

  const aslam::Distortion& distortion = camera.getDistortion();
  if(distortion.getType() != aslam::Distortion::Type::kNoDistortion) {
    Node distortion_node;
    switch(distortion.getType()) {
      case aslam::Distortion::Type::kEquidistant:
        distortion_node["type"] = "equidistant";
        break;
      case aslam::Distortion::Type::kFisheye:
        distortion_node["type"] = "fisheye";
        break;
      case aslam::Distortion::Type::kRadTan:
        distortion_node["type"] = "radial-tangential";
        break;
      default:
        LOG(ERROR) << "Unknown distortion model: "
          << static_cast<std::underlying_type<aslam::Distortion::Type>::type>(
              distortion.getType());
    }
    distortion_node["parameters"] = distortion.getParameters();
    camera_node["distortion"] = distortion_node;
  }
  return camera_node;
}

}  // namespace YAML

