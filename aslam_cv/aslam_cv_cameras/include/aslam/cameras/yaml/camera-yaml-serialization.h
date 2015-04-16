#ifndef ASLAM_CV_CAMERA_YAML_SERIALIZATION_H
#define ASLAM_CV_CAMERA_YAML_SERIALIZATION_H

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

namespace aslam {
class Camera;
}  // namespace aslam

namespace YAML {

template<>
struct convert<std::shared_ptr<aslam::Camera>> {
  /// This function will attempt to parse a camera from the yaml node.
  /// By default, yaml-cpp will throw and exception if the parsing fails.
  /// This function was written to *not* throw exceptions. Hence, decode always
  /// returns true, but when it fails, the shared pointer will be null.
  static bool decode(const Node& node, std::shared_ptr<aslam::Camera>& camera);
  static Node encode(const std::shared_ptr<aslam::Camera>& camera);
};

template<>
struct convert<aslam::Camera> {
  static bool decode(const Node& node, aslam::Camera& camera);
  static Node encode(const aslam::Camera& camera);
};

}  // namespace YAML

#endif // ASLAM_CV_CAMERA_YAML_SERIALIZATION_H
