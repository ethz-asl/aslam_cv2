#ifndef ASLAM_CV_NCAMERA_YAML_SERIALIZATION_H
#define ASLAM_CV_NCAMERA_YAML_SERIALIZATION_H

#include <glog/logging.h>
#include <yaml-cpp/yaml.h>

#include <aslam/cameras/ncamera.h>

namespace YAML {

template<>
struct convert<std::shared_ptr<aslam::NCamera>> {
  /// This function will attempt to parse an ncamera from the yaml node.
  /// By default, yaml-cpp will throw an exception if the parsing fails.
  /// This function was written to *not* throw exceptions. Hence, decode always
  /// returns true, but when it fails, the shared pointer will be null.
  static bool decode(const Node& node, std::shared_ptr<aslam::NCamera>& ncamera);
  static Node encode(const std::shared_ptr<aslam::NCamera>& ncamera);
};

template<>
struct convert<aslam::NCamera> {
  static bool decode(const Node& node, aslam::NCamera& ncamera);
  static Node encode(const aslam::NCamera& ncamera);
};

}  // namespace YAML

#endif // ASLAM_CV_NCAMERA_YAML_SERIALIZATION_H
