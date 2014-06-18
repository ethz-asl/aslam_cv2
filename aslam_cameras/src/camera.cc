#include <aslam/cameras/camera.h>
#include <glog/logging.h>
//#include <sm/PropertyTree.hpp>

namespace aslam {
Camera::Camera() : line_delay_nano_seconds_(0) {}

// TODO(slynen)
//Camera::Camera(const sm::PropertyTree& property_tree) {
//  double value = property_tree.getDouble("line_delay_nano_seconds", -1.0);
//  if (value == -1.0) {
//    value = 0.0;
//    VLOG(3) << "Failed to load line delay property for camera. Using " << value << ".";
//  }
//}

Camera::~Camera() { }

bool Camera::operator==(const Camera& other) const {
  return line_delay_nano_seconds_ == other.line_delay_nano_seconds_;
}
}  // namespace aslam
