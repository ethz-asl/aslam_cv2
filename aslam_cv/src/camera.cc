#include <aslam/cameras/camera.h>
#include <glog/logging.h>
// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {
Camera::Camera() :
    line_delay_nano_seconds_(0),
    image_width_(0),
    image_height_(0){}

// TODO(slynen) Enable commented out PropertyTree support
//Camera::Camera(const sm::PropertyTree& property_tree) {
//  double value = property_tree.getDouble("line_delay_nano_seconds", -1.0);
//  if (value == -1.0) {
//    value = 0.0;
//    VLOG(3) << "Failed to load line delay property for camera. Using " << value << ".";
//  }
//}

Camera::~Camera() { }

void Camera::printParameters(std::ostream& out, const std::string& text) {
  out << text << std::endl;
  out << "Camera(" << this->id_ << "): " << this->label_ << std::endl;
  out << "  line delay: " << this->line_delay_nano_seconds_ << std::endl;
  out << "  image (cols,rows): " << imageWidth() << ", " << imageHeight() << std::endl;
}

bool Camera::operator==(const Camera& other) const {

  //test if internal state is equal
  // \todo(slynen) should we include the id and name here?
  return (this->line_delay_nano_seconds_ == other.line_delay_nano_seconds_) &&
         (this->image_width_ == other.image_width_) &&
         (this->image_height_ == other.image_height_);

}

}  // namespace aslam
