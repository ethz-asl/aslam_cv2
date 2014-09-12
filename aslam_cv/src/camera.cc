#include <aslam/cameras/camera.h>
#include <glog/logging.h>
// TODO(slynen) Enable commented out PropertyTree support
//#include <sm/PropertyTree.hpp>

namespace aslam {

// TODO(slynen) Enable commented out PropertyTree support
//Camera::Camera(const sm::PropertyTree& property_tree) {
//  double value = property_tree.getDouble("line_delay_nano_seconds", -1.0);
//  if (value == -1.0) {
//    value = 0.0;
//    VLOG(3) << "Failed to load line delay property for camera. Using " << value << ".";
//  }
//}

Camera::Camera(const Eigen::VectorXd& intrinsics)
    : intrinsics_(intrinsics) { }

void Camera::printParameters(std::ostream& out, const std::string& text) const {
  if(text.size() > 0) {
    out << text << std::endl;
  }
  out << "Camera(" << this->id_ << "): " << this->label_ << std::endl;
  out << "  line delay: " << this->line_delay_nano_seconds_ << std::endl;
  out << "  image (cols,rows): " << imageWidth() << ", " << imageHeight() << std::endl;
}

bool Camera::operator==(const Camera& other) const {
  // \TODO(slynen) should we include the id and name here?
  return (this->line_delay_nano_seconds_ == other.line_delay_nano_seconds_) &&
         (this->image_width_ == other.image_width_) &&
         (this->image_height_ == other.image_height_);
}

const ProjectionResult Camera::project4(const Eigen::Vector4d& point_4d,
                                       Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  Eigen::Vector3d point_3d;
  if (point_4d[3] < 0)
    point_3d = -point_4d.head<3>();
  else
    point_3d =  point_4d.head<3>();

  return project3(point_3d, out_keypoint);
}

const ProjectionResult Camera::project4(const Eigen::Vector4d& point_4d,
                                       Eigen::Vector2d* out_keypoint,
                                       Eigen::Matrix<double, 2, 4>* out_jacobian) const {
  CHECK_NOTNULL(out_keypoint);
  CHECK_NOTNULL(out_jacobian);

  Eigen::Vector3d point_3d;
  if (point_4d[3] < 0)
    point_3d = -point_4d.head<3>();
  else
    point_3d =  point_4d.head<3>();

  Eigen::Matrix<double, 2, 3> Je;
  ProjectionResult ret = project3(point_3d, out_keypoint, &Je);
  out_jacobian->setZero();
  out_jacobian->topLeftCorner<2, 3>() = Je;

  return ret;
}

bool Camera::backProject4(const Eigen::Vector2d& keypoint,
                          Eigen::Vector4d* out_point4d) const {
  CHECK_NOTNULL(out_point4d);

  Eigen::Vector3d point_3d;
  bool success = backProject3(keypoint, &point_3d);
  (*out_point4d) << point_3d, 0.0;

  return success;
}

bool Camera::isProjectable3(const Eigen::Vector3d& p) const {
  Eigen::Vector2d k;
  const ProjectionResult& ret = project3(p, &k);
  return ret.isKeypointVisible();
}

bool Camera::isProjectable4(const Eigen::Vector4d& ph) const {
  Eigen::Vector2d k;
  const ProjectionResult& ret = project4(ph, &k);
  return ret.isKeypointVisible();
}

bool Camera::isKeypointVisible(const Eigen::Vector2d& keypoint) const {
  return keypoint[0] >= 0.0
      && keypoint[1] >= 0.0
      && keypoint[0] < static_cast<double>(imageWidth())
      && keypoint[1] < static_cast<double>(imageHeight());
}

}  // namespace aslam
