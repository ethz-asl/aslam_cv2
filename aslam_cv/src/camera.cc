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
    : intrinsics_(intrinsics) {}

void Camera::printParameters(std::ostream& out, const std::string& text) const {
  out << text << std::endl;
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

const ProjectionState Camera::project4(const Eigen::Vector4d& point_4d,
                                       Eigen::Vector2d* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  Eigen::Vector3d point_3d;
  if (point_4d[3] < 0)
    point_3d = -point_4d.head<3>();
  else
    point_3d =  point_4d.head<3>();

  return project3(point_3d, out_keypoint);
}

const ProjectionState Camera::project4(const Eigen::Vector4d& point_4d,
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
  ProjectionState ret = project3(point_3d, out_keypoint, &Je);
  out_jacobian->setZero();
  out_jacobian->topLeftCorner<2, 3>() = Je;

  return ret;
}

void Camera::backProject4(const Eigen::Vector2d& keypoint,
                          Eigen::Vector4d* out_point4d) const {
  CHECK_NOTNULL(out_point4d);

  Eigen::Vector3d point_3d;
  backProject3(keypoint, &point_3d);
  (*out_point4d) << point_3d, 0.0;
}

bool Camera::isProjectable3(const Eigen::Vector3d& p) const {
  Eigen::Vector2d k;
  const ProjectionState& ret = project3(p, &k);
  return ret.isKeypointVisible();
}

bool Camera::isProjectable4(const Eigen::Vector4d& ph) const {
  Eigen::Vector2d k;
  const ProjectionState& ret = project4(ph, &k);
  return ret.isKeypointVisible();
}

bool Camera::isKeypointVisible(const Eigen::Vector2d& keypoint) const {
  return keypoint[0] >= 0.0
      && keypoint[1] >= 0.0
      && keypoint[0] < static_cast<double>(imageWidth())
      && keypoint[1] < static_cast<double>(imageHeight());
}

Eigen::Vector2d Camera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  out(0) = std::abs(out(0)) * imageWidth();
  out(1) = std::abs(out(1)) * imageHeight();
  return out;
}

Eigen::Vector3d Camera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

void Camera::getBorderRays(Eigen::MatrixXd& rays) {
  rays.resize(4, 8);
  Eigen::Vector4d ray;
  backProject4(Eigen::Vector2d(0.0, 0.0), &ray);
  rays.col(0) = ray;
  backProject4(Eigen::Vector2d(0.0, imageHeight() * 0.5), &ray);
  rays.col(1) = ray;
  backProject4(Eigen::Vector2d(0.0, imageHeight() - 1.0), &ray);
  rays.col(2) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, 0.0), &ray);
  rays.col(3) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, imageHeight() * 0.5), &ray);
  rays.col(4) = ray;
  backProject4(Eigen::Vector2d(imageWidth() - 1.0, imageHeight() - 1.0), &ray);
  rays.col(5) = ray;
  backProject4(Eigen::Vector2d(imageWidth() * 0.5, 0.0), &ray);
  rays.col(6) = ray;
  backProject4(Eigen::Vector2d(imageWidth() * 0.5, imageHeight() - 1.0), &ray);
  rays.col(7) = ray;
}

}  // namespace aslam
