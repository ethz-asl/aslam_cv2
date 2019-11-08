#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/camera-lidar.h>
#include <aslam/common/types.h>

#include <memory>
#include <utility>

#include "aslam/cameras/random-camera-generator.h"

namespace aslam {
std::ostream& operator<<(std::ostream& out, const LidarCamera& camera) {
  camera.printParameters(out,
                         std::string("Lidar cameras don't have parameters."));
  return out;
}

LidarCamera::LidarCamera()
    : Base(Eigen::Vector4d::Zero(), 0, 0, Camera::Type::kLidar) {}

LidarCamera::LidarCamera(uint32_t image_width, uint32_t image_height)
    : Base(Eigen::Vector4d::Zero(), image_width, image_height,
           Camera::Type::kLidar) {}

bool LidarCamera::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    Eigen::Vector3d* out_point_3d) const {
  // Assumtion: The point gets reporjected onto the "unit cylinder", meaning a
  // cylinder with unit radius and infinite height.
  CHECK_NOTNULL(out_point_3d);

  double yaw = 0;
  //(keypoint[0] - cu()) / fu() * 2 * M_PI;  // rotation around camera Y axis.
  double pitch = 0;  // (-keypoint[1] + cv()) / fv() * 2 *
                     // M_PI;  // Elevation around camera X axis.

  (*out_point_3d)[0] = -sin(yaw);
  (*out_point_3d)[1] = -tan(pitch);
  (*out_point_3d)[2] = -cos(yaw);

  // Always valid for the lidar model.
  return true;
}

const ProjectionResult LidarCamera::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const {
  CHECK_NOTNULL(out_keypoint);

  return ProjectionResult::Status::KEYPOINT_VISIBLE;
}

Eigen::Vector2d LidarCamera::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  // Unit tests often fail when the point is near the border. Keep the point
  // away from the border.
  double border = std::min(imageWidth(), imageHeight()) * 0.1;

  out(0) = border + std::abs(out(0)) * (imageWidth() - border * 2.0);
  out(1) = border + std::abs(out(1)) * (imageHeight() - border * 2.0);

  return out;
}

Eigen::Vector3d LidarCamera::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

bool LidarCamera::areParametersValid(const Eigen::VectorXd& parameters) {
  return true;
}

bool LidarCamera::intrinsicsValid(const Eigen::VectorXd& intrinsics) const {
  return true;
}

void LidarCamera::printParameters(std::ostream& out,
                                  const std::string& text) const {
  Camera::printParameters(out, text);
}

bool LidarCamera::isValidImpl() const { return true; }

void LidarCamera::setRandomImpl() {
  LidarCamera::Ptr test_camera = LidarCamera::createTestCamera();
  CHECK(test_camera);
  line_delay_nanoseconds_ = test_camera->line_delay_nanoseconds_;
  image_width_ = test_camera->image_width_;
  image_height_ = test_camera->image_height_;
  mask_ = test_camera->mask_;
  intrinsics_ = test_camera->intrinsics_;
  camera_type_ = test_camera->camera_type_;
  if (test_camera->distortion_) {
    distortion_ = std::move(test_camera->distortion_);
  }
}

bool LidarCamera::isEqualImpl(const Sensor& other,
                              const bool /*verbose*/) const {
  const LidarCamera* other_camera = dynamic_cast<const LidarCamera*>(&other);
  if (other_camera == nullptr) {
    return false;
  }

  return true;
}

LidarCamera::Ptr LidarCamera::createTestCamera() {
  LidarCamera::Ptr camera(new LidarCamera(0, 0));
  CameraId id;
  generateId(&id);
  camera->setId(id);
  return camera;
}

}  // namespace aslam
