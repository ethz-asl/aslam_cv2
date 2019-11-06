#include "aslam/cameras/camera-3d-lidar.h"

#include <memory>
#include <utility>

#include <aslam/cameras/camera-factory.h>
#include <aslam/common/types.h>

#include "aslam/cameras/random-camera-generator.h"

namespace aslam {
std::ostream& operator<<(std::ostream& out, const CameraLidar3D& camera) {
  camera.printParameters(out, std::string(""));
  return out;
}

CameraLidar3D::CameraLidar3D()
    : Base(Eigen::Vector3d::Zero(), 0, 0, Camera::Type::kLidar3D) {}

CameraLidar3D::CameraLidar3D(
    const Eigen::VectorXd& intrinsics, uint32_t image_width,
    uint32_t image_height)
    : Base(intrinsics, image_width, image_height, Camera::Type::kLidar3D) {
  CHECK(intrinsicsValid(intrinsics));
}

bool CameraLidar3D::backProject3(
    const Eigen::Ref<const Eigen::Vector2d>& keypoint,
    Eigen::Vector3d* out_point_3d) const {
  CHECK_NOTNULL(out_point_3d);

  Eigen::Vector2d kp = keypoint;
  kp[0] = kp[0] * horizontalResolution() - horizontalCenter();
  kp[1] = (kp[1] * verticalResolution()) - verticalCenter();
  (*out_point_3d)[0] = std::sin(kp[0]);
  (*out_point_3d)[1] = std::tan(kp[1]);
  (*out_point_3d)[2] = std::cos(kp[0]);

  // Always valid for the pinhole model.
  return true;
}

const ProjectionResult CameraLidar3D::project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    const Eigen::VectorXd* intrinsics_external,
    const Eigen::VectorXd* distortion_coefficients_external,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const {
  CHECK_NOTNULL(out_keypoint);

  CHECK(distortion_coefficients_external == nullptr)
      << "External distortion is not implemented!";
  CHECK(out_jacobian_point == nullptr)
      << "Jacobians are currently not implemented!";
  CHECK(out_jacobian_intrinsics == nullptr)
      << "Jacobians are currently not implemented!";
  CHECK(out_jacobian_distortion == nullptr)
      << "Jacobians are currently not implemented!";

  // Determine the parameter source. (if nullptr, use internal)
  const Eigen::VectorXd* intrinsics;
  if (!intrinsics_external)
    intrinsics = &getParameters();
  else
    intrinsics = intrinsics_external;
  CHECK_EQ(intrinsics->size(), kNumOfParams) << "intrinsics: invalid size!";

  // This camera does not have distortion, so we do not use this.
  const Eigen::VectorXd dummy_distortion_coefficients;

  return project3Functional<double, NullDistortion>(
      point_3d, *intrinsics, dummy_distortion_coefficients, out_keypoint);
}

Eigen::Vector2d CameraLidar3D::createRandomKeypoint() const {
  Eigen::Vector2d out;
  out.setRandom();
  // Unit tests often fail when the point is near the border. Keep the point
  // away from the border.
  double border = std::min(imageWidth(), imageHeight()) * 0.1;

  out(0) = border + std::abs(out(0)) * (imageWidth() - border * 2.0);
  out(1) = border + std::abs(out(1)) * (imageHeight() - border * 2.0);

  return out;
}

Eigen::Vector3d CameraLidar3D::createRandomVisiblePoint(double depth) const {
  CHECK_GT(depth, 0.0) << "Depth needs to be positive!";
  Eigen::Vector3d point_3d;

  Eigen::Vector2d y = createRandomKeypoint();
  backProject3(y, &point_3d);
  point_3d /= point_3d.norm();

  // Muck with the depth. This doesn't change the pointing direction.
  return point_3d * depth;
}

void CameraLidar3D::getBorderRays(Eigen::MatrixXd& rays) const {
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

bool CameraLidar3D::areParametersValid(const Eigen::VectorXd& parameters) {
  return (parameters.size() == parameterCount()) &&
         (parameters[Parameters::kHorizontalResolutionRad] > 0.0) &&
         (parameters[Parameters::kVerticalResolutionRad] > 0.0) &&
         (parameters[Parameters::kVerticalCenterRad] >= 0.0) &&
         (parameters[Parameters::kHorizontalCenterRad] >= 0.0);
}

bool CameraLidar3D::intrinsicsValid(const Eigen::VectorXd& intrinsics) const {
  return areParametersValid(intrinsics);
}

void CameraLidar3D::printParameters(
    std::ostream& out, const std::string& text) const {
  Camera::printParameters(out, text);
  out << "  horizontal angular resolution [rad]: " << horizontalResolution()
      << std::endl;
  out << "  vertical angular resolution [rad]: " << verticalResolution()
      << std::endl;
  out << "  line delay [ns]: " << line_delay_nanoseconds_ << std::endl;
}

const double CameraLidar3D::kSquaredMinimumDepth = 0.0001;

bool CameraLidar3D::isValidImpl() const {
  return intrinsicsValid(intrinsics_);
}

void CameraLidar3D::setRandomImpl() {
  CameraLidar3D::Ptr test_camera = CameraLidar3D::createTestCamera();
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

bool CameraLidar3D::isEqualImpl(const Sensor& other, const bool verbose) const {
  const CameraLidar3D* other_camera =
      dynamic_cast<const CameraLidar3D*>(&other);
  if (other_camera == nullptr) {
    LOG_IF(ERROR, verbose) << "Other camera is not a CameraLidar3D!";
    return false;
  }

  // Verify that the base members are equal.
  if (!isEqualCameraImpl(*other_camera, verbose)) {
    LOG_IF(ERROR, verbose) << "Base camera is not the same!";
    return false;
  }

  // Compare the distortion model (if distortion is set for both).
  if (!(*(this->distortion_) == *(other_camera->distortion_))) {
    LOG_IF(ERROR, verbose) << "Distortion is not the same!";
    return false;
  }

  return true;
}

}  // namespace aslam
