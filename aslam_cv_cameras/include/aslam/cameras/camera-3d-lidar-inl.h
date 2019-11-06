#ifndef ASLAM_CAMERAS_CAMERA_3D_LIDAR_INL_H_
#define ASLAM_CAMERAS_CAMERA_3D_LIDAR_INL_H_

#include <memory>

namespace aslam {

template <
    typename ScalarType, typename DistortionType, typename MIntrinsics,
    typename MDistortion>
const ProjectionResult CameraLidar3D::project3Functional(
    const Eigen::Matrix<ScalarType, 3, 1>& point_3d,
    const Eigen::MatrixBase<MIntrinsics>& intrinsics_external,
    const Eigen::MatrixBase<MDistortion>& distortion_coefficients_external,
    Eigen::Matrix<ScalarType, 2, 1>* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);
  CHECK_EQ(intrinsics_external.size(), kNumOfParams)
      << "intrinsics: invalid size!";

  const ScalarType x_z_scalar_inv =
      std::sqrt(point_3d.x() * point_3d.x() + point_3d.z() * point_3d.z());
  CHECK_GT(x_z_scalar_inv, 1e-6);

  const Eigen::Matrix<ScalarType, 3, 1> bearing_vec_on_cylinder =
      point_3d / x_z_scalar_inv;

  const ScalarType x_z_norm = std::sqrt(
      bearing_vec_on_cylinder.x() * bearing_vec_on_cylinder.x() +
      bearing_vec_on_cylinder.z() * bearing_vec_on_cylinder.z());
  CHECK_LT(std::abs(x_z_norm - 1.0), 1e-6);

  (*out_keypoint)[0] =
      (std::atan2(bearing_vec_on_cylinder.x(), bearing_vec_on_cylinder.z()) +
       horizontalCenter()) /
      horizontalResolution();

  if ((*out_keypoint)[0] < 0.0) {
    (*out_keypoint)[0] = imageWidth() + (*out_keypoint)[0];
  }

  (*out_keypoint)[1] =
      ((std::atan(bearing_vec_on_cylinder.y()) + verticalCenter()) /
       verticalResolution());

  return evaluateProjectionResult(*out_keypoint, point_3d);
}

template <typename DerivedKeyPoint, typename DerivedPoint3d>
inline const ProjectionResult CameraLidar3D::evaluateProjectionResult(
    const Eigen::MatrixBase<DerivedKeyPoint>& keypoint,
    const Eigen::MatrixBase<DerivedPoint3d>& point_3d) const {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DerivedKeyPoint, 2, 1);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DerivedPoint3d, 3, 1);

  Eigen::Matrix<typename DerivedKeyPoint::Scalar, 2, 1> kp = keypoint;
  const bool visibility = isKeypointVisible(kp);

  const double squaredNorm = point_3d.squaredNorm();
  const bool is_outside_min_depth = squaredNorm > kSquaredMinimumDepth;

  if (visibility && is_outside_min_depth)
    return ProjectionResult(ProjectionResult::Status::KEYPOINT_VISIBLE);
  else if (!visibility && is_outside_min_depth)
    return ProjectionResult(
        ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX);
  else
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}

}  // namespace aslam
#endif  // ASLAM_CAMERAS_CAMERA_3D_LIDAR_INL_H_
