#ifndef ASLAM_CAMERAS_GENERIC_CAMERA_INL_H_
#define ASLAM_CAMERAS_GENERIC_CAMERA_INL_H_

#include <memory>

namespace aslam {

template <typename DerivedKeyPoint, typename DerivedPoint3d>
inline const ProjectionResult GenericCamera::evaluateProjectionResult(
    const Eigen::MatrixBase<DerivedKeyPoint>& keypoint,
    const Eigen::MatrixBase<DerivedPoint3d>& point_3d) const {
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DerivedKeyPoint, 2, 1);
  EIGEN_STATIC_ASSERT_MATRIX_SPECIFIC_SIZE(DerivedPoint3d, 3, 1);

  Eigen::Matrix<typename DerivedKeyPoint::Scalar, 2, 1> kp = keypoint;
  const bool visibility = isKeypointVisible(kp);

  if (visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionResult(ProjectionResult::Status::KEYPOINT_VISIBLE);
  else if (!visibility && (point_3d[2] > kMinimumDepth))
    return ProjectionResult(ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX);
  else if (point_3d[2] < 0.0)
    return ProjectionResult(ProjectionResult::Status::POINT_BEHIND_CAMERA);
  else
    return ProjectionResult(ProjectionResult::Status::PROJECTION_INVALID);
}

}  // namespace aslam
#endif  // ASLAM_CAMERAS_GENERIC_CAMERA_INL_H_
