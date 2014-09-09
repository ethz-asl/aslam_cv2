#ifndef ASLAM_CAMERAS_PINHOLE_CAMERA_INL_H_
#define ASLAM_CAMERAS_PINHOLE_CAMERA_INL_H_

#include <memory>

namespace aslam {

// TODO(dymczykm) actually, I'm not sure if it wouldn't be better if
// specialized versions of these function (for double) would use this
// implementation instead of repeating it twice. But that would mean we need
// to template the whole class on ScalarType (or at least intrinsics and few
// methods).

template <typename ScalarType, typename DistortionType>
const ProjectionState PinholeCamera::project3Functional(
    const Eigen::Matrix<ScalarType, 3, 1>& point_3d,
    const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& intrinsics,
    Eigen::Matrix<ScalarType, 2, 1>* out_keypoint) const {
  CHECK_NOTNULL(out_keypoint);

  ScalarType rz = static_cast<ScalarType>(1.0) / point_3d[2];
  Eigen::Matrix<ScalarType, 2, 1> keypoint;
  keypoint[0] = point_3d[0] * rz;
  keypoint[1] = point_3d[1] * rz;

  distortion_->undistort(&keypoint);

  (*out_keypoint)[0] = intrinsics(0) * (*out_keypoint)[0] + intrinsics(2);
  (*out_keypoint)[1] = intrinsics(1) * (*out_keypoint)[1] + intrinsics(3);

  return evaluateProjectionState(out_keypoint, point_3d);
}

}  // namespace aslam
#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_INL_H_
