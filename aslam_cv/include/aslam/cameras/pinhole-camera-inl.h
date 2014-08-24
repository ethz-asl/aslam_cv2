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
bool PinholeCamera::project3(
    const Eigen::Matrix<ScalarType, 3, 1>& point,
    const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& intrinsics,
    Eigen::Matrix<ScalarType, 2, 1>* out_point) const {
  CHECK_NOTNULL(out_point);

  ScalarType rz = static_cast<ScalarType>(1.0) / point[2];
  Eigen::Matrix<ScalarType, 2, 1> keypoint;
  keypoint[0] = point[0] * rz;
  keypoint[1] = point[1] * rz;

  CHECK_NOTNULL(_distortion.get());
  std::shared_ptr<DistortionType> distortion_ptr =
      std::dynamic_pointer_cast<DistortionType>(_distortion);
  CHECK(distortion_ptr);
  distortion_ptr->distort(
      Eigen::Map<Eigen::VectorXd>(intrinsics.tail(distortion_ptr->getParameterSize()).data(), distortion_ptr->getParameterSize()),
      keypoint,
      out_point);

  (*out_point)[0] = intrinsics(0) * (*out_point)[0] + intrinsics(2);
  (*out_point)[1] = intrinsics(1) * (*out_point)[1] + intrinsics(3);

  return isValid(*out_point) && point[2] > static_cast<ScalarType>(0);
}

template <typename ScalarType>
bool PinholeCamera::isKeypointVisible(
    const Eigen::Matrix<ScalarType, 2, 1>& keypoint) const {
  return keypoint[0] >= static_cast<ScalarType>(0)
      && keypoint[1] >= static_cast<ScalarType>(0)
      && keypoint[0] < static_cast<ScalarType>(_ru)
      && keypoint[1] < static_cast<ScalarType>(_rv);
}

}  // namespace aslam

#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_INL_H_
