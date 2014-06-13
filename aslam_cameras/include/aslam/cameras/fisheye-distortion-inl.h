#ifndef TEST_FISHEYE_DISTORTION_INL_H_
#define TEST_FISHEYE_DISTORTION_INL_H_

#include <Eigen/Core>

namespace aslam {

template <typename ScalarType>
void TestFisheyeDistortion::distort(
    Eigen::Matrix<ScalarType, 2, 1>* keypoint) const {
  CHECK_NOTNULL(keypoint);

  // Evaluate the camera distortion.
  const ScalarType r_u = (*keypoint).norm();
  ScalarType r_rd;
  if (w_ * w_ < 1e-5) {
    // Limit w->0.
    r_rd = static_cast<ScalarType>(1);
  } else {
    const ScalarType mul2tanwby2 =
        static_cast<ScalarType>(2.0 * ceres::tan(w_ / 2.0));
    const ScalarType mul2tanwby2byw = mul2tanwby2 / static_cast<ScalarType>(w_);

    if (r_u * r_u < static_cast<ScalarType>(1e-5)) {
      // Limit r_u->0.
      r_rd = mul2tanwby2byw;
    } else {
      r_rd = ceres::atan(r_u * mul2tanwby2) /
          (r_u * static_cast<ScalarType>(w_));
    }
  }

  *keypoint *= r_rd;
}

template <typename ScalarType>
void TestFisheyeDistortion::undistort(
    Eigen::Matrix<ScalarType, 2, 1>* y) const {
  CHECK_NOTNULL(y);

  // TODO(dymczykm) will be needed for tests

}

} // namespace aslam

#endif /* TEST_FISHEYE_DISTORTION_INL_H_ */
