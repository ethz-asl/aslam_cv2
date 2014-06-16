#ifndef FISHEYE_DISTORTION_INL_H_
#define FISHEYE_DISTORTION_INL_H_

// TODO(dymczykm) it would be better (if not necessary) to drop
// ceres here
#include <ceres/ceres.h>

namespace aslam {

template <typename ScalarType>
void FisheyeDistortion::distort(
    const Eigen::Matrix<ScalarType, 2, 1>& point,
    Eigen::Matrix<ScalarType, 2, 1>* out_point) const {
  CHECK_NOTNULL(out_point);

  // Evaluate the camera distortion.
  const ScalarType r_u = point.norm();
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

  *out_point = point;
  *out_point *= r_rd;
}

} // namespace aslam

#endif /* FISHEYE_DISTORTION_INL_H_ */
