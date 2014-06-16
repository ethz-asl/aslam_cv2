#ifndef FISHEYE_DISTORTION_INL_H_
#define FISHEYE_DISTORTION_INL_H_

// TODO(dymczykm) it would be better (if not necessary) to drop
// ceres here
#include <ceres/ceres.h>

namespace aslam {

template <typename ScalarType>
void FisheyeDistortion::distort(
    const Eigen::Matrix<ScalarType, 2, 1>& point,
    const Eigen::Matrix<ScalarType, kNumOfParams, 1>& params,
    Eigen::Matrix<ScalarType, 2, 1>* out_point) const {
  CHECK_NOTNULL(out_point);

  const ScalarType& w = params(0);

  // Evaluate the camera distortion.
  const ScalarType r_u = point.norm();
  ScalarType r_rd;
  if (w * w < 1e-5) {
    // Limit w->0.
    r_rd = static_cast<ScalarType>(1);
  } else {
    const ScalarType mul2tanwby2 =
        static_cast<ScalarType>(2.0 * ceres::tan(w / 2.0));
    const ScalarType mul2tanwby2byw = mul2tanwby2 / w;

    if (r_u * r_u < static_cast<ScalarType>(1e-5)) {
      // Limit r_u->0.
      r_rd = mul2tanwby2byw;
    } else {
      r_rd = ceres::atan(r_u * mul2tanwby2) /
          (r_u * w);
    }
  }

  *out_point = point;
  *out_point *= r_rd;
}

} // namespace aslam

#endif /* FISHEYE_DISTORTION_INL_H_ */
