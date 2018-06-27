#ifndef FISHEYE_DISTORTION_INL_H_
#define FISHEYE_DISTORTION_INL_H_

#include "aslam/cameras/jet-trigonometric.h"

namespace aslam {

template <typename ScalarType, typename MDistortion>
void FisheyeDistortion::distortUsingExternalCoefficients(
    const Eigen::MatrixBase<MDistortion>& dist_coeffs,
    const Eigen::Matrix<ScalarType, 2, 1>& point,
    Eigen::Matrix<ScalarType, 2, 1>* out_point) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(out_point);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(MDistortion);

  const ScalarType& w = dist_coeffs(0);

  // Evaluate the camera distortion.
  const ScalarType r_u = point.norm();
  ScalarType r_rd;
  if (w * w < 1e-5) {
    // Limit w->0.
    r_rd = static_cast<ScalarType>(1);
  } else {
    const ScalarType mul2tanwby2 = static_cast<ScalarType>(2.0 * jet_trigonometric::tan(w / 2.0));
    const ScalarType mul2tanwby2byw = mul2tanwby2 / w;

    if (r_u * r_u < static_cast<ScalarType>(1e-5)) {
      // Limit r_u->0.
      r_rd = mul2tanwby2byw;
    } else {
      r_rd = jet_trigonometric::atan(r_u * mul2tanwby2) /
          (r_u * w);
    }
  }

  *out_point = point;
  *out_point *= r_rd;
}

} // namespace aslam

#endif /* FISHEYE_DISTORTION_INL_H_ */
