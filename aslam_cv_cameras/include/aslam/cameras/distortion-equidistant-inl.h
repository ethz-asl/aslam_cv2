#include "aslam/cameras/ceres-jet-math.h"

namespace aslam {

template <typename ScalarType, typename MDistortion>
void EquidistantDistortion::distortUsingExternalCoefficients(
    const Eigen::MatrixBase<MDistortion>& dist_coeffs,
    const Eigen::Matrix<ScalarType, 2, 1>& point,
    Eigen::Matrix<ScalarType, 2, 1>* out_point) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(out_point);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(MDistortion);

  *out_point = point;
  ScalarType& x = (*out_point)(0);
  ScalarType& y = (*out_point)(1);

  const ScalarType& k1 = dist_coeffs(0);
  const ScalarType& k2 = dist_coeffs(1);
  const ScalarType& k3 = dist_coeffs(2);
  const ScalarType& k4 = dist_coeffs(3);

  const ScalarType x2 = x * x;
  const ScalarType y2 = y * y;
  const ScalarType r = jet_math::sqrt(x2 + y2);

  const ScalarType theta = jet_math::atan(r);
  const ScalarType theta2 = theta * theta;
  const ScalarType theta4 = theta2 * theta2;
  const ScalarType theta6 = theta2 * theta4;
  const ScalarType theta8 = theta4 * theta4;
  const ScalarType thetad = theta * (ScalarType(1.0) + k1 * theta2 +
                                     k2 * theta4 + k3 * theta6 + k4 * theta8);
  const ScalarType scaling =
      (r > ScalarType(1e-8)) ? thetad / r : ScalarType(1.0);
  x *= scaling;
  y *= scaling;
}

}  // namespace aslam
