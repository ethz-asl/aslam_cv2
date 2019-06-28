#ifndef ASLAM_CAMERAS_DISTORTION_RADTAN_INLINE_H_
#define ASLAM_CAMERAS_DISTORTION_RADTAN_INLINE_H_

namespace aslam {

template <typename ScalarType, typename MDistortion>
void RadTanDistortion::distortUsingExternalCoefficients(
    const Eigen::MatrixBase<MDistortion>& dist_coeffs,
    Eigen::Matrix<ScalarType, 2, 1>* point) const {
  CHECK_EQ(dist_coeffs.size(), kNumOfParams) << "dist_coeffs: invalid size!";
  CHECK_NOTNULL(point);
  EIGEN_STATIC_ASSERT_VECTOR_ONLY(MDistortion);
  ScalarType& x = (*point)(0);
  ScalarType& y = (*point)(1);

  const ScalarType& k1 = dist_coeffs(0);
  const ScalarType& k2 = dist_coeffs(1);
  const ScalarType& p1 = dist_coeffs(2);
  const ScalarType& p2 = dist_coeffs(3);

  const ScalarType mx2_u = x * x;
  const ScalarType my2_u = y * y;
  const ScalarType mxy_u = x * y;
  const ScalarType rho2_u = mx2_u + my2_u;
  const ScalarType rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;

  x += x * rad_dist_u + ScalarType(2.0) * p1 * mxy_u +
       p2 * (rho2_u + ScalarType(2.0) * mx2_u);
  y += y * rad_dist_u + ScalarType(2.0) * p2 * mxy_u +
       p1 * (rho2_u + ScalarType(2.0) * my2_u);
}

}  // namespace aslam

#endif  // ASLAM_CAMERAS_DISTORTION_RADTAN_INLINE_H_
