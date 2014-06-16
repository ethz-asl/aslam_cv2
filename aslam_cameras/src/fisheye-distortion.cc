#include "aslam/cameras/fisheye-distortion.h"

namespace aslam {

void FisheyeDistortion::distort(
    const Eigen::Matrix<double, 2, 1>* point) const {
  CHECK_NOTNULL(point);
  Eigen::Matrix2Xd jacobian;
  distort(point, &jacobian);
}

void FisheyeDistortion::distort(const Eigen::Matrix<double, 2, 1>& point,
                                Eigen::Matrix<double, 2, 1>* out_point) const {
  CHECK_NOTNULL(out_point);
  *out_point = point;
  Eigen::Matrix2Xd jacobian;
  distort(out_point, &jacobian);
}

void FisheyeDistortion::distort(
    const Eigen::Matrix<double, 2, 1>* point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_NOTNULL(point);
  CHECK_NOTNULL(out_jacobian);

  const double r_u = point->norm();
  const double r_u_cubed = r_u * r_u * r_u;
  const double tanwhalf = tan(w_ / 2.);
  const double tanwhalfsq = tanwhalf * tanwhalf;
  const double atan_w_rd = atan(2. * tanwhalf * r_u);
  double r_rd;

  if (w_ * w_ < 1e-5) {
    // Limit w > 0.
    r_rd = 1.0;
  } else {
    if (r_u * r_u < 1e-5) {
      // Limit r_u > 0.
      r_rd = 2. * tanwhalf / w_;
    } else {
      r_rd = atan_w_rd / (r_u * w_);
    }
  }

  const double& u = (*point)(0);
  const double& v = (*point)(1);

  const double duf_du = (atan_w_rd) / (w_ * r_u)
        - (u * u * atan_w_rd) / (w_ * r_u_cubed)
        + (2 * u * u * tanwhalf)
        / (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1));
  const double duf_dv = (2 * u * v * tanwhalf)
        / (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1))
        - (u * v * atan_w_rd) / (w_ * r_u_cubed);
  const double dvf_du = (2 * u * v * tanwhalf)
        / (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1))
        - (u * v * atan_w_rd) / (w_ * r_u_cubed);
  const double dvf_dv = (atan_w_rd) / (w_ * r_u)
        - (v * v * atan_w_rd) / (w_ * r_u_cubed)
        + (2 * v * v * tanwhalf)
        / (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1));
  out_jacobian->resize(2, 2);
  *out_jacobian <<
      duf_du, duf_dv,
      dvf_du, dvf_dv;

  *const_cast<Eigen::Matrix<double, 2, 1>*>(point) *= r_rd;
}

void FisheyeDistortion::undistort(Eigen::Matrix<double, 2, 1>* point) const {
  CHECK_NOTNULL(point);

  double mul2tanwby2 = tan(w_ / 2.0) * 2.0;

  // Calculate distance from point to center.
  double r_d = point->norm();

  // Calculate undistorted radius of point.
  double r_u;
  if (fabs(r_d * w_) <= kMaxValidAngle) {
    r_u = tan(r_d * w_) / (r_d * mul2tanwby2);
  } else {
    r_u = std::numeric_limits<double>::infinity();
  }

  (*point) *= r_u;
}

void FisheyeDistortion::distortParameterJacobian(
    const Eigen::Matrix<double, 2, 1>& point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const {
  CHECK_NOTNULL(out_jacobian);
  CHECK_EQ(out_jacobian->cols(), 1);

  const double tanwhalf = tan(w_ / 2.);
  const double tanwhalfsq = tanwhalf * tanwhalf;
  const double r_u = point.norm();
  const double atan_w_rd = atan(2. * tanwhalf * r_u);

  const double& u = point(0);
  const double& v = point(1);

  const double dxd_d_w = (2 * u * (tanwhalfsq / 2 + 0.5))
      / (w_ * (4 * tanwhalfsq * r_u * r_u + 1))
      - (u * atan_w_rd) / (w_ * w_ * r_u);

  const double dyd_d_w = (2 * v * (tanwhalfsq / 2 + 0.5))
      / (w_ * (4 * tanwhalfsq * r_u * r_u + 1))
      - (v * atan_w_rd) / (w_ * w_ * r_u);

  *out_jacobian << dxd_d_w, dyd_d_w;
}

} // namespace aslam
