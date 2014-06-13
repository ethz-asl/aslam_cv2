#include "aslam/cameras/fisheye-distortion.h"

namespace aslam {

void FisheyeDistortion::distort(const Eigen::Matrix<double, 2, 1>* y) const {
  CHECK_NOTNULL(y);
  Eigen::Matrix2Xd outJy;
  distort(y, &outJy);
}

void FisheyeDistortion::distort(const Eigen::Matrix<double, 2, 1>& y,
                                Eigen::Matrix<double, 2, 1>* outPoint) const {
  CHECK_NOTNULL(outPoint);
  *outPoint = y;
  Eigen::Matrix2Xd outJy;
  distort(outPoint, &outJy);
}

void FisheyeDistortion::distort(
    const Eigen::Matrix<double, 2, 1>* point,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJy) const {
  CHECK_NOTNULL(point);
  CHECK_NOTNULL(outJy);

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
  outJy->resize(2, 2);
  *outJy <<
      duf_du, duf_dv,
      dvf_du, dvf_dv;

  *const_cast<Eigen::Matrix<double, 2, 1>*>(point) *= r_rd;
}

void FisheyeDistortion::undistort(
    Eigen::Matrix<double, 2, 1>* y,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJy) const {
  CHECK_NOTNULL(y);
  CHECK_NOTNULL(outJy);

  // TODO(dymczykm) will be needed for tests
}

void FisheyeDistortion::distortParameterJacobian(
    Eigen::Matrix<double, 2, 1>* imageY,
    Eigen::Matrix<double, 2, Eigen::Dynamic>* outJd) const {
  CHECK_NOTNULL(outJd);
  CHECK_EQ(outJd->cols(), 1);

  const double tanwhalf = tan(w_ / 2.);
  const double tanwhalfsq = tanwhalf * tanwhalf;
  const double r_u = imageY->norm();
  const double atan_w_rd = atan(2. * tanwhalf * r_u);

  const double& u = (*imageY)(0);
  const double& v = (*imageY)(1);

  const double dxd_d_w = (2 * u * (tanwhalfsq / 2 + 0.5))
      / (w_ * (4 * tanwhalfsq * r_u * r_u + 1))
      - (u * atan_w_rd) / (w_ * w_ * r_u);

  const double dyd_d_w = (2 * v * (tanwhalfsq / 2 + 0.5))
      / (w_ * (4 * tanwhalfsq * r_u * r_u + 1))
      - (v * atan_w_rd) / (w_ * w_ * r_u);

  *outJd << dxd_d_w, dyd_d_w;
}

} // namespace aslam
