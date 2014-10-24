#ifndef TRIANGULATION_TRIANGULATION_TOOLBOX_H_
#define TRIANGULATION_TRIANGULATION_TOOLBOX_H_
#include <vector>

#include <aslam/common/pose-types.h>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <glog/logging.h>
#include <aslam/common/memory.h>

namespace aslam {
// From Google Tango.
inline bool LinearTriangulateFromNViews(
    const Aligned<std::vector, Eigen::Vector2d>::type& measurements,
    const Aligned<std::vector, aslam::Transformation>::type& T_G_I,
    const aslam::Transformation& T_I_C,
    Eigen::Vector3d* G_point) {
  CHECK_NOTNULL(G_point);
  CHECK_EQ(measurements.size(), T_G_I.size());
  if (measurements.size() < 2u) {
    return false;
  }

  size_t rows = 3 * measurements.size();
  size_t cols = 3 + measurements.size();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

  const Eigen::Matrix3d I_R_C = T_I_C.getRotation().getRotationMatrix();

  // Fill in A and b.
  for (size_t i = 0; i < measurements.size(); ++i) {
    Eigen::Vector3d v(measurements[i](0), measurements[i](1), 1.);
    Eigen::Matrix3d R_G_I = T_G_I[i].getRotation().getRotationMatrix();
    Eigen::Vector3d p_G_I = T_G_I[i].getPosition();
    A.block<3, 3>(3 * i, 0) = Eigen::Matrix3d::Identity();
    A.block<3, 1>(3 * i, 3 + i) = -R_G_I * I_R_C * v;
    b.segment<3>(3 * i) = p_G_I + R_G_I * T_I_C.getPosition();
  }

  *G_point = A.colPivHouseholderQr().solve(b).head<3>();
  return true;
}
}  // namespace aslam
#endif  // TRIANGULATION_TRIANGULATION_TOOLBOX_H_
