#ifndef TRIANGULATION_H_
#define TRIANGULATION_H_
#include <vector>

#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <glog/logging.h>

namespace aslam {

/// brief Triangulate a 3d point from a set of n keypoint measurements on the normalized camera
///       plane.
/// @param measurements Keypoint measurements on normalized camera plane.
/// @param T_W_B Pose of the body frame of reference w.r.t. the global frame, expressed
///              in the global frame.
/// @param T_B_C Pose of the camera w.r.t. the body frame expressed in the body frame of reference.
/// @param G_point Triangulated point in global frame.
/// @return Was the triangulation successful?
inline bool linearTriangulateFromNViews(
    const Aligned<std::vector, Eigen::Vector2d>::type& measurements,
    const Aligned<std::vector, aslam::Transformation>::type& T_G_B,
    const aslam::Transformation& T_B_C, Eigen::Vector3d* G_point) {
  CHECK_NOTNULL(G_point);
  CHECK_EQ(measurements.size(), T_G_B.size());
  if (measurements.size() < 2u) {
    return false;
  }

  const size_t rows = 3 * measurements.size();
  const size_t cols = 3 + measurements.size();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

  const Eigen::Matrix3d R_B_C = T_B_C.getRotationMatrix();

  // Fill in A and b.
  for (size_t i = 0; i < measurements.size(); ++i) {
    Eigen::Vector3d v(measurements[i](0), measurements[i](1), 1.);
    Eigen::Matrix3d R_G_B = T_G_B[i].getRotationMatrix();
    Eigen::Vector3d G_p_B = T_G_B[i].getPosition();
    A.block<3, 3>(3 * i, 0) = Eigen::Matrix3d::Identity();
    A.block<3, 1>(3 * i, 3 + i) = -R_G_B * R_B_C * v;
    b.segment<3>(3 * i) = G_p_B + R_G_B * T_B_C.getPosition();
  }

  // Check that the disparity angles between the rays are sufficiently large for triangulation.
  Eigen::Vector3d G_ray_mean;
  G_ray_mean.setZero();

  for (size_t i = 0; i < measurements.size(); ++i) {
    G_ray_mean += A.block<3, 1>(3 * i, 3 + i);
  }
  G_ray_mean /= measurements.size();
  G_ray_mean.normalize();

  static constexpr double kRayAngleDisparityThreshold = 0.1 / 180.0 * M_PI;
  bool condition_satisfied = false;
  for (size_t i = 0; i < measurements.size(); ++i) {
    const double disparity_angle = std::acos(
        G_ray_mean.dot(A.block<3, 1>(3 * i, 3 + i).normalized()));
    if (disparity_angle >= kRayAngleDisparityThreshold) {
      condition_satisfied = true;
      break;
    }
  }
  if (!condition_satisfied) {
    return false;
  }

  *G_point = A.colPivHouseholderQr().solve(b).head<3>();
  return true;
}
}  // namespace aslam
#endif  // TRIANGULATION_H_
