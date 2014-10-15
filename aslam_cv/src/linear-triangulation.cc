#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/pose-types.h>
#include <aslam/geometric-vision/linear-triangulation.h>

namespace aslam {
namespace geometric_vision {

bool triangulateFromNormalizedTwoViewsHomogeneous(
    const Eigen::Vector2d& measurement0,
    const aslam::Transformation& camera_pose0,
    const Eigen::Vector2d& measurement1,
    const aslam::Transformation& camera_pose1,
    Eigen::Vector3d* triangulated_point) {
  CHECK_NOTNULL(triangulated_point);

  const Eigen::Vector3d& G_p_C0 = camera_pose0.getPosition();
  const Eigen::Vector3d& G_p_C1 = camera_pose1.getPosition();
  const Eigen::Quaterniond& G_q_C0 =
      camera_pose0.getRotation().toImplementation();
  const Eigen::Quaterniond& G_q_C1 =
      camera_pose1.getRotation().toImplementation();
  Eigen::Matrix3d G_R_C0, G_R_C1;
  common::toRotationMatrix(G_q_C0.coeffs(), &G_R_C0);
  common::toRotationMatrix(G_q_C1.coeffs(), &G_R_C1);

  Eigen::Matrix<double, 3, 4> P0, P1;
  P0.block<3, 3>(0, 0) = G_R_C0;
  P1.block<3, 3>(0, 0) = G_R_C1;
  P0.block<3, 1>(0, 3) = G_p_C0;
  P1.block<3, 1>(0, 3) = G_p_C1;

  Eigen::Matrix<double, 4, 4> A;
  A.row(0) = measurement0(0) * P0.row(2) - P0.row(0);
  A.row(1) = measurement0(1) * P0.row(2) - P0.row(1);
  A.row(2) = measurement1(0) * P1.row(2) - P1.row(0);
  A.row(3) = measurement1(1) * P1.row(2) - P1.row(1);

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
  Eigen::Vector4d triangulated_point_homogeneous =
      svd.matrixV().rightCols<1>();

  if (triangulated_point_homogeneous[3] == 0) {
    return false;
  }

  *triangulated_point = triangulated_point_homogeneous.hnormalized().head<3>();
  return true;
}

bool triangulateFromNormalizedTwoViews(
    const Eigen::Vector2d& measurement0,
    const aslam::Transformation& camera_pose0,
    const Eigen::Vector2d& measurement1,
    const aslam::Transformation& camera_pose1,
    Eigen::Vector3d* triangulated_point) {
  CHECK_NOTNULL(triangulated_point);

  const Eigen::Vector3d& G_p_C0 = camera_pose0.getPosition();
  const Eigen::Vector3d& G_p_C1 = camera_pose1.getPosition();
  const Eigen::Quaterniond& G_q_C0 =
      camera_pose0.getRotation().toImplementation();
  const Eigen::Quaterniond& G_q_C1 =
      camera_pose1.getRotation().toImplementation();
  Eigen::Matrix3d G_R_C0, G_R_C1;
  common::toRotationMatrix(G_q_C0.coeffs(), &G_R_C0);
  common::toRotationMatrix(G_q_C1.coeffs(), &G_R_C1);

  Eigen::Matrix<double, 4, 3> A;
  Eigen::Vector4d b;

  A.row(0) = measurement0(0) * G_R_C0.row(2) - G_R_C0.row(0);
  A.row(1) = measurement0(1) * G_R_C0.row(2) - G_R_C0.row(1);
  A.row(2) = measurement1(0) * G_R_C1.row(2) - G_R_C1.row(0);
  A.row(3) = measurement1(1) * G_R_C1.row(2) - G_R_C1.row(1);

  b(0) = -(measurement0(0) * G_p_C0(2) - G_p_C0(0));
  b(1) = -(measurement0(1) * G_p_C0(2) - G_p_C0(1));
  b(2) = -(measurement1(0) * G_p_C1(2) - G_p_C1(0));
  b(3) = -(measurement1(1) * G_p_C1(2) - G_p_C1(1));

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.nonzeroSingularValues() < 3) {
    VLOG(3) << "At least one singular value is zero";
    return false;
  }

  double condition_number;
  const Eigen::Vector3d singular_values = svd.singularValues();
  condition_number = singular_values.maxCoeff() / singular_values.minCoeff();
  VLOG(3) << "Condition number: " << condition_number;

  *triangulated_point = svd.solve(b);
  return true;
}

bool triangulateFromNormalizedNViews(
    const Aligned<std::vector, Eigen::Vector2d>::type& measurements,
    const Aligned<std::vector, aslam::Transformation>::type& camera_poses,
    Eigen::Vector3d* triangulated_point) {
  CHECK_NOTNULL(triangulated_point);
  CHECK_EQ(measurements.size(), camera_poses.size());
  CHECK_GE(measurements.size(), 2u);

  Eigen::Matrix<double, Eigen::Dynamic, 3> A;
  Eigen::VectorXd b;
  A.resize(2 * measurements.size(), Eigen::NoChange);
  b.resize(2 * measurements.size());

  for (unsigned int i = 0; i < measurements.size(); ++i) {
    Eigen::Matrix3d G_R_C;
    common::toRotationMatrix(
        camera_poses[i].getRotation().toImplementation().coeffs(), &G_R_C);
    const Eigen::Vector3d& G_p_C = camera_poses[i].getPosition();

    A.row(2 * i + 0) = measurements[i](0) * G_R_C.row(2) - G_R_C.row(0);
    A.row(2 * i + 1) = measurements[i](1) * G_R_C.row(2) - G_R_C.row(1);

    b(2 * i + 0) = -(measurements[i](0) * G_p_C(2) - G_p_C(0));
    b(2 * i + 1) = -(measurements[i](1) * G_p_C(2) - G_p_C(1));
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      A, Eigen::ComputeThinU | Eigen::ComputeThinV);
  *triangulated_point = svd.solve(b);

  return true;
}

} // namespace geometric_vision
} // namespace aslam
