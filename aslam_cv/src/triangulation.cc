#include "aslam/triangulation/triangulation.h"

#include <Eigen/QR>
#include <glog/logging.h>

namespace aslam {

TriangulationResult::Status TriangulationResult::SUCCESSFUL =
    TriangulationResult::Status::kSuccessful;
TriangulationResult::Status TriangulationResult::TOO_FEW_MEASUREMENTS =
    TriangulationResult::Status::kTooFewMeasurments;
TriangulationResult::Status TriangulationResult::UNOBSERVABLE =
    TriangulationResult::Status::kUnobservable;
TriangulationResult::Status TriangulationResult::UNINITIALIZED =
    TriangulationResult::Status::kUninitialized;

TriangulationResult linearTriangulateFromNViews(
    const Aligned<std::vector, Eigen::Vector2d>::type& measurements_normalized,
    const Aligned<std::vector, aslam::Transformation>::type& T_G_B,
    const aslam::Transformation& T_B_C, Eigen::Vector3d* G_point) {
  CHECK_NOTNULL(G_point);
  CHECK_EQ(measurements_normalized.size(), T_G_B.size());
  if (measurements_normalized.size() < 2u) {
    return TriangulationResult(TriangulationResult::TOO_FEW_MEASUREMENTS);
  }

  const size_t rows = 3 * measurements_normalized.size();
  const size_t cols = 3 + measurements_normalized.size();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

  const Eigen::Matrix3d R_B_C = T_B_C.getRotationMatrix();

  // Fill in A and b.
  for (size_t i = 0; i < measurements_normalized.size(); ++i) {
    Eigen::Vector3d v(measurements_normalized[i](0),
        measurements_normalized[i](1), 1.);
    Eigen::Matrix3d R_G_B = T_G_B[i].getRotationMatrix();
    const Eigen::Vector3d& p_G_B = T_G_B[i].getPosition();
    A.block<3, 3>(3 * i, 0) = Eigen::Matrix3d::Identity();
    A.block<3, 1>(3 * i, 3 + i) = -R_G_B * R_B_C * v;
    b.segment<3>(3 * i) = p_G_B + R_G_B * T_B_C.getPosition();
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr = A.colPivHouseholderQr();
  static constexpr double kRankLossTolerance = 0.001;
  qr.setThreshold(kRankLossTolerance);
  const size_t rank = qr.rank();
  if ((rank - measurements_normalized.size()) < 3) {
    return TriangulationResult(TriangulationResult::UNOBSERVABLE);
  }

  *G_point = qr.solve(b).head<3>();
  return TriangulationResult(TriangulationResult::SUCCESSFUL);
}

TriangulationResult linearTriangulateFromNViews(
    const Eigen::Matrix3Xd& t_G_bv,
    const Eigen::Matrix3Xd& p_G_C,
    Eigen::Vector3d* p_G_P) {
  CHECK_NOTNULL(p_G_P);

  const int num_measurements = t_G_bv.cols();
  if (num_measurements < 2) {
    return TriangulationResult(TriangulationResult::TOO_FEW_MEASUREMENTS);
  }

  // 1.) Formulate the geometrical problem
  // p_G_P + alpha[i] * t_G_bv[i] = p_G_C[i]      (+ alpha intended)
  // as linear system Ax = b, where
  // x = [p_G_P; alpha[0]; alpha[1]; ... ] and b = [p_G_C[0]; p_G_C[1]; ...]
  //
  // 2.) Apply the approximation AtAx = Atb
  // AtA happens to be composed of mostly more convenient blocks than A:
  // - Top left = N * Eigen::Matrix3d::Identity()
  // - Top right and bottom left = t_G_bv
  // - Bottom right = t_G_bv.colwise().squaredNorm().asDiagonal()

  // - Atb.head(3) = p_G_C.rowwise().sum()
  // - Atb.tail(N) = columnwise dot products between t_G_bv and p_G_C
  //               = t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose()
  //
  // 3.) Apply the Schur complement to solve after p_G_P only
  // AtA = [E B; C D] (same blocks as above) ->
  // (E - B * D.inverse() * C) * p_G_P = Atb.head(3) - B * D.inverse() * Atb.tail(N)

  const Eigen::MatrixXd BiD = t_G_bv *
      t_G_bv.colwise().squaredNorm().asDiagonal().inverse();
  const Eigen::Matrix3d AxtAx = num_measurements * Eigen::Matrix3d::Identity() -
      BiD * t_G_bv.transpose();
  const Eigen::Vector3d Axtbx = p_G_C.rowwise().sum() - BiD *
      t_G_bv.cwiseProduct(p_G_C).colwise().sum().transpose();

  Eigen::ColPivHouseholderQR<Eigen::Matrix3d> qr = AxtAx.colPivHouseholderQr();
  static constexpr double kRankLossTolerance = 1e-5;
  qr.setThreshold(kRankLossTolerance);
  const size_t rank = qr.rank();
  if (rank < 3) {
    return TriangulationResult(TriangulationResult::UNOBSERVABLE);
  }

  *p_G_P = qr.solve(Axtbx);
  return TriangulationResult(TriangulationResult::SUCCESSFUL);
}

bool linearTriangulateFromNViewsMultiCam(
    const Aligned<std::vector, Eigen::Vector2d>::type& measurements_normalized,
    const std::vector<int>& measurement_camera_indices,
    const Aligned<std::vector, aslam::Transformation>::type& T_G_B,
    const Aligned<std::vector, aslam::Transformation>::type& T_B_C,
    Eigen::Vector3d* G_point) {
  CHECK_NOTNULL(G_point);
  CHECK_EQ(measurements_normalized.size(), T_G_B.size());
  CHECK_EQ(measurements_normalized.size(), measurement_camera_indices.size());
  if (measurements_normalized.size() < 2u) {
    return false;
  }

  const size_t rows = 3 * measurements_normalized.size();
  const size_t cols = 3 + measurements_normalized.size();
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);

  // Fill in A and b.
  for (size_t i = 0; i < measurements_normalized.size(); ++i) {
    int cam_index = measurement_camera_indices[i];
    CHECK_LT(cam_index, T_B_C.size());
    Eigen::Vector3d v(measurements_normalized[i](0),
        measurements_normalized[i](1), 1.);
    const Eigen::Vector3d& t_B_C = T_B_C[cam_index].getPosition();

    A.block<3, 3>(3 * i, 0) = Eigen::Matrix3d::Identity();
    A.block<3, 1>(3 * i, 3 + i) = -1.0 * T_G_B[i].getRotation().rotate(
        T_B_C[cam_index].getRotation().rotate(v));
    b.segment<3>(3 * i) = T_G_B[i] * t_B_C;
  }

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr = A.colPivHouseholderQr();
  static constexpr double kRankLossTolerance = 0.001;
  qr.setThreshold(kRankLossTolerance);
  const size_t rank = qr.rank();
  if ((rank - measurements_normalized.size()) < 3) {
    return false;
  }

  *G_point = qr.solve(b).head<3>();
  return true;
}

}  // namespace aslam

