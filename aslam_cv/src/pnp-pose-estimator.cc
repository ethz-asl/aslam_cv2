#include "geometric-vision/pnp-pose-estimator.h"

#include <vector>

#include <multiagent_mapping_common/aligned_allocation.h>

namespace geometric_vision {

void PnpPoseEstimator::absolutePoseRansacPinholeCam(
    const Eigen::Matrix2Xd& measurements,
    const Eigen::Matrix3Xd& landmark_positions, double pixel_sigma,
    unsigned int max_ransac_iters, aslam::Camera::ConstPtr camera_ptr,
    pose::Transformation* G_T_C, std::vector<int>* inliers,
    unsigned int* num_iters) {
  CHECK_NOTNULL(G_T_C);
  CHECK_NOTNULL(inliers);
  CHECK_NOTNULL(num_iters);
  CHECK_EQ(measurements.cols(), landmark_positions.cols());

  // This method is designed for pinhole camera, so we should be able to cast
  // the base camera ptr to the derived pinhole type.
  aslam::PinholeCamera::ConstPtr pinhole_camera_ptr =
      std::dynamic_pointer_cast<const aslam::PinholeCamera>(camera_ptr);
  CHECK(pinhole_camera_ptr)
      << "Couldn't cast camera pointer to pinhole camera type.";

  // Assuming the mean of lens focal lengths is the best estimate here.
  const double focal_length =
      (pinhole_camera_ptr->fu() + pinhole_camera_ptr->fv()) / 2.0;
  const double ransac_threshold = 1.0 - cos(atan(pixel_sigma / focal_length));

  absolutePoseRansac(measurements, landmark_positions, ransac_threshold,
                     max_ransac_iters, camera_ptr, G_T_C, inliers, num_iters);
}

void PnpPoseEstimator::absolutePoseRansac(
    const Eigen::Matrix2Xd& measurements,
    const Eigen::Matrix3Xd& landmark_positions, double ransac_threshold,
    unsigned int max_ransac_iters, aslam::Camera::ConstPtr camera_ptr,
    pose::Transformation* G_T_C, std::vector<int>* inliers,
    unsigned int* num_iters) {
  CHECK_NOTNULL(G_T_C);
  CHECK_NOTNULL(inliers);
  CHECK_NOTNULL(num_iters);
  CHECK_EQ(measurements.cols(), landmark_positions.cols());

  opengv::points_t points;
  opengv::bearingVectors_t bearing_vectors;
  points.resize(measurements.cols());
  bearing_vectors.resize(measurements.cols());
  for (unsigned int i = 0; i < measurements.cols(); ++i) {
    camera_ptr->keypointToEuclidean(measurements.col(i),
                                    &bearing_vectors[i]);
    bearing_vectors[i].normalize();
    points[i] = landmark_positions.col(i);
  }

  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
                                                        points);
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  boost::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::GAO));
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ = ransac_threshold;
  ransac.max_iterations_ = max_ransac_iters;
  ransac.computeModel();

  G_T_C->getPosition() = ransac.model_coefficients_.rightCols(1);
  Eigen::Matrix3d C_R_G = ransac.model_coefficients_.leftCols(3);

  G_T_C->getRotation().toImplementation() =
      Eigen::Quaterniond(C_R_G.transpose());

  *inliers = ransac.inliers_;
  *num_iters = static_cast<unsigned int>(ransac.iterations_);
}

}  // namespace geometric_vision
