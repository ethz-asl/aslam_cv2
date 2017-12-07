#include "aslam/calibration/target-algorithms.h"

#include <Eigen/Core>
#include <aslam/geometric-vision/pnp-pose-estimator.h>

namespace aslam {
namespace calibration {

bool estimateTargetTransformation(
    const TargetObservation& target_observation,
    const aslam::Camera::ConstPtr& camera_ptr, aslam::Transformation* T_G_C) {
  constexpr bool kRunNonlinearRefinement = true;
  constexpr double kRansacPixelSigma = 1.0;
  constexpr int kRansacMaxIters = 200;
  return estimateTargetTransformation(
      target_observation, camera_ptr, kRunNonlinearRefinement,
      kRansacPixelSigma, kRansacMaxIters, T_G_C);
}

bool estimateTargetTransformation(
    const TargetObservation& target_observation,
    const aslam::Camera::ConstPtr& camera_ptr,
    const bool run_nonlinear_refinement, const double ransac_pixel_sigma,
    const int ransac_max_iters, aslam::Transformation* T_G_C) {
  CHECK(camera_ptr);
  CHECK_GT(ransac_pixel_sigma, 0.0);
  CHECK_GT(ransac_max_iters, 0);
  CHECK_NOTNULL(T_G_C);
  const Eigen::Matrix2Xd& observed_corners =
      target_observation.getObservedCorners();
  const Eigen::Matrix3Xd G_corner_positions =
      target_observation.getCorrespondingTargetPoints();
  aslam::geometric_vision::PnpPoseEstimator pnp(run_nonlinear_refinement);
  std::vector<int> inliers;
  int num_iters = 0;
  bool pnp_success = pnp.absolutePoseRansacPinholeCam(
      observed_corners, G_corner_positions, ransac_pixel_sigma,
      ransac_max_iters, camera_ptr, T_G_C, &inliers, &num_iters);
  VLOG(3) << "Num inliers: " << inliers.size() << "/"
          << observed_corners.cols();
  return pnp_success;
}

}  // namespace calibration
}  // namespace aslam
