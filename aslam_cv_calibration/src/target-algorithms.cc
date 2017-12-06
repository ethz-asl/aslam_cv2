#include "aslam/calibration/target-algorithms.h"

#include <Eigen/Core>
#include <aslam/geometric-vision/pnp-pose-estimator.h>

namespace aslam {
namespace calibration {

bool estimateTargetTransformation(
    const TargetObservation& target_observation,
    const aslam::Camera::ConstPtr& camera_ptr, aslam::Transformation* T_G_C) {
  const Eigen::Matrix2Xd& observed_corners =
      target_observation.getObservedCorners();
  const Eigen::Matrix3Xd G_corner_positions =
      target_observation.getCorrespondingTargetPoints();

  // TODO(fabianbl): Create second function which takes these as input args.
  constexpr bool kRunNonlinearRefinement = true;
  const double kPixelSigma = 1.0;
  const int kMaxRansacIters = 200;
  aslam::geometric_vision::PnpPoseEstimator pnp(kRunNonlinearRefinement);

  std::vector<int> inliers;
  int num_iters = 0;
  // bool pnp_success = pnp.absolutePoseRansacPinholeCam(
  // keypoints_measured, G_corner_positions, kPixelSigma, kMaxRansacIters,
  // camera, &T_G_C, &inliers, &num_iters);
}

}  // namespace calibration
}  // namespace aslam
