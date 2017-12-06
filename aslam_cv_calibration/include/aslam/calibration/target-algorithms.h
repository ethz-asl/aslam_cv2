#ifndef ASLAM_CALIBRATION_TARGET_ALGORITHMS_H
#define ASLAM_CALIBRATION_TARGET_ALGORITHMS_H

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>

#include "aslam/calibration/target-observation.h"

namespace aslam {
namespace calibration {

// TODO(fabianbl): Add description.
// TODO(fabianbl): Add option to reproject keypoints.
bool estimateTargetTransformation(
    const TargetObservation& target_observation,
    const aslam::Camera::ConstPtr& camera_ptr, aslam::Transformation* T_G_C);

bool estimateTargetTransformation(
    const TargetObservation& target_observation,
    const aslam::Camera::ConstPtr& camera_ptr,
    const bool run_nonlinear_refinement, const double ransac_pixel_sigma,
    const int ransac_max_iters, aslam::Transformation* T_G_C);

}  // namespace calibration
}  // namespace aslam

#endif  // ASLAM_CALIBRATION_TARGET_ALGORITHMS_H
