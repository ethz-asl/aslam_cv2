#ifndef ASLAM_CALIBRATION_TARGET_ALGORITHMS_H
#define ASLAM_CALIBRATION_TARGET_ALGORITHMS_H

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>

#include "aslam/calibration/target-observation.h"

namespace aslam {
namespace calibration {

// TODO(fabianbl): Add description.
bool estimateTargetTransformation(
    const TargetObservation& target_observation,
    const aslam::Camera::ConstPtr& camera_ptr, aslam::Transformation* T_G_C);

}  // namespace calibration
}  // namespace aslam

#endif  // ASLAM_CALIBRATION_TARGET_ALGORITHMS_H
