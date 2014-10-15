#ifndef FIVE_POINT_POSE_ESTIMATOR_H_
#define FIVE_POINT_POSE_ESTIMATOR_H_

#include <Eigen/Core>
#include <glog/logging.h>

#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-pinhole.h>

namespace aslam {
namespace geometric_vision {

struct FivePointPoseEstimator {
  // Only static methods, no instances allowed!
  FivePointPoseEstimator() = delete;

  static int computePinhole(const Eigen::Matrix2Xd& measurements_a,
                            const Eigen::Matrix2Xd& measurements_b,
                            std::shared_ptr<Camera> camera_ptr, double pixel_sigma,
                            unsigned int max_ransac_iters, aslam::Transformation* output_transform);

  static int compute(const Eigen::Matrix2Xd& measurements_a, const Eigen::Matrix2Xd& measurements_b,
                     std::shared_ptr<Camera> camera_A, std::shared_ptr<Camera> camera_B,
                     double ransac_threshold, unsigned int max_ransac_iters,
                     aslam::Transformation* output_transform);
};

}  // namespace geometric_vision
}  // namespace aslam

#endif  // FIVE_POINT_POSE_ESTIMATOR_H_
