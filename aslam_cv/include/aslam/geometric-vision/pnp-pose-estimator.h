#ifndef GEOMETRIC_VISION_PNP_POSE_ESTIMATOR_H_
#define GEOMETRIC_VISION_PNP_POSE_ESTIMATOR_H_

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/pinhole-camera.h>
#include <multiagent_mapping_common/pose_types.h>
#include <multiagent_mapping_common/quaternion-math.h>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

namespace geometric_vision {

class PnpPoseEstimator {
 public:
  void absolutePoseRansac(const Eigen::Matrix2Xd& measurements,
                          const Eigen::Matrix3Xd& landmark_positions,
                          double pixel_sigma, unsigned int max_ransac_iters,
                          aslam::Camera::ConstPtr camera_ptr,
                          pose::Transformation* G_T_C,
                          std::vector<int>* inliers, unsigned int* num_iters);

  void absolutePoseRansacPinholeCam(const Eigen::Matrix2Xd& measurements,
                                    const Eigen::Matrix3Xd& landmark_positions,
                                    double pixel_sigma,
                                    unsigned int max_ransac_iters,
                                    aslam::Camera::ConstPtr camera_ptr,
                                    pose::Transformation* G_T_C,
                                    std::vector<int>* inliers,
                                    unsigned int* num_iters);
};

}  // namespace geometric_vision

#endif  // GEOMETRIC_VISION_PNP_POSE_ESTIMATOR_H_
