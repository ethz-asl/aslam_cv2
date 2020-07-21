#ifndef GEOMETRIC_VISION_PNP_POSE_ESTIMATOR_H_
#define GEOMETRIC_VISION_PNP_POSE_ESTIMATOR_H_

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/ncamera.h>
#include <aslam/common/pose-types.h>

namespace aslam {
namespace geometric_vision {

class PnpPoseEstimator {
 public:
  explicit PnpPoseEstimator(bool run_nonlinear_refinement)
      : random_seed_(true),
        run_nonlinear_refinement_(run_nonlinear_refinement) {}
  /// This constructor should be used for when a deterministic seed (set from
  /// outside) is necessary, such as for testing.
  PnpPoseEstimator(bool run_nonlinear_refinement, bool random_seed)
      : random_seed_(random_seed),
        run_nonlinear_refinement_(run_nonlinear_refinement) {}

  /// The pinhole variants of these methods are wrappers that determine an
  /// appropriate ransac_threshold from pixel_sigma and camera focal lengths
  /// for pinhole cameras.
  bool absolutePoseRansacPinholeCam(
      const Eigen::Matrix2Xd& measurements,
      const Eigen::Matrix3Xd& G_landmark_positions, double pixel_sigma,
      int max_ransac_iters, aslam::Camera::ConstPtr camera_ptr,
      aslam::Transformation* T_G_C, std::vector<int>* inliers, int* num_iters);

  bool absoluteMultiPoseRansacPinholeCam(
      const Eigen::Matrix2Xd& measurements,
      const std::vector<int>& measurement_camera_indices,
      const Eigen::Matrix3Xd& G_landmark_positions, double pixel_sigma,
      int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
      aslam::Transformation* T_G_I, std::vector<int>* inliers, int* num_iters);
  bool absoluteMultiPoseRansacPinholeCam(
      const Eigen::Matrix2Xd& measurements,
      const std::vector<int>& measurement_camera_indices,
      const Eigen::Matrix3Xd& G_landmark_positions, double pixel_sigma,
      int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
      aslam::Transformation* T_G_I, std::vector<int>* inliers,
      std::vector<double>* inlier_distances_to_model, int* num_iters);

  bool absolutePoseRansac(const Eigen::Matrix2Xd& measurements,
                          const Eigen::Matrix3Xd& G_landmark_positions,
                          double pixel_sigma, int max_ransac_iters,
                          aslam::Camera::ConstPtr camera_ptr,
                          aslam::Transformation* T_G_C,
                          std::vector<int>* inliers, int* num_iters);

  /// Same as the above functions, but supports multiple cameras. Only additional
  /// information is the NCamera (instead of Camera) pointer and a vector,
  /// measurement_camera_indices, of the same length as measurements
  /// that maps each measurement to a camera index (corresponding to the index
  /// in NCamera).
  bool absoluteMultiPoseRansac(
      const Eigen::Matrix2Xd& measurements,
      const std::vector<int>& measurement_camera_indices,
      const Eigen::Matrix3Xd& G_landmark_positions, double ransac_threshold,
      int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
      aslam::Transformation* T_G_I, std::vector<int>* inliers, int* num_iters);
  bool absoluteMultiPoseRansac(
      const Eigen::Matrix2Xd& measurements,
      const std::vector<int>& measurement_camera_indices,
      const Eigen::Matrix3Xd& G_landmark_positions, double ransac_threshold,
      int max_ransac_iters, aslam::NCamera::ConstPtr ncamera_ptr,
      aslam::Transformation* T_G_I, std::vector<int>* inliers,
      std::vector<double>* inlier_distances_to_model, int* num_iters);


 private:
  /// Whether to let RANSAC pick a timestamp-based random seed or not. If false,
  /// a seed can be set with srand().
  const bool random_seed_;

  /// Run nonlinear refinement over all inliers.
  const bool run_nonlinear_refinement_;
};

}  // namespace geometric_vision
}  // namespace aslam

#endif  // GEOMETRIC_VISION_PNP_POSE_ESTIMATOR_H_
