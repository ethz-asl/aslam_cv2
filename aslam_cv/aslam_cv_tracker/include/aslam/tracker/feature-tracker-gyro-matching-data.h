#ifndef ASLAM_TRACKER_FEATURE_TRACKER_GYRO_MATCHING_DATA_H_
#define ASLAM_TRACKER_FEATURE_TRACKER_GYRO_MATCHING_DATA_H_

#include <Eigen/Core>
#include <vector>

#include <aslam/frames/visual-frame.h>
#include <aslam/common/pose-types.h>
#include <aslam/common-private/feature-descriptor-ref.h>

namespace aslam {

struct KeypointData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KeypointData(const Eigen::Vector2d& measurement, const int& index);
  Eigen::Vector2d measurement;
  int index;
};

struct GyroTrackerMatchingData {
  GyroTrackerMatchingData(const Quaternion& q_Ckp1_Ck,
                          const VisualFrame& frame_kp1,
                          const VisualFrame& frame_k);
  // The current frame.
  const VisualFrame& frame_kp1;
  // The previous frame.
  const VisualFrame& frame_k;
  // Rotation matrix that describes the camera rotation between the
  // two frames that are matched.
  const Quaternion& q_Ckp1_Ck;
  /// Descriptor size in bytes.
  const size_t descriptor_size_bytes_;
  /// Number of keypoints/descriptors in frame (k+1).
  const int num_points_kp1;
  /// Number of keypoints/descriptors in frame k.
  const int num_points_k;
  // Predicted locations of the keypoints in frame k
  // in frame (k+1) based on camera rotation.
  Eigen::Matrix2Xd predicted_keypoint_positions_kp1;
  // Index marking keypoints of frame (k+1) as valid or invalid
  // depending on the success of the prediction.
  std::vector<unsigned char> prediction_success;
  // Descriptors of frame (k+1).
  std::vector<common::FeatureDescriptorConstRef> descriptors_kp1_wrapped;
  // Descriptors of frame k.
  std::vector<common::FeatureDescriptorConstRef> descriptors_k_wrapped;
  // Keypoints of frame (k+1) sorted from small to large y coordinates.
  Aligned<std::vector, KeypointData>::type keypoints_kp1_by_y;

 private:
  void setupData();
};

} // namespace aslam

#endif // ASLAM_TRACKER_FEATURE_TRACKER_GYRO_MATCHING_DATA_H_
