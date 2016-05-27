#include "aslam/tracker/feature-tracker-gyro-matching-data.h"

#include <aslam/tracker/tracking-helpers.h>
#include <glog/logging.h>

namespace aslam {

KeypointData::KeypointData(const Eigen::Vector2d& measurement, const int& index)
  : measurement(measurement), index(index) {}

GyroTrackerMatchingData::GyroTrackerMatchingData(
    const Quaternion& q_Ckp1_Ck,
    const VisualFrame& frame_kp1,
    const VisualFrame& frame_k)
  : frame_kp1(frame_kp1), frame_k(frame_k), q_Ckp1_Ck(q_Ckp1_Ck),
    descriptor_size_bytes_(frame_kp1.getDescriptorSizeBytes()),
    num_points_kp1(frame_kp1.getKeypointMeasurements().cols()),
    num_points_k(frame_k.getKeypointMeasurements().cols()) {
  CHECK(frame_kp1.isValid());
  CHECK(frame_k.isValid());
  CHECK(frame_kp1.hasDescriptors());
  CHECK(frame_k.hasDescriptors());
  CHECK(frame_kp1.hasKeypointMeasurements());
  CHECK(frame_k.hasKeypointMeasurements());
  CHECK_GT(num_points_kp1, 0);
  CHECK_GT(num_points_k, 0);
  CHECK_EQ(num_points_kp1, frame_kp1.getDescriptors().cols()) <<
      "Number of keypoints and descriptors in frame k+1 is not the same.";
  CHECK_EQ(num_points_k, frame_k.getDescriptors().cols()) <<
      "Number of keypoints and descriptors in frame k is not the same.";

  setupData();
}

void GyroTrackerMatchingData::setupData() {
  // Predict keypoint positions.
  predictKeypointsByRotation(frame_k, q_Ckp1_Ck, &predicted_keypoint_positions_kp1, &prediction_success);

  // Prepare descriptors for efficient matching.
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_kp1 =
      frame_kp1.getDescriptors();
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_k =
      frame_k.getDescriptors();

  descriptors_kp1_wrapped.clear();
  descriptors_k_wrapped.clear();
  descriptors_kp1_wrapped.reserve(num_points_kp1);
  descriptors_k_wrapped.reserve(num_points_k);

  for (int descriptor_kp1_idx = 0; descriptor_kp1_idx < num_points_kp1;
      ++descriptor_kp1_idx) {
    descriptors_kp1_wrapped.emplace_back(
        &(descriptors_kp1.coeffRef(0, descriptor_kp1_idx)), descriptor_size_bytes_);
  }

  for (int descriptor_k_idx = 0; descriptor_k_idx < num_points_k;
      ++descriptor_k_idx) {
    descriptors_k_wrapped.emplace_back(
        &(descriptors_k.coeffRef(0, descriptor_k_idx)), descriptor_size_bytes_);
  }

  // Sort keypoints of frame (k+1) from small to large y coordinates.
  keypoints_kp1_by_y.reserve(num_points_kp1);

  for (int i = 0; i < num_points_kp1; ++i) {
    keypoints_kp1_by_y.emplace_back(frame_kp1.getKeypointMeasurement(i), i);
  }

  std::sort(keypoints_kp1_by_y.begin(), keypoints_kp1_by_y.end(),
            [](const KeypointData& lhs, const KeypointData& rhs)-> bool {
              return lhs.measurement(1) < rhs.measurement(1);
            });
}


} // namespace aslam
