#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/common/memory.h>
#include <aslam/common/statistics/statistics.h>
#include <aslam/common-private/feature-descriptor-ref.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker-gyro.h>
#include <aslam/tracker/feature-tracker-gyro-matching-data.h>
#include <aslam/tracker/tracking-helpers.h>

namespace aslam {

GyroTracker::GyroTracker(const Camera& camera)
    : camera_(camera) {}

void GyroTracker::track(const Quaternion& q_Ckp1_Ck,
                        const VisualFrame& frame_k,
                        VisualFrame* frame_kp1,
                        MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(frame_k.hasKeypointMeasurements());
  CHECK(CHECK_NOTNULL(frame_kp1)->hasKeypointMeasurements());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(frame_k.getCameraGeometry().get())->getId());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(frame_kp1->getCameraGeometry().get())->getId());
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();
  CHECK(frame_k.hasTrackIds());
  CHECK(frame_kp1->hasTrackIds());
  // Make sure the frames are in order time-wise
  CHECK_GT(frame_kp1->getTimestampNanoseconds(), frame_k.getTimestampNanoseconds());
  // Check that the required data is available in the frame
  CHECK(frame_kp1->hasDescriptors());
  CHECK_EQ(frame_kp1->getDescriptors().rows(), frame_kp1->getDescriptorSizeBytes());
  CHECK_EQ(frame_kp1->getKeypointMeasurements().cols(), frame_kp1->getDescriptors().cols());

  // Match the descriptors of frame (k+1) with those of frame k.
  matchFeatures(q_Ckp1_Ck, *frame_kp1, frame_k, matches_with_score_kp1_k);
}

void GyroTracker::matchFeatures(const Quaternion& q_Ckp1_Ck,
                                const VisualFrame& frame_kp1,
                                const VisualFrame& frame_k,
                                MatchesWithScore* matches_with_score_kp1_k) const {
  CHECK_NOTNULL(matches_with_score_kp1_k);
  matches_with_score_kp1_k->clear();

  GyroTrackerMatchingData matching_data(
      q_Ckp1_Ck, frame_kp1, frame_k);

  // corner_row_LUT[i] is the number of keypoints that has an y position
  // lower than i in the image.
  std::vector<int> corner_row_LUT;
  const uint32_t image_height = camera_.imageHeight();
  corner_row_LUT.reserve(image_height);
  int v = 0;
  for (size_t y = 0; y < image_height; ++y) {
    while (v < static_cast<int>(matching_data.num_points_kp1) &&
        y > matching_data.keypoints_kp1_by_y[v].measurement(1)) {
      ++v;
    }
    corner_row_LUT.push_back(v);
  }
  CHECK_EQ(static_cast<int>(corner_row_LUT.size()), image_height);

  const static unsigned int kdescriptorSizeBytes = matching_data.descriptor_size_bytes_;
  // usually binary descriptors size is less or equal to 512 bits.
  CHECK_LE(kdescriptorSizeBytes * 8, 512u);

  matches_with_score_kp1_k->reserve(matching_data.num_points_k);

  // Keep track of matched keypoints of frame (k+1) such that they
  // are not matched again.
  // TODO(magehrig): Improve this by allowing duplicate matches
  // and discarding duplicate matches according to descriptor distances.
  // TODO(magehrig): Use prediction success/fail info.
  std::vector<bool> is_keypoint_kp1_matched;
  is_keypoint_kp1_matched.resize(matching_data.num_points_kp1, false);
  size_t number_of_matches = 0u;

  for (int i = 0; i < matching_data.num_points_k; ++i) {
    Eigen::Matrix<double, 2, 1> predicted_keypoint_position_kp1 =
        matching_data.predicted_keypoint_positions_kp1.block<2, 1>(0, i);
    const common::FeatureDescriptorConstRef& descriptor_k =
        matching_data.descriptors_k_wrapped.at(i);

    // Get search area for LUT iterators (rowwise).
    int idxnearest[2];  // Min search region.
    idxnearest[0] = clamp(0, image_height - 1, predicted_keypoint_position_kp1(1) + 0.5 - kSmallSearchDistance);
    idxnearest[1] = clamp(0, image_height - 1, predicted_keypoint_position_kp1(1) + 0.5 + kSmallSearchDistance);
    int idxnear[2];  // Max search region.
    idxnear[0] = clamp(0, image_height - 1, predicted_keypoint_position_kp1(1) + 0.5 - kLargeSearchDistance);
    idxnear[1] = clamp(0, image_height - 1, predicted_keypoint_position_kp1(1) + 0.5 + kLargeSearchDistance);

    CHECK_LE(idxnearest[0], idxnearest[1]);
    CHECK_LE(idxnear[0], idxnear[1]);

    CHECK_GE(idxnearest[0], 0);
    CHECK_GE(idxnearest[1], 0);
    CHECK_GE(idxnear[0], 0);
    CHECK_GE(idxnear[1], 0);
    CHECK_LT(idxnearest[0], image_height);
    CHECK_LT(idxnearest[1], image_height);
    CHECK_LT(idxnear[0], image_height);
    CHECK_LT(idxnear[1], image_height);

    int nearest_top = std::min<int>(idxnearest[0], image_height - 1);
    int nearest_bottom = std::min<int>(idxnearest[1] + 1, image_height - 1);
    int near_top = std::min<int>(idxnear[0], image_height - 1);
    int near_bottom = std::min<int>(idxnear[1] + 1, image_height - 1);

    // Get corners in this area.
    typedef typename Aligned<std::vector, KeypointData>::type::const_iterator KeyPointIterator;
    KeyPointIterator nearest_corners_begin = matching_data.keypoints_kp1_by_y.begin() + corner_row_LUT[nearest_top];
    KeyPointIterator nearest_corners_end = matching_data.keypoints_kp1_by_y.begin() + corner_row_LUT[nearest_bottom];
    KeyPointIterator near_corners_begin = matching_data.keypoints_kp1_by_y.begin() + corner_row_LUT[near_top];
    KeyPointIterator near_corners_end = matching_data.keypoints_kp1_by_y.begin() + corner_row_LUT[near_bottom];

    // Get descriptors and match.
    bool found = false;
    int n_processed_corners = 0;
    KeyPointIterator it_best;
    const static unsigned int kdescriptorSizeBits = kdescriptorSizeBytes*8;
    int best_score = static_cast<int>(kdescriptorSizeBits*kMatchingThresholdBitsRatio);
    // Keep track of processed corners s.t. we don't process them again in the
    // large window.
    std::vector<bool> processed_corners_kp1;
    processed_corners_kp1.resize(matching_data.num_points_kp1, false);

    const int bound_left_nearest = predicted_keypoint_position_kp1(0) - kSmallSearchDistance;
    const int bound_right_nearest = predicted_keypoint_position_kp1(0) + kSmallSearchDistance;

    // First search small window.
    for (KeyPointIterator it = nearest_corners_begin; it != nearest_corners_end; ++it) {
      if (it->measurement(0) < bound_left_nearest
          || it->measurement(0) > bound_right_nearest) {
        continue;
      }
      if (is_keypoint_kp1_matched.at(it->index)) {
        continue;
      }

      CHECK_LT(it->index, matching_data.num_points_kp1);
      CHECK_GE(it->index, 0u);
      const common::FeatureDescriptorConstRef& descriptor_kp1 =
          matching_data.descriptors_kp1_wrapped.at(it->index);
      int current_score = kdescriptorSizeBits - common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
      if (current_score > best_score) {
        best_score = current_score;
        it_best = it;
        found = true;
        CHECK_LT((predicted_keypoint_position_kp1 - it_best->measurement).norm(), kSmallSearchDistance * 2);
      }
      processed_corners_kp1[it->index] = true;
      ++n_processed_corners;
    }

    // If no match in small window, increase window and search again.
    if (!found) {
      const int bound_left_near = predicted_keypoint_position_kp1(0) - kLargeSearchDistance;
      const int bound_right_near = predicted_keypoint_position_kp1(0) + kLargeSearchDistance;

      for (KeyPointIterator it = near_corners_begin; it != near_corners_end; ++it) {
        if (processed_corners_kp1[it->index] || is_keypoint_kp1_matched.at(it->index)) {
          continue;
        }
        if (it->measurement(0) < bound_left_near || it->measurement(0) > bound_right_near) {
          continue;
        }
        CHECK_LT(it->index, matching_data.num_points_kp1);
        CHECK_GE(it->index, 0);
        const common::FeatureDescriptorConstRef& descriptor_kp1 =
            matching_data.descriptors_kp1_wrapped.at(it->index);
        int current_score = kdescriptorSizeBits - common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
        if (current_score > best_score) {
          best_score = current_score;
          it_best = it;
          found = true;
          CHECK_LT((predicted_keypoint_position_kp1 - it_best->measurement).norm(), kLargeSearchDistance * 2);
        }
        processed_corners_kp1[it->index] = true;
        ++n_processed_corners;
      }
    }

    if (found) {
      ++number_of_matches;
      is_keypoint_kp1_matched.at(it_best->index) = true;
      // The larger the matching score (which is smaller or equal to 1),
      // the higher the probability that a true match occurred.
      const double matching_score = static_cast<double>(best_score)/kdescriptorSizeBits;
      matches_with_score_kp1_k->emplace_back(
          static_cast<int>(it_best->index), i, matching_score);
      aslam::statistics::StatsCollector stats_distance_match("GyroTracker: number of matching bits");
      stats_distance_match.AddSample(best_score);
    } else {
      aslam::statistics::StatsCollector stats_count_processed("GyroTracker: number of processed keypoints per keypoint");
      stats_count_processed.AddSample(n_processed_corners);
    }
  }
  aslam::statistics::StatsCollector stats_number_of_matches("GyroTracker: number of matched keypoints");
  stats_number_of_matches.AddSample(number_of_matches);
  aslam::statistics::StatsCollector stats_number_of_no_matches("GyroTracker: number of unmatched keypoints");
  stats_number_of_no_matches.AddSample(matching_data.num_points_k - number_of_matches);
}

inline int GyroTracker::clamp(const int& lower, const int& upper, const int& in) const {
  return std::min<int>(std::max<int>(in, lower), upper);
}

}  //namespace aslam
