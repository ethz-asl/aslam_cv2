#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/common/memory.h>
#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/tracker/feature-tracker-gyro.h>
#include <aslam/tracker/tracking-helpers.h>

namespace aslam {

GyroTracker::GyroTracker(const aslam::Camera& camera)
    : camera_(camera) {}

void GyroTracker::track(const aslam::Quaternion& q_Ckp1_Ck,
                        const aslam::VisualFrame& previous_frame,
                        aslam::VisualFrame* current_frame,
                        aslam::MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(previous_frame.hasKeypointMeasurements());
  CHECK(CHECK_NOTNULL(current_frame)->hasKeypointMeasurements());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(previous_frame.getCameraGeometry().get())->getId());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(current_frame->getCameraGeometry().get())->getId());
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();
  CHECK(previous_frame.hasTrackIds());
  CHECK(current_frame->hasTrackIds());
  // Make sure the frames are in order time-wise
  CHECK_GT(current_frame->getTimestampNanoseconds(), previous_frame.getTimestampNanoseconds());
  // Check that the required data is available in the frame
  CHECK(current_frame->hasDescriptors());
  CHECK_EQ(current_frame->getDescriptors().rows(), current_frame->getDescriptorSizeBytes());
  CHECK_EQ(current_frame->getKeypointMeasurements().cols(), current_frame->getDescriptors().cols());

  aslam::timing::Timer timer_tracking("FeatureTrackerLk: track");

  // Match the keypoints in the current frame to the previous one.
  matchFeatures(q_Ckp1_Ck, *current_frame, previous_frame, matches_with_score_kp1_k);
}

void GyroTracker::matchFeatures(const aslam::Quaternion& q_Ckp1_Ck,
                                const VisualFrame& current_frame,
                                const VisualFrame& previous_frame,
                                aslam::MatchesWithScore* matches_with_score_kp1_k) const {
  CHECK_NOTNULL(matches_with_score_kp1_k);
  matches_with_score_kp1_k->clear();

  struct KeypointAndData {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d measurement;
    double response;
    int index;
  };

  // Sort keypoints by y-coordinate.
  const int current_num_pts = current_frame.getKeypointMeasurements().cols();
  Aligned<std::vector, KeypointAndData>::type current_keypoints_by_y;
  current_keypoints_by_y.resize(current_num_pts);

  for (int i = 0; i < current_num_pts; ++i) {
    current_keypoints_by_y[i].measurement = current_frame.getKeypointMeasurement(i);
    current_keypoints_by_y[i].response = current_frame.getKeypointScore(i);
    current_keypoints_by_y[i].index = i;
  }

  std::sort(current_keypoints_by_y.begin(), current_keypoints_by_y.end(),
            [](const KeypointAndData& lhs, const KeypointAndData& rhs)->bool {
              return lhs.measurement(1) < rhs.measurement(1);
            });

  // corner_row_LUT[i] is the number of keypoints that has an y position
  // lower than i in the image.
  std::vector<int> corner_row_LUT;
  const uint32_t image_height = camera_.imageHeight();
  corner_row_LUT.reserve(image_height);
  int v = 0;
  for (size_t y = 0; y < image_height; ++y) {
    while (v < current_num_pts && y > current_keypoints_by_y[v].measurement(1)) {
      ++v;
    }
    corner_row_LUT.push_back(v);
  }
  CHECK_EQ(static_cast<int>(corner_row_LUT.size()), image_height);

  // Undistort and predict previous keypoints.
  const int prev_num_pts =  previous_frame.getKeypointMeasurements().cols();

  // Predict the keypoint locations from the frame (k) to the frame (k+1) using the rotation prior.
  // The initial keypoint location is kept if the prediction failed.
  Eigen::Matrix2Xd C2_previous_image_points;
  std::vector<unsigned char> prediction_success;
  predictKeypointsByRotation(previous_frame, q_Ckp1_Ck, &C2_previous_image_points, &prediction_success);

  const static unsigned int kdescriptorSizeBytes = current_frame.getDescriptorSizeBytes();
  // usually binary descriptors size is less or equal to 512 bits.
  CHECK_LE(kdescriptorSizeBytes * 8, 512);
  std::function<unsigned int(const unsigned char*, const unsigned char*)> hammingDistance512 =
      [kdescriptorSizeBytes](const unsigned char* x, const unsigned char* y)->unsigned int {
        unsigned int distance = 0;
        for(unsigned int i = 0; i < kdescriptorSizeBytes; i++) {
          unsigned char val = *(x + i) ^ *(y + i);
          while(val) {
            ++distance;
            val &= val - 1;
          }
        }
        CHECK_LE(distance, kdescriptorSizeBytes * 8);
        return distance;
      };

  matches_with_score_kp1_k->reserve(prev_num_pts);

  // Keep track of matched keypoints in the current frame such that they
  // are not matched again.
  // TODO(magehrig): Improve this by allowing duplicate matches
  // and discarding duplicate matches according to descriptor distances. Efficiency?
  std::vector<bool> is_current_keypoint_matched;
  is_current_keypoint_matched.resize(current_num_pts, false);

  for (int i = 0; i < prev_num_pts; ++i) {
    Eigen::Matrix<double, 2, 1> previous_predicted = C2_previous_image_points.block<2, 1>(0, i);
    const unsigned char* previous_descriptor = previous_frame.getDescriptor(i);

    const int bound_left_nearest = previous_predicted(0) - kSmallSearchDistance;
    const int bound_right_nearest = previous_predicted(0) + kSmallSearchDistance;
    const int bound_left_near = previous_predicted(0) - kLargeSearchDistance;
    const int bound_right_near = previous_predicted(0) + kLargeSearchDistance;

    std::function<int(int, int, int)> clamp = [](int lower, int upper, int in) {
      return std::min<int>(std::max<int>(in, lower), upper);
    };

    // Get search area for LUT iterators (rowwise).
    int idxnearest[2];  // Min search region.
    idxnearest[0] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 - kSmallSearchDistance);
    idxnearest[1] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 + kSmallSearchDistance);
    int idxnear[2];  // Max search region.
    idxnear[0] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 - kLargeSearchDistance);
    idxnear[1] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 + kLargeSearchDistance);

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
    typedef typename Aligned<std::vector, KeypointAndData>::type::const_iterator KeyPointIterator;
    KeyPointIterator nearest_corners_begin = current_keypoints_by_y.begin() + corner_row_LUT[nearest_top];
    KeyPointIterator nearest_corners_end = current_keypoints_by_y.begin() + corner_row_LUT[nearest_bottom];
    KeyPointIterator near_corners_begin = current_keypoints_by_y.begin() + corner_row_LUT[near_top];
    KeyPointIterator near_corners_end = current_keypoints_by_y.begin() + corner_row_LUT[near_bottom];

    // Get descriptors and match.
    bool found = false;
    int n_processed_corners = 0;
    KeyPointIterator it_best;
    const static unsigned int kdescriptorSizeBits = kdescriptorSizeBytes*8;
    int best_score = static_cast<int>(kdescriptorSizeBits*kMatchingThresholdBitsRatio);
    // Keep track of processed corners s.t. we don't process them again in the
    // large window.
    std::vector<bool> processed_current_corners;
    processed_current_corners.resize(current_num_pts, false);

    // First search small window.
    for (KeyPointIterator it = nearest_corners_begin; it != nearest_corners_end; ++it) {
      if (it->measurement(0) < bound_left_nearest
          || it->measurement(0) > bound_right_nearest) {
        continue;
      }
      if (is_current_keypoint_matched.at(it->index)) {
        continue;
      }

      CHECK_LT(it->index, current_num_pts);
      CHECK_GE(it->index, 0);
      const unsigned char* const current_descriptor = current_frame.getDescriptor(it->index);
      int current_score = kdescriptorSizeBits - hammingDistance512(previous_descriptor, current_descriptor);
      if (current_score > best_score) {
        best_score = current_score;
        it_best = it;
        found = true;
        CHECK_LT((previous_predicted - it_best->measurement).norm(), kSmallSearchDistance * 2);

        aslam::statistics::DebugStatsCollector stats_distance_match("GyroTracker distance to match min");
        stats_distance_match.AddSample((previous_predicted - it_best->measurement).norm());
      }
      processed_current_corners[it->index] = true;
      ++n_processed_corners;
    }

    // If no match in small window, increase window and search again.
    if (!found) {
      for (KeyPointIterator it = near_corners_begin; it != near_corners_end; ++it) {
        if (processed_current_corners[it->index] || is_current_keypoint_matched.at(it->index)) {
          continue;
        }
        if (it->measurement(0) < bound_left_near || it->measurement(0) > bound_right_near) {
          continue;
        }
        CHECK_LT(it->index, current_num_pts);
        CHECK_GE(it->index, 0);
        const unsigned char* const current_descriptor = current_frame.getDescriptor(it->index);
        int current_score = kdescriptorSizeBits - hammingDistance512(previous_descriptor, current_descriptor);
        if (current_score > best_score) {
          best_score = current_score;
          it_best = it;
          found = true;
          CHECK_LT((previous_predicted - it_best->measurement).norm(), kLargeSearchDistance * 2);

          aslam::statistics::DebugStatsCollector stats_distance_match("GyroTracker distance to match norm");
          stats_distance_match.AddSample((previous_predicted - it_best->measurement).norm());
        }
        processed_current_corners[it->index] = true;
        ++n_processed_corners;
      }
    }

    if (found) {
      is_current_keypoint_matched.at(it_best->index) = true;
      // TODO(magehrig): Replace keypoints score with descriptor distance score.
      matches_with_score_kp1_k->emplace_back(it_best->index, i, it_best->response);
      aslam::statistics::DebugStatsCollector stats_distance_match("GyroTracker match bits");
      stats_distance_match.AddSample(best_score);
    } else {
      aslam::statistics::DebugStatsCollector stats_distance_no_match("GyroTracker no-match num_checked");
      stats_distance_no_match.AddSample(n_processed_corners);
    }
  }
}

}  //namespace aslam
