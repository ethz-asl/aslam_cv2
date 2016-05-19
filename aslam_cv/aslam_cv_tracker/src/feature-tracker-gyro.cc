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
    : camera_(camera),
      track_lengths_initialized_(false),
      current_track_id_(0) {}

void GyroTracker::track(const aslam::Quaternion& q_Ckp1_Ck,
                        const aslam::VisualFrame& previous_frame,
                        aslam::VisualFrame* current_frame,
                        aslam::MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(previous_frame.hasKeypointMeasurements());
  CHECK(CHECK_NOTNULL(current_frame)->hasKeypointMeasurements());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(previous_frame.getCameraGeometry().get())->getId());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(current_frame->getCameraGeometry().get())->getId());
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();

  aslam::timing::Timer timer_tracking("FeatureTrackerLk: track");

  if (!track_lengths_initialized_) {
    const size_t num_keypoints = previous_frame.getKeypointMeasurements().cols();
    previous_track_lengths_.resize(num_keypoints, 0);
    track_lengths_initialized_ = true;
    return;
  }

  // Make sure the frames are in order time-wise
  // TODO(schneith): Maybe also enforce that deltaT < tolerance?
  CHECK_GT(current_frame->getTimestampNanoseconds(), previous_frame.getTimestampNanoseconds());

  // Check that the required data is available in the frame
  CHECK(current_frame->hasDescriptors());
  CHECK_EQ(current_frame->getDescriptors().rows(), current_frame->getDescriptorSizeBytes());
  CHECK_EQ(current_frame->getKeypointMeasurements().cols(), current_frame->getDescriptors().cols());

  // Match the keypoints in the current frame to the previous one.
  matchFeatures(q_Ckp1_Ck, *current_frame, previous_frame, matches_with_score_kp1_k);
  // Convenience access reference
  const aslam::MatchesWithScore& matches_current_prev = *matches_with_score_kp1_k;

  aslam::statistics::DebugStatsCollector stats_num_matches("GyroTracker num. keypoint matches");
  stats_num_matches.AddSample(matches_with_score_kp1_k->size());

  // Prepare buckets.
  std::vector<std::vector<int> > buckets;
  std::vector<std::pair<int, int> > candidates_new_tracks;
  candidates_new_tracks.reserve(matches_with_score_kp1_k->size());
  buckets.resize(kNumberOfTrackingBuckets * kNumberOfTrackingBuckets);

  float bucket_width_x = static_cast<float>(camera_.imageWidth()) / kNumberOfTrackingBuckets;
  float bucket_width_y = static_cast<float>(camera_.imageHeight()) / kNumberOfTrackingBuckets;

  std::function<int(const Eigen::Block<Eigen::Matrix2Xd, 2, 1>&)> compute_bin_index =
      [buckets, bucket_width_x, bucket_width_y, this](const Eigen::Vector2d &kp) -> int {
        float bin_x = kp[0] / bucket_width_x;
        float bin_y = kp[1] / bucket_width_y;

        int bin_index = static_cast<int>(std::floor(bin_y)) * kNumberOfTrackingBuckets +
                        static_cast<int>(std::floor(bin_x));

        CHECK_GE(bin_index, 0);
        CHECK_LT(bin_index, static_cast<int>(buckets.size()));
        return bin_index;
      };

  const size_t current_num_pts = current_frame->getKeypointMeasurements().cols();
  Eigen::VectorXi current_track_ids(current_num_pts);
  current_track_ids.fill(-1); // Initialize as untracked.
  current_track_lengths_.clear();
  current_track_lengths_.resize(current_num_pts, 0);

  for (size_t i = 0; i < matches_with_score_kp1_k->size(); ++i) {
    CHECK_LE(matches_current_prev[i].getIndexBanana(), static_cast<int>(previous_frame.getTrackIds().rows()));
    CHECK_LE(matches_current_prev[i].getIndexApple(), static_cast<int>(current_track_ids.size()));
    current_track_ids(matches_current_prev[i].getIndexApple()) = previous_frame.getTrackId(
        matches_current_prev[i].getIndexBanana());
    current_track_lengths_[matches_current_prev[i].getIndexApple()] =
        previous_track_lengths_[matches_current_prev[i].getIndexBanana()] + 1;

    // Check if this is a continued track.
    if (current_track_ids(matches_current_prev[i].getIndexApple()) >= 0) {
      // Put the current keypoint into the bucket.
      CHECK_LE(matches_current_prev[i].getIndexApple(), current_num_pts);
      const Eigen::Block<Eigen::Matrix2Xd, 2, 1>& keypoint = current_frame->getKeypointMeasurement(
          matches_current_prev[i].getIndexApple());
      int bin_index = compute_bin_index(keypoint);
      buckets[bin_index].push_back(0);
      candidates_new_tracks.emplace_back(
          std::make_pair(matches_current_prev[i].getIndexApple(),
                         matches_current_prev[i].getIndexBanana()));
    }
  }

  VLOG(4) << "Got " << candidates_new_tracks.size() << " continued tracks";

  std::vector<std::pair<int, float> > candidates;
  candidates.reserve(matches_with_score_kp1_k->size());
  for (size_t i = 0; i < matches_with_score_kp1_k->size(); ++i) {
    int index_in_curr = matches_current_prev[i].getIndexApple();
    const double& keypoint_score = current_frame->getKeypointScore(index_in_curr);
    aslam::statistics::DebugStatsCollector stats_laplacian_score("GyroTracker keypoint score");
    stats_laplacian_score.AddSample(keypoint_score);

    if (current_track_ids(index_in_curr) < 0) {
      candidates.emplace_back(i, keypoint_score);
    }
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<int, float> & lhs, const std::pair<int, float> & rhs) {
              return lhs.second > rhs.second;
            });

  // Unconditionally push the first very strong points.
  int candidate_idx = 0;
  for (; candidate_idx < std::min<int>(kNumberOfKeyPointsUseUnconditional, candidates.size());
      ++candidate_idx) {
    int match_idx = candidates[candidate_idx].first;
    int index_in_curr = matches_current_prev[match_idx].getIndexApple();
    const Eigen::Block<Eigen::Matrix2Xd, 2, 1>& keypoint = current_frame->getKeypointMeasurement(
        index_in_curr);
    const double& keypoint_score = current_frame->getKeypointScore(index_in_curr);
    if (keypoint_score < kKeypointScoreThresholdUnconditional) {
      aslam::statistics::DebugStatsCollector stats_too_low_laplacian_score(
          "GyroTracker Too low laplacian score for unconditional");
      stats_too_low_laplacian_score.AddSample(keypoint_score);
      continue;
    }

    int bin_index = compute_bin_index(keypoint);
    buckets[bin_index].push_back(0);
    candidates_new_tracks.emplace_back(
        std::make_pair(matches_current_prev[match_idx].getIndexApple(),
                       matches_current_prev[match_idx].getIndexBanana()));
    aslam::statistics::DebugStatsCollector stats_unconditionally(
        "GyroTracker Unconditionally accepted");
    stats_unconditionally.AddSample(keypoint_score);
  }

  int bucket_too_full = 0;
  // Now push as many strong points as there is space in the buckets.
  int num_pts_per_bucket = kNumberOfKeyPointsUseStrong / buckets.size();
  for (; candidate_idx < std::min<int>(kNumberOfKeyPointsUseStrong, candidates.size());
      ++candidate_idx) {
    int match_idx = candidates[candidate_idx].first;
    int index_in_curr = matches_current_prev[match_idx].getIndexApple();
    const Eigen::Block<Eigen::Matrix2Xd, 2, 1>& keypoint = current_frame->getKeypointMeasurement(
        index_in_curr);
    const double& keypoint_score = current_frame->getKeypointScore(index_in_curr);
    if (keypoint_score < kKeypointScoreThresholdStrong) {
      aslam::statistics::DebugStatsCollector stats_too_low_keypoint_score_strong(
          "GyroTracker Too low score for strong");
      stats_too_low_keypoint_score_strong.AddSample(keypoint_score);
      continue;
    }
    int bin_index = compute_bin_index(keypoint);

    if (static_cast<int>(buckets[bin_index].size()) < num_pts_per_bucket) {
      buckets[bin_index].push_back(0);
      candidates_new_tracks.emplace_back(
          std::make_pair(matches_current_prev[match_idx].getIndexApple(),
                         matches_current_prev[match_idx].getIndexBanana()));
      aslam::statistics::DebugStatsCollector stats_strong_acc("GyroTracker Strong accepted");
      stats_strong_acc.AddSample(keypoint_score);
    } else {
      ++bucket_too_full;
      aslam::statistics::DebugStatsCollector stats_bucket_too_full("GyroTracker Bucket too full");
      stats_bucket_too_full.AddSample(keypoint_score);
    }
  }

  // Assign new Id's to all candidates that are not continued tracks.
  aslam::statistics::DebugStatsCollector stats_track_length("GyroTracker Track lengths");
  for (size_t i = 0; i < candidates_new_tracks.size(); ++i) {
    int index_in_curr = candidates_new_tracks[i].first;
    if (current_track_ids(index_in_curr) == -1) {
      current_track_ids(index_in_curr) = ++current_track_id_;
      current_track_lengths_[index_in_curr] = 1;
    }
    stats_track_length.AddSample(current_track_lengths_[index_in_curr]);
  }

  // Save the output track-ids to the channel in the current frame.
  current_frame->swapTrackIds(&current_track_ids);

  // Keep the current track length and the current frame
  previous_track_lengths_.swap(current_track_lengths_);

  // Print some statistics now and then.
  static int count = 0;
  if (count++ % 30 == 0) {
    VLOG(3) << statistics::Statistics::Print();
  }
}

void GyroTracker::matchFeatures(const aslam::Quaternion& q_Ckp1_Ck,
                                const VisualFrame& current_frame,
                                const VisualFrame& previous_frame,
                                aslam::MatchesWithScore* matches_with_score_kp1_k) const {
  CHECK_NOTNULL(matches_with_score_kp1_k);
  matches_with_score_kp1_k->clear();

  struct KeypointAndIndex {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector2d measurement;
    int index;
  };

  // Sort keypoints by y-coordinate.
  const int current_num_pts = current_frame.getKeypointMeasurements().cols();
  Aligned<std::vector, KeypointAndIndex>::type current_keypoints_by_y;
  current_keypoints_by_y.resize(current_num_pts);

  for (int i = 0; i < current_num_pts; ++i) {
    current_keypoints_by_y[i].measurement = current_frame.getKeypointMeasurement(i),
    current_keypoints_by_y[i].index = i;
  }

  std::sort(current_keypoints_by_y.begin(), current_keypoints_by_y.end(),
            [](const KeypointAndIndex& lhs, const KeypointAndIndex& rhs)->bool {
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

  const unsigned int descriptorSizeBytes = current_frame.getDescriptorSizeBytes();
  CHECK_LE(descriptorSizeBytes * 8, 512);
  std::function<unsigned int(const unsigned char*, const unsigned char*)> hammingDistance512 =
      [descriptorSizeBytes](const unsigned char* x, const unsigned char* y)->unsigned int {
        unsigned int distance = 0;
        for(unsigned int i = 0; i < descriptorSizeBytes; i++) {
          unsigned char val = *(x + i) ^ *(y + i);
          while(val) {
            ++distance;
            val &= val - 1;
          }
        }
        CHECK_LE(distance, descriptorSizeBytes * 8);
        return distance;
      };

  // Look for matches in a small area around the predicted keypoint location.
  static constexpr int kMinSearchRadius = 5;
  static constexpr int kSearchRadius = 10;

  matches_with_score_kp1_k->reserve(prev_num_pts);

  // Keep track of matched keypoints in the current frame such that they
  // are not matched again.
  std::vector<bool> is_current_keypoint_matched;
  is_current_keypoint_matched.resize(current_num_pts, false);

  for (int i = 0; i < prev_num_pts; ++i) {
    Eigen::Matrix<double, 2, 1> previous_predicted = C2_previous_image_points.block<2, 1>(0, i);
    const unsigned char* previous_descriptor = previous_frame.getDescriptor(i);

    const int bound_left_nearest = previous_predicted(0) - kMinSearchRadius;
    const int bound_right_nearest = previous_predicted(0) + kMinSearchRadius;
    const int bound_left_near = previous_predicted(0) - kSearchRadius;
    const int bound_right_near = previous_predicted(0) + kSearchRadius;

    std::function<int(int, int, int)> clamp = [](int lower, int upper, int in) {
      return std::min<int>(std::max<int>(in, lower), upper);
    };

    // Get search area for LUT iterators (rowwise).
    int idxnearest[2];  // Min search region.
    idxnearest[0] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 - kMinSearchRadius);
    idxnearest[1] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 + kMinSearchRadius);
    int idxnear[2];  // Max search region.
    idxnear[0] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 - kSearchRadius);
    idxnear[1] = clamp(0, image_height - 1, previous_predicted(1) + 0.5 + kSearchRadius);

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
    typedef typename Aligned<std::vector, KeypointAndIndex>::type::const_iterator KeyPointIterator;
    KeyPointIterator nearest_corners_begin = current_keypoints_by_y.begin() + corner_row_LUT[nearest_top];
    KeyPointIterator nearest_corners_end = current_keypoints_by_y.begin() + corner_row_LUT[nearest_bottom];
    KeyPointIterator near_corners_begin = current_keypoints_by_y.begin() + corner_row_LUT[near_top];
    KeyPointIterator near_corners_end = current_keypoints_by_y.begin() + corner_row_LUT[near_bottom];

    // Get descriptors and match.
    bool found = false;
    int n_processed_corners = 0;
    KeyPointIterator it_best;
    const unsigned int descriptorSizeBits = descriptorSizeBytes*8;
    int best_score = static_cast<int>(descriptorSizeBits*kMatchingThresholdBitsRatio);
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
      int current_score = descriptorSizeBits - hammingDistance512(previous_descriptor, current_descriptor);
      if (current_score > best_score) {
        best_score = current_score;
        it_best = it;
        found = true;
        CHECK_LT((previous_predicted - it_best->measurement).norm(), kMinSearchRadius * 2);

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
        int current_score = descriptorSizeBits - hammingDistance512(previous_descriptor, current_descriptor);
        if (current_score > best_score) {
          best_score = current_score;
          it_best = it;
          found = true;
          CHECK_LT((previous_predicted - it_best->measurement).norm(), kSearchRadius * 2);

          aslam::statistics::DebugStatsCollector stats_distance_match("GyroTracker distance to match norm");
          stats_distance_match.AddSample((previous_predicted - it_best->measurement).norm());
        }
        processed_current_corners[it->index] = true;
        ++n_processed_corners;
      }
    }

    if (found) {
      is_current_keypoint_matched.at(it_best->index) = true;
      // TODO(maehrig): add true response of keypoint_kp1 to matches_with_score_kp1_k instead of 0.0.
      matches_with_score_kp1_k->emplace_back(it_best->index, i, 0.0);
      aslam::statistics::DebugStatsCollector stats_distance_match("GyroTracker match bits");
      stats_distance_match.AddSample(best_score);
    } else {
      aslam::statistics::DebugStatsCollector stats_distance_no_match("GyroTracker no-match num_checked");
      stats_distance_no_match.AddSample(n_processed_corners);
    }
  }
}

// TODO(magehrig): Implement functionality to abort tracks in the rest of the GyroTracker.
// This does not do anything right now.
void GyroTracker::swapKeypointIndicesToAbort(
    const aslam::FrameId& frame_id, std::unordered_set<size_t>* keypoint_indices_to_abort) {
  CHECK_NOTNULL(keypoint_indices_to_abort);
  CHECK(frame_id.isValid());
  keypoint_indices_to_abort_.swap(*keypoint_indices_to_abort);
  abort_keypoints_wrt_frame_id_ = frame_id;
}

}  //namespace aslam
