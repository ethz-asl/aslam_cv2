#include "aslam/matcher/gyro-two-frame-matcher.h"

#include <aslam/common/statistics/statistics.h>

namespace aslam {

GyroTwoFrameMatcher::GyroTwoFrameMatcher(
    const Quaternion& q_Ckp1_Ck,
    const VisualFrame& frame_kp1,
    const VisualFrame& frame_k,
    const uint32_t image_height,
    MatchesWithScore* matches_with_score_kp1_k)
  : frame_kp1_(frame_kp1), frame_k_(frame_k), q_Ckp1_Ck_(q_Ckp1_Ck),
    kDescriptorSizeBytes(frame_kp1.getDescriptorSizeBytes()),
    kNumPointsKp1(frame_kp1.getKeypointMeasurements().cols()),
    kNumPointsK(frame_k.getKeypointMeasurements().cols()),
    kImageHeight(image_height),
    matches_with_score_kp1_k_(matches_with_score_kp1_k),
    is_keypoint_kp1_matched_(kNumPointsKp1, false),
    iteration_processed_keypoints_kp1_(kNumPointsKp1, false) {
  CHECK(frame_kp1.isValid());
  CHECK(frame_k.isValid());
  CHECK(frame_kp1.hasDescriptors());
  CHECK(frame_k.hasDescriptors());
  CHECK(frame_kp1.hasKeypointMeasurements());
  CHECK(frame_k.hasKeypointMeasurements());
  CHECK_GT(frame_kp1.getTimestampNanoseconds(), frame_k.getTimestampNanoseconds());
  CHECK_NOTNULL(matches_with_score_kp1_k_)->clear();
  CHECK_GT(kNumPointsKp1, 0);
  CHECK_GT(kNumPointsK, 0);
  CHECK_EQ(kNumPointsKp1, frame_kp1.getDescriptors().cols()) <<
      "Number of keypoints and descriptors in frame k+1 is not the same.";
  CHECK_EQ(kNumPointsK, frame_k.getDescriptors().cols()) <<
      "Number of keypoints and descriptors in frame k is not the same.";
  // Usually binary descriptors' size is less or equal to 512 bits.
  // Adapt the following check if this framework uses larger binary descriptors.
  CHECK_LE(kDescriptorSizeBytes*8, 512u);
  CHECK_GT(kImageHeight, 0u);
  CHECK_EQ(iteration_processed_keypoints_kp1_.size(), kNumPointsKp1);
  CHECK_EQ(is_keypoint_kp1_matched_.size(), kNumPointsKp1);

  descriptors_kp1_wrapped_.reserve(kNumPointsKp1);
  keypoints_kp1_sorted_by_y_.reserve(kNumPointsKp1);
  descriptors_k_wrapped_.reserve(kNumPointsK);
  matches_with_score_kp1_k_->reserve(kNumPointsK);
  corner_row_LUT_.reserve(kImageHeight);
}

void GyroTwoFrameMatcher::Initialize() {
  // Predict keypoint positions.
  predictKeypointsByRotation(frame_k_, q_Ckp1_Ck_, &predicted_keypoint_positions_kp1_, &prediction_success_);
  CHECK_EQ(prediction_success_.size(), predicted_keypoint_positions_kp1_.cols());

  // Prepare descriptors for efficient matching.
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_kp1 =
      frame_kp1_.getDescriptors();
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_k =
      frame_k_.getDescriptors();

  for (int descriptor_kp1_idx = 0; descriptor_kp1_idx < kNumPointsKp1;
      ++descriptor_kp1_idx) {
    descriptors_kp1_wrapped_.emplace_back(
        &(descriptors_kp1.coeffRef(0, descriptor_kp1_idx)), kDescriptorSizeBytes);
  }

  for (int descriptor_k_idx = 0; descriptor_k_idx < kNumPointsK;
      ++descriptor_k_idx) {
    descriptors_k_wrapped_.emplace_back(
        &(descriptors_k.coeffRef(0, descriptor_k_idx)), kDescriptorSizeBytes);
  }

  // Sort keypoints of frame (k+1) from small to large y coordinates.
  for (int i = 0; i < kNumPointsKp1; ++i) {
    keypoints_kp1_sorted_by_y_.emplace_back(frame_kp1_.getKeypointMeasurement(i), i);
  }

  std::sort(keypoints_kp1_sorted_by_y_.begin(), keypoints_kp1_sorted_by_y_.end(),
            [](const KeypointData& lhs, const KeypointData& rhs)-> bool {
              return lhs.measurement(1) < rhs.measurement(1);
            });

  // Lookup table construction.
  int v = 0;
  for (size_t y = 0; y < kImageHeight; ++y) {
    while (v < kNumPointsKp1 &&
        y > keypoints_kp1_sorted_by_y_[v].measurement(1)) {
      ++v;
    }
    corner_row_LUT_.push_back(v);
  }
  CHECK_EQ(static_cast<int>(corner_row_LUT_.size()), kImageHeight);
}

void GyroTwoFrameMatcher::Match() {
  Initialize();

  for (int i = 0; i < kNumPointsK; ++i) {
    MatchKeypoint(i);
  }

  std::vector<bool> is_inferior_keypoint_kp1_matched(
      is_keypoint_kp1_matched_);
  for (size_t i = 0u; i < kInferiorIterations; ++i) {
    if(!MatchInferiorMatches(&is_inferior_keypoint_kp1_matched)) return;
  }
}

bool GyroTwoFrameMatcher::MatchInferiorMatches(std::vector<bool>* is_inferior_keypoint_kp1_matched) {
  CHECK_EQ(is_inferior_keypoint_kp1_matched->size(), is_keypoint_kp1_matched_.size());

  bool found_something = false;

  std::unordered_set<int> erase_inferior_match_keypoint_idx_k;
  for (const int inferior_keypoint_idx_k: inferior_match_keypoint_idx_k) {
    const MatchData& match_data = idx_k_to_attempted_match_data_map.at(inferior_keypoint_idx_k);
    bool found = false;
    double best_matching_score = static_cast<double>(kMatchingThresholdBitsRatio);
    KeyPointIterator it_best;

    for (size_t i = 0u; i < match_data.keypoint_match_candidates_kp1.size(); ++i) {
      const KeyPointIterator& keypoint_kp1 = match_data.keypoint_match_candidates_kp1.at(i);
      const double matching_score = match_data.match_candidate_matching_scores.at(i);
      // Make sure that we don't try to match with already matched keypoints
      // of frame (k+1) (also previous inferior matches).
      if (is_keypoint_kp1_matched_.at(keypoint_kp1->channel_index)) continue;
      if (matching_score > best_matching_score) {
        it_best = keypoint_kp1;
        best_matching_score = matching_score;
        found = true;
      }
    }

    if (found) {
      if (!found_something) found_something = true;
      const int best_match_keypoint_idx_kp1 = it_best->channel_index;
      if (is_inferior_keypoint_kp1_matched->at(best_match_keypoint_idx_kp1)) {
        if (best_matching_score > kp1_idx_to_matches_with_score_iterator_map_.at(
            best_match_keypoint_idx_kp1)->score) {
          // The current match is better than a previous match associated with the
          // current keypoint of frame (k+1). Hence, the revoked match is the
          // previous match associated with the current keypoint of frame (k+1).
          const int revoked_inferior_keypoint_idx_k =
              kp1_idx_to_matches_with_score_iterator_map_.at(
                  best_match_keypoint_idx_kp1)->getIndexBanana();
          // The current keypoint k does not have to be matched anymore
          // in the next iteration.
          erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);
          // The keypoint k that was revoked. That means that it can be matched
          // again in the next iteration.
          erase_inferior_match_keypoint_idx_k.erase(revoked_inferior_keypoint_idx_k);

          kp1_idx_to_matches_with_score_iterator_map_.at(
              best_match_keypoint_idx_kp1)->setScore(best_matching_score);
          kp1_idx_to_matches_with_score_iterator_map_.at(
              best_match_keypoint_idx_kp1)->setIndexApple(best_match_keypoint_idx_kp1);
          kp1_idx_to_matches_with_score_iterator_map_.at(
              best_match_keypoint_idx_kp1)->setIndexBanana(inferior_keypoint_idx_k);
        }
      } else {
        is_inferior_keypoint_kp1_matched->at(best_match_keypoint_idx_kp1) = true;
        matches_with_score_kp1_k_->emplace_back(
            best_match_keypoint_idx_kp1, inferior_keypoint_idx_k, best_matching_score);
        erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);

        CHECK(matches_with_score_kp1_k_->end() != matches_with_score_kp1_k_->begin())
          << "Match vector should not be empty.";
        CHECK(kp1_idx_to_matches_with_score_iterator_map_.emplace(
            best_match_keypoint_idx_kp1, matches_with_score_kp1_k_->end() - 1).second);
      }
    }
  }

  if (erase_inferior_match_keypoint_idx_k.size() > 0u) {
    // Do not iterate again over newly matched keypoints of frame k.
    // Hence, remove the matched keypoints.
    std::vector<int>::iterator iter_erase_from = std::remove_if(
        inferior_match_keypoint_idx_k.begin(), inferior_match_keypoint_idx_k.end(),
        [&erase_inferior_match_keypoint_idx_k](const int element) -> bool {
          return erase_inferior_match_keypoint_idx_k.count(element) == 1u;
        }
    );
    inferior_match_keypoint_idx_k.erase(iter_erase_from, inferior_match_keypoint_idx_k.end());
  }

  // Subsequent iterations should not mess with the current matches.
  is_keypoint_kp1_matched_ = *is_inferior_keypoint_kp1_matched;

  return found_something;
}


void GyroTwoFrameMatcher::MatchKeypoint(const int idx_k) {
  if (!prediction_success_.at(idx_k)) {
    return;
  }

  std::fill(iteration_processed_keypoints_kp1_.begin(),
            iteration_processed_keypoints_kp1_.end(),
            false);

  Eigen::Matrix<double, 2, 1> predicted_keypoint_position_kp1 =
      predicted_keypoint_positions_kp1_.block<2, 1>(0, idx_k);
  const common::FeatureDescriptorConstRef& descriptor_k =
      descriptors_k_wrapped_.at(idx_k);

  // Compute search area for LUT iterators row-wise.
  int y_nearest[2];  // Small search region.
  y_nearest[0] = Clamp(0, kImageHeight - 1, predicted_keypoint_position_kp1(1) + 0.5 - kSmallSearchDistance);
  y_nearest[1] = Clamp(0, kImageHeight - 1, predicted_keypoint_position_kp1(1) + 0.5 + kSmallSearchDistance);
  int y_near[2];  // Large search region.
  y_near[0] = Clamp(0, kImageHeight - 1, predicted_keypoint_position_kp1(1) + 0.5 - kLargeSearchDistance);
  y_near[1] = Clamp(0, kImageHeight - 1, predicted_keypoint_position_kp1(1) + 0.5 + kLargeSearchDistance);

  CHECK_LE(y_nearest[0], y_nearest[1]);
  CHECK_LE(y_near[0], y_near[1]);
  CHECK_GE(y_nearest[0], 0);
  CHECK_GE(y_nearest[1], 0);
  CHECK_GE(y_near[0], 0);
  CHECK_GE(y_near[1], 0);
  CHECK_LT(y_nearest[0], kImageHeight);
  CHECK_LT(y_nearest[1], kImageHeight);
  CHECK_LT(y_near[0], kImageHeight);
  CHECK_LT(y_near[1], kImageHeight);

  int nearest_top = std::min<int>(y_nearest[0], kImageHeight - 1);
  int nearest_bottom = std::min<int>(y_nearest[1] + 1, kImageHeight - 1);
  int near_top = std::min<int>(y_near[0], kImageHeight - 1);
  int near_bottom = std::min<int>(y_near[1] + 1, kImageHeight - 1);

  KeyPointIterator nearest_corners_begin = keypoints_kp1_sorted_by_y_.begin() + corner_row_LUT_[nearest_top];
  KeyPointIterator nearest_corners_end = keypoints_kp1_sorted_by_y_.begin() + corner_row_LUT_[nearest_bottom];
  KeyPointIterator near_corners_begin = keypoints_kp1_sorted_by_y_.begin() + corner_row_LUT_[near_top];
  KeyPointIterator near_corners_end = keypoints_kp1_sorted_by_y_.begin() + corner_row_LUT_[near_bottom];

  bool found = false;
  int n_processed_corners = 0;
  KeyPointIterator it_best;
  const static unsigned int kDescriptorSizeBits = 8*kDescriptorSizeBytes;
  int best_score = static_cast<int>(kDescriptorSizeBits*kMatchingThresholdBitsRatio);

  const int bound_left_nearest = predicted_keypoint_position_kp1(0) - kSmallSearchDistance;
  const int bound_right_nearest = predicted_keypoint_position_kp1(0) + kSmallSearchDistance;

  MatchData current_match_data;

  // First search small window.
  for (KeyPointIterator it = nearest_corners_begin; it != nearest_corners_end; ++it) {
    if (it->measurement(0) < bound_left_nearest ||
        it->measurement(0) > bound_right_nearest) {
      continue;
    }

    CHECK_LT(it->channel_index, kNumPointsKp1);
    CHECK_GE(it->channel_index, 0u);
    const common::FeatureDescriptorConstRef& descriptor_kp1 =
        descriptors_kp1_wrapped_.at(it->channel_index);
    int current_score = kDescriptorSizeBits - common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
    if (current_score > best_score) {
      best_score = current_score;
      it_best = it;
      found = true;
    }
    iteration_processed_keypoints_kp1_.at(it->channel_index) = true;
    ++n_processed_corners;
    const double current_matching_score = ComputeMatchingScore(current_score, kDescriptorSizeBits);
    current_match_data.AddCandidate(it, current_matching_score);
  }

  // If no match in small window, increase window and search again.
  if (!found) {
    const int bound_left_near = predicted_keypoint_position_kp1(0) - kLargeSearchDistance;
    const int bound_right_near = predicted_keypoint_position_kp1(0) + kLargeSearchDistance;

    for (KeyPointIterator it = near_corners_begin; it != near_corners_end; ++it) {
      if (iteration_processed_keypoints_kp1_.at(it->channel_index)) {
        continue;
      }
      if (it->measurement(0) < bound_left_near ||
          it->measurement(0) > bound_right_near) {
        continue;
      }
      CHECK_LT(it->channel_index, kNumPointsKp1);
      CHECK_GE(it->channel_index, 0);
      const common::FeatureDescriptorConstRef& descriptor_kp1 =
          descriptors_kp1_wrapped_.at(it->channel_index);
      int current_score = kDescriptorSizeBits - common::GetNumBitsDifferent(descriptor_k, descriptor_kp1);
      if (current_score > best_score) {
        best_score = current_score;
        it_best = it;
        found = true;
      }
      ++n_processed_corners;
      const double current_matching_score = ComputeMatchingScore(current_score, kDescriptorSizeBits);
      current_match_data.AddCandidate(it, current_matching_score);
    }
  }

  if (found) {
    CHECK(idx_k_to_attempted_match_data_map.insert(
        std::make_pair(idx_k, current_match_data)).second);
    const int best_match_keypoint_idx_kp1 = it_best->channel_index;
    const double matching_score = ComputeMatchingScore(
        best_score, kDescriptorSizeBits);
    if (is_keypoint_kp1_matched_.at(best_match_keypoint_idx_kp1)) {
      if (matching_score > kp1_idx_to_matches_with_score_iterator_map_.at(
          best_match_keypoint_idx_kp1)->score) {
        // The current match is better than a previous match associated with the
        // current keypoint of frame (k+1). Hence, the inferior match is the
        // previous match associated with the current keypoint of frame (k+1).
        const int inferior_keypoint_idx_k =
            kp1_idx_to_matches_with_score_iterator_map_.at(
                best_match_keypoint_idx_kp1)->getIndexBanana();
        inferior_match_keypoint_idx_k.push_back(inferior_keypoint_idx_k);

        kp1_idx_to_matches_with_score_iterator_map_.at(
            best_match_keypoint_idx_kp1)->setScore(matching_score);
        kp1_idx_to_matches_with_score_iterator_map_.at(
            best_match_keypoint_idx_kp1)->setIndexApple(best_match_keypoint_idx_kp1);
        kp1_idx_to_matches_with_score_iterator_map_.at(
            best_match_keypoint_idx_kp1)->setIndexBanana(idx_k);
      } else {
        // The current match is inferior to a previous match associated with the
        // current keypoint of frame (k+1).
        inferior_match_keypoint_idx_k.push_back(idx_k);
        }
    } else {
      is_keypoint_kp1_matched_.at(best_match_keypoint_idx_kp1) = true;
      matches_with_score_kp1_k_->emplace_back(
          best_match_keypoint_idx_kp1, idx_k, matching_score);

      CHECK(matches_with_score_kp1_k_->end() != matches_with_score_kp1_k_->begin())
        << "Match vector should not be empty.";
      CHECK(kp1_idx_to_matches_with_score_iterator_map_.emplace(
          best_match_keypoint_idx_kp1, matches_with_score_kp1_k_->end() - 1).second);
    }

    aslam::statistics::StatsCollector stats_distance_match(
        "GyroTracker: number of matching bits");
    stats_distance_match.AddSample(best_score);
  }
  aslam::statistics::StatsCollector stats_count_processed(
      "GyroTracker: number of computed distances per keypoint");
  stats_count_processed.AddSample(n_processed_corners);
}


} // namespace aslam
