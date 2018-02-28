#include "aslam/matcher/stereo-matcher.h"

#include <aslam/common/statistics/statistics.h>
#include <dense-reconstruction/stereo-pair-detection.h>
#include <glog/logging.h>

namespace aslam {

void StereoMatcher::match(const VisualFrame& frame0, const VisualFrame& frame1,  
      StereoMatchesWithScore* matches_frame0_frame1) {
  CHECK(frame0.isValid());
  CHECK(frame1.isValid());
  CHECK(frame0.hasDescriptors());
  CHECK(frame1.hasDescriptors());
  CHECK(frame0.hasKeypointMeasurements());
  CHECK(frame1.hasKeypointMeasurements());
  CHECK_EQ(frame0.getTimestampNanoseconds(), frame1.getTimestampNanoseconds()) << 
      "The two frames have different time stamps.";
  CHECK_NOTNULL(matches_frame0_frame1_)->clear();

  const int kNumPointsFrame0 = frame0.getKeypointMeasruements().cols();
  const int kNumPointsFrame1 = frame1.getKeypointMeasurements().cols();
  const size_t kDescriptorSizeBytes =  frame0.getDescriptorSizeBytes();

  is_keypoint_frame1_matched_(kNumPointsframe1, false),
  iteration_processed_keypoints_frame1_(kNumPointsframe1, false) {

  CHECK_EQ(kNumPointsframe1, frame_frame1.getDescriptors().cols()) <<
      "Number of keypoints and descriptors in frame k+1 is not the same.";
  CHECK_EQ(kNumPointsK, frame_k.getDescriptors().cols()) <<
      "Number of keypoints and descriptors in frame k is not the same.";
  CHECK_LE(kDescriptorSizeBytes*8, 512u) << "Usually binary descriptors' size "
      "is less or equal to 512 bits. Adapt the following check if this "
      "framework uses larger binary descriptors.";

  CHECK_GT(kImageHeight, 0u);
  CHECK_EQ(iteration_processed_keypoints_frame1_.size(), kNumPointsframe1);
  CHECK_EQ(is_keypoint_frame1_matched_.size(), kNumPointsFrame1);
  CHECK_EQ(prediction_success_.size(), predicted_keypoint_positions_frame1_.cols());
  CHECK_GT(small_search_distance_px_, 0);
  CHECK_GT(large_search_distance_px_, 0);
  CHECK_GE(large_search_distance_px_, small_search_distance_px_);

  std::vector<common::FeatureDescriptorConstRef> descriptors_frame0_wrapped;
  descriptors_frame0_wrapped.reserve(kNumPointsFrame0);
  std::vector<common::FeatureDescriptorConstRef> descriptors_frame1_wrapped;
  descriptors_frame1_wrapped.reserve(kNumPointsFrame1);
  keypoints_frame1_sorted_by_y_.reserve(kNumPointsframe1);
  descriptors_frame0_wrapped_.reserve(kNumPointsK);
  matches_frame1_frame0_->reserve(kNumPointsK);
  corner_row_LUT_.reserve(kImageHeight);
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_frame1 =
      frame_frame1_.getDescriptors();
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& descriptors_k =
      frame_frame0_.getDescriptors();

  for (int descriptor_frame1_idx = 0; descriptor_frame1_idx < kNumPointsframe1;
      ++descriptor_frame1_idx) {
    descriptors_frame1_wrapped_.emplace_back(
        &(descriptors_frame1.coeffRef(0, descriptor_frame1_idx)), kDescriptorSizeBytes);
  }

  for (int descriptor_frame0_idx = 0; descriptor_frame0_idx < kNumPointsK;
      ++descriptor_frame0_idx) {
    descriptors_frame0_wrapped_.emplace_back(
        &(descriptors_k.coeffRef(0, descriptor_frame0_idx)), kDescriptorSizeBytes);
  }

  // Sort keypoints of frame (k+1) from small to large y coordinates.
  for (int i = 0; i < kNumPointsframe1; ++i) {
    keypoints_frame1_sorted_by_y_.emplace_back(frame_frame1_.getKeypointMeasurement(i), i);
  }

  std::sort(keypoints_frame1_sorted_by_y_.begin(), keypoints_frame1_sorted_by_y_.end(),
            [](const KeypointData& lhs, const KeypointData& rhs)-> bool {
              return lhs.measurement(1) < rhs.measurement(1);
            });

  // Lookup table construction.
  // TODO(magehrig):  Sort by y if image height >= image width,
  //                  otherwise sort by x.
  int v = 0;
  for (size_t y = 0u; y < kImageHeight; ++y) {
    while (v < kNumPointsframe1 &&
        y > static_cast<size_t>(keypoints_frame1_sorted_by_y_[v].measurement(1))) {
      ++v;
    }
    corner_row_LUT_.push_back(v);
  }
  CHECK_EQ(static_cast<int>(corner_row_LUT_.size()), kImageHeight);initialize();

  if (kNumPointsFrame0 == 0 || kNumPointsFrame1 == 0) {
    return;
  }

  for (int i = 0; i < kNumPointsK; ++i) {
    matchKeypoint(i);
  }

  std::vector<bool> is_inferior_keypoint_frame1_matched(
      is_keypoint_frame1_matched_);
  for (size_t i = 0u; i < kMaxNumInferiorIterations; ++i) {
    if(!matchInferiorMatches(&is_inferior_keypoint_frame1_matched)) return;
  }
}

void StereoMatcher::matchKeypoint(const int idx_k) {
  if (!prediction_success_[idx_k]) {
    return;
  }

  std::fill(iteration_processed_keypoints_frame1_.begin(),
            iteration_processed_keypoints_frame1_.end(),
            false);

  bool found = false;
  bool passed_ratio_test = false;
  int n_processed_corners = 0;
  KeyPointIterator it_best;
  const static unsigned int kDescriptorSizeBits = 8 * kDescriptorSizeBytes;
  int best_score = static_cast<int>(
      kDescriptorSizeBits * kMatchingThresholdBitsRatioRelaxed);
  unsigned int distance_best = kDescriptorSizeBits + 1;
  unsigned int distance_second_best = kDescriptorSizeBits + 1;
  const common::FeatureDescriptorConstRef& descriptor_k =
      descriptors_frame0_wrapped_[idx_k];

  Eigen::Vector2d predicted_keypoint_position_frame1 =
      predicted_keypoint_positions_frame1_.block<2, 1>(0, idx_k);
  KeyPointIterator nearest_corners_begin, nearest_corners_end;
  getKeypointIteratorsInWindow(
      predicted_keypoint_position_frame1, small_search_distance_px_, &nearest_corners_begin, &nearest_corners_end);

  const int bound_left_nearest =
      predicted_keypoint_position_frame1(0) - small_search_distance_px_;
  const int bound_right_nearest =
      predicted_keypoint_position_frame1(0) + small_search_distance_px_;

  MatchData current_match_data;

  // First search small window.
  for (KeyPointIterator it = nearest_corners_begin; it != nearest_corners_end; ++it) {
    if (it->measurement(0) < bound_left_nearest ||
        it->measurement(0) > bound_right_nearest) {
      continue;
    }

    CHECK_LT(it->channel_index, kNumPointsframe1);
    CHECK_GE(it->channel_index, 0u);
    const common::FeatureDescriptorConstRef& descriptor_frame1 =
        descriptors_frame1_wrapped_[it->channel_index];
    unsigned int distance = common::GetNumBitsDifferent(descriptor_k, descriptor_frame1);
    int current_score = kDescriptorSizeBits - distance;
    if (current_score > best_score) {
      best_score = current_score;
      distance_second_best = distance_best;
      distance_best = distance;
      it_best = it;
      found = true;
    } else if (distance < distance_second_best) {
      // The second best distance can also belong
      // to two descriptors that do not qualify as match.
      distance_second_best = distance;
    }
    iteration_processed_keypoints_frame1_[it->channel_index] = true;
    ++n_processed_corners;
    const double current_matching_score =
        computeMatchingScore(current_score, kDescriptorSizeBits);
    current_match_data.addCandidate(it, current_matching_score);
  }

  // If no match in small window, increase window and search again.
  if (!found) {
    const int bound_left_near =
        predicted_keypoint_position_frame1(0) - large_search_distance_px_;
    const int bound_right_near =
        predicted_keypoint_position_frame1(0) + large_search_distance_px_;

    KeyPointIterator near_corners_begin, near_corners_end;
    getKeypointIteratorsInWindow(
        predicted_keypoint_position_frame1, large_search_distance_px_, &near_corners_begin, &near_corners_end);

    for (KeyPointIterator it = near_corners_begin; it != near_corners_end; ++it) {
      if (iteration_processed_keypoints_frame1_[it->channel_index]) {
        continue;
      }
      if (it->measurement(0) < bound_left_near ||
          it->measurement(0) > bound_right_near) {
        continue;
      }
      CHECK_LT(it->channel_index, kNumPointsframe1);
      CHECK_GE(it->channel_index, 0);
      const common::FeatureDescriptorConstRef& descriptor_frame1 =
          descriptors_frame1_wrapped_[it->channel_index];
      unsigned int distance =
          common::GetNumBitsDifferent(descriptor_k, descriptor_frame1);
      int current_score = kDescriptorSizeBits - distance;
      if (current_score > best_score) {
        best_score = current_score;
        distance_second_best = distance_best;
        distance_best = distance;
        it_best = it;
        found = true;
      } else if (distance < distance_second_best) {
        // The second best distance can also belong
        // to two descriptors that do not qualify as match.
        distance_second_best = distance;
      }
      ++n_processed_corners;
      const double current_matching_score =
          computeMatchingScore(current_score, kDescriptorSizeBits);
      current_match_data.addCandidate(it, current_matching_score);
    }
  }

  if (found) {
    passed_ratio_test = ratioTest(kDescriptorSizeBits, distance_best,
                                  distance_second_best);
  }

  if (passed_ratio_test) {
    CHECK(idx_frame0_to_attempted_match_data_map_.insert(
        std::make_pair(idx_k, current_match_data)).second);
    const int best_match_keypoint_idx_frame1 = it_best->channel_index;
    const double matching_score = computeMatchingScore(
        best_score, kDescriptorSizeBits);
    if (is_keypoint_frame1_matched_[best_match_keypoint_idx_frame1]) {
      if (matching_score > frame1_idx_to_matches_iterator_map_
          [best_match_keypoint_idx_frame1]->getScore()) {
        // The current match is better than a previous match associated with the
        // current keypoint of frame (k+1). Hence, the inferior match is the
        // previous match associated with the current keypoint of frame (k+1).
        const int inferior_keypoint_idx_k =
            frame1_idx_to_matches_iterator_map_
            [best_match_keypoint_idx_frame1]->getKeypointIndexBananaFrame();
        inferior_match_keypoint_idx_frame0_.push_back(inferior_keypoint_idx_k);

        frame1_idx_to_matches_iterator_map_
        [best_match_keypoint_idx_frame1]->setScore(matching_score);
        frame1_idx_to_matches_iterator_map_
        [best_match_keypoint_idx_frame1]->setIndexApple(best_match_keypoint_idx_frame1);
        frame1_idx_to_matches_iterator_map_
        [best_match_keypoint_idx_frame1]->setIndexBanana(idx_k);
      } else {
        // The current match is inferior to a previous match associated with the
        // current keypoint of frame (k+1).
        inferior_match_keypoint_idx_frame0_.push_back(idx_k);
        }
    } else {
      is_keypoint_frame1_matched_[best_match_keypoint_idx_frame1] = true;
      matches_frame1_frame0_->emplace_back(
          best_match_keypoint_idx_frame1, idx_k, matching_score);

      CHECK(matches_frame1_frame0_->end() != matches_frame1_frame0_->begin())
        << "Match vector should not be empty.";
      CHECK(frame1_idx_to_matches_iterator_map_.emplace(
          best_match_keypoint_idx_frame1, matches_frame1_frame0_->end() - 1).second);
    }

    statistics::StatsCollector stats_distance_match(
        "GyroTracker: number of matching bits");
    stats_distance_match.AddSample(best_score);
  }
  statistics::StatsCollector stats_count_processed(
      "GyroTracker: number of computed distances per keypoint");
  stats_count_processed.AddSample(n_processed_corners);
}

bool StereoMatcher::matchInferiorMatches(
    std::vector<bool>* is_inferior_keypoint_frame1_matched) {
  CHECK_NOTNULL(is_inferior_keypoint_frame1_matched);
  CHECK_EQ(is_inferior_keypoint_frame1_matched->size(), is_keypoint_frame1_matched_.size());

  bool found_inferior_match = false;

  std::unordered_set<int> erase_inferior_match_keypoint_idx_k;
  for (const int inferior_keypoint_idx_k : inferior_match_keypoint_idx_frame0_) {
    const MatchData& match_data =
        idx_frame0_to_attempted_match_data_map_[inferior_keypoint_idx_k];
    bool found = false;
    double best_matching_score = static_cast<double>(kMatchingThresholdBitsRatioStrict);
    KeyPointIterator it_best;

    for (size_t i = 0u; i < match_data.keypoint_match_candidates_frame1.size(); ++i) {
      const KeyPointIterator& keypoint_frame1 = match_data.keypoint_match_candidates_frame1[i];
      const double matching_score = match_data.match_candidate_matching_scores[i];
      // Make sure that we don't try to match with already matched keypoints
      // of frame (k+1) (also previous inferior matches).
      if (is_keypoint_frame1_matched_[keypoint_frame1->channel_index]) continue;
      if (matching_score > best_matching_score) {
        it_best = keypoint_frame1;
        best_matching_score = matching_score;
        found = true;
      }
    }

    if (found) {
      found_inferior_match = true;
      const int best_match_keypoint_idx_frame1 = it_best->channel_index;
      if ((*is_inferior_keypoint_frame1_matched)[best_match_keypoint_idx_frame1]) {
        if (best_matching_score > frame1_idx_to_matches_iterator_map_
            [best_match_keypoint_idx_frame1]->getScore()) {
          // The current match is better than a previous match associated with the
          // current keypoint of frame (k+1). Hence, the revoked match is the
          // previous match associated with the current keypoint of frame (k+1).
          const int revoked_inferior_keypoint_idx_k =
              frame1_idx_to_matches_iterator_map_
              [best_match_keypoint_idx_frame1]->getKeypointIndexBananaFrame();
          // The current keypoint k does not have to be matched anymore
          // in the next iteration.
          erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);
          // The keypoint k that was revoked. That means that it can be matched
          // again in the next iteration.
          erase_inferior_match_keypoint_idx_k.erase(revoked_inferior_keypoint_idx_k);

          frame1_idx_to_matches_iterator_map_
          [best_match_keypoint_idx_frame1]->setScore(best_matching_score);
          frame1_idx_to_matches_iterator_map_
          [best_match_keypoint_idx_frame1]->setIndexApple(best_match_keypoint_idx_frame1);
          frame1_idx_to_matches_iterator_map_
          [best_match_keypoint_idx_frame1]->setIndexBanana(inferior_keypoint_idx_k);
        }
      } else {
        (*is_inferior_keypoint_frame1_matched)[best_match_keypoint_idx_frame1] = true;
        matches_frame1_frame0_->emplace_back(
            best_match_keypoint_idx_frame1, inferior_keypoint_idx_k, best_matching_score);
        erase_inferior_match_keypoint_idx_k.insert(inferior_keypoint_idx_k);

        CHECK(matches_frame1_frame0_->end() != matches_frame1_frame0_->begin())
          << "Match vector should not be empty.";
        CHECK(frame1_idx_to_matches_iterator_map_.emplace(
            best_match_keypoint_idx_frame1, matches_frame1_frame0_->end() - 1).second);
      }
    }
  }

  if (erase_inferior_match_keypoint_idx_k.size() > 0u) {
    // Do not iterate again over newly matched keypoints of frame k.
    // Hence, remove the matched keypoints.
    std::vector<int>::iterator iter_erase_from = std::remove_if(
        inferior_match_keypoint_idx_frame0_.begin(), inferior_match_keypoint_idx_frame0_.end(),
        [&erase_inferior_match_keypoint_idx_k](const int element) -> bool {
          return erase_inferior_match_keypoint_idx_k.count(element) == 1u;
        }
    );
    inferior_match_keypoint_idx_frame0_.erase(
        iter_erase_from, inferior_match_keypoint_idx_frame0_.end());
  }

  // Subsequent iterations should not mess with the current matches.
  is_keypoint_frame1_matched_ = *is_inferior_keypoint_frame1_matched;

  return found_inferior_match;
}


} // namespace aslam
