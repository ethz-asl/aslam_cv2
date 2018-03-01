#include "aslam/matcher/stereo-matcher.h"

#include <aslam/common/statistics/statistics.h>
#include <glog/logging.h>

namespace aslam {

StereoMatcher::StereoMatcher(
    const size_t first_camera_idx, const size_t second_camera_idx,
    const aslam::NCamera::ConstPtr camera_rig,
    const aslam::VisualFrame::ConstPtr frame0,
    const aslam::VisualFrame::ConstPtr frame1,
    StereoMatchesWithScore* matches_frame0_frame1)
    : first_camera_idx_(first_camera_idx),
      second_camera_idx_(second_camera_idx),
      camera_rig_(camera_rig),
      frame0_(frame0),
      frame1_(frame1),
      matches_frame0_frame1_(matches_frame0_frame1),
      kImageHeight(camera_rig->getCameraShared(first_camera_idx)->imageHeight()),
      kNumPointsFrame0(frame0->getKeypointMeasurements().cols()),
      kNumPointsFrame1(frame1->getKeypointMeasurements().cols()),
      kDescriptorSizeBytes(frame0->getDescriptorSizeBytes()),
      is_keypoint_frame1_matched_(kNumPointsFrame1, false),
      iteration_processed_keypoints_frame1_(kNumPointsFrame1, false) {
  CHECK(frame0_.isValid());
  CHECK(frame1_.isValid());
  CHECK(frame0_.hasDescriptors());
  CHECK(frame1_.hasDescriptors());
  CHECK(frame0_.hasKeypointMeasurements());
  CHECK(frame1_.hasKeypointMeasurements());
  CHECK_EQ(frame0_.getTimestampNanoseconds(), frame1_.getTimestampNanoseconds())
      << "The two frames have different time stamps.";
  CHECK_NOTNULL(matches_frame0_frame1_)->clear();

  if (kNumPointsFrame0 == 0 || kNumPointsFrame1 == 0) {
    return;
  }
  CHECK_EQ(kNumPointsFrame0, frame0_.getDescriptors().cols())
      << "Number of keypoints and descriptors in frame0 is not the same.";
  CHECK_EQ(kNumPointsFrame1, frame1_.getDescriptors().cols())
      << "Number of keypoints and descriptors in frame1 is not the same.";
  CHECK_LE(kDescriptorSizeBytes * 8, 512u)
      << "Usually binary descriptors' size "
         "is less or equal to 512 bits. Adapt the following check if this "
         "framework uses larger binary descriptors.";
  CHECK_GT(kImageHeight, 0u);
  CHECK_EQ(iteration_processed_keypoints_frame1_.size(), kNumPointsFrame1);
  CHECK_EQ(is_keypoint_frame1_matched_.size(), kNumPointsFrame1);

  descriptors_frame0_wrapped_.reserve(kNumPointsFrame0);
  descriptors_frame1_wrapped_.reserve(kNumPointsFrame1);

  keypoints_frame1_sorted_by_y_.reserve(kNumPointsFrame1);
  matches_frame0_frame1_->reserve(std::min(kNumPointsFrame0, kNumPointsFrame1));

  corner_row_LUT_.reserve(kImageHeight);
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&
      descriptors_frame0 = frame0.getDescriptors();
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&
      descriptors_frame1 = frame1.getDescriptors();

  // Warp descriptors.
  for (int descriptor_frame0_idx = 0; descriptor_frame0_idx < kNumPointsFrame0;
       ++descriptor_frame0_idx) {
    descriptors_frame0_wrapped_.emplace_back(
        &(descriptors_0.coeffRef(0, descriptor_frame0_idx)),
        kDescriptorSizeBytes);
  }
  for (int descriptor_frame1_idx = 0; descriptor_frame1_idx < kNumPointsFrame1;
       ++descriptor_frame1_idx) {
    descriptors_frame1_wrapped_.emplace_back(
        &(descriptors_frame1.coeffRef(0, descriptor_frame1_idx)),
        kDescriptorSizeBytes);
  }
}

void StereoMatcher::match() {
  // Sort keypoints of frame1 from small to large y coordinates.
  for (int i = 0; i < kNumPointsFrame1; ++i) {
    keypoints_frame1_sorted_by_y_.emplace_back(
        frame1_.getKeypointMeasurement(i), i);
  }

  std::sort(
      keypoints_frame1_sorted_by_y_.begin(),
      keypoints_frame1_sorted_by_y_.end(),
      [](const KeypointData& lhs, const KeypointData& rhs) -> bool {
        return lhs.measurement(1) < rhs.measurement(1);
      });

  // Lookup table construction.
  // TODO(magehrig):  Sort by y if image height >= image width,
  //                  otherwise sort by x.
  int v = 0;
  for (size_t y = 0u; y < kImageHeight; ++y) {
    while (v < kNumPointsFrame1 &&
           y > static_cast<size_t>(
                   keypoints_frame1_sorted_by_y_[v].measurement(1))) {
      ++v;
    }
    corner_row_LUT_.push_back(v);
  }
  CHECK_EQ(static_cast<int>(corner_row_LUT_.size()), kImageHeight);

  // Remember matched keypoints of frame1.
  for (int i = 0; i < kNumPointsFrame0; ++i) {
    matchKeypoint(i);
  }

  std::vector<bool> is_inferior_keypoint_frame1_matched(
      is_keypoint_frame1_matched_);
  for (size_t i = 0u; i < kMaxNumInferiorIterations; ++i) {
    if (!matchInferiorMatches(&is_inferior_keypoint_frame1_matched)) {
      return;
    }
  }
}

void StereoMatcher::matchKeypoint(const int idx_frame0) {
  std::fill(
      iteration_processed_keypoints_frame1_.begin(),
      iteration_processed_keypoints_frame1_.end(), false);

  bool found = false;
  bool passed_ratio_test = false;
  int n_processed_corners = 0;
  KeyPointIterator it_best;

  const static unsigned int kDescriptorSizeBits = 8 * kDescriptorSizeBytes;
  int best_score = static_cast<int>(
      kDescriptorSizeBits * kMatchingThresholdBitsRatioRelaxed);
  unsigned int distance_best = kDescriptorSizeBits + 1;
  unsigned int distance_second_best = kDescriptorSizeBits + 1;
  const common::FeatureDescriptorConstRef& descriptor_frame0 =
      descriptors_frame0_wrapped_[idx_frame0];

  MatchData current_match_data;

  // Perform the search.
  for (KeyPointIterator it = descriptor_frame1_wrapped_.begin();
       it != descriptors_frame1_wrapped_.end(); ++it) {
    CHECK_LT(it->channel_index, kNumPointsFrame1);
    CHECK_GE(it->channel_index, 0u);
    const common::FeatureDescriptorConstRef& descriptor_frame1 =
        descriptors_frame1_wrapped_[it->channel_index];

    if (!epipolarConstraint(descriptor_frame0, descriptor_frame1)) {
      continue;
    }

    unsigned int distance =
        common::GetNumBitsDifferent(descriptor_frame0, descriptor_frame1);
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

  if (found) {
    passed_ratio_test =
        ratioTest(kDescriptorSizeBits, distance_best, distance_second_best);
  }

  if (passed_ratio_test) {
    CHECK(
        idx_frame0_to_attempted_match_data_map_
            .insert(std::make_pair(idx_frame0, current_match_data))
            .second);
    const int best_match_keypoint_idx_frame1 = it_best->channel_index;
    const double matching_score =
        computeMatchingScore(best_score, kDescriptorSizeBits);
    if (is_keypoint_frame1_matched_[best_match_keypoint_idx_frame1]) {
      if (matching_score >
          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->getScore()) {
        // The current match is better than a previous match associated with the
        // current keypoint of frame1. Hence, the inferior match is the
        // previous match associated with the current keypoint of frame1.
        const int inferior_keypoint_idx_frame0 =
            frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
                ->getKeypointIndexBananaFrame();
        inferior_match_keypoint_idx_frame0_.push_back(
            inferior_keypoint_idx_frame0);

        frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
            ->setScore(matching_score);
        frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
            ->setIndexApple(best_match_keypoint_idx_frame1);
        frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
            ->setIndexBanana(idx_frame0);
      } else {
        // The current match is inferior to a previous match associated with the
        // current keypoint of frame (k+1).
        inferior_match_keypoint_idx_frame0_.push_back(idx_frame0);
      }
    } else {
      is_keypoint_frame1_matched_[best_match_keypoint_idx_frame1] = true;
      matches_frame0_frame1_->emplace_back(
          idx_frame0, best_match_keypoint_idx_frame1, matching_score);

      CHECK(matches_frame0_frame1_->end() != matches_frame0_frame1_->begin())
          << "Match vector should not be empty.";
      CHECK(
          frame1_idx_to_matches_iterator_map_
              .emplace(
                  best_match_keypoint_idx_frame1,
                  matches_frame0_frame1_->end() - 1)
              .second);
    }

    statistics::StatsCollector stats_distance_match(
        "StereoTracker: number of matching bits");
    stats_distance_match.AddSample(best_score);
  }
  statistics::StatsCollector stats_count_processed(
      "StereoTracker: number of computed distances per keypoint");
  stats_count_processed.AddSample(n_processed_corners);
}

bool StereoMatcher::matchInferiorMatches(
    std::vector<bool>* is_inferior_keypoint_frame1_matched) {
  CHECK_NOTNULL(is_inferior_keypoint_frame1_matched);
  CHECK_EQ(
      is_inferior_keypoint_frame1_matched->size(),
      is_keypoint_frame1_matched_.size());

  bool found_inferior_match = false;

  std::unordered_set<int> erase_inferior_match_keypoint_idx_frame0;
  for (const int inferior_keypoint_idx_frame0 :
       inferior_match_keypoint_idx_frame0_) {
    const MatchData& match_data =
        idx_frame0_to_attempted_match_data_map_[inferior_keypoint_idx_frame0];
    bool found = false;
    double best_matching_score =
        static_cast<double>(kMatchingThresholdBitsRatioStrict);
    KeyPointIterator it_best;

    for (size_t i = 0u; i < match_data.keypoint_match_candidates_frame1.size();
         ++i) {
      const KeyPointIterator& keypoint_frame1 =
          match_data.keypoint_match_candidates_frame1[i];
      const double matching_score =
          match_data.match_candidate_matching_scores[i];
      // Make sure that we don't try to match with already matched keypoints
      // of frame (k+1) (also previous inferior matches).
      if (is_keypoint_frame1_matched_[keypoint_frame1->channel_index])
        continue;
      if (matching_score > best_matching_score) {
        it_best = keypoint_frame1;
        best_matching_score = matching_score;
        found = true;
      }
    }

    if (found) {
      found_inferior_match = true;
      const int best_match_keypoint_idx_frame1 = it_best->channel_index;
      if ((*is_inferior_keypoint_frame1_matched)
              [best_match_keypoint_idx_frame1]) {
        if (best_matching_score >
            frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
                ->getScore()) {
          // The current match is better than a previous match associated with
          // the
          // current keypoint of frame (k+1). Hence, the revoked match is the
          // previous match associated with the current keypoint of frame (k+1).
          const int revoked_inferior_keypoint_idx_frame0 =
              frame1_idx_to_matches_iterator_map_
                  [best_match_keypoint_idx_frame1]
                      ->getKeypointIndexBananaFrame();
          // The current keypoint k does not have to be matched anymore
          // in the next iteration.
          erase_inferior_match_keypoint_idx_frame0.insert(
              inferior_keypoint_idx_frame0);
          // The keypoint k that was revoked. That means that it can be matched
          // again in the next iteration.
          erase_inferior_match_keypoint_idx_frame0.erase(
              revoked_inferior_keypoint_idx_frame0);

          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->setScore(best_matching_score);
          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->setIndexApple(best_match_keypoint_idx_frame1);
          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->setIndexBanana(inferior_keypoint_idx_frame0);
        }
      } else {
        (*is_inferior_keypoint_frame1_matched)[best_match_keypoint_idx_frame1] =
            true;
        matches_frame0_frame1_->emplace_back(
            best_match_keypoint_idx_frame0, inferior_keypoint_idx_frame0,
            best_matching_score);
        erase_inferior_match_keypoint_idx_frame0.insert(
            inferior_keypoint_idx_frame0);

        CHECK(matches_frame0_frame1_->end() != matches_frame0_frame1_->begin())
            << "Match vector should not be empty.";
        CHECK(
            frame1_idx_to_matches_iterator_map_
                .emplace(
                    best_match_keypoint_idx_frame1,
                    matches_frame0_frame1_->end() - 1)
                .second);
      }
    }
  }

  if (erase_inferior_match_keypoint_idx_frame0.size() > 0u) {
    // Do not iterate again over newly matched keypoints of frame k.
    // Hence, remove the matched keypoints.
    std::vector<int>::iterator iter_erase_from = std::remove_if(
        inferior_match_keypoint_idx_frame0_.begin(),
        inferior_match_keypoint_idx_frame0_.end(),
        [&erase_inferior_match_keypoint_idx_frame0](const int element) -> bool {
          return erase_inferior_match_keypoint_idx_frame0.count(element) == 1u;
        });
    inferior_match_keypoint_idx_frame0_.erase(
        iter_erase_from, inferior_match_keypoint_idx_frame0_.end());
  }

  // Subsequent iterations should not mess with the current matches.
  is_keypoint_frame1_matched_ = *is_inferior_keypoint_frame1_matched;

  return found_inferior_match;
}

}  // namespace aslam
