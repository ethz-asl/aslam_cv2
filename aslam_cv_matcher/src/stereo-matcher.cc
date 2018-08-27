#include "aslam/matcher/stereo-matcher.h"

#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <glog/logging.h>

DEFINE_double(
    stereo_matcher_epipolar_threshold, 0.01,
    "Threshold whether a point is considered as fulfilling the epipolar "
    "constraint. The higher this value, the more points are considered for the "
    "correspondance search.");
DEFINE_bool(
    use_stereo_matching, false,
    "Decide whether you want to use the stereo matching for overlapping "
    "cameras. So far, only cameras calibrated with the pinhole model are "
    "supported.");

DEFINE_double(stereo_matcher_min_depth_for_match_m, 0.3, "bla");
DEFINE_double(stereo_matcher_max_depth_for_match_m, 2, "bla");

namespace aslam {

StereoMatcher::StereoMatcher(
    const size_t first_camera_idx, const size_t second_camera_idx,
    const aslam::NCamera::ConstPtr camera_rig,
    const Eigen::Matrix3d& fundamental_matrix,
    const Eigen::Matrix3d& rotation_C1_C0,
    const Eigen::Vector3d& translation_C1_C0,
    const std::shared_ptr<MappedUndistorter> first_mapped_undistorter,
    const std::shared_ptr<MappedUndistorter> second_mapped_undistorter,

    const aslam::VisualFrame::Ptr frame0, const aslam::VisualFrame::Ptr frame1,

    StereoMatchesWithScore* matches_frame0_frame1)
    : first_camera_idx_(first_camera_idx),
      second_camera_idx_(second_camera_idx),
      camera_rig_(camera_rig),
      fundamental_matrix_(fundamental_matrix),

      rotation_C1_C0_(rotation_C1_C0),
      translation_C1_C0_(translation_C1_C0),

      first_mapped_undistorter_(first_mapped_undistorter),
      second_mapped_undistorter_(second_mapped_undistorter),
      frame0_(frame0),
      frame1_(frame1),
      matches_frame0_frame1_(matches_frame0_frame1),
      kImageHeight(
          camera_rig->getCameraShared(first_camera_idx)->imageHeight()),
      kEpipolarThreshold(FLAGS_stereo_matcher_epipolar_threshold),
      kNumPointsFrame0(frame0->getKeypointMeasurements().cols()),
      kNumPointsFrame1(frame1->getKeypointMeasurements().cols()),
      kDescriptorSizeBytes(frame0->getDescriptorSizeBytes()),
      is_keypoint_frame1_matched_(
          frame1->getKeypointMeasurements().cols(), false),
      iteration_processed_keypoints_frame1_(
          frame1->getKeypointMeasurements().cols(), false) {
  CHECK(frame0_->isValid());
  CHECK(frame1_->isValid());
  CHECK(frame0_->hasDescriptors());
  CHECK(frame1_->hasDescriptors());
  CHECK(frame0_->hasKeypointMeasurements());
  CHECK(frame1_->hasKeypointMeasurements());
  CHECK_EQ(
      frame0_->getTimestampNanoseconds(), frame1_->getTimestampNanoseconds())
      << "The two frames have different time stamps.";
  CHECK_NOTNULL(matches_frame0_frame1_)->clear();
  if (kNumPointsFrame0 == 0 || kNumPointsFrame1 == 0) {
    return;
  }
  CHECK_EQ(kNumPointsFrame0, frame0_->getDescriptors().cols())
      << "Number of keypoints and descriptors in frame0 is not the same.";
  CHECK_EQ(kNumPointsFrame1, frame1_->getDescriptors().cols())
      << "Number of keypoints and descriptors in frame1 is not the same.";
  CHECK_LE(kDescriptorSizeBytes * 8, 512u)
      << "Usually binary descriptors' size "
         "is less or equal to 512 bits. Adapt the following check if this "
         "framework uses larger binary descriptors.";
  CHECK_GT(kImageHeight, 0u);
  CHECK_GT(kEpipolarThreshold, 0.0) << "The epipolar constraint threshold "
                                       "should be higher than 0 to enable "
                                       "correspondance search for noisy input.";
  CHECK_EQ(iteration_processed_keypoints_frame1_.size(), kNumPointsFrame1);
  CHECK_EQ(is_keypoint_frame1_matched_.size(), kNumPointsFrame1);

  descriptors_frame0_wrapped_.reserve(kNumPointsFrame0);
  descriptors_frame1_wrapped_.reserve(kNumPointsFrame1);

  keypoints_frame1_sorted_by_y_.reserve(kNumPointsFrame1);
  matches_frame0_frame1_->reserve(std::min(kNumPointsFrame0, kNumPointsFrame1));

  corner_row_LUT_.reserve(kImageHeight);
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&
      descriptors_frame0 = frame0->getDescriptors();
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>&
      descriptors_frame1 = frame1->getDescriptors();

  // Warp descriptors.
  for (int descriptor_frame0_idx = 0; descriptor_frame0_idx < kNumPointsFrame0;
       ++descriptor_frame0_idx) {
    descriptors_frame0_wrapped_.emplace_back(
        &(descriptors_frame0.coeffRef(0, descriptor_frame0_idx)),
        kDescriptorSizeBytes);
  }
  for (int descriptor_frame1_idx = 0; descriptor_frame1_idx < kNumPointsFrame1;
       ++descriptor_frame1_idx) {
    descriptors_frame1_wrapped_.emplace_back(
        &(descriptors_frame1.coeffRef(0, descriptor_frame1_idx)),
        kDescriptorSizeBytes);
  }

  const aslam::PinholeCamera::ConstPtr camera0 =
      std::dynamic_pointer_cast<const aslam::PinholeCamera>(
          camera_rig_->getCameraShared(first_camera_idx_));
  const aslam::PinholeCamera::ConstPtr camera1 =
      std::dynamic_pointer_cast<const aslam::PinholeCamera>(
          camera_rig_->getCameraShared(second_camera_idx_));
  // Get inverse of camera matrices.
  camera_matrix_C0_inv_ = camera0->getCameraMatrix().inverse();
  camera_matrix_C1_inv_ = camera1->getCameraMatrix().inverse();
}

void StereoMatcher::match() {
  // Sort keypoints of frame1 from small to large y coordinates.
  for (int i = 0; i < kNumPointsFrame1; ++i) {
    keypoints_frame1_sorted_by_y_.emplace_back(
        frame1_->getKeypointMeasurement(i), i);
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

  timing::Timer match_timer("StereoMatcher: Matching");
  // Remember matched keypoints of frame1.
  for (int i = 0; i < kNumPointsFrame0; ++i) {
    matchKeypoint(i);
  }
  match_timer.Stop();

  std::vector<bool> is_inferior_keypoint_frame1_matched(
      is_keypoint_frame1_matched_);
  for (size_t i = 0u; i < kMaxNumInferiorIterations; ++i) {
    if (!matchInferiorMatches(&is_inferior_keypoint_frame1_matched)) {
      // No more inferior matches, now triangulate depth for each matched
      // keypoint and add to frame.
      timing::Timer triangulation_timer("StereoMatcher: Triangulation");
      for (aslam::StereoMatchesWithScore::iterator it =
               matches_frame0_frame1_->begin();
           it != matches_frame0_frame1_->end();) {
        const std::pair<double, double> depths = calculateDepth(
            frame0_->getKeypointMeasurement(it->getKeypointIndexFrame0()),
            frame1_->getKeypointMeasurement(it->getKeypointIndexFrame1()));
        if (depths.first < FLAGS_stereo_matcher_min_depth_for_match_m ||
            depths.first > FLAGS_stereo_matcher_max_depth_for_match_m ||
            depths.second < FLAGS_stereo_matcher_min_depth_for_match_m ||
            depths.second > FLAGS_stereo_matcher_max_depth_for_match_m) {
          matches_frame0_frame1_->erase(it);
        } else {
          it->setDepthFrame0(depths.first);
          it->setDepthFrame1(depths.second);
          ++it;
        }
      }
      triangulation_timer.Stop();
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
  for (KeyPointIterator it = keypoints_frame1_sorted_by_y_.begin();
       it != keypoints_frame1_sorted_by_y_.end(); ++it) {
    CHECK_LT(it->channel_index, kNumPointsFrame1);
    CHECK_GE(it->channel_index, 0u);
    const common::FeatureDescriptorConstRef& descriptor_frame1 =
        descriptors_frame1_wrapped_[it->channel_index];

    if (!epipolarConstraint(
            frame0_->getKeypointMeasurementVector(idx_frame0),
            it->measurement)) {
      continue;
    }

    unsigned int distance =
        common::GetNumBitsDifferent(descriptor_frame0, descriptor_frame1);
    int current_score = kDescriptorSizeBits - distance;
    const std::pair<double, double> depths = calculateDepth(
        frame0_->getKeypointMeasurementVector(idx_frame0),
        frame1_->getKeypointMeasurementVector(it->channel_index));
    if (depths.first < FLAGS_stereo_matcher_min_depth_for_match_m ||
        depths.first > FLAGS_stereo_matcher_max_depth_for_match_m ||
        depths.second < FLAGS_stereo_matcher_min_depth_for_match_m ||
        depths.second > FLAGS_stereo_matcher_max_depth_for_match_m) {
      continue;
    }
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
    CHECK(idx_frame0_to_attempted_match_data_map_
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
                ->getKeypointIndexFrame0();
        inferior_match_keypoint_idx_frame0_.push_back(
            inferior_keypoint_idx_frame0);

        frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
            ->setScore(matching_score);
        frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
            ->setIndexBanana(best_match_keypoint_idx_frame1);
        frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
            ->setIndexApple(idx_frame0);
      } else {
        // The current match is inferior to a previous match associated with the
        // current keypoint of frame1.
        inferior_match_keypoint_idx_frame0_.push_back(idx_frame0);
      }
    } else {
      is_keypoint_frame1_matched_[best_match_keypoint_idx_frame1] = true;
      matches_frame0_frame1_->emplace_back(
          idx_frame0, best_match_keypoint_idx_frame1, matching_score);

      CHECK(matches_frame0_frame1_->end() != matches_frame0_frame1_->begin())
          << "Match vector should not be empty.";
      CHECK(frame1_idx_to_matches_iterator_map_
                .emplace(
                    best_match_keypoint_idx_frame1,
                    matches_frame0_frame1_->end() - 1)
                .second);
    }

    statistics::StatsCollector stats_distance_match(
        "StereoMatcher: number of matching bits");
    stats_distance_match.AddSample(best_score);
  }
  statistics::StatsCollector stats_count_processed(
      "StereoMatcher: number of computed distances per keypoint");
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
          // the current keypoint of frame1. Hence, the revoked match is the
          // previous match associated with the current keypoint of frame1.
          const int revoked_inferior_keypoint_idx_frame0 =
              frame1_idx_to_matches_iterator_map_
                  [best_match_keypoint_idx_frame1]
                      ->getKeypointIndexFrame1();
          // The current keypoint of frame0 does not have to be matched anymore
          // in the next iteration.
          erase_inferior_match_keypoint_idx_frame0.insert(
              inferior_keypoint_idx_frame0);
          // The keypoint of frame0 that was revoked. That means that it can be
          // matched again in the next iteration.
          erase_inferior_match_keypoint_idx_frame0.erase(
              revoked_inferior_keypoint_idx_frame0);

          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->setScore(best_matching_score);
          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->setIndexBanana(best_match_keypoint_idx_frame1);
          frame1_idx_to_matches_iterator_map_[best_match_keypoint_idx_frame1]
              ->setIndexApple(inferior_keypoint_idx_frame0);
        }
      } else {
        (*is_inferior_keypoint_frame1_matched)[best_match_keypoint_idx_frame1] =
            true;
        matches_frame0_frame1_->emplace_back(
            inferior_keypoint_idx_frame0, best_match_keypoint_idx_frame1,
            best_matching_score);
        erase_inferior_match_keypoint_idx_frame0.insert(
            inferior_keypoint_idx_frame0);

        CHECK(matches_frame0_frame1_->end() != matches_frame0_frame1_->begin())
            << "Match vector should not be empty.";
        CHECK(frame1_idx_to_matches_iterator_map_
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

bool StereoMatcher::epipolarConstraint(
    const Eigen::Vector2d& keypoint_frame0,
    const Eigen::Vector2d& keypoint_frame1) const {
  // Convert points to homogenous coordinates.
  Eigen::Vector2d keypoint_frame0_undistorted;
  first_mapped_undistorter_->processPoint(
      keypoint_frame0, &keypoint_frame0_undistorted);
  Eigen::Vector2d keypoint_frame1_undistorted;
  second_mapped_undistorter_->processPoint(
      keypoint_frame1, &keypoint_frame1_undistorted);

  /* for higher performance rewritten
     bool result = std::abs(
                    keypoint_hat_frame1.transpose() * fundamental_matrix_ *
                    keypoint_hat_frame0) < kEpipolarThreshold;

  into: */
  double epipole =
      keypoint_frame0(0) *
          (keypoint_frame1_undistorted(0) * fundamental_matrix_(0, 0) +
           keypoint_frame1_undistorted(1) * fundamental_matrix_(1, 0) +
           fundamental_matrix_(2, 0)) +
      keypoint_frame0_undistorted(1) *
          (keypoint_frame1_undistorted(0) * fundamental_matrix_(0, 1) +
           keypoint_frame1_undistorted(1) * fundamental_matrix_(1, 1) +
           fundamental_matrix_(2, 1)) +
      keypoint_frame1_undistorted(0) * fundamental_matrix_(0, 2) +
      keypoint_frame1_undistorted(1) * fundamental_matrix_(1, 2) +
      fundamental_matrix_(2, 2);

  VLOG(250) << "KP0: " << keypoint_frame0_undistorted
            << ", KP1: " << keypoint_frame1_undistorted << ", e = " << epipole;
  bool result = (epipole > -kEpipolarThreshold && epipole < kEpipolarThreshold);
  return result;
}

// bool StereoMatcher::calculateDepth(aslam::StereoMatchWithScore* match) {
std::pair<double, double> StereoMatcher::calculateDepth(
    const Eigen::Vector2d& keypoint_frame0,
    const Eigen::Vector2d& keypoint_frame1) {
  /* Triangulate point using method from Trucco E., Verri A. 1998.
   * Introductory Techniques for 3-D Computer Vision. See
   * https://pdfs.semanticscholar.org/675a/75494f55b0ac6092f6beef6ac413c296faf4.pdf
   * page 20 for a summary.
   *
   *  Solve vector triangle:
   *      a * K0.inv() * u0 + b * (K0.inv() x R * K1.inv() * u1) + c * R
   *          K1.inv() * u1 = T
   *   => a * p0 + b * d + c * p1 = T
   *
   *      a = (d2 p11 t0 - d1 p12 t0 - d2 p10 t1 + d0 p12 t1 + d1 p10 t2 -
   *            d0 p11 t2)/(d2 p01 p10 - d1 p02 p10 - d2 p00 p11 + d0 p02 p11
   *            + d1 p00 p12 - d0 p01 p12)
   *   => b = -((p02 p11 t0 - p01 p12 t0 - p02 p10 t1 + p00 p12 t1 + p01 p10
   * t2
   *            - p00 p11 t2)/(-d2 p01 p10 + d1 p02 p10 + d2 p00 p11 -
   *            d0 p02 p11 - d1 p00 p12 + d0 p01 p12))
   *      c = -((d2 p01 t0 - d1 p02 t0 - d2 p00 t1 + d0 p02 t1 + d1 p00 t2 -
   *            d0 p01 t2)/(-d2 p01 p10 + d1 p02 p10 + d2 p00 p11 - d0 p02 p11
   *            - d1 p00 p12 + d0 p01 p12))
   *
   *  Final 3d point can then be found as X = a * p0 + b/2 * d.
   *  The depth is calculated as D = X(2)
   */
  // Eigen::Vector2d keypoint_frame0 =
  //     frame0_->getKeypointMeasurement(match->getKeypointIndexFrame0());
  Eigen::Vector2d keypoint_frame0_undistorted;
  Eigen::Vector3d u0;
  first_mapped_undistorter_->processPoint(
      keypoint_frame0, &keypoint_frame0_undistorted);
  u0 << keypoint_frame0_undistorted, Eigen::Matrix<double, 1, 1>(1.0);

  // Eigen::Vector2d keypoint_frame1 =
  //     frame1_->getKeypointMeasurement(match->getKeypointIndexFrame1());
  Eigen::Vector2d keypoint_frame1_undistorted;
  Eigen::Vector3d u1;
  first_mapped_undistorter_->processPoint(
      keypoint_frame1, &keypoint_frame1_undistorted);
  u1 << keypoint_frame1_undistorted, Eigen::Matrix<double, 1, 1>(1.0);

  Eigen::Vector3d p0 = camera_matrix_C0_inv_ * u0;
  Eigen::Vector3d p1 = rotation_C1_C0_.transpose() * camera_matrix_C1_inv_ * u1;
  Eigen::Vector3d d = p0.cross(p1);
  Eigen::Vector3d t = -translation_C1_C0_;
  double a =
      -((d(2) * p1(1) * t(0) - d(1) * p1(2) * t(0) - d(2) * p1(0) * t(1) +
         d(0) * p1(2) * t(1) + d(1) * p1(0) * t(2) - d(0) * p1(1) * t(2)) /
        (d(2) * p0(1) * p1(0) - d(1) * p0(2) * p1(0) - d(2) * p0(0) * p1(1) +
         d(0) * p0(2) * p1(1) + d(1) * p0(0) * p1(2) - d(0) * p0(1) * p1(2)));
  double b =
      -((p0(2) * p1(1) * t(0) - p0(1) * p1(2) * t(0) - p0(2) * p1(0) * t(1) +
         p0(0) * p1(2) * t(1) + p0(1) * p1(0) * t(2) - p0(0) * p1(1) * t(2)) /
        (-d(2) * p0(1) * p1(0) + d(1) * p0(2) * p1(0) + d(2) * p0(0) * p1(1) -
         d(0) * p0(2) * p1(1) - d(1) * p0(0) * p1(2) + d(0) * p0(1) * p1(2)));
  /* Calculation of c is actually not needed.
  double c =
      -((d(2) * p0(1) * t(0) - d(1) * p0(2) * t(0) - d(2) * p0(0) * t(1) +
         d(0) * p0(2) * t(1) + d(1) * p0(0) * t(2) - d(0) * p0(1) * t(2)) /
        (-d(2) * p0(1) * p1(0) + d(1) * p0(2) * p1(0) + d(2) * p0(0) * p1(1) -
         d(0) * p0(2) * p1(1) - d(1) * p0(0) * p1(2) + d(0) * p0(1) * p1(2)));
         */

  Eigen::Vector3d X_cam0 = a * p0 + b / 2.0 * d;
  const double depth0 = calculateDepth(X_cam0, nullptr);
  Eigen::Vector3d X_cam1 = rotation_C1_C0_ * (X_cam0 - translation_C1_C0_);
  const double depth1 = calculateDepth(X_cam1, nullptr);
  return std::make_pair(depth0, depth1);
}

double StereoMatcher::calculateDepth(
    const Eigen::Vector3d& landmark,
    Eigen::Matrix<double, 1, 3>* out_jacobian_point) {
  if (out_jacobian_point) {
    const double dd_dx = 0;
    const double dd_dy = 0;
    const double dd_dz = 1;
    (*out_jacobian_point) << dd_dx, dd_dy, dd_dz;
  }

  return landmark(2);
}
}  // namespace aslam
