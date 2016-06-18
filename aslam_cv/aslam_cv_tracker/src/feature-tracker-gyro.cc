#include "aslam/tracker/feature-tracker-gyro.h"

#include <algorithm>

#include <Eigen/Core>
#include <glog/logging.h>

#include <aslam/cameras/camera.h>
#include <aslam/common/memory.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/gyro-two-frame-matcher.h>
#include <aslam/matcher/matching-helpers.h>
#include <opencv2/video/tracking.hpp>

#include "aslam/tracker/tracking-helpers.h"

DEFINE_double(gyro_lk_candidate_ratio, 0.4, "This ratio defines the number of "
    "unmatched (from frame k to (k+1)) keypoints that will be tracked with "
    "the lk tracker to the next frame. If we detect N keypoints in frame (k+1), "
    "we track at most 'N times this ratio' keypoints to frame (k+1) with the "
    "lk tracker.");
DEFINE_uint64(gyro_lk_max_status_track_length, 3u, "Status track length is the "
    "track length since the status of the keypoint has changed (e.g. from lk "
    "tracked to detected or the reverse). The lk tracker will not track "
    "keypoints with longer status track length than this value.");
DEFINE_int32(gyro_lk_window_size, 21, "Size of the search window at each "
    "pyramid level.");
DEFINE_int32(gyro_lk_max_pyramid_levels, 3, "If set to 0, pyramids are not "
    "used (single level), if set to 1, two levels are used, and so on. "
    "If pyramids are passed to the input then the algorithm will use as many "
    "levels as possible but not more than this threshold.");
DEFINE_double(gyro_lk_min_eigenvalue_threshold, 0.001, "The algorithm "
    "calculates the minimum eigenvalue of a 2x2 normal matrix of optical flow "
    "equations, divided by number of pixels in a window. If this value is less "
    "than this threshold, the corresponding feature is filtered out and its "
    "flow is not processed, so it allows to remove bad points and get a "
    "performance boost.");

namespace aslam {

GyroTrackerSettings::GyroTrackerSettings()
  : lk_max_num_candidates_ratio_kp1(FLAGS_gyro_lk_candidate_ratio),
    lk_max_status_track_length(FLAGS_gyro_lk_max_status_track_length),
    lk_termination_criteria(
        cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 50, 0.005)),
    lk_window_size(FLAGS_gyro_lk_window_size, FLAGS_gyro_lk_window_size),
    lk_max_pyramid_levels(FLAGS_gyro_lk_max_pyramid_levels),
    lk_operation_flag(cv::OPTFLOW_USE_INITIAL_FLOW),
    lk_min_eigenvalue_threshold(FLAGS_gyro_lk_min_eigenvalue_threshold) {
  CHECK_GE(lk_max_num_candidates_ratio_kp1, 0.0);
  CHECK_LE(lk_max_num_candidates_ratio_kp1, 1.0) <<
      "Higher values than 1.0 are possible. Change this check if you really "
      "want to enable tracking more keypoints with the lk-tracker than you "
      "are detecting in the next frame.";
  CHECK_GT(lk_max_status_track_length, 0u);
  CHECK_GT(FLAGS_gyro_lk_window_size, 0);
  CHECK_GE(lk_max_pyramid_levels, 0);
  CHECK_GT(lk_min_eigenvalue_threshold, 0.0);
}

GyroTracker::GyroTracker(const Camera& camera,
                         const size_t min_distance_to_image_border,
                         cv::Ptr<cv::DescriptorExtractor> extractor_ptr)
    : camera_(camera) ,
      kMinDistanceToImageBorderPx(min_distance_to_image_border),
      initialized_(false),
      extractor_(extractor_ptr) {
}

void GyroTracker::track(const Quaternion& q_Ckp1_Ck,
                        const VisualFrame& frame_k,
                        VisualFrame* frame_kp1,
                        MatchesWithScore* matches_with_score_kp1_k) {
  CHECK(frame_k.isValid());
  CHECK(frame_k.hasKeypointMeasurements());
  CHECK(frame_k.hasKeypointOrientations());
  CHECK(frame_k.hasKeypointScales());
  CHECK(frame_k.hasKeypointScores());
  CHECK(frame_k.hasTrackIds());
  CHECK(frame_k.hasDescriptors());
  CHECK(frame_k.hasRawImage());
  CHECK(CHECK_NOTNULL(frame_kp1)->isValid());
  CHECK(frame_kp1->hasKeypointMeasurements());
  CHECK(frame_kp1->hasKeypointOrientations());
  CHECK(frame_kp1->hasKeypointScales());
  CHECK(frame_kp1->hasKeypointScores());
  CHECK(frame_kp1->hasTrackIds());
  CHECK(frame_kp1->hasDescriptors());
  CHECK(frame_kp1->hasRawImage());
  CHECK_EQ(frame_kp1->getDescriptors().rows(),
           frame_kp1->getDescriptorSizeBytes());
  CHECK_EQ(frame_kp1->getKeypointMeasurements().cols(),
           frame_kp1->getDescriptors().cols());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(
      frame_k.getCameraGeometry().get())->getId());
  CHECK_EQ(camera_.getId(), CHECK_NOTNULL(
      frame_kp1->getCameraGeometry().get())->getId());
  CHECK_NOTNULL(matches_with_score_kp1_k)->clear();
  // Make sure the frames are in order time-wise
  CHECK_GT(frame_kp1->getTimestampNanoseconds(),
           frame_k.getTimestampNanoseconds());
  UpdateFramePointerDeque(frame_k);
  if (!initialized_) {
    InitializeFeatureStatusDeque();
  }

  // Predict keypoint positions for all keypoints in current frame k.
  Eigen::Matrix2Xd predicted_keypoint_positions_kp1;
  std::vector<unsigned char> prediction_success;
  predictKeypointsByRotation(frame_k, q_Ckp1_Ck,
                             &predicted_keypoint_positions_kp1,
                             &prediction_success);
  CHECK_EQ(prediction_success.size(), predicted_keypoint_positions_kp1.cols());

  // Match descriptors of frame k with those of frame (k+1).
  GyroTwoFrameMatcher matcher(
      q_Ckp1_Ck, *frame_kp1, frame_k, camera_.imageHeight(),
      predicted_keypoint_positions_kp1,
      prediction_success, matches_with_score_kp1_k);
  matcher.Match();

  // Compute LK candidates and track them.
  FrameStatusTrackLength status_track_length_k;
  std::vector<TrackedMatch> tracked_matches;
  std::vector<int> lk_candidate_indices_k;

  ComputeTrackedMatches(&tracked_matches);
  ComputeStatusTrackLengthOfFrameK(tracked_matches, &status_track_length_k);
  ComputeLKCandidates(*matches_with_score_kp1_k, status_track_length_k,
                      *frame_kp1, &lk_candidate_indices_k);
  LKTracking(predicted_keypoint_positions_kp1, prediction_success,
             lk_candidate_indices_k, frame_kp1, matches_with_score_kp1_k);

  status_track_length_km1_.swap(status_track_length_k);
  initialized_ = true;
}

void GyroTracker::LKTracking(
      const Eigen::Matrix2Xd& predicted_keypoint_positions_kp1,
      const std::vector<unsigned char>& prediction_success,
      const std::vector<int>& lk_candidate_indices_k,
      VisualFrame* frame_kp1,
      MatchesWithScore* matches_with_score_kp1_k) {
  CHECK_NOTNULL(frame_kp1);
  CHECK_NOTNULL(matches_with_score_kp1_k);
  CHECK_GT(frames_k_km1_.size(), 0u);
  CHECK_EQ(prediction_success.size(), predicted_keypoint_positions_kp1.cols());

  if (lk_candidate_indices_k.empty()) {
    VLOG(3) << "No LK candidates to track.";
    return;
  }

  // Definite lk indices are the subset of lk candidate indices with
  // successfully predicted keypoint locations in frame (k+1).
  std::vector<int> lk_definite_indices_k;
  for (const int candidate_index_k: lk_candidate_indices_k) {
    if (prediction_success[candidate_index_k] == 1) {
      lk_definite_indices_k.push_back(candidate_index_k);
    }
  }

  // Get definite lk keypoint locations in OpenCV format.
  std::vector<cv::Point2f> lk_cv_points_k;
  std::vector<cv::Point2f> lk_cv_points_kp1;
  lk_cv_points_k.reserve(lk_definite_indices_k.size());
  lk_cv_points_kp1.reserve(lk_definite_indices_k.size());
  for (const int lk_definite_index_k: lk_definite_indices_k) {
    // Compute Cv points in frame k.
    const Eigen::Matrix<double, 2, 1>& lk_keypoint_location_k =
        frames_k_km1_[0]->getKeypointMeasurement(lk_definite_index_k);
    lk_cv_points_k.emplace_back(
        static_cast<float>(lk_keypoint_location_k(0,0)),
        static_cast<float>(lk_keypoint_location_k(0,1)));
    // Compute predicted locations in frame (k+1).
    lk_cv_points_kp1.emplace_back(
        static_cast<float>(predicted_keypoint_positions_kp1(0, lk_definite_index_k)),
        static_cast<float>(predicted_keypoint_positions_kp1(1, lk_definite_index_k)));
  }

  std::vector<unsigned char> lk_tracking_success;
  std::vector<float> lk_tracking_errors;

  // TODO(magehrig): use predicted locations. Otherwise optflow will fail.
  cv::calcOpticalFlowPyrLK(
      frames_k_km1_[0]->getRawImage(), frame_kp1->getRawImage(), lk_cv_points_k,
      lk_cv_points_kp1, lk_tracking_success, lk_tracking_errors,
      settings.lk_window_size, settings.lk_max_pyramid_levels,
      settings.lk_termination_criteria, settings.lk_operation_flag,
      settings.lk_min_eigenvalue_threshold);

  CHECK_EQ(lk_tracking_success.size(), lk_candidate_indices_k.size());
  CHECK_EQ(lk_cv_points_kp1.size(), lk_tracking_success.size());
  CHECK_EQ(lk_cv_points_k.size(), lk_cv_points_kp1.size());

  std::function<bool(const cv::Point2f&)> is_outside_roi =
      [this](const cv::Point2f& point) -> bool {
    return point.x < kMinDistanceToImageBorderPx ||
         point.x >= (camera_.imageWidth() - kMinDistanceToImageBorderPx) ||
         point.y < kMinDistanceToImageBorderPx ||
         point.y >= (camera_.imageHeight() - kMinDistanceToImageBorderPx);
  };

  std::unordered_set<size_t> indices_to_erase;
  for (size_t i = 0u; i < lk_tracking_success.size(); ++i) {
    if (lk_tracking_success[i] == 0u || is_outside_roi(lk_cv_points_kp1[i])) {
      indices_to_erase.insert(i);
    }
  }
  EraseVectorElementsHelper(indices_to_erase, &lk_definite_indices_k);
  EraseVectorElementsHelper(indices_to_erase, &lk_cv_points_kp1);

  const size_t num_points_successfully_tracked = lk_cv_points_kp1.size();

  // Convert Cv points to Cv keypoints because this format is
  // required for descriptor extraction. Take relevant keypoint information
  // (such as score and size) from frame k.
  // Assign unique class_id to keypoints because some of them will get removed
  // during the extraction phase and we want to be able to identify them.
  std::vector<cv::KeyPoint> lk_cv_keypoints_kp1;
  lk_cv_keypoints_kp1.reserve(num_points_successfully_tracked);
  for (size_t i = 0u; i < num_points_successfully_tracked; ++i) {
    const size_t channel_idx = lk_definite_indices_k[i];
    const int class_id = static_cast<int>(i);
    lk_cv_keypoints_kp1.emplace_back(
        lk_cv_points_kp1[i], frames_k_km1_[0]->getKeypointScale(channel_idx),
        frames_k_km1_[0]->getKeypointOrientation(channel_idx),
        frames_k_km1_[0]->getKeypointScore(channel_idx),
        0 /* octave info not used by extractor */, class_id);
  }

  cv::Mat lk_descriptors_kp1;
  extractor_->compute(frame_kp1->getRawImage(), lk_cv_keypoints_kp1, lk_descriptors_kp1);

  CHECK_EQ(lk_descriptors_kp1.type(), CV_8UC1);
  CHECK(lk_descriptors_kp1.isContinuous());

  const size_t num_points_after_extraction = lk_cv_keypoints_kp1.size();

  const int initial_size_kp1 = frame_kp1->getTrackIds().size();
  for (int i = 0; i < static_cast<int>(num_points_after_extraction); ++i) {
    matches_with_score_kp1_k->emplace_back(
        initial_size_kp1 + i, lk_definite_indices_k[lk_cv_keypoints_kp1[i].class_id], 0.0);
  }

  // Add keypoints and descriptors to frame (k+1).
  insertAdditionalCvKeypointsAndDescriptorsToVisualFrame(
      lk_cv_keypoints_kp1, lk_descriptors_kp1,
      settings.kKeypointUncertaintyPx, frame_kp1);

  // Update feature status for next iteration.
  const size_t extended_size_pk1 = static_cast<size_t>(initial_size_kp1) +
      num_points_after_extraction;
  FrameFeatureStatus frame_feature_status_kp1(extended_size_pk1);
  std::fill(frame_feature_status_kp1.begin(), frame_feature_status_kp1.begin() + initial_size_kp1, FeatureStatus::kDetected);
  std::fill(frame_feature_status_kp1.begin() + initial_size_kp1, frame_feature_status_kp1.end(), FeatureStatus::kLkTracked);
  UpdateFeatureStatusDeque(frame_feature_status_kp1);
}

void GyroTracker::ComputeTrackedMatches(
      std::vector<TrackedMatch>* tracked_matches) const {
  CHECK_NOTNULL(tracked_matches)->clear();
  if (!initialized_) {
    return;
  }
  CHECK_EQ(frames_k_km1_.size(), 2u);

  // Get the index of an integer value in a vector of unique elements.
  // Return -1 if there is no such value in the vector.
  std::function<int(const std::vector<int>&, const int)> GetIndexOfValue =
      [](const std::vector<int>& vec, const int value) -> int {
    std::vector<int>::const_iterator iter = std::find(vec.begin(), vec.end(), value);
    if (iter == vec.end()) {
      return -1;
    } else {
      return static_cast<int>(std::distance(vec.begin(), iter));
    }
  };

  const Eigen::VectorXi& track_ids_k_eigen = frames_k_km1_[0]->getTrackIds();
  const Eigen::VectorXi& track_ids_km1_eigen = frames_k_km1_[1]->getTrackIds();
  const std::vector<int> track_ids_km1(
      track_ids_km1_eigen.data(),
      track_ids_km1_eigen.data() + track_ids_km1_eigen.size());
  for (int index_k = 0; index_k < track_ids_k_eigen.size(); ++index_k) {
    const int index_km1 = GetIndexOfValue(track_ids_km1, track_ids_k_eigen(index_k));
    if (index_km1 >= 0) {
      tracked_matches->emplace_back(index_k, index_km1);
    }
  }
}

void GyroTracker::ComputeLKCandidates(
    const MatchesWithScore& matches_with_score_kp1_k,
    const FrameStatusTrackLength& status_track_length_k,
    const VisualFrame& frame_kp1,
    std::vector<int>* lk_candidate_indices_k) const {
  CHECK_NOTNULL(lk_candidate_indices_k)->clear();
  CHECK_EQ(status_track_length_k.size(), frames_k_km1_[0]->getTrackIds().size());

  std::vector<int> unmatched_indices_k;
  ComputeUnmatchedIndicesOfFrameK(
      matches_with_score_kp1_k, &unmatched_indices_k);

  std::vector<std::pair<int, size_t>> indices_detected_and_tracked;
  std::vector<std::pair<int, double>> indices_detected_and_untracked;
  std::vector<std::pair<int, size_t>> indices_lktracked;
  for (const int unmatched_index_k: unmatched_indices_k) {
    const size_t current_status_track_length =
        status_track_length_k.at(unmatched_index_k);
    const FeatureStatus current_feature_status =
        feature_status_k_km1_[0].at(unmatched_index_k);
    if (current_feature_status == FeatureStatus::kDetected) {
      if (current_status_track_length > 0u) {
        // These candidates have the highest priority as lk candidates.
        // The most valuable candidates have the longest status track length.
        indices_detected_and_tracked.emplace_back(
            unmatched_index_k, current_status_track_length);
      } else {
        // These candidates have the medium priority as lk candidates.
        // The most valuable candidates have the highest keypoint scores.
        const double keypoint_score =
            frames_k_km1_[0]->getKeypointScore(unmatched_index_k);
        indices_detected_and_untracked.emplace_back(unmatched_index_k, keypoint_score);
      }
    } else if (current_feature_status == FeatureStatus::kLkTracked) {
      if (current_status_track_length < settings.lk_max_status_track_length) {
        // These candidates have the lowest priority as lk candidates.
        // The most valuable candidates have the shortest status track length.
        indices_lktracked.emplace_back(
            unmatched_index_k, current_status_track_length);
      }
    } else {
      LOG(FATAL) << "Unknown feature status.";
    }
  }

  const size_t kNumPointsKp1 = frame_kp1.getTrackIds().size();
  const size_t kLkNumCandidatesBeforeCutoff =
      indices_detected_and_tracked.size() +
      indices_detected_and_untracked.size() + indices_lktracked.size();
  const size_t kLkNumMaxCandidates = static_cast<size_t>(
      kNumPointsKp1*settings.lk_max_num_candidates_ratio_kp1);
  const size_t kNumLkCandidatesAfterCutoff = std::min(
      kLkNumCandidatesBeforeCutoff, kLkNumMaxCandidates);
  lk_candidate_indices_k->reserve(kNumLkCandidatesAfterCutoff);

  // Only sort the indices that are possible candidates.
  if (kLkNumCandidatesBeforeCutoff > kLkNumMaxCandidates) {
    std::sort(indices_detected_and_tracked.begin(),
              indices_detected_and_tracked.end(),
              [](const std::pair<int, size_t>& lhs,
                  const std::pair<int, size_t>& rhs) -> bool {
      return lhs.second > rhs.second;
    });
    if (indices_detected_and_tracked.size() < kLkNumMaxCandidates) {
      std::sort(indices_detected_and_untracked.begin(),
                indices_detected_and_untracked.end(),
                [](const std::pair<int, double>& lhs,
                    const std::pair<int, double>& rhs) -> bool {
        return lhs.second > rhs.second;
      });
      if (indices_detected_and_tracked.size() +
          indices_detected_and_untracked.size() < kLkNumMaxCandidates) {
        std::sort(indices_lktracked.begin(),
                  indices_lktracked.end(),
                  [](const std::pair<int, size_t>& lhs,
                      const std::pair<int, size_t>& rhs) -> bool {
          return lhs.second < rhs.second;
        });
      }
    }
  }

  // Construct candidate vector based on sorted candidate indices
  // until max number of candidates is reached.
  size_t counter = 0u;
  for (const std::pair<int, size_t>& pair: indices_detected_and_tracked) {
    if (counter == kLkNumMaxCandidates) break;
    lk_candidate_indices_k->push_back(pair.first);
    ++counter;
  }
  for (const std::pair<int, double>& pair: indices_detected_and_untracked) {
    if (counter == kLkNumMaxCandidates) break;
    lk_candidate_indices_k->push_back(pair.first);
    ++counter;
  }
  for (const std::pair<int, size_t>& pair: indices_lktracked) {
    if (counter == kLkNumMaxCandidates) break;
    lk_candidate_indices_k->push_back(pair.first);
    ++counter;
  }
}

void GyroTracker::ComputeUnmatchedIndicesOfFrameK(
    const MatchesWithScore& matches_with_score_kp1_k,
    std::vector<int>* unmatched_indices_k) const {
  CHECK_GT(frames_k_km1_.size(), 0u);
  CHECK_GE(frames_k_km1_[0]->getTrackIds().size(), matches_with_score_kp1_k.size());
  CHECK_NOTNULL(unmatched_indices_k)->clear();

  const size_t num_points_k = frames_k_km1_[0]->getTrackIds().size();
  const size_t num_matches_k = matches_with_score_kp1_k.size();
  const size_t num_unmatched_k = num_points_k - num_matches_k;

  unmatched_indices_k->reserve(num_unmatched_k);
  std::vector<bool> is_unmatched(num_points_k, true);

  for (const MatchWithScore& match: matches_with_score_kp1_k) {
    is_unmatched.at(match.correspondence[1]) = false;
  }

  for (int i = 0; i < static_cast<int>(num_points_k); ++i) {
    if (is_unmatched[i]) {
      unmatched_indices_k->push_back(i);
    }
  }

  CHECK_EQ(unmatched_indices_k->size(), num_unmatched_k);
  CHECK_EQ(num_matches_k + unmatched_indices_k->size(),
           num_points_k);
}

void GyroTracker::ComputeStatusTrackLengthOfFrameK(
    const std::vector<TrackedMatch>& tracked_matches,
    FrameStatusTrackLength* status_track_length_k) {
  CHECK_NOTNULL(status_track_length_k)->clear();
  CHECK_GT(frames_k_km1_.size(), 0u);

  const int num_points_k = frames_k_km1_[0]->getTrackIds().size();
  status_track_length_k->assign(num_points_k, 0u);

  if (!initialized_) {
    return;
  }
  CHECK_EQ(feature_status_k_km1_.size(), 2u);
  CHECK_GT(status_track_length_km1_.size(), 0u);

  for (const TrackedMatch& match: tracked_matches) {
    const int match_index_k = match.correspondence[0];
    const int match_index_km1 = match.correspondence[1];
    if (feature_status_k_km1_[1].at(match_index_km1) !=
        feature_status_k_km1_[0].at(match_index_k)) {
      // Reset the status track length to 1 because the status of this
      // particular tracked keypoint has changed from frame (k-1) to k.
      status_track_length_k->at(match_index_k) = 1u;
    } else {
      status_track_length_k->at(match_index_k) =
          status_track_length_km1_.at(match_index_km1) + 1u;
    }
  }
}

void GyroTracker::InitializeFeatureStatusDeque() {
  CHECK_EQ(frames_k_km1_.size(), 1u);
  const size_t num_points_k = frames_k_km1_[0]->getTrackIds().size();
  FrameFeatureStatus frame_feature_status_k(num_points_k, FeatureStatus::kDetected);
  UpdateFeatureStatusDeque(frame_feature_status_k);
}

void GyroTracker::UpdateFeatureStatusDeque(
    const FrameFeatureStatus& frame_feature_status_kp1) {
  feature_status_k_km1_.emplace_front(
      frame_feature_status_kp1.begin(), frame_feature_status_kp1.end());
  if (feature_status_k_km1_.size() == 3u) {
    feature_status_k_km1_.pop_back();
  }
}

void GyroTracker::UpdateFramePointerDeque(
    const VisualFrame& new_frame_k) {
  CHECK(new_frame_k.hasKeypointMeasurements());
  CHECK(new_frame_k.hasDescriptors());
  CHECK(new_frame_k.hasKeypointMeasurementUncertainties());
  CHECK(new_frame_k.hasKeypointOrientations());
  CHECK(new_frame_k.hasKeypointScales());
  CHECK(new_frame_k.hasKeypointScores());
  CHECK(new_frame_k.hasTrackIds());

  frames_k_km1_.push_front(&new_frame_k);
  if (frames_k_km1_.size() == 3u) {
    frames_k_km1_.pop_back();
  }
  if (frames_k_km1_.size() >= 1u) {
    CHECK(frames_k_km1_[0]->hasKeypointMeasurements());
    CHECK(frames_k_km1_[0]->hasDescriptors());
    CHECK(frames_k_km1_[0]->hasKeypointMeasurementUncertainties());
    CHECK(frames_k_km1_[0]->hasKeypointOrientations());
    CHECK(frames_k_km1_[0]->hasKeypointScales());
    CHECK(frames_k_km1_[0]->hasKeypointScores());
    CHECK(frames_k_km1_[0]->hasTrackIds());
    if (frames_k_km1_.size() == 2u) {
      CHECK(frames_k_km1_[1]->hasKeypointMeasurements());
      CHECK(frames_k_km1_[1]->hasDescriptors());
      CHECK(frames_k_km1_[1]->hasKeypointMeasurementUncertainties());
      CHECK(frames_k_km1_[1]->hasKeypointOrientations());
      CHECK(frames_k_km1_[1]->hasKeypointScales());
      CHECK(frames_k_km1_[1]->hasKeypointScores());
      CHECK(frames_k_km1_[1]->hasTrackIds());
    }
  }
}

}  //namespace aslam
