#ifndef ASLAM_MATCHER_MATCH_HELPERS_INL_H_
#define ASLAM_MATCHER_MATCH_HELPERS_INL_H_

#include <aslam/common/stl-helpers.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-frame-to-frame.h"
#include "aslam/matcher/matching-problem-landmarks-to-frame.h"

namespace aslam {

template<typename MatchWithScore, typename Match>
void convertMatches(
    const typename Aligned<std::vector, MatchWithScore>::type& matches_with_score_A_B,
    typename Aligned<std::vector, Match>::type* matches_A_B) {
  CHECK_NOTNULL(matches_A_B)->clear();
  matches_A_B->reserve(matches_with_score_A_B.size());
  for (const MatchWithScore& match : matches_with_score_A_B) {
    CHECK_GE(match.getIndexApple(), 0) << "The apple index is negative.";
    CHECK_GE(match.getIndexBanana(), 0) << "The banana index is negative.";
    matches_A_B->emplace_back(static_cast<size_t>(match.getIndexApple()),
                              static_cast<size_t>(match.getIndexBanana()));
  }
  CHECK_EQ(matches_with_score_A_B.size(), matches_A_B->size());
}

template<typename MatchWithScore>
void convertMatches(
    const typename Aligned<std::vector, MatchWithScore>::type& matches_with_score_A_B,
    OpenCvMatches* matches_A_B) {
  CHECK_NOTNULL(matches_A_B)->clear();
  matches_A_B->reserve(matches_with_score_A_B.size());
  for (const MatchWithScore& match : matches_with_score_A_B) {
    CHECK_GE(match.getIndexApple(), 0) << "The apple index is negative.";
    CHECK_GE(match.getIndexBanana(), 0) << "The banana index is negative.";
    matches_A_B->emplace_back(match.getIndexApple(), match.getIndexBanana(),
                              static_cast<float>(match.getScore()));
  }
  CHECK_EQ(matches_with_score_A_B.size(), matches_A_B->size());
}

inline void convertMatches(const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B) {
  convertMatches<MatchWithScore, Match>(matches_with_score_A_B, matches_A_B);
}

template<typename MatchesWithScore>
void convertMatches(const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B) {
  aslam::MatchesWithScore aslam_matches_with_score_A_B;
  aslam_matches_with_score_A_B.reserve(matches_with_score_A_B.size());
  for (const typename MatchesWithScore::value_type& match : matches_with_score_A_B) {
    aslam_matches_with_score_A_B.emplace_back(match);
  }
  convertMatches(aslam_matches_with_score_A_B, matches_A_B);
}

/// Convert MatchesWithScore to Matches.
template<typename MatchingProblem>
void convertMatches(const typename MatchingProblem::MatchesWithScore& matches_with_score_A_B,
                    typename MatchingProblem::Matches* matches_A_B) {
  convertMatches<typename MatchingProblem::MatchWithScore, typename MatchingProblem::Match>(
      matches_with_score_A_B, matches_A_B);
}

/// Get number of matches for a rig match list. (outer vector = cameras, inner vector = match list)
template<typename MatchType>
size_t countRigMatches(
    const typename Aligned<std::vector,
                           typename Aligned<std::vector,MatchType>::type>::type& rig_matches) {
  size_t num_matches = 0;
  for (const typename Aligned<std::vector, MatchType>::type& camera_matches : rig_matches) {
    num_matches += camera_matches.size();
  }
  return num_matches;
}

/// Select and return N random matches for each camera in the rig.
template<typename MatchesType>
typename Aligned<std::vector, MatchesType>::type pickNRandomRigMatches(
    size_t n_per_camera, const typename Aligned<std::vector, MatchesType>::type& rig_matches) {
  CHECK_GT(n_per_camera, 0u);
  size_t num_cameras = rig_matches.size();
  typename Aligned<std::vector, MatchesType>::type subsampled_rig_matches(num_cameras);

  for (size_t cam_idx = 0; cam_idx < num_cameras; ++cam_idx) {
    const MatchesType& camera_matches = rig_matches[cam_idx];
    if (camera_matches.size() <= n_per_camera) {
      subsampled_rig_matches[cam_idx] = camera_matches;
    } else {
      common::drawNRandomElements(n_per_camera, camera_matches, &subsampled_rig_matches[cam_idx]);
    }
  }
  CHECK_EQ(rig_matches.size(), subsampled_rig_matches.size());
  return subsampled_rig_matches;
}

/// Get the matches based on the track id channels for one VisualFrame.
template<typename MatchesType>
size_t extractMatchesFromTrackIdChannel(const aslam::VisualFrame& frame_kp1,
                                        const aslam::VisualFrame& frame_k,
                                        MatchesType* matches_kp1_kp) {
  CHECK_NOTNULL(matches_kp1_kp);
  CHECK_EQ(frame_kp1.getRawCameraGeometry().get(), frame_k.getRawCameraGeometry().get());
  const Eigen::VectorXi& track_ids_kp1 = frame_kp1.getTrackIds();
  const Eigen::VectorXi& track_ids_k = frame_k.getTrackIds();

  // Build trackid <-> keypoint_idx lookup table.
  typedef std::unordered_map<int, size_t> TrackIdKeypointIdxMap;
  TrackIdKeypointIdxMap track_id_kp1_keypoint_idx_kp1_map;
  for (int keypoint_idx_kp1 = 0; keypoint_idx_kp1 < track_ids_kp1.rows(); ++keypoint_idx_kp1) {
    int track_id_kp1 = track_ids_kp1(keypoint_idx_kp1);
    // Skip unassociated keypoints.
    if(track_id_kp1 < 0)
      continue;
    track_id_kp1_keypoint_idx_kp1_map.insert(std::make_pair(track_id_kp1, keypoint_idx_kp1));
  }
  CHECK_LE(track_id_kp1_keypoint_idx_kp1_map.size(), frame_kp1.getNumKeypointMeasurements());

  // Create indices matches vector using the lookup table.
  matches_kp1_kp->clear();
  matches_kp1_kp->reserve(1000);
  for (int keypoint_idx_k = 0; keypoint_idx_k < track_ids_k.rows(); ++keypoint_idx_k) {
    int track_id_k = track_ids_k(keypoint_idx_k);
    if(track_id_k < 0)
      continue;
    TrackIdKeypointIdxMap::const_iterator it = track_id_kp1_keypoint_idx_kp1_map.find(track_id_k);
    if (it != track_id_kp1_keypoint_idx_kp1_map.end()) {
      size_t keypoint_idx_kp1 = it->second;
      matches_kp1_kp->emplace_back(keypoint_idx_kp1, keypoint_idx_k);
    }
  }
  return matches_kp1_kp->size();
}

/// Get the matches based on the track id channels for one VisualNFrame.
template<typename MatchesType>
size_t extractMatchesFromTrackIdChannels(
    const aslam::VisualNFrame& nframe_kp1, const aslam::VisualNFrame& nframe_k,
    typename Aligned<std::vector, MatchesType>::type* rig_matches_kp1_kp) {
  CHECK_NOTNULL(rig_matches_kp1_kp);
  CHECK_EQ(nframe_kp1.getNCameraShared().get(), nframe_k.getNCameraShared().get());

  size_t num_cameras = nframe_kp1.getNumCameras();
  rig_matches_kp1_kp->clear();
  rig_matches_kp1_kp->resize(num_cameras);

  size_t num_matches = 0;
  for (size_t cam_idx = 0; cam_idx < nframe_kp1.getNumCameras(); ++cam_idx) {
    num_matches += extractMatchesFromTrackIdChannel(nframe_kp1.getFrame(cam_idx),
                                                    nframe_k.getFrame(cam_idx),
                                                    &(*rig_matches_kp1_kp)[cam_idx]);
  }
  return num_matches;
}

/// Get the median pixel disparity for all matches.
template<typename MatchesType>
 double getMatchPixelDisparityMedian(
      const aslam::VisualNFrame& nframe_kp1, const aslam::VisualNFrame& nframe_k,
      const typename Aligned<std::vector, MatchesType>::type& matches_kp1_kp) {
  aslam::Quaternion q_kp1_kp;
  q_kp1_kp.setIdentity();
  return getUnrotatedMatchPixelDisparityMedian(nframe_kp1, nframe_k, matches_kp1_kp, q_kp1_kp);
}

/// Get the median pixel disparity for all matches, taking into account the relative
/// orientation of the frames.
template<typename MatchType>
double getUnrotatedMatchPixelDisparityMedian(
    const aslam::VisualNFrame& nframe_kp1, const aslam::VisualNFrame& nframe_k,
    const typename Aligned<std::vector,
                           typename Aligned<std::vector, MatchType>::type>::type& matches_kp1_k,
    const aslam::Quaternion& q_kp1_k) {
  CHECK_EQ(nframe_kp1.getNCameraShared().get(), nframe_k.getNCameraShared().get());

  const size_t num_cameras = nframe_kp1.getNumCameras();
  CHECK_EQ(matches_kp1_k.size(), num_cameras);
  const size_t num_matches = countRigMatches<MatchType>(matches_kp1_k);
  std::vector<double> disparity_px;
  disparity_px.reserve(num_matches);
  size_t projection_failed_counter = 0u;

  if (std::fabs(q_kp1_k.w()) == 1.0) {
    // Case with no rotation specified, directly calculate the disparity from the image plane
    // measurements.
    for (size_t cam_idx = 0u; cam_idx < num_cameras; ++cam_idx) {
      const Eigen::Matrix2Xd& keypoints_kp1 =
          nframe_kp1.getFrame(cam_idx).getKeypointMeasurements();
      const Eigen::Matrix2Xd& keypoints_k = nframe_k.getFrame(cam_idx).getKeypointMeasurements();
      for (const MatchType& match_kp1_kp : matches_kp1_k[cam_idx]) {
        CHECK_LT(static_cast<int>(match_kp1_kp.first), keypoints_kp1.cols());
        CHECK_LT(static_cast<int>(match_kp1_kp.second), keypoints_k.cols());
        disparity_px.emplace_back((keypoints_kp1.col(match_kp1_kp.first)
            - keypoints_k.col(match_kp1_kp.second)).norm());
      }
    }
  } else {
    // Non-identity rotation case.
    for (size_t cam_idx = 0u; cam_idx < num_cameras; ++cam_idx) {
      std::vector<size_t> keypoint_indices_k;
      keypoint_indices_k.reserve(matches_kp1_k[cam_idx].size());
      for (const MatchType& match_kp1_kp : matches_kp1_k[cam_idx]) {
        CHECK_LT(static_cast<int>(match_kp1_kp.second),
                 nframe_k.getFrame(cam_idx).getNumKeypointMeasurements());
        keypoint_indices_k.emplace_back(match_kp1_kp.second);
      }

      std::vector<unsigned char> success;
      success.reserve(keypoint_indices_k.size());
      Eigen::Matrix3Xd bearing_vectors_k =
          nframe_k.getFrame(cam_idx).getNormalizedBearingVectors(
              keypoint_indices_k, &success);
      CHECK_EQ(static_cast<int>(success.size()), bearing_vectors_k.cols());
      CHECK_EQ(static_cast<int>(matches_kp1_k[cam_idx].size()), bearing_vectors_k.cols());

      if (bearing_vectors_k.cols() == 0u) {
        continue;
      }

      // Rotate the bearing vectors into the frame_kp1 coordinates.
      Eigen::Matrix3Xd bearing_vectors_k_kp1 = q_kp1_k.rotateVectorized(bearing_vectors_k);

      // Project the bearing vectors to the frame kp1 and calculate the disparity.
      const Eigen::Matrix2Xd& keypoints_kp1 =
          nframe_kp1.getFrame(cam_idx).getKeypointMeasurements();
      for (int i = 0; i < bearing_vectors_k_kp1.cols(); ++i) {
        if (success[i]) {
          Eigen::Vector2d rotated_k_keypoint;
          aslam::ProjectionResult projection_result =
              nframe_kp1.getCamera(cam_idx).project3(bearing_vectors_k_kp1.col(i),
                                                     &rotated_k_keypoint);
          if (projection_result == aslam::ProjectionResult::KEYPOINT_VISIBLE) {
            const size_t kp1_match_index = matches_kp1_k[cam_idx][i].first;
            CHECK_LT(static_cast<int>(kp1_match_index), keypoints_kp1.cols());
            disparity_px.emplace_back((keypoints_kp1.col(kp1_match_index)
                - rotated_k_keypoint).norm());
          } else {
            ++projection_failed_counter;
          }
        } else {
          ++projection_failed_counter;
        }
      }
    }
  }
  CHECK_EQ(disparity_px.size() + projection_failed_counter, num_matches);
  return aslam::common::median(disparity_px.begin(), disparity_px.end());
}

/// Return the normalized bearing vectors for a list of single camera matches.
template<typename MatchType>
void getBearingVectorsFromMatches(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const typename Aligned<std::vector, MatchType>::type& matches_kp1_k,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_kp1,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_k) {
  CHECK_NOTNULL(bearing_vectors_kp1);
  CHECK_NOTNULL(bearing_vectors_k);

  const size_t num_matches = matches_kp1_k.size();
  std::vector<size_t> keypoint_indices_kp1;
  keypoint_indices_kp1.reserve(num_matches);
  std::vector<size_t> keypoint_indices_k;
  keypoint_indices_k.reserve(num_matches);

  keypoint_indices_kp1.reserve(matches_kp1_k.size());
  keypoint_indices_k.reserve(matches_kp1_k.size());
  for (const MatchType& match_kp1_k : matches_kp1_k) {
    keypoint_indices_kp1.emplace_back(match_kp1_k.first);
    keypoint_indices_k.emplace_back(match_kp1_k.second);
  }

  std::vector<unsigned char> success;
  aslam::common::convertEigenToStlVector(frame_kp1.getNormalizedBearingVectors(
      keypoint_indices_kp1, &success), bearing_vectors_kp1);
  aslam::common::convertEigenToStlVector(frame_k.getNormalizedBearingVectors(
            keypoint_indices_k, &success), bearing_vectors_k);
}

}  // namespace aslam

#endif  // ASLAM_MATCHER_MATCH_HELPERS_INL_H_
