#include "aslam/matcher/match-helpers.h"

#include <aslam/cameras/camera.h>
#include <aslam/common/stl-helpers.h>
#include <aslam/frames/visual-frame.h>
#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {

/// Select and return N random matches for each camera in the rig.
void pickNRandomRigMatches(
    size_t n_per_camera, const FrameToFrameMatchesList& rig_matches,
    FrameToFrameMatchesList* selected_rig_matches) {
  CHECK_NOTNULL(selected_rig_matches)->clear();
  CHECK_GT(n_per_camera, 0u);
  const size_t num_cameras = rig_matches.size();
  selected_rig_matches->resize(num_cameras);

  for (size_t cam_idx = 0u; cam_idx < num_cameras; ++cam_idx) {
    const aslam::FrameToFrameMatches& camera_matches = rig_matches[cam_idx];
    if (camera_matches.size() <= n_per_camera) {
      (*selected_rig_matches)[cam_idx] = camera_matches;
    } else {
      common::drawNRandomElements(n_per_camera, camera_matches, &(*selected_rig_matches)[cam_idx]);
    }
  }
  CHECK_EQ(num_cameras, selected_rig_matches->size());
}

/// Get the matches based on the track id channels for one VisualFrame.
size_t extractMatchesFromTrackIdChannel(const VisualFrame& frame_kp1,
                                        const VisualFrame& frame_k,
                                        FrameToFrameMatches* matches_kp1_kp) {
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
size_t extractMatchesFromTrackIdChannels(
    const VisualNFrame& nframe_kp1, const VisualNFrame& nframe_k,
    FrameToFrameMatchesList* rig_matches_kp1_kp) {
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
double getMatchPixelDisparityMedian(
      const VisualNFrame& nframe_kp1, const VisualNFrame& nframe_k,
      const FrameToFrameMatchesList& matches_kp1_kp) {
  aslam::Quaternion q_kp1_kp;
  q_kp1_kp.setIdentity();
  return getUnrotatedMatchPixelDisparityMedian(nframe_kp1, nframe_k, matches_kp1_kp, q_kp1_kp);
}

/// Get the median pixel disparity for all matches, taking into account the relative
/// orientation of the frames.
double getUnrotatedMatchPixelDisparityMedian(
    const VisualNFrame& nframe_kp1, const VisualNFrame& nframe_k,
    const FrameToFrameMatchesList& matches_kp1_k,
    const aslam::Quaternion& q_kp1_k) {
  CHECK_EQ(nframe_kp1.getNCameraShared().get(), nframe_k.getNCameraShared().get());

  const size_t num_cameras = nframe_kp1.getNumCameras();
  CHECK_EQ(matches_kp1_k.size(), num_cameras);
  const size_t num_matches =
      aslam::common::countNumberOfElementsInNestedList(matches_kp1_k);
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
      for (const FrameToFrameMatch& match_kp1_kp : matches_kp1_k[cam_idx]) {
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
      for (const FrameToFrameMatch& match_kp1_kp : matches_kp1_k[cam_idx]) {
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
void getBearingVectorsFromMatches(
    const VisualFrame& frame_kp1, const VisualFrame& frame_k,
    const FrameToFrameMatches& matches_kp1_k,
    Aligned<std::vector, Eigen::Vector3d>* bearing_vectors_kp1,
    Aligned<std::vector, Eigen::Vector3d>* bearing_vectors_k) {
  CHECK_NOTNULL(bearing_vectors_kp1);
  CHECK_NOTNULL(bearing_vectors_k);

  const size_t num_matches = matches_kp1_k.size();
  std::vector<size_t> keypoint_indices_kp1;
  keypoint_indices_kp1.reserve(num_matches);
  std::vector<size_t> keypoint_indices_k;
  keypoint_indices_k.reserve(num_matches);

  keypoint_indices_kp1.reserve(matches_kp1_k.size());
  keypoint_indices_k.reserve(matches_kp1_k.size());
  for (const FrameToFrameMatch& match_kp1_k : matches_kp1_k) {
    keypoint_indices_kp1.emplace_back(match_kp1_k.first);
    keypoint_indices_k.emplace_back(match_kp1_k.second);
  }

  std::vector<unsigned char> success;
  aslam::common::convertEigenToStlVector(frame_kp1.getNormalizedBearingVectors(
      keypoint_indices_kp1, &success), bearing_vectors_kp1);
  aslam::common::convertEigenToStlVector(frame_k.getNormalizedBearingVectors(
            keypoint_indices_k, &success), bearing_vectors_k);
}

void predictKeypointsByRotation(const VisualFrame& frame_k,
                                const aslam::Quaternion& q_Ckp1_Ck,
                                Eigen::Matrix2Xd* predicted_keypoints_kp1,
                                std::vector<unsigned char>* prediction_success) {
  CHECK_NOTNULL(predicted_keypoints_kp1);
  CHECK_NOTNULL(prediction_success)->clear();
  CHECK(frame_k.hasKeypointMeasurements());
  if (frame_k.getNumKeypointMeasurements() == 0u) {
    return;
  }
  const aslam::Camera& camera = *CHECK_NOTNULL(frame_k.getCameraGeometry().get());

  // Early exit for identity rotation.
  if (std::abs(q_Ckp1_Ck.w() - 1.0) < 1e-8) {
    *predicted_keypoints_kp1 = frame_k.getKeypointMeasurements();
    prediction_success->resize(predicted_keypoints_kp1->size(), true);
  }

  // Backproject the keypoints to bearing vectors.
  Eigen::Matrix3Xd bearing_vectors_k;
  camera.backProject3Vectorized(frame_k.getKeypointMeasurements(), &bearing_vectors_k,
                                prediction_success);
  CHECK_EQ(static_cast<int>(prediction_success->size()), bearing_vectors_k.cols());
  CHECK_EQ(static_cast<int>(frame_k.getNumKeypointMeasurements()), bearing_vectors_k.cols());

  // Rotate the bearing vectors into the frame_kp1 coordinates.
  const Eigen::Matrix3Xd bearing_vectors_kp1 = q_Ckp1_Ck.rotateVectorized(bearing_vectors_k);

  // Project the bearing vectors to the frame_kp1.
  std::vector<ProjectionResult> projection_results;
  camera.project3Vectorized(bearing_vectors_kp1, predicted_keypoints_kp1, &projection_results);
  CHECK_EQ(predicted_keypoints_kp1->cols(), bearing_vectors_k.cols());
  CHECK_EQ(static_cast<int>(projection_results.size()), bearing_vectors_k.cols());

  // Set the success based on the backprojection and projection results and output the initial
  // unrotated keypoint for failed predictions.
  const Eigen::Matrix2Xd& keypoints_k = frame_k.getKeypointMeasurements();
  CHECK_EQ(keypoints_k.cols(), predicted_keypoints_kp1->cols());

  for (size_t idx = 0u; idx < projection_results.size(); ++idx) {
    (*prediction_success)[idx] = (*prediction_success)[idx] &&
                                 projection_results[idx].isKeypointVisible();

    // Set the initial keypoint location for failed predictions.
    if (!(*prediction_success)[idx]) {
      predicted_keypoints_kp1->col(idx) = keypoints_k.col(idx);
    }
  }
}

}  // namespace aslam
