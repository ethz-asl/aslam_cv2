#ifndef ASLAM_MATCHER_MATCH_HELPERS_H_
#define ASLAM_MATCHER_MATCH_HELPERS_H_

#include <aslam/common/pose-types.h>

#include "aslam/matcher/match.h"

namespace aslam {

// Convert any kind of matches with score to any kind of match.
template<typename MatchWithScore, typename Match>
void convertMatchesWithScoreToMatches(
    const Aligned<std::vector, MatchWithScore>& matches_with_score_A_B,
    Aligned<std::vector, Match>* matches_A_B);

// Convert aslam::MatchesWithScore to aslam::Matches.
inline void convertMatchesWithScoreToMatches(
    const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B);

// Convert any (derived) MatchesWithScore to aslam::Matches.
template<typename MatchesWithScore>
void convertMatchesWithScoreToMatches(
    const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B);

/// Convert MatchesWithScore of a matching problem to the corresponding Matches.
template<typename MatchingProblem>
void convertMatchesWithScoreToMatches(
    const typename MatchingProblem::MatchesWithScore& matches_with_score_A_B,
    typename MatchingProblem::Matches* matches_A_B);

/// Convert MatchesWithScore to cv::DMatches.
template<typename MatchWithScore>
void convertMatchesWithScoreToOpenCvMatches(
    const Aligned<std::vector, MatchWithScore>& matches_with_score_A_B,
    OpenCvMatches* matches_A_B);

/// Select and return N random matches for each camera in the rig.
void pickNRandomRigMatches(
    size_t n_per_camera, const FrameToFrameMatchesList& rig_matches,
    FrameToFrameMatchesList* selected_rig_matches);

/// Get the matches based on the track id channels for one VisualFrame.
size_t extractMatchesFromTrackIdChannel(const VisualFrame& frame_kp1,
                                        const VisualFrame& frame_k,
                                        FrameToFrameMatches* matches_kp1_kp);

/// Get the matches based on the track id channels for one VisualNFrame.
size_t extractMatchesFromTrackIdChannels(
    const VisualNFrame& nframe_kp1, const VisualNFrame& nframe_k,
    FrameToFrameMatchesList* rig_matches_kp1_kp);

/// Get the median pixel disparity for all matches.
double getMatchPixelDisparityMedian(
      const VisualNFrame& nframe_kp1, const VisualNFrame& nframe_k,
      const FrameToFrameMatchesList& matches_kp1_kp);

/// Get the median pixel disparity for all matches, taking into account the relative
/// orientation of the frames.
double getUnrotatedMatchPixelDisparityMedian(
    const VisualNFrame& nframe_kp1, const VisualNFrame& nframe_k,
    const FrameToFrameMatchesList& matches_kp1_k,
    const aslam::Quaternion& q_kp1_k);

/// Return the normalized bearing vectors for a list of single camera matches.
void getBearingVectorsFromMatches(
    const VisualFrame& frame_kp1, const VisualFrame& frame_k,
    const FrameToFrameMatches& matches_kp1_k,
    Aligned<std::vector, Eigen::Vector3d>* bearing_vectors_kp1,
    Aligned<std::vector, Eigen::Vector3d>* bearing_vectors_k);

/// Rotate keypoints from a VisualFrame using a specified rotation. Note that if the back-,
/// projection fails or the keypoint leaves the image region, the predicted keypoint will be left
/// unchanged and the prediction_success will be set to false.
void predictKeypointsByRotation(const VisualFrame& frame_k,
                                const aslam::Quaternion& q_Ckp1_Ck,
                                Eigen::Matrix2Xd* predicted_keypoints_kp1,
                                std::vector<unsigned char>* prediction_success);

}  // namespace aslam

#include "./match-helpers-inl.h"

#endif  // ASLAM_MATCHER_MATCH_HELPERS_H_
