#ifndef ASLAM_MATCHER_MATCH_HELPERS_H_
#define ASLAM_MATCHER_MATCH_HELPERS_H_

#include "aslam/matcher/match.h"

namespace aslam {
inline void convertMatches(const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B);

template<typename MatchesWithScore>
void convertMatches(const MatchesWithScore& matches_with_score_A_B, Matches* matches_A_B);

/// Convert MatchesWithScore to Matches.
template<typename MatchingProblem>
void convertMatches(const typename MatchingProblem::MatchesWithScore& matches_with_score_A_B,
                    typename MatchingProblem::Matches* matches_A_B);

/// Convert MatchesWithScore to cv::DMatches.
template<typename MatchWithScore>
void convertMatches(
    const typename Aligned<std::vector, MatchWithScore>::type& matches_with_score_A_B,
    OpenCvMatches* matches_A_B);

/// Select and return N random matches for each camera in the rig.
FrameToFrameMatchesList pickNRandomRigMatches(
    size_t n_per_camera, const FrameToFrameMatchesList& rig_matches);

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
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_kp1,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_k);
}  // namespace aslam

#include "./match-helpers-inl.h"

#endif  // ASLAM_MATCHER_MATCH_HELPERS_H_
