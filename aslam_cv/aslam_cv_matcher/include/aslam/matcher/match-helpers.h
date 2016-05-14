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

/// Get number of matches for a rig match list. (outer vector = cameras, inner vector = match list)
template<typename MatchType>
size_t countRigMatches(
    const typename Aligned<std::vector,
                           typename Aligned<std::vector,MatchType>::type>::type& rig_matches);

/// Select and return N random matches for each camera in the rig.
template<typename MatchesType>
typename Aligned<std::vector, MatchesType>::type pickNRandomRigMatches(
    size_t n_per_camera, const typename Aligned<std::vector, MatchesType>::type& rig_matches);

/// Get the matches based on the track id channels for one VisualFrame.
template<typename MatchesType>
size_t extractMatchesFromTrackIdChannel(const aslam::VisualFrame& frame_kp1,
                                        const aslam::VisualFrame& frame_k,
                                        MatchesType* matches_kp1_kp);

/// Get the matches based on the track id channels for one VisualNFrame.
template<typename MatchesType>
size_t extractMatchesFromTrackIdChannels(
    const aslam::VisualNFrame& nframe_kp1, const aslam::VisualNFrame& nframe_k,
    typename Aligned<std::vector, MatchesType>::type* rig_matches_kp1_kp);

/// Get the median pixel disparity for all matches.
template<typename MatchesType>
double getMatchPixelDisparityMedian(
      const aslam::VisualNFrame& nframe_kp1, const aslam::VisualNFrame& nframe_k,
      const typename Aligned<std::vector, MatchesType>::type& matches_kp1_kp);

/// Get the median pixel disparity for all matches, taking into account the relative
/// orientation of the frames.
template<typename MatchType>
double getUnrotatedMatchPixelDisparityMedian(
    const aslam::VisualNFrame& nframe_kp1, const aslam::VisualNFrame& nframe_k,
    const typename Aligned<std::vector,
                           typename Aligned<std::vector, MatchType>::type>::type& matches_kp1_k,
    const aslam::Quaternion& q_kp1_k);

/// Return the normalized bearing vectors for a list of single camera matches.
template<typename MatchType>
void getBearingVectorsFromMatches(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const typename Aligned<std::vector, MatchType>::type& matches_kp1_k,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_kp1,
    Aligned<std::vector, Eigen::Vector3d>::type* bearing_vectors_k);
}  // namespace aslam

#include "./match-helpers-inl.h"

#endif  // ASLAM_MATCHER_MATCH_HELPERS_H_
