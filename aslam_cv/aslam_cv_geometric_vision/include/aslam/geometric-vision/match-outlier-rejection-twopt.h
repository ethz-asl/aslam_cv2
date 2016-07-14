#ifndef ASLAM_MATCH_OUTLIER_REJECTION_TWOPT_H_
#define ASLAM_MATCH_OUTLIER_REJECTION_TWOPT_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/pose-types.h>
#include <aslam/matcher/match.h>

namespace aslam {
class VisualFrame;

namespace geometric_vision {

/// \brief Runs two RANSAC schemes over the specified match list to separate matches into
///        out-/inliers. Both a translation only and rotation only RANSAC are computed.
///        The final set of inliers is the union of both inlier sets.
/// @param[in]  frame_kp1  Current frame.
/// @param[in]  frame_k    Previous frame.
/// @param[in]  q_Ckp1_Ck  Rotation taking points from the camera frame k to the camera frame k+1.
/// @param[in]  matches_kp1_k The matches between the frames.
/// @param[in]  fix_random_seed Use a fixed random seed for RANSAC.
/// @param[in]  ransac_threshold  RANSAC threshold to consider a sample as inlier.
///                               The threshold is defined as:  1 - cos(max_ray_disparity_angle).
/// @param[in]  ransac_max_iterations Max. RANSAC iterations.
/// @param[out] inlier_matches_kp1_k The list of inlier matches.
/// @param[out] outlier_matches_kp1_k The list of outlier matches.
/// @return RANSAC successful?
bool rejectOutlierFeatureMatchesTranslationRotationSAC(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Quaternion& q_Ckp1_Ck,
    const aslam::FrameToFrameMatchesWithScore& matches_kp1_k, bool fix_random_seed,
    double ransac_threshold, size_t ransac_max_iterations,
    aslam::FrameToFrameMatchesWithScore* inlier_matches_kp1_k,
    aslam::FrameToFrameMatchesWithScore* outlier_matches_kp1_k);

}  // namespace geometric_vision

}  // namespace aslam
#endif  // ASLAM_MATCH_OUTLIER_REJECTION_TWOPT_H_
