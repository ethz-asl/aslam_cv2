#ifndef ASLAM_MATCH_OUTLIER_REJECTION_TWOPT_H_
#define ASLAM_MATCH_OUTLIER_REJECTION_TWOPT_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/pose-types.h>
#include <aslam/matcher/match.h>

namespace aslam {
class VisualFrame;

namespace geometric_vision {

/// \brief Runs a 2-pt RANSAC scheme over the specified match list to separate matches into
///        out-/inliers. The bearing vectors are unrotated using a given rotation q_Ckp1_Ck
///        and a model is used that only estimates the translational motion.
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
bool rejectOutlierFeatureMatchesTranslationSAC(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Quaternion& q_Ckp1_Ck, const aslam::MatchesWithScore& matches_kp1_k,
    bool fix_random_seed, double ransac_threshold,
    size_t ransac_max_iterations, aslam::MatchesWithScore* inlier_matches_kp1_k,
    aslam::MatchesWithScore* outlier_matches_kp1_k);

bool rejectOutlierFeatureMatchesRelativePoseRotationSAC(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::Quaternion& q_Ckp1_Ck, const aslam::MatchesWithScore& matches_kp1_k,
    bool fix_random_seed, double ransac_threshold,
    size_t ransac_max_iterations, aslam::MatchesWithScore* inlier_matches_kp1_k,
    aslam::MatchesWithScore* outlier_matches_kp1_k);

}  // namespace gv

}  // namespace aslam
#endif  // ASLAM_MATCH_OUTLIER_REJECTION_TWOPT_H_
