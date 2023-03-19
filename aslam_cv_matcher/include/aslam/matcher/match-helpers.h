#ifndef ASLAM_MATCHER_MATCH_HELPERS_H_
#define ASLAM_MATCHER_MATCH_HELPERS_H_

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>

#include "aslam/matcher/match.h"

namespace aslam {

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
                                std::vector<unsigned char>* prediction_success,
                                int descriptor_type = 0);
void predictKeypointsByRotation(const aslam::Camera& camera,
                                const Eigen::Matrix2Xd keypoints_k,
                                const aslam::Quaternion& q_Ckp1_Ck,
                                Eigen::Matrix2Xd* predicted_keypoints_kp1,
                                std::vector<unsigned char>* prediction_success);

}  // namespace aslam

#endif  // ASLAM_MATCHER_MATCH_HELPERS_H_
