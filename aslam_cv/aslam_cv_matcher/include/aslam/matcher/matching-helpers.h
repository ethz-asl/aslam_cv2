#ifndef ASLAM_MATCHING_HELPERS_H_
#define ASLAM_MATCHING_HELPERS_H_

#include <vector>

#include <aslam/common/pose-types.h>
#include <Eigen/Core>

namespace aslam {
class VisualFrame;

/// Rotate keypoints from a VisualFrame using a specified rotation. Note that if the back-,
/// projection fails or the keypoint leaves the image region, the predicted keypoint will be left
/// unchanged and the prediction_success will be set to false.
void predictKeypointsByRotation(const VisualFrame& frame_k,
                                const aslam::Quaternion& q_Ckp1_Ck,
                                Eigen::Matrix2Xd* predicted_keypoints_kp1,
                                std::vector<unsigned char>* prediction_success);
} // namespace aslam

#endif // ASLAM_MATCHING_HELPERS_H_
