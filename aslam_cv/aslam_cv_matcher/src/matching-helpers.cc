#include "aslam/matcher/matching-helpers.h"

#include <aslam/cameras/camera.h>
#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>
#include <Eigen/Core>

namespace aslam {

void predictKeypointsByRotation(const VisualFrame& frame_k,
                                const aslam::Quaternion& q_Ckp1_Ck,
                                Eigen::Matrix2Xd* predicted_keypoints_kp1,
                                std::vector<unsigned char>* prediction_success) {
  CHECK_NOTNULL(predicted_keypoints_kp1);
  CHECK_NOTNULL(prediction_success)->clear();
  CHECK(frame_k.hasKeypointMeasurements());
  CHECK_GT(frame_k.getNumKeypointMeasurements(), 0u);
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
