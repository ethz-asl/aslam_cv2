#include "aslam/tracker/tracking-helpers.h"

#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

namespace aslam {

void convertKeypointVectorToCvPointList(const Eigen::Matrix2Xd& keypoints,
                                        std::vector<cv::Point2f>* keypoints_cv) {
  CHECK_NOTNULL(keypoints_cv);
  keypoints_cv->reserve(keypoints.cols());
  for (int idx = 0u; idx < keypoints.cols(); ++idx) {
    keypoints_cv->emplace_back(keypoints.col(idx)(0), keypoints.col(idx)(1));
  }
}

void convertCvPointListToKeypointVector(const std::vector<cv::Point2f>& keypoints,
                                        Eigen::Matrix2Xd* keypoints_eigen) {
  CHECK_NOTNULL(keypoints_eigen);
  keypoints_eigen->resize(Eigen::NoChange, keypoints.size());
  for (size_t idx = 0u; idx < keypoints.size(); ++idx) {
    keypoints_eigen->col(idx)(0) = keypoints[idx].x;
    keypoints_eigen->col(idx)(0) = keypoints[idx].y;
  }
}

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

void insertAdditionalKeypointsToVisualFrame(const Eigen::Matrix2Xd& new_keypoints,
                                            double fixed_keypoint_uncertainty_px,
                                            aslam::VisualFrame* frame) {
  CHECK_NOTNULL(frame);
  CHECK_GT(fixed_keypoint_uncertainty_px, 0.0);
  const size_t num_new_keypoints = static_cast<size_t>(new_keypoints.cols());

  // Add keypoints at the back if the frame already contains keypoints, otherwise just insert
  // the keypoints directly.
  if (frame->hasKeypointMeasurements()) {
    CHECK(frame->hasTrackIds());
    CHECK(frame->hasKeypointMeasurementUncertainties());
    const size_t initial_size = frame->getNumKeypointMeasurements();
    const size_t extended_size = initial_size + num_new_keypoints;

    // Resize the existing vectors.
    Eigen::Matrix2Xd* keypoints = CHECK_NOTNULL(frame->getKeypointMeasurementsMutable());
    Eigen::VectorXi* track_ids = CHECK_NOTNULL(frame->getTrackIdsMutable());
    Eigen::VectorXd* uncertainties =
        CHECK_NOTNULL(frame->getKeypointMeasurementUncertaintiesMutable());
    CHECK_EQ(keypoints->cols(), track_ids->rows());
    CHECK_EQ(keypoints->cols(), uncertainties->rows());

    keypoints->conservativeResize(Eigen::NoChange, extended_size);
    track_ids->conservativeResize(extended_size);
    uncertainties->conservativeResize(extended_size);

    // Insert new keypoints at the back.
    keypoints->block(initial_size, num_new_keypoints, 2, num_new_keypoints) = new_keypoints;
    track_ids->segment(initial_size, num_new_keypoints).setConstant(-1);
    uncertainties->segment(initial_size, num_new_keypoints).setConstant(
        fixed_keypoint_uncertainty_px);

    CHECK_EQ(static_cast<int>(extended_size), frame->getKeypointMeasurements().cols());
    CHECK_EQ(static_cast<int>(extended_size), frame->getKeypointMeasurementUncertainties().rows());
    CHECK_EQ(static_cast<int>(extended_size), frame->getTrackIds().rows());
  } else {
    // Just set the keypoints, invalid track ids and constant measurement uncertainties.
    frame->setKeypointMeasurements(new_keypoints);

    Eigen::VectorXi track_ids(num_new_keypoints);
    track_ids.setConstant(-1);
    frame->swapTrackIds(&track_ids);

    Eigen::VectorXd uncertainties(num_new_keypoints);
    uncertainties.setConstant(fixed_keypoint_uncertainty_px);
    frame->swapKeypointMeasurementUncertainties(&uncertainties);
  }
}

void insertAdditionalKeypointsToVisualFrame(const Verctor2dList& keypoints,
                                            double fixed_keypoint_uncertainty_px,
                                            aslam::VisualFrame* frame) {
  // Convert std::vector to Eigen vector.
  const size_t num_new_keypoints = keypoints.size();
  Eigen::Matrix2Xd keypoints_eigen(2, num_new_keypoints);
  for (size_t idx = 0; idx < num_new_keypoints; ++idx) {
    keypoints_eigen.col(idx) = keypoints[idx];
  }

  insertAdditionalKeypointsToVisualFrame(keypoints_eigen, fixed_keypoint_uncertainty_px, frame);
}

}  // namespace aslam
