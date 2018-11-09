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
  for (int idx = 0; idx < keypoints.cols(); ++idx) {
    keypoints_cv->emplace_back(keypoints.col(idx)(0), keypoints.col(idx)(1));
  }
}

void convertCvPointListToKeypointVector(const std::vector<cv::Point2f>& keypoints,
                                        Eigen::Matrix2Xd* keypoints_eigen) {
  CHECK_NOTNULL(keypoints_eigen);
  keypoints_eigen->resize(Eigen::NoChange, keypoints.size());
  for (size_t idx = 0u; idx < keypoints.size(); ++idx) {
    keypoints_eigen->col(idx)(0) = keypoints[idx].x;
    keypoints_eigen->col(idx)(1) = keypoints[idx].y;
  }
}

void insertCvKeypointsAndDescriptorsIntoEmptyVisualFrame(
    const std::vector<cv::KeyPoint>& new_cv_keypoints, const cv::Mat& new_cv_descriptors,
    const double fixed_keypoint_uncertainty_px, aslam::VisualFrame* frame) {
  CHECK_NOTNULL(frame);
  CHECK(!frame->hasKeypointMeasurements() || frame->getNumKeypointMeasurements() == 0u);
  CHECK(!frame->hasDescriptors() || frame->getDescriptors().cols() == 0);
  CHECK(!frame->hasKeypointMeasurementUncertainties() ||
        frame->getKeypointMeasurementUncertainties().rows() == 0);
  CHECK(!frame->hasKeypointOrientations() || frame->getKeypointOrientations().rows() == 0);
  CHECK(!frame->hasKeypointScales() || frame->getKeypointScales().rows() == 0);
  CHECK(!frame->hasKeypointScores() || frame->getKeypointScores().rows() == 0);
  CHECK(!frame->hasTrackIds() || frame->getTrackIds().rows() == 0);
  CHECK_GT(fixed_keypoint_uncertainty_px, 0.0);
  CHECK_EQ(new_cv_keypoints.size(), static_cast<size_t>(new_cv_descriptors.rows));
  CHECK_EQ(new_cv_descriptors.type(), CV_8UC1);
  CHECK(new_cv_descriptors.isContinuous());

  const size_t num_new_keypoints = new_cv_keypoints.size();

  Eigen::Matrix2Xd new_keypoints_measurements;
  Eigen::VectorXd new_keypoint_scores;
  Eigen::VectorXd new_keypoint_scales;
  Eigen::VectorXd new_keypoint_orientations;
  new_keypoints_measurements.resize(Eigen::NoChange, num_new_keypoints);
  new_keypoint_scores.resize(num_new_keypoints);
  new_keypoint_scales.resize(num_new_keypoints);
  new_keypoint_orientations.resize(num_new_keypoints);
  for (size_t idx = 0u; idx < num_new_keypoints; ++idx) {
    const cv::KeyPoint& cv_keypoint = new_cv_keypoints[idx];
    new_keypoints_measurements.col(idx)(0) = static_cast<double>(cv_keypoint.pt.x);
    new_keypoints_measurements.col(idx)(1) = static_cast<double>(cv_keypoint.pt.y);
    new_keypoint_scores(idx) = static_cast<double>(cv_keypoint.response);
    new_keypoint_scales(idx) = static_cast<double>(cv_keypoint.size);
    new_keypoint_orientations(idx) = static_cast<double>(cv_keypoint.angle);
  }

  frame->swapKeypointMeasurements(&new_keypoints_measurements);
  frame->swapKeypointScores(&new_keypoint_scores);
  frame->swapKeypointScales(&new_keypoint_scales);
  frame->swapKeypointOrientations(&new_keypoint_orientations);

  Eigen::VectorXd uncertainties(num_new_keypoints);
  uncertainties.setConstant(fixed_keypoint_uncertainty_px);
  frame->swapKeypointMeasurementUncertainties(&uncertainties);

  // Set invalid track ids.
  Eigen::VectorXi track_ids(num_new_keypoints);
  track_ids.setConstant(-1);
  frame->swapTrackIds(&track_ids);

  frame->setDescriptors(
      // Switch cols/rows as Eigen is col-major and cv::Mat is row-major.
      Eigen::Map<aslam::VisualFrame::DescriptorsT>(
          new_cv_descriptors.data, new_cv_descriptors.cols, new_cv_descriptors.rows));
}

}  // namespace aslam
