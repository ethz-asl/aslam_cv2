#ifndef ASLAM_TRACKING_HELPERS_H_
#define ASLAM_TRACKING_HELPERS_H_

#include <vector>

#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace aslam {
class VisualFrame;
typedef Aligned<std::vector, Eigen::Vector2d> Verctor2dList;

void convertKeypointVectorToCvPointList(const Eigen::Matrix2Xd& keypoints,
                                        std::vector<cv::Point2f>* keypoints_cv);

void convertCvPointListToKeypointVector(const std::vector<cv::Point2f>& keypoints,
                                        Eigen::Matrix2Xd* keypoints_eigen);

/// Insert a list of OpenCV keypoints and descriptors into an empty VisualFrame.
void insertCvKeypointsAndDescriptorsIntoEmptyVisualFrame(
    const std::vector<cv::KeyPoint>& new_cv_keypoints, const cv::Mat& new_cv_descriptors,
    const double fixed_keypoint_uncertainty_px, aslam::VisualFrame* frame);

/// Append a list of OpenCV kepoints and descriptors to a VisualFrame.
void insertAdditionalCvKeypointsAndDescriptorsToVisualFrame(
    const std::vector<cv::KeyPoint>& new_cv_keypoints, const cv::Mat& new_cv_descriptors,
    const double fixed_keypoint_uncertainty_px, aslam::VisualFrame* frame);

/// Append a list of kepoints to a VisualFrame.
void insertAdditionalKeypointsToVisualFrame(const Eigen::Matrix2Xd& new_keypoints,
                                            const double fixed_keypoint_uncertainty_px,
                                            aslam::VisualFrame* frame);
void insertAdditionalKeypointsToVisualFrame(const Verctor2dList& keypoints,
                                            const double fixed_keypoint_uncertainty_px,
                                            aslam::VisualFrame* frame);

}  // namespace aslam

#endif  // ASLAM_TRACKING_HELPERS_H_
