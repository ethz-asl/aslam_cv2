#ifndef ASLAM_CV_MATCH_VISUALIZATION_H_
#define ASLAM_CV_MATCH_VISUALIZATION_H_

#include <string>
#include <vector>

#include <aslam/frames/visual-frame.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "aslam/matcher/match.h"

namespace aslam {

enum class FeatureVisualizationType {
  kHorizontal,  // Images side by side, matches joined across images.
  kVertical,    // Images stacked, matches joined across images.
  kInSitu       // Images side by side, matches joined within both images.
};

void getCvKeyPointsFromVisualFrame(const VisualFrame& frame,
                                   std::vector<cv::KeyPoint>* cv_key_points);

void drawVisualFrameKeyPointsAndMatches(const VisualFrame& frame_A,
                                        const VisualFrame& frame_B,
                                        aslam::FeatureVisualizationType visualization_type,
                                        const FrameToFrameMatches& matches_A_B,
                                        cv::Mat* image_w_feature_matches);

void drawAslamKeyPointsAndMatches(const cv::Mat& image_A,
                                  const Eigen::Matrix2Xd& key_points_A,
                                  const cv::Mat& image_B,
                                  const Eigen::Matrix2Xd& key_points_B,
                                  FeatureVisualizationType type,
                                  const FrameToFrameMatches& matches_A_B,
                                  cv::Mat* image_w_feature_matches);

void drawKeyPointsAndMatches(const cv::Mat& image_A,
                             const std::vector<cv::KeyPoint>& key_points_A,
                             const cv::Mat& image_B,
                             const std::vector<cv::KeyPoint>& key_points_B,
                             const aslam::OpenCvMatches& matches_A_B,
                             FeatureVisualizationType type,
                             cv::Mat* image_w_feature_matches);

}  // namespace aslam

#endif //ASLAM_CV_MATCH_VISUALIZATION_H_
