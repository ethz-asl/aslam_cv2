#ifndef ASLAM_CV_MATCH_VISUALIZATION_H_
#define ASLAM_CV_MATCH_VISUALIZATION_H_

#include <string>
#include <vector>

#include <aslam/matcher/match.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace aslam {

enum class FeatureVisualizationType {
  kHorizontal,  // Images side by side, matches joined across images.
  kVertical,    // Images stacked, matches joined across images.
  kInSitu       // Images side by side, matches joined within both images.
};

void drawKeyPointsAndMatches(const cv::Mat& image_A,
                             const std::vector<cv::KeyPoint>& key_points_A,
                             const cv::Mat& image_B,
                             const std::vector<cv::KeyPoint>& key_points_B,
                             const std::vector<cv::DMatch>& matches_A_B,
                             FeatureVisualizationType type,
                             cv::Mat* image_w_feature_matches);

}  // namespace aslam

#endif //ASLAM_CV_MATCH_VISUALIZATION_H_
