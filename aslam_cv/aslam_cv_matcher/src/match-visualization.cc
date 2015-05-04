#include <vector>

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "aslam/matcher/match-visualization.h"

namespace aslam {

void drawKeyPointsAndMatches(const cv::Mat& image_A,
                             const std::vector<cv::KeyPoint>& key_points_A,
                             const cv::Mat& image_B,
                             const std::vector<cv::KeyPoint>& key_points_B,
                             const std::vector<cv::DMatch>& matches_A_B,
                             FeatureVisualizationType type,
                             cv::Mat* image_w_feature_matches) {
  CHECK_NOTNULL(image_w_feature_matches);
  CHECK(!image_A.empty());
  CHECK(!image_B.empty());
  CHECK_EQ(image_A.type(), CV_8UC1) << "Please provide grayscale images!";
  CHECK_EQ(image_B.type(), CV_8UC1) << "Please provide grayscale images!";
  CHECK(!(type == FeatureVisualizationType::kInSitu) ||
        image_A.size() == image_B.size())
      << "The in situ mode is only available for images of the same size!";

  const cv::Scalar kGreen = cv::Scalar(0, 255, 0);
  const cv::Scalar kRed = cv::Scalar(0, 0, 255);

  // Arrange the two images.
  int new_width = 0;
  int new_height = 0;
  cv::Mat sub_image_A, sub_image_B;
  switch (type) {
    case FeatureVisualizationType::kHorizontal:
    // Fall through intended.
    case FeatureVisualizationType::kInSitu:
      new_width = image_A.cols + image_B.cols;
      new_height = std::max(image_A.rows, image_B.rows);
      image_w_feature_matches->create(new_height, new_width, CV_8UC3);
      sub_image_A = (*image_w_feature_matches)(
          cv::Rect(0, 0, image_A.cols, image_A.rows));
      sub_image_B = (*image_w_feature_matches)(
          cv::Rect(image_A.cols, 0, image_B.cols, image_B.rows));
      break;
    case FeatureVisualizationType::kVertical:
      new_width = std::max(image_A.cols, image_B.cols);
      new_height = image_A.rows + image_B.rows;
      image_w_feature_matches->create(new_height, new_width, CV_8UC3);
      sub_image_A = (*image_w_feature_matches)(
          cv::Rect(0, 0, image_A.cols, image_A.rows));
      sub_image_B = (*image_w_feature_matches)(
          cv::Rect(0, image_A.rows, image_B.cols, image_B.rows));
      break;
    default:
      LOG(FATAL) << "Unknown FeatureVisualizationType: "
                 << static_cast<int>(type);
  }
  cvtColor(image_A, sub_image_A, CV_GRAY2BGR);
  cvtColor(image_B, sub_image_B, CV_GRAY2BGR);

  // Draw keypoints.
  cv::drawKeypoints(sub_image_A, key_points_A, sub_image_A, kGreen,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(sub_image_B, key_points_B, sub_image_B, kRed,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // Draw matches.
  for (cv::DMatch match_A_B : matches_A_B) {
    int idx_A = match_A_B.queryIdx;
    int idx_B = match_A_B.trainIdx;
    CHECK_LT(idx_A, key_points_A.size());
    CHECK_LT(idx_B, key_points_B.size());
    const cv::KeyPoint& key_point_A = key_points_A[idx_A];
    const cv::KeyPoint& key_point_B = key_points_B[idx_B];
    cv::Point start(cvRound(key_point_A.pt.x), cvRound(key_point_A.pt.y));
    cv::Point end(cvRound(key_point_B.pt.x), cvRound(key_point_B.pt.y));
    switch (type) {
      case FeatureVisualizationType::kHorizontal:
        end.x += image_A.cols;
        cv::line(*image_w_feature_matches, start, end, kGreen, 1);
        break;
      case FeatureVisualizationType::kVertical:
        end.y += image_A.rows;
        cv::line(*image_w_feature_matches, start, end, kGreen, 1);
        break;
      case FeatureVisualizationType::kInSitu:
        cv::line(sub_image_A, start, end, kGreen, 1);
        cv::line(sub_image_B, start, end, kGreen, 1);
        break;
      default:
        LOG(FATAL) << "Unknown FeatureVisualizationType: "
                   << static_cast<int>(type);
    }
  }
}
}  // namespace aslam
