#include <vector>

#include <glog/logging.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "aslam/matcher/match-visualization.h"

namespace aslam {

void getCvKeyPointsFromVisualFrame(const VisualFrame& frame,
                                   std::vector<cv::KeyPoint>* cv_key_points){
  CHECK_NOTNULL(cv_key_points)->clear();
  cv_key_points->reserve(frame.getNumKeypointMeasurements());
  for(unsigned int i = 0; i < frame.getNumKeypointMeasurements(); ++i){
    Eigen::Matrix2Xd key_point = frame.getKeypointMeasurement(i);
    cv_key_points->emplace_back(
        cv::KeyPoint(key_point(0,0),
                     key_point(1,0),
                     frame.getKeypointScale(i),
                     frame.getKeypointOrientation(i),
                     frame.getKeypointScore(i), 0, -1));
  }
}

void drawVisualFrameKeyPointsAndMatches(const VisualFrame& frame_A,
                                        const VisualFrame& frame_B,
                                        aslam::FeatureVisualizationType type,
                                        const FrameToFrameMatches& matches_A_B,
                                        cv::Mat* image_w_feature_matches) {
  CHECK_NOTNULL(image_w_feature_matches);

  cv::Mat image_A = frame_A.getRawImage();
  cv::Mat image_B = frame_B.getRawImage();
  if(image_A.empty() || image_B.empty()){
    LOG(FATAL) << "Cannot draw key points. No images found.";
  }

  // Extract cv::KeyPoints
  std::vector<cv::KeyPoint> cv_key_points_A, cv_key_points_B;
  getCvKeyPointsFromVisualFrame(frame_A, &cv_key_points_A);
  getCvKeyPointsFromVisualFrame(frame_B, &cv_key_points_B);

  // Extract matches
  aslam::OpenCvMatches cv_matches_A_B;
  if(!matches_A_B.empty()) {
    cv_matches_A_B.reserve(matches_A_B.size());
    for(const FrameToFrameMatch& match: matches_A_B) {
      cv_matches_A_B.emplace_back(cv::DMatch(
          static_cast<int>(match.getKeypointIndexAppleFrame()),
          static_cast<int>(match.getKeypointIndexBananaFrame()), 0.0));
    }
  }

  drawKeyPointsAndMatches(image_A, cv_key_points_A, image_B, cv_key_points_B,
                          cv_matches_A_B, type, image_w_feature_matches);
}

void drawAslamKeyPointsAndMatches(const cv::Mat& image_A,
                                  const Eigen::Matrix2Xd& key_points_A,
                                  const cv::Mat& image_B,
                                  const Eigen::Matrix2Xd& key_points_B,
                                  FeatureVisualizationType type,
                                  const FrameToFrameMatches& matches_A_B,
                                  cv::Mat* image_w_feature_matches) {
  CHECK_NOTNULL(image_w_feature_matches);

  // Extract cv::KeyPoints
  std::vector<cv::KeyPoint> cv_key_points_A;
  cv_key_points_A.reserve(key_points_A.size());
  for(int i = 0; i < key_points_A.cols(); ++i) {
    cv::KeyPoint key_point_A(key_points_A(0,i), key_points_A(1,i), 0.0f, -1.0, 0.0, 0, -1);
    cv_key_points_A.emplace_back(key_point_A);
  }
  std::vector<cv::KeyPoint> cv_key_points_B;
  cv_key_points_B.reserve(key_points_B.size());
  for(int i = 0; i < key_points_B.cols(); ++i) {
    cv::KeyPoint key_point_B(key_points_B(0,i), key_points_B(1,i), 0.0f, -1.0, 0.0, 0, -1);
    cv_key_points_B.emplace_back(key_point_B);
  }

  // Extract matches
  aslam::OpenCvMatches cv_matches_A_B;
  if(!matches_A_B.empty()) {
    cv_matches_A_B.reserve(matches_A_B.size());
    for(const FrameToFrameMatch& match: matches_A_B) {
      cv_matches_A_B.emplace_back(cv::DMatch(
          static_cast<int>(match.getKeypointIndexAppleFrame()),
          static_cast<int>(match.getKeypointIndexBananaFrame()), 0.0));
    }
  }

  drawKeyPointsAndMatches(image_A, cv_key_points_A, image_B, cv_key_points_B,
                          cv_matches_A_B, type, image_w_feature_matches);
}

void drawKeyPointsAndMatches(const cv::Mat& image_A,
                             const std::vector<cv::KeyPoint>& key_points_A,
                             const cv::Mat& image_B,
                             const std::vector<cv::KeyPoint>& key_points_B,
                             const aslam::OpenCvMatches& matches_A_B,
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
  cv::drawKeypoints(sub_image_A, key_points_A, sub_image_A, kRed,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(sub_image_B, key_points_B, sub_image_B, kRed,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

  // Draw matches.
  std::vector<cv::KeyPoint> matched_keypoint_A, matched_keypoint_B;
  matched_keypoint_A.reserve(matches_A_B.size());
  matched_keypoint_B.reserve(matches_A_B.size());
  for (const cv::DMatch& match_A_B : matches_A_B) {
    const int idx_A = match_A_B.queryIdx;
    const int idx_B = match_A_B.trainIdx;
    CHECK_LT(idx_A, key_points_A.size());
    CHECK_LT(idx_B, key_points_B.size());
    const cv::KeyPoint& key_point_A = key_points_A[idx_A];
    const cv::KeyPoint& key_point_B = key_points_B[idx_B];
    matched_keypoint_A.emplace_back(key_point_A);
    matched_keypoint_B.emplace_back(key_point_B);
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
  cv::drawKeypoints(sub_image_A, matched_keypoint_A, sub_image_A, kGreen,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
  cv::drawKeypoints(sub_image_B, matched_keypoint_B, sub_image_B, kGreen,
                    cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}
}  // namespace aslam
