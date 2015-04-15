#include "aslam/visualization/basic-visualization.h"

namespace aslam_cv_visualization {

void drawKeypoints(const aslam::VisualFrame& frame, cv::Mat* image) {
  CHECK_NOTNULL(image);
  const size_t num_keypoints = frame.getNumKeypointMeasurements();
  for (size_t keypoint_idx = 0u; keypoint_idx < num_keypoints; ++keypoint_idx) {
    Eigen::Vector2d keypoint;
    if (frame.getRawCameraGeometry()) {
      frame.getKeypointInRawImageCoordinates(keypoint_idx, &keypoint);
    } else {
      keypoint = frame.getKeypointMeasurement(keypoint_idx);
    }
    cv::circle(*image, cv::Point(keypoint[0], keypoint[1]), 1,
               cv::Scalar(0, 255, 255), 1, CV_AA);
  }
}

void drawKeypointMatches(const aslam::VisualFrame& frame_kp1,
                         const aslam::VisualFrame& frame_k,
                         const aslam::Matches& matches_kp1_k,
                         cv::Scalar color_keypoint_kp1, cv::Scalar line_color,
                         cv::Mat* image) {
  CHECK_NOTNULL(image);

  for (const aslam::Match& match_kp1_k : matches_kp1_k) {
    Eigen::Vector2d keypoint_k;
    Eigen::Vector2d keypoint_kp1;
    CHECK_LT(match_kp1_k.second, frame_k.getNumKeypointMeasurements());
    if (frame_k.getRawCameraGeometry()) {
      frame_k.getKeypointInRawImageCoordinates(match_kp1_k.second, &keypoint_k);
      CHECK_LT(match_kp1_k.first, frame_kp1.getNumKeypointMeasurements());
      frame_kp1.getKeypointInRawImageCoordinates(match_kp1_k.first, &keypoint_kp1);
    } else {
      keypoint_k = frame_k.getKeypointMeasurement(match_kp1_k.second);
      CHECK_LT(match_kp1_k.first, frame_kp1.getNumKeypointMeasurements());
      keypoint_kp1 = frame_kp1.getKeypointMeasurement(match_kp1_k.first);
    }

    cv::circle(*image, cv::Point(keypoint_kp1[0], keypoint_kp1[1]), 4, color_keypoint_kp1);
    cv::line(*image, cv::Point(keypoint_k[0], keypoint_k[1]),
             cv::Point(keypoint_kp1[0], keypoint_kp1[1]), line_color);
  }
}
  
}  // namespace aslam_cv_visualization
