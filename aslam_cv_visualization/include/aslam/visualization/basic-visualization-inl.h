#ifndef ASLAM_VISUALIZATION_BASIC_VISUALIZATION_INL_H_
#define ASLAM_VISUALIZATION_BASIC_VISUALIZATION_INL_H_

#include <aslam/matcher/match.h>
#include <aslam/matcher/match-helpers.h>

namespace aslam_cv_visualization {

inline void visualizeMatches(
    const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
    const aslam::FrameToFrameMatches& matches, cv::Mat* image) {
  CHECK_NOTNULL(image);

  cv::cvtColor(frame_kp1.getRawImage(), *image, CV_GRAY2BGR);
  CHECK_NOTNULL(image->data);
  drawKeypointMatches(
      frame_kp1, frame_k, matches, cv::Scalar(255, 255, 0),
      cv::Scalar(255, 0, 255), image);
}

}  // namespace aslam_cv_visualization

#endif // ASLAM_VISUALIZATION_BASIC_VISUALIZATION_INL_H_
