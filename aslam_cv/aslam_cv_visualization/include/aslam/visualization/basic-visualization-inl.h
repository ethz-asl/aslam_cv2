#ifndef ASLAM_VISUALIZATION_BASIC_VISUALIZATION_INL_H_
#define ASLAM_VISUALIZATION_BASIC_VISUALIZATION_INL_H_

#include <aslam/matcher/match.h>
#include <aslam/matcher/match-helpers.h>

namespace aslam_cv_visualization {

template<typename MatchesWithScore>
void visualizeMatches(const aslam::VisualFrame& frame_kp1,
                      const aslam::VisualFrame& frame_k,
                      const MatchesWithScore& matches_with_score,
                      cv::Mat* image) {
  CHECK_NOTNULL(image);

  cv::cvtColor(frame_kp1.getRawImage(), *image, CV_GRAY2BGR);

  VLOG(4) << "Converted raw image from grayscale to color.";

  aslam::Matches matches;
  aslam::convertMatchesWithScoreToMatches<MatchesWithScore>(matches_with_score, &matches);
  VLOG(4) << "Converted the matches.";

  CHECK_NOTNULL(image->data);

  drawKeypointMatches(
      frame_kp1, frame_k, matches, cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), image);
}

}  // namespace aslam_cv_visualization

#endif // ASLAM_VISUALIZATION_BASIC_VISUALIZATION_INL_H_
