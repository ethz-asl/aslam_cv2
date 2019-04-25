#ifndef ASLAM_MATCHER_MATCHING_PROBLEM_INFINITE_LINES_TO_FRAME_H_
#define ASLAM_MATCHER_MATCHING_PROBLEM_INFINITE_LINES_TO_FRAME_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <map>
#include <memory>
#include <vector>

#include <Eigen/Core>
#include <aslam/cameras/camera.h>
#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/lines/line-2d-with-angle.h>
#include <nabo/nabo.h>

#include "aslam/matcher/match.h"
#include "aslam/matcher/matching-problem.h"

namespace aslam {

bool reprojectInfiniteLineIntoImage(
    const Line3d& line_3d_C, const Camera& camera,
    Line2dWithAngle* reprojected_line);

class MatchingProblemInfiniteLinesToFrame : public MatchingProblem {
 public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemInfiniteLinesToFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemInfiniteLinesToFrame);
  ASLAM_ADD_MATCH_TYPEDEFS(InfiniteLinesToFrame);

  MatchingProblemInfiniteLinesToFrame() = delete;
  MatchingProblemInfiniteLinesToFrame(
      const Lines2dWithAngle& lines_2d_frame, const Lines3d& lines_3d_C_lines,
      const Camera& camera, const double search_radius_px,
      const double angle_threshold_deg,
      const double lateral_distance_threshold_px);

  virtual ~MatchingProblemInfiniteLinesToFrame() {}

  size_t numApples() const override {
    return lines_2d_frame_.size();
  }

  size_t numBananas() const override {
    return lines_3d_C_lines_.size();
  }

  bool doSetup() override;
  void getCandidates(CandidatesList* candidates_for_3d_lines) override;

 protected:
  inline double computeMatchScore(
      const double distance_px2, const double angle_diff_rad,
      const double lateral_distance_px) {
    constexpr double kAngleDiffWeighting = 0.5;
    return kAngleDiffWeighting * angle_diff_rad +
           (1.0 - kAngleDiffWeighting) * lateral_distance_px;
  }


 private:
  Lines2dWithAngle lines_2d_frame_;
  Lines3d lines_3d_C_lines_;
  Lines2dWithAngle lines_2d_reprojected_;
  std::vector<int> lines_2d_reprojected_index_to_lines_3d_index_;
  std::vector<unsigned char> is_line_3d_valid_;

  const Camera& camera_;

  Eigen::MatrixXd lines_2d_index_;
  std::unique_ptr<Nabo::NNSearchD> nabo_;

  static constexpr double kSearchNNEpsilon = 0.0;
  static constexpr unsigned kSearchOptionFlags =
      Nabo::NNSearchD::ALLOW_SELF_MATCH;
  static constexpr int kSearchDimension = 1;

  const double search_radius_rad_;
  const double angle_threshold_rad_;
  const double lateral_distance_threshold_px_;
};
}  // namespace aslam
#endif  // ASLAM_MATCHER_MATCHING_PROBLEM_INFINITE_LINES_TO_FRAME_H_
