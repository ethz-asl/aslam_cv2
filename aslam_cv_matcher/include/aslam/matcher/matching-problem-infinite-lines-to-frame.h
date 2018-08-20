#ifndef ASLAM_CV_MATCHING_PROBLEM_INFINITE_LINES_TO_FRAME_H_
#define ASLAM_CV_MATCHING_PROBLEM_INFINITE_LINES_TO_FRAME_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <map>
#include <memory>
#include <vector>

#include <aslam/cameras/camera.h>
#include <aslam/common/line.h>
#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <Eigen/Core>
#include <nabo/nabo.h>

#include "aslam/matcher/match.h"
#include "aslam/matcher/matching-problem.h"

namespace aslam {

bool reprojectInfiniteLineIntoImage(
    const aslam::Line3d& line_3d_C, const aslam::Camera& camera,
    Line* reprojected_line);

class MatchingProblemInfiniteLinesToFrame : public MatchingProblem {
 public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemInfiniteLinesToFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemInfiniteLinesToFrame);
  ASLAM_ADD_MATCH_TYPEDEFS(InfiniteLinesToFrame);

  MatchingProblemInfiniteLinesToFrame() = delete;

  /// \brief Constructor for a landmarks-to-frame matching problem.
  ///
  /// @param[in]  frame                                       Visual frame.
  /// @param[in]  landmarks                                   List of landmarks with descriptors.
  /// @param[in]  image_space_distance_threshold_pixels       Max image space distance threshold
  ///                                                         for a keypoint and a projected
  ///                                                         land to become match candidates.
  /// @param[in]  hamming_distance_threshold                  Max hamming distance for a keypoint
  ///                                                         and a projected landmark
  ///                                                         to become candidates.
  MatchingProblemInfiniteLinesToFrame(
      const aslam::Lines& lines_2d_frame,
      const aslam::Lines3d& lines_3d_C_lines, const aslam::Camera& camera,
      const double search_radius_px, const double angle_threshold_deg,
      const double lateral_distance_threshold_px);

  virtual ~MatchingProblemInfiniteLinesToFrame() {};

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
      const double distance_px2, const double angle_diff_deg,
      const double lateral_distance_px) {
    constexpr double kAngleDiffWeighting = 0.5;
    return kAngleDiffWeighting * angle_diff_deg +
        (1.0 - kAngleDiffWeighting) * lateral_distance_px;
  }


 private:
  aslam::Lines lines_2d_frame_;
  aslam::Lines3d lines_3d_C_lines_;
  aslam::Lines lines_2d_reprojected_;
  std::vector<int> lines_2d_reprojected_index_to_lines_3d_index_;
  std::vector<unsigned char> is_line_3d_valid_;

  const aslam::Camera& camera_;

  //Eigen::MatrixXd lines_2d_start_index_;
  //Eigen::MatrixXd lines_2d_end_index_;
  //Eigen::MatrixXd lines_2d_reference_index_;
  Eigen::MatrixXd lines_2d_index_;

  //std::unique_ptr<Nabo::NNSearchD> nabo_start_;
  //std::unique_ptr<Nabo::NNSearchD> nabo_end_;
  //std::unique_ptr<Nabo::NNSearchD> nabo_reference_;
  std::unique_ptr<Nabo::NNSearchD> nabo_;


  static constexpr double kSearchNNEpsilon = 0.0;
  static constexpr unsigned kSearchOptionFlags =
      Nabo::NNSearchD::ALLOW_SELF_MATCH;
  static constexpr int kSearchDimension = 1;

  const double search_radius_deg_;
  const double angle_threshold_deg_;
  const double lateral_distance_threshold_px_;
};
}  // namespace aslam
#endif  // ASLAM_CV_MATCHING_PROBLEM_INFINITE_LINES_TO_FRAME_H_
