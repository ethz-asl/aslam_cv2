#include "aslam/matcher/matching-problem-lines-to-frame.h"

#include <aslam/lines/line-2d-with-angle-helpers.h>

namespace aslam {

bool reprojectLineIntoImage(
    const Line3d& line_3d_C, const Camera& camera, Line2d* line_2d) {
  CHECK_NOTNULL(line_2d);

  const Eigen::Vector3d p_C_start = line_3d_C.getStartPoint();
  Eigen::Vector2d map_line_start;
  const ProjectionResult start_projection_result =
      camera.project3(p_C_start, &map_line_start);

  const Eigen::Vector3d p_C_end = line_3d_C.getEndPoint();
  Eigen::Vector2d map_line_end;
  const ProjectionResult end_projection_result =
      camera.project3(p_C_end, &map_line_end);

  *line_2d = Line2d(map_line_start, map_line_end);

  return start_projection_result.isKeypointVisible() &&
      end_projection_result.isKeypointVisible();
}

constexpr double kDeg2Rad = M_PI / 180.0;

MatchingProblemLinesToFrame::MatchingProblemLinesToFrame(
    const Lines2dWithAngle& lines_2d_frame, const Lines3d& lines_3d_C_lines,
    const Camera& camera, const double search_radius_px,
    const double angle_threshold_deg)
    : lines_2d_frame_(lines_2d_frame),
      lines_3d_C_lines_(lines_3d_C_lines),
      camera_(camera),
      search_radius_px_(search_radius_px),
      angle_threshold_rad_(angle_threshold_deg * kDeg2Rad) {
  CHECK_GT(search_radius_px_, 0.0);
  CHECK_GT(angle_threshold_rad_, 0.0);
}

bool MatchingProblemLinesToFrame::doSetup() {
  LOG_IF(FATAL, lines_2d_frame_.empty())
      << "Please don't try to match nothing.";

  const size_t num_2d_lines = lines_2d_frame_.size();

  lines_2d_index_.resize(kLinesDimension, num_2d_lines);

  for (size_t line_2d_idx = 0u; line_2d_idx < num_2d_lines; ++line_2d_idx) {
    const Line2dWithAngle& line = lines_2d_frame_[line_2d_idx];
    lines_2d_index_.block<2, 1>(0, line_2d_idx) = line.getStartPoint();
    lines_2d_index_.block<2, 1>(2, line_2d_idx) = line.getEndPoint();
  }
  nabo_.reset(
      Nabo::NNSearchD::createKDTreeLinearHeap(
          lines_2d_index_, kLinesDimension));

  const size_t num_3d_lines = lines_3d_C_lines_.size();
  is_line_3d_valid_.resize(num_3d_lines, false);

  lines_2d_reprojected_.reserve(num_3d_lines);
  lines_2d_reprojected_index_to_lines_3d_index_.clear();

  for (size_t line_3d_idx = 0u; line_3d_idx < num_3d_lines; ++line_3d_idx) {
    Line2dWithAngle line_2d;
    if (reprojectLineIntoImage(
        lines_3d_C_lines_[line_3d_idx], camera_, &line_2d)) {
      is_line_3d_valid_[line_3d_idx] = true;
      lines_2d_reprojected_.emplace_back(line_2d);
      lines_2d_reprojected_index_to_lines_3d_index_.emplace_back(line_3d_idx);
    }
  }

  return true;
}

void MatchingProblemLinesToFrame::getCandidates(
    CandidatesList* candidates_for_3d_lines) {
  CHECK_NOTNULL(candidates_for_3d_lines)->clear();
  candidates_for_3d_lines->resize(numBananas());

  const int num_neighbors = static_cast<int>(numApples());
  const int num_valid_query_lines_3d =
      static_cast<int>(lines_2d_reprojected_.size());
  LOG(INFO) << "Num valid query lines 3d: " << num_valid_query_lines_3d;

  Eigen::MatrixXi result_indices(num_neighbors, num_valid_query_lines_3d);
  Eigen::MatrixXd distances2(num_neighbors, num_valid_query_lines_3d);

  Eigen::MatrixXd query(kLinesDimension, num_valid_query_lines_3d);

  for (int line_3d_idx = 0; line_3d_idx < num_valid_query_lines_3d;
      ++line_3d_idx) {
    const Line2d& line = lines_2d_reprojected_[line_3d_idx];

    query.block<2, 1>(0, line_3d_idx) = line.getStartPoint();
    query.block<2, 1>(2, line_3d_idx) = line.getEndPoint();
  }

  CHECK(nabo_);
  nabo_->knn(
      query, result_indices, distances2, num_neighbors, kSearchNNEpsilon,
      kSearchOptionFlags, search_radius_px_);

  for (int line_2d_reprojected_idx = 0;
       line_2d_reprojected_idx < num_valid_query_lines_3d;
       ++line_2d_reprojected_idx) {
    CHECK_LT(line_2d_reprojected_idx, static_cast<int>(
        lines_2d_reprojected_index_to_lines_3d_index_.size()));
    const int line_3d_index =
        lines_2d_reprojected_index_to_lines_3d_index_[line_2d_reprojected_idx];
    CHECK_GE(line_3d_index, 0);
    CHECK_LT(line_3d_index, lines_3d_C_lines_.size());

    const Line2dWithAngle& line_2d_reprojected =
        lines_2d_reprojected_[line_2d_reprojected_idx];

    Candidates line_2d_candidates;
    for (int neighbor_idx = 0; neighbor_idx < num_neighbors; ++neighbor_idx) {
      const double distance2 =
          distances2(neighbor_idx, line_2d_reprojected_idx);
      if (distance2 < std::numeric_limits<double>::infinity()) {
        const int matching_2d_line_index =
            result_indices(neighbor_idx, line_2d_reprojected_idx);
        CHECK_GE(matching_2d_line_index, 0);
        CHECK_LT(matching_2d_line_index,
                 static_cast<int>(lines_2d_frame_.size()));
        const Line2dWithAngle& line_2d =
            lines_2d_frame_[matching_2d_line_index];

        const double angle_diff_rad =
            getAngleInRadiansBetweenLines2d(line_2d_reprojected, line_2d);

        if (angle_diff_rad < angle_threshold_rad_) {
          const double score = computeMatchScore(distance2, angle_diff_rad);
          constexpr int kPriority = 0;
          line_2d_candidates.emplace_back(
              matching_2d_line_index, line_3d_index, score, kPriority);
        }
      }
    }

    (*candidates_for_3d_lines)[line_3d_index] = line_2d_candidates;
  }
}

}  // namespace aslam
