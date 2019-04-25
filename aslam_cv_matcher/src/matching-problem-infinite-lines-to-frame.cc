#include "aslam/matcher/matching-problem-infinite-lines-to-frame.h"

#include <aslam/lines/homogeneous-line-helpers.h>
#include <aslam/lines/homogeneous-line.h>
#include <aslam/lines/line-2d-with-angle-helpers.h>

namespace aslam {

bool reprojectInfiniteLineIntoImage(
    const Line3d& line_3d_C, const Camera& camera,
    Line2dWithAngle* reprojected_line) {
  CHECK_NOTNULL(reprojected_line);

  const Eigen::Vector3d p_C_start = line_3d_C.getStartPoint();
  Eigen::Vector2d map_line_start;
  const ProjectionResult start_projection_result =
      camera.project3(p_C_start, &map_line_start);

  const Eigen::Vector3d p_C_end = line_3d_C.getEndPoint();
  Eigen::Vector2d map_line_end;
  const ProjectionResult end_projection_result =
      camera.project3(p_C_end, &map_line_end);

  const HomogeneousLine2d projected_line(map_line_start, map_line_end);
  const Eigen::Vector2d x_limits(0.0, static_cast<double>(camera.imageWidth()));
  const Eigen::Vector2d y_limits(
      0.0, static_cast<double>(camera.imageHeight()));

  Vector2dList intersections;
  getLineIntersectionWithRectangle(
      projected_line, x_limits, y_limits, &intersections);

  if (intersections.empty()) {
    return false;
  }

  *reprojected_line = Line2dWithAngle(map_line_start, map_line_end);
  return true;
}

constexpr double kDegToRad = M_PI / 180.0;

double getLateralDistanceToInfiniteLinePixels(
    const Line2d& line_in_image, const Line2d& infinite_line) {
  const Eigen::Vector2d v_a =
      line_in_image.getStartPoint() - line_in_image.getEndPoint();
  const Eigen::Vector2d v_a_orth(-v_a(1), v_a(0));
  const Line2d line_a_orth(
      line_in_image.getMidpoint(), line_in_image.getMidpoint() + v_a_orth);
  Eigen::Vector2d intersecting_point_line_b;
  getIntersectingPoint(line_a_orth, infinite_line, &intersecting_point_line_b);
  const double lateral_distance_a =
      (line_in_image.getMidpoint() - intersecting_point_line_b).norm();
  return lateral_distance_a;
}

MatchingProblemInfiniteLinesToFrame::MatchingProblemInfiniteLinesToFrame(
    const Lines2dWithAngle& lines_2d_frame, const Lines3d& lines_3d_C_lines,
    const Camera& camera, const double search_radius_deg,
    const double angle_threshold_deg,
    const double lateral_distance_threshold_px)
    : lines_2d_frame_(lines_2d_frame),
      lines_3d_C_lines_(lines_3d_C_lines),
      camera_(camera),
      search_radius_rad_(search_radius_deg * kDegToRad),
      angle_threshold_rad_(angle_threshold_deg * kDegToRad),
      lateral_distance_threshold_px_(lateral_distance_threshold_px) {
  CHECK_GT(search_radius_rad_, 0.0);
  CHECK_LE(search_radius_rad_, M_PI);
  CHECK_GT(angle_threshold_rad_, 0.0);
  CHECK_LE(angle_threshold_rad_, M_PI);
  CHECK_GT(lateral_distance_threshold_px_, 0.0);
}

bool MatchingProblemInfiniteLinesToFrame::doSetup() {
  LOG_IF(FATAL, lines_2d_frame_.empty())
      << "Please don't try to match nothing.";

  const size_t num_2d_lines = lines_2d_frame_.size();

  lines_2d_index_.resize(kSearchDimension, num_2d_lines);

  for (size_t line_2d_idx = 0u; line_2d_idx < num_2d_lines; ++line_2d_idx) {
    const Line2dWithAngle& line = lines_2d_frame_[line_2d_idx];
    lines_2d_index_(0, line_2d_idx) = line.getAngleWrtXAxisRad();
  }
  nabo_.reset(
      Nabo::NNSearchD::createKDTreeLinearHeap(
          lines_2d_index_, kSearchDimension));

  const size_t num_3d_lines = lines_3d_C_lines_.size();
  is_line_3d_valid_.resize(num_3d_lines, false);

  lines_2d_reprojected_.reserve(num_3d_lines);
  lines_2d_reprojected_index_to_lines_3d_index_.clear();

  for (size_t line_3d_idx = 0u; line_3d_idx < num_3d_lines; ++line_3d_idx) {
    Line2dWithAngle reprojected_line;
    if (reprojectInfiniteLineIntoImage(
        lines_3d_C_lines_[line_3d_idx], camera_, &reprojected_line)) {
      is_line_3d_valid_[line_3d_idx] = true;
      lines_2d_reprojected_.emplace_back(reprojected_line);
      lines_2d_reprojected_index_to_lines_3d_index_.emplace_back(line_3d_idx);
    }
  }

  return true;
}

void MatchingProblemInfiniteLinesToFrame::getCandidates(
    CandidatesList* candidates_for_3d_lines) {
  CHECK_NOTNULL(candidates_for_3d_lines)->clear();
  candidates_for_3d_lines->resize(numBananas());

  const int num_neighbors = static_cast<int>(numApples());
  const int num_valid_query_lines_3d =
      static_cast<int>(lines_2d_reprojected_.size());
  LOG(INFO) << "Num valid query lines 3d: " << num_valid_query_lines_3d;

  Eigen::MatrixXi result_indices(num_neighbors, num_valid_query_lines_3d);
  Eigen::MatrixXd distances2(num_neighbors, num_valid_query_lines_3d);

  Eigen::MatrixXd query(kSearchDimension, num_valid_query_lines_3d);

  for (int line_3d_idx = 0; line_3d_idx < num_valid_query_lines_3d;
      ++line_3d_idx) {
    const Line2dWithAngle& line = lines_2d_reprojected_[line_3d_idx];

    query(0, line_3d_idx) = line.getAngleWrtXAxisRad();
  }

  CHECK(nabo_);
  nabo_->knn(
      query, result_indices, distances2, num_neighbors, kSearchNNEpsilon,
      kSearchOptionFlags, search_radius_rad_);

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
          const double lateral_distance_px =
              getLateralDistanceToInfiniteLinePixels(
                  line_2d, line_2d_reprojected);
          if (lateral_distance_px < lateral_distance_threshold_px_) {
            const double score = computeMatchScore(
                distance2, angle_diff_rad, lateral_distance_px);
            constexpr int kPriority = 0;
            line_2d_candidates.emplace_back(
                matching_2d_line_index, line_3d_index, score, kPriority);
          }
        }
      }
    }

    (*candidates_for_3d_lines)[line_3d_index] = line_2d_candidates;
  }
}

}  // namespace aslam
