#include "aslam/matcher/matching-problem-infinite-lines-to-frame.h"

#include <aslam/common/line-helpers.h>

namespace aslam {

bool reprojectInfiniteLineIntoImage(
    const aslam::Line3d& line_3d_C, const aslam::Camera& camera,
    Line* reprojected_line) {
  CHECK_NOTNULL(reprojected_line);

  const Eigen::Vector3d p_C_start = line_3d_C.getStartPoint();
  Eigen::Vector2d map_line_start;
  const ProjectionResult start_projection_result =
      camera.project3(p_C_start, &map_line_start);

  LOG(INFO) << "repr. start: " << map_line_start.transpose();

  const Eigen::Vector3d p_C_end = line_3d_C.getEndPoint();
  Eigen::Vector2d map_line_end;
  const aslam::ProjectionResult end_projection_result =
      camera.project3(p_C_end, &map_line_end);

  LOG(INFO) << "repr. end: " << map_line_end.transpose();

  const HomogeneousLine2d projected_line(map_line_start, map_line_end);
  const Eigen::Vector2d x_limits(0.0, static_cast<double>(camera.imageWidth()));
  const Eigen::Vector2d y_limits(0.0, static_cast<double>(camera.imageHeight()));

  Vector2dList intersections;
  getLineIntersectionWithRectangle(
      projected_line, x_limits, y_limits, &intersections);

  if (intersections.empty()) {
    return false;
  }

  LOG(INFO) << "Creating line";
  *reprojected_line = Line(map_line_start, map_line_end);
  LOG(INFO) << "Done";
  return true;
}

double getLateralDistanceToInfiniteLinePixels(
    const aslam::Line& line_in_image, const aslam::Line& infinite_line) {
  const Eigen::Vector2d v_a =
      line_in_image.getStartPoint() - line_in_image.getEndPoint();
  const Eigen::Vector2d v_a_orth(-v_a(1), v_a(0));
  const aslam::Line line_a_orth(
      line_in_image.getReferencePoint(), line_in_image.getReferencePoint() + v_a_orth);
  Eigen::Vector2d intersecting_point_line_b;
  getIntersectingPoint(line_a_orth, infinite_line, &intersecting_point_line_b);
  const double lateral_distance_a =
      (line_in_image.getReferencePoint() - intersecting_point_line_b).norm();
  return lateral_distance_a;
}

MatchingProblemInfiniteLinesToFrame::MatchingProblemInfiniteLinesToFrame(
    const aslam::Lines& lines_2d_frame, const aslam::Lines3d& lines_3d_C_lines,
    const aslam::Camera& camera, const double search_radius_deg,
    const double angle_threshold_deg, const double lateral_distance_threshold_px)
    : lines_2d_frame_(lines_2d_frame), lines_3d_C_lines_(lines_3d_C_lines),
      camera_(camera), search_radius_deg_(search_radius_deg),
      angle_threshold_deg_(angle_threshold_deg),
      lateral_distance_threshold_px_(lateral_distance_threshold_px) {
  CHECK_GT(search_radius_deg_, 0.0);
  CHECK_LE(search_radius_deg_, 180.0);
  CHECK_GT(angle_threshold_deg_, 0.0);
  CHECK_LE(angle_threshold_deg_, 180.0);
  CHECK_GT(lateral_distance_threshold_px_, 0.0);
}

bool MatchingProblemInfiniteLinesToFrame::doSetup() {
  LOG_IF(FATAL, lines_2d_frame_.empty())
      << "Please don't try to match nothing.";

  const size_t num_2d_lines = lines_2d_frame_.size();

  lines_2d_index_.resize(kSearchDimension, num_2d_lines);
  /*
  lines_2d_start_index_.resize(kLinesDimension, num_2d_lines);
  lines_2d_end_index_.resize(kLinesDimension, num_2d_lines);
  lines_2d_reference_index_.resize(kLinesDimension, num_2d_lines);*/

  for (size_t line_2d_idx = 0u; line_2d_idx < num_2d_lines; ++line_2d_idx) {
    const aslam::Line& line = lines_2d_frame_[line_2d_idx];
    lines_2d_index_(0, line_2d_idx) = line.getAngleDeg();
    /*
    lines_2d_index_.block<2, 1>(0, line_2d_idx) = line.getStartPoint();
    lines_2d_index_.block<2, 1>(2, line_2d_idx) = line.getEndPoint();
    */
    /*
    lines_2d_start_index_.col(line_2d_idx) = line.getStartPoint();
    lines_2d_end_index_.col(line_2d_idx) = line.getEndPoint();

    const Eigen::Vector2d reference_point = line.getReferencePoint();
    lines_2d_reference_index_.col(line_2d_idx) = reference_point;*/
  }
  nabo_.reset(
      Nabo::NNSearchD::createKDTreeLinearHeap(
          lines_2d_index_, kSearchDimension));

  /*
  nabo_start_.reset(
      Nabo::NNSearchD::createKDTreeLinearHeap(
          lines_2d_start_index_, kLinesDimension));

  nabo_end_.reset(
      Nabo::NNSearchD::createKDTreeLinearHeap(
          lines_2d_end_index_, kLinesDimension));

  nabo_reference_.reset(
      Nabo::NNSearchD::createKDTreeLinearHeap(
          lines_2d_reference_index_, kLinesDimension));*/

  const size_t num_3d_lines = lines_3d_C_lines_.size();
  is_line_3d_valid_.resize(num_3d_lines, false);

  lines_2d_reprojected_.reserve(num_3d_lines);
  lines_2d_reprojected_index_to_lines_3d_index_.clear();

  for (size_t line_3d_idx = 0u; line_3d_idx < num_3d_lines; ++line_3d_idx) {
    Line reprojected_line;
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

  /*
  Eigen::MatrixXi result_indices_start(num_neighbors, num_valid_query_lines_3d);
  Eigen::MatrixXi distances2_start(num_neighbors, num_valid_query_lines_3d);

  Eigen::MatrixXi result_indices_end(num_neighbors, num_valid_query_lines_3d);
  Eigen::MatrixXi distances2_end(num_neighbors, num_valid_query_lines_3d);

  Eigen::MatrixXi result_indices_reference(num_neighbors, num_valid_query_lines_3d);
  Eigen::MatrixXi distances2_reference(num_neighbors, num_valid_query_lines_3d);*/

  Eigen::MatrixXd query(kSearchDimension, num_valid_query_lines_3d);
  /*
  Eigen::MatrixXd query_start(kLinesDimension, num_neighbors);
  Eigen::MatrixXd query_end(kLinesDimension, num_neighbors);
  Eigen::MatrixXd query_reference(kLinesDimension, num_neighbors);*/

  for (int line_3d_idx = 0; line_3d_idx < num_valid_query_lines_3d;
      ++line_3d_idx) {
    const aslam::Line& line = lines_2d_reprojected_[line_3d_idx];
    //query_start.col(line_3d_idx) = line.getStartPoint();
    //query_end.col(line_3d_idx) = line.getEndPoint();
    //query_reference.col(line_3d_idx) = line.getReferencePoint();;

    query(0, line_3d_idx) = line.getAngleDeg();
  }

  CHECK(nabo_);
  nabo_->knn(
      query, result_indices, distances2, num_neighbors, kSearchNNEpsilon,
      kSearchOptionFlags, search_radius_deg_);

  /*
  CHECK(nabo_start_);
  nabo_start_->knn(
      query_start, result_indices_start, distances2_start, num_neighbors,
      kSearchNNEpsilon, kSearchOptionFlags, kMaxRadiusCornersPixels);

  CHECK(nabo_end_);
  nabo_end_->knn(
      query_end, result_indices_end, distances2_end, num_neighbors,
      kSearchNNEpsilon, kSearchOptionFlags, kMaxRadiusCornersPixels);

  CHECK(nabo_reference_);
  nabo_reference_->knn(
      query_reference, result_indices_reference, distances2_reference,
      num_neighbors, kSearchNNEpsilon, kSearchOptionFlags,
      kMaxRadiusCornersPixels);*/

  for (int line_2d_reprojected_idx = 0; line_2d_reprojected_idx < num_valid_query_lines_3d;
      ++line_2d_reprojected_idx) {
    CHECK_LT(line_2d_reprojected_idx, static_cast<int>(
        lines_2d_reprojected_index_to_lines_3d_index_.size()));
    const int line_3d_index =
        lines_2d_reprojected_index_to_lines_3d_index_[line_2d_reprojected_idx];
    CHECK_GE(line_3d_index, 0);
    CHECK_LT(line_3d_index, lines_3d_C_lines_.size());

    const aslam::Line& line_2d_reprojected =
        lines_2d_reprojected_[line_2d_reprojected_idx];

    Candidates line_2d_candidates;
    for (int neighbor_idx = 0; neighbor_idx < num_neighbors; ++neighbor_idx) {

      const double distance2 = distances2(neighbor_idx, line_2d_reprojected_idx);
      if (distance2 < std::numeric_limits<double>::infinity()) {

        const int matching_2d_line_index =
            result_indices(neighbor_idx, line_2d_reprojected_idx);
        CHECK_GE(matching_2d_line_index, 0);
        CHECK_LT(matching_2d_line_index,
                 static_cast<int>(lines_2d_frame_.size()));
        const aslam::Line& line_2d = lines_2d_frame_[matching_2d_line_index];

        const double angle_diff_deg =
            getAngleDifferenceDegrees(
                line_2d_reprojected.getAngleDeg(), line_2d.getAngleDeg());

        if (angle_diff_deg < angle_threshold_deg_) {
          const double lateral_distance_px =
              getLateralDistanceToInfiniteLinePixels(
                  line_2d, line_2d_reprojected);
          if (lateral_distance_px < lateral_distance_threshold_px_) {
            const double score = computeMatchScore(
                distance2, angle_diff_deg, lateral_distance_px);
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
