#ifndef ASLAM_LINES_LINE_2D_WITH_ANGLE_HELPERS_H_
#define ASLAM_LINES_LINE_2D_WITH_ANGLE_HELPERS_H_

#include <algorithm>
#include <limits>

#include <glog/logging.h>

#include "aslam/lines/line-2d-with-angle.h"

namespace aslam {

// Returns the angle difference between two 2D lines in radians. Range: [0,
// PI/2].
inline double getAngleInRadiansBetweenLines2d(
    const Line2dWithAngle& line_a, const Line2dWithAngle& line_b) {
  const double angle_wrt_x_axis_line_a_rad = line_a.getAngleWrtXAxisRad();
  const double angle_wrt_x_axis_line_b_rad = line_b.getAngleWrtXAxisRad();
  CHECK_GE(angle_wrt_x_axis_line_a_rad, 0.0);
  CHECK_LT(angle_wrt_x_axis_line_a_rad, M_PI);
  CHECK_GE(angle_wrt_x_axis_line_b_rad, 0.0);
  CHECK_LT(angle_wrt_x_axis_line_b_rad, M_PI);
  double angle_difference_rad =
      std::abs(angle_wrt_x_axis_line_a_rad - angle_wrt_x_axis_line_b_rad);
  angle_difference_rad =
      std::min(angle_difference_rad, M_PI - angle_difference_rad);
  CHECK_GE(angle_difference_rad, 0.0);
  CHECK_LE(angle_difference_rad, M_PI_2);
  return angle_difference_rad;
}

// Returns the intersecting point between two 2D lines.
inline bool getIntersectingPoint(
    const Line2d& line_a, const Line2d& line_b,
    Line2d::PointType* intersecting_point) {
  CHECK_NOTNULL(intersecting_point);

  const Line2d::PointType& line_a_start_point = line_a.getStartPoint();
  const Line2d::PointType line_a_vector_in_line_direction =
      line_a.getEndPoint() - line_a.getStartPoint();
  CHECK_GT(line_a_vector_in_line_direction.squaredNorm(), 0.0);

  const Line2d::PointType& line_b_start_point = line_b.getStartPoint();
  const Line2d::PointType line_b_vector_in_line_direction =
      line_b.getEndPoint() - line_b.getStartPoint();
  CHECK_GT(line_b_vector_in_line_direction.squaredNorm(), 0.0);

  const Line2d::PointType b = line_b_start_point - line_a_start_point;
  Eigen::Matrix2d J;
  J << line_a_vector_in_line_direction(0), -line_b_vector_in_line_direction(0),
      line_a_vector_in_line_direction(1), -line_b_vector_in_line_direction(1);

  if (std::abs(J.determinant()) < std::numeric_limits<double>::epsilon()) {
    // If the determinant is zero, the two lines are colinear. No intersecting
    // point can be computed.
    return false;
  }

  const Line2d::PointType lambda = J.householderQr().solve(b);

  *intersecting_point =
      line_a_start_point + lambda(0) * line_a_vector_in_line_direction;

  return true;
}

// Computes a scalar metric for the lateral distance between two lines.
// For each of the two lines, the distance between their midpoint, and the
// closest point on the other line (closest to the midpoint) is computed, and
// the arithmetic average of both distances is returned.
//
//  a
//  |      b
//  |   q  |
//  x------o
//  |      |
//  |      |
//     w   |
//  o------x
//         |
//         |
//         |
//         |
//
//
// Assigns 0.5 (q + w) to average_lateral_distance.
//
// Note that the intersection points (point on line y closest to the midpoint
// of line x, marked by 'o') may also lie on the
// extrapolated line y (as is the case for line b in the example above).
//
// If the two lines are orthogonal to each other, false is returned, as no
// lateral distance can be computed. Otherwise, the function returns true.
inline double getAverageLateralDistance(
    const Line2d& line_a, const Line2d& line_b) {
  const Line2d::PointType line_a_vector_in_line_direction =
      line_a.getStartPoint() - line_a.getEndPoint();
  const Line2d::PointType line_b_vector_in_line_direction =
      line_b.getStartPoint() - line_b.getEndPoint();

  if (std::abs(
          line_a_vector_in_line_direction.dot(
              line_b_vector_in_line_direction)) <
      std::numeric_limits<double>::epsilon()) {
    // The two lines are orthogonal to each other. A lateral distance cannot
    // be computed.
    return std::numeric_limits<double>::infinity();
  }

  const Line2d::PointType vector_orthogonal_to_line_a(
      -line_a_vector_in_line_direction(1), line_a_vector_in_line_direction(0));
  const Line2d line_orthogonal_to_line_a(
      line_a.getMidpoint(), line_a.getMidpoint() + vector_orthogonal_to_line_a);
  Line2d::PointType intersecting_point_line_b;
  CHECK(
      getIntersectingPoint(
          line_orthogonal_to_line_a, line_b, &intersecting_point_line_b));
  const double lateral_distance_a =
      (line_a.getMidpoint() - intersecting_point_line_b).norm();

  const Line2d::PointType vector_orthogonal_to_line_b(
      -line_b_vector_in_line_direction(1), line_b_vector_in_line_direction(0));
  const Line2d line_orthogonal_to_line_b(
      line_b.getMidpoint(), line_b.getMidpoint() + vector_orthogonal_to_line_b);
  Line2d::PointType intersecting_point_line_a;
  CHECK(
      getIntersectingPoint(
          line_orthogonal_to_line_b, line_a, &intersecting_point_line_a));
  const double lateral_distance_b =
      (line_b.getMidpoint() - intersecting_point_line_a).norm();

  return 0.5 * (lateral_distance_a + lateral_distance_b);
}

}  // namespace aslam

#endif  // ASLAM_LINES_LINE_2D_WITH_ANGLE_HELPERS_H_
