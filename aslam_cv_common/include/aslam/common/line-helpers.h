#ifndef INCLUDE_ASLAM_COMMON_LINE_HELPERS_H_
#define INCLUDE_ASLAM_COMMON_LINE_HELPERS_H_

#include "aslam/common/line.h"

#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {

inline Line3d operator*(const Transformation& T_B_A, const Line3d& A_line) {
  return Line3d(T_B_A * A_line.getStartPoint(), T_B_A * A_line.getEndPoint());
}

template <typename T>
using Vector2dListTemplate = Aligned<std::vector, Eigen::Matrix<T, 2, 1>>;

typedef Vector2dListTemplate<double> Vector2dList;

template <typename T>
inline void getLineIntersectionWithRectangle(
    const HomogeneousLine2dTemplate<T>& line, const Eigen::Matrix<T, 2, 1>& x_limits,
    const Eigen::Matrix<T, 2, 1>& y_limits,
    Vector2dListTemplate<T>* intersections) {
  CHECK_NOTNULL(intersections)->clear();
  // TODO(fabianbl): Use Liang–Barsky algorithm instead to make this a bit more
  // efficient.
  CHECK_GT(x_limits(1), x_limits(0));
  CHECK_GT(y_limits(1), y_limits(0));
  // Intersection coordinates with x_limits(0), x_limits(1), y_limits(0),
  // y_limits(1).
  Eigen::Matrix<T, 2, 1> x_limits_intersection, y_limits_intersection;
  for (int i = 0; i < 2; ++i) {
    if (line.getY(x_limits(i), &x_limits_intersection(i))) {
      if (x_limits_intersection(i) > y_limits(0)) {
        if (x_limits_intersection(i) < y_limits(1)) {
          intersections->emplace_back(x_limits(i), x_limits_intersection(i));
        }
      }
    }
    if (line.getX(y_limits(i), &y_limits_intersection(i))) {
      if (y_limits_intersection(i) > x_limits(0)) {
        if (y_limits_intersection(i) < x_limits(1)) {
          intersections->emplace_back(y_limits_intersection(i), y_limits(i));
        }
      }
    }
  }
}

template <typename T>
inline HomogeneousLine2dTemplate<T> projectLine3dToImagePlane(
    const aslam::LineImpl<T, 3>& C_line_3d,
    const Eigen::Matrix<T, 3, 3>& camera_matrix) {
  const Eigen::Matrix<T, 3, 1>& start_point = C_line_3d.getStartPoint();
  const Eigen::Matrix<T, 3, 1>& end_point = C_line_3d.getEndPoint();
  CHECK((end_point - start_point).squaredNorm() > T(1e-10));
  if (end_point(2) * end_point(2) <= T(1e-10)) {
    CHECK(start_point(2) * start_point(2) > T(1e-10));
  }
  const Eigen::Matrix<T, 3, 1> homogeneous_projected_start_point =
      camera_matrix * start_point;
  const Eigen::Matrix<T, 3, 1> homogeneous_projected_end_point =
      camera_matrix * end_point;
  Eigen::Matrix<T, 2, 1> projected_start_point(T(0.0), T(0.0));
  Eigen::Matrix<T, 2, 1> projected_end_point(T(0.0), T(0.0));
  CHECK(camera_matrix(0, 0) > T(1e-10));
  CHECK(camera_matrix(0, 2) > T(1e-10));
  CHECK(camera_matrix(1, 1) > T(1e-10));
  CHECK(camera_matrix(1, 2) > T(1e-10));
  CHECK(camera_matrix(2, 2) > T(1e-10));
  if (homogeneous_projected_start_point(2) *
          homogeneous_projected_start_point(2) >
      T(1e-10)) {
    projected_start_point(0) = homogeneous_projected_start_point(0) /
                               homogeneous_projected_start_point(2);
    projected_start_point(1) = homogeneous_projected_start_point(1) /
                               homogeneous_projected_start_point(2);
  }
  if (homogeneous_projected_end_point(2) * homogeneous_projected_end_point(2) >
      T(1e-10)) {
    projected_end_point(0) =
        homogeneous_projected_end_point(0) / homogeneous_projected_end_point(2);
    projected_end_point(1) =
        homogeneous_projected_end_point(1) / homogeneous_projected_end_point(2);
  }
  CHECK((projected_start_point - projected_end_point).squaredNorm() > T(1e-10));
  return HomogeneousLine2dTemplate<T>(
      projected_start_point, projected_end_point);
}

inline double getAngleDifferenceDegrees(
    const double angle_deg_1, const double angle_deg_2) {
  CHECK_GE(angle_deg_1, 0.0);
  CHECK_LT(angle_deg_1, 180.0);
  CHECK_GE(angle_deg_2, 0.0);
  CHECK_LT(angle_deg_2, 180.0);
  double diff_degrees = std::abs(angle_deg_1 - angle_deg_2);
  diff_degrees = std::min(diff_degrees, 180.0 - diff_degrees);
  CHECK_GE(diff_degrees, 0.0);
  CHECK_LE(diff_degrees, 90.0);
  return diff_degrees;
}

inline void getIntersectingPoint(
    const aslam::Line& line_a, const aslam::Line& line_b,
    Eigen::Vector2d* intersecting_point) {
  CHECK_NOTNULL(intersecting_point);

  const Eigen::Vector2d& ref_a = line_a.getStartPoint();
  const Eigen::Vector2d v_a = line_a.getEndPoint() - line_a.getStartPoint();
  CHECK_GT(v_a.squaredNorm(), 0.0);

  const Eigen::Vector2d& ref_b = line_b.getStartPoint();
  const Eigen::Vector2d v_b = line_b.getEndPoint() - line_b.getStartPoint();
  CHECK_GT(v_b.squaredNorm(), 0.0);

  const Eigen::Vector2d b = ref_b - ref_a;
  Eigen::Matrix2d J;
  J << v_a(0), -v_b(0), v_a(1), -v_b(1);
  const Eigen::Vector2d lambda = J.householderQr().solve(b);

  const Eigen::Vector2d ip1 = ref_a + lambda(0) * v_a;
  const Eigen::Vector2d ip2 = ref_b + lambda(1) * v_b;

  *intersecting_point = ip1;
}

inline double getAverageLateralDistance(
    const aslam::Line& line_a, const aslam::Line& line_b,
    const cv::Mat& image) {
  const Eigen::Vector2d v_a = line_a.getStartPoint() - line_a.getEndPoint();
  const Eigen::Vector2d v_a_orth(-v_a(1), v_a(0));
  const aslam::Line line_a_orth(
      line_a.getReferencePoint(), line_a.getReferencePoint() + v_a_orth);
  Eigen::Vector2d intersecting_point_line_b;
  getIntersectingPoint(line_a_orth, line_b, &intersecting_point_line_b);
  const double lateral_distance_a =
      (line_a.getReferencePoint() - intersecting_point_line_b).norm();

  const Eigen::Vector2d v_b = line_b.getStartPoint() - line_b.getEndPoint();
  const Eigen::Vector2d v_b_orth(-v_b(1), v_b(0));
  const aslam::Line line_b_orth(
      line_b.getReferencePoint(), line_b.getReferencePoint() + v_b_orth);
  Eigen::Vector2d intersecting_point_line_a;
  getIntersectingPoint(line_b_orth, line_a, &intersecting_point_line_a);
  const double lateral_distance_b =
      (line_b.getReferencePoint() - intersecting_point_line_a).norm();

  return 0.5 * (lateral_distance_a + lateral_distance_b);
}

}  // namespace aslam

#endif /* INCLUDE_ASLAM_COMMON_LINE_HELPERS_H_ */
