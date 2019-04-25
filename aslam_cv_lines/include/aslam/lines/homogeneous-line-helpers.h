#ifndef ASLAM_LINES_HOMOGENEOUS_LINE_HELPERS_H_
#define ASLAM_LINES_HOMOGENEOUS_LINE_HELPERS_H_

#include <limits>

#include <aslam/common/memory.h>
#include <glog/logging.h>

#include "aslam/lines/homogeneous-line.h"
#include "aslam/lines/line.h"

namespace aslam {

template <typename T>
using Vector2dListTemplate = Aligned<std::vector, Eigen::Matrix<T, 2, 1>>;

typedef Vector2dListTemplate<double> Vector2dList;

template <typename T>
inline void getLineIntersectionWithRectangle(
    const HomogeneousLine2dTemplate<T>& line,
    const Eigen::Matrix<T, 2, 1>& x_limits,
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
    if (line.getYFromX(x_limits(i), &x_limits_intersection(i))) {
      if (x_limits_intersection(i) > y_limits(0)) {
        if (x_limits_intersection(i) < y_limits(1)) {
          intersections->emplace_back(x_limits(i), x_limits_intersection(i));
        }
      }
    }
    if (line.getXFromY(y_limits(i), &y_limits_intersection(i))) {
      if (y_limits_intersection(i) > x_limits(0)) {
        if (y_limits_intersection(i) < x_limits(1)) {
          intersections->emplace_back(y_limits_intersection(i), y_limits(i));
        }
      }
    }
  }
}

}  // namespace aslam

#endif  // ASLAM_LINES_HOMOGENEOUS_LINE_HELPERS_H_
