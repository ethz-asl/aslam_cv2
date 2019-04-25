#ifndef ASLAM_LINES_LINE_2D_WITH_ANGLE_H_
#define ASLAM_LINES_LINE_2D_WITH_ANGLE_H_

#include <limits>

#include <Eigen/Core>
#include <glog/logging.h>

#include "aslam/lines/line.h"

namespace aslam {

struct Line2dWithAngle final : public Line2d {
  static constexpr double kInvalidAngleRad = -1.0;
  Line2dWithAngle() : angle_rad_(kInvalidAngleRad) {}
  Line2dWithAngle(const PointType& start, const PointType& end)
      : Line2d(start, end) {
    setAngleRad();
  }
  ~Line2dWithAngle() = default;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // Returns the angle wrt. x-axis in radians. In the range [0, PI).
  double getAngleWrtXAxisRad() const {
    return angle_rad_;
  }

 private:
  void setAngleRad() {
    const Eigen::Vector2d ref(1.0, 0.0);
    Eigen::Vector2d l;
    if (getStartPoint()(1) > getEndPoint()(1)) {
      l = Eigen::Vector2d(getStartPoint() - getEndPoint());
    } else {
      l = Eigen::Vector2d(getEndPoint() - getStartPoint());
    }
    CHECK_GT(l.norm(), 0.0);
    angle_rad_ = std::acos(l.dot(ref) / l.norm());

    if (std::abs(angle_rad_ - M_PI) < std::numeric_limits<double>::epsilon()) {
      angle_rad_ = 0.0;
    }
    CHECK_GE(angle_rad_, 0.0);
    CHECK_LT(angle_rad_, M_PI);
  }

  // Angle wrt. y-axis, in range [0, PI) radians.
  double angle_rad_;
};
typedef Aligned<std::vector, Line2dWithAngle> Lines2dWithAngle;

}  // namespace aslam

#endif  // ASLAM_LINES_LINE_2D_WITH_ANGLE_H_
