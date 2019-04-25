#ifndef ASLAM_LINES_HOMOGENEOUS_LINE_H_
#define ASLAM_LINES_HOMOGENEOUS_LINE_H_

#include <limits>

#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {

template <typename T>
class HomogeneousLine2dTemplate final {
 public:
  using PointType = Eigen::Matrix<T, 2, 1>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  HomogeneousLine2dTemplate() = delete;
  HomogeneousLine2dTemplate(
      const PointType& start_point, const PointType& end_point) {
    const PointType direction_vector = end_point - start_point;
    normalized_normal_vector_ << -direction_vector.y(), direction_vector.x();
    CHECK_GT(
        normalized_normal_vector_.squaredNorm(),
        std::numeric_limits<T>::epsilon());
    normalized_normal_vector_.normalize();
    // Find the distance d by inserting one of the points.
    positive_distance_to_origin_ =
        -(normalized_normal_vector_.x() * start_point.x() +
          normalized_normal_vector_.y() * start_point.y());
    // Make distance always positive and invert the normal vector if necessary.
    if (positive_distance_to_origin_ < static_cast<T>(0.0)) {
      normalized_normal_vector_ = -normalized_normal_vector_;
      positive_distance_to_origin_ = -positive_distance_to_origin_;
    }
  }
  explicit HomogeneousLine2dTemplate(const LineImpl<T, 2>& line_2d)
      : HomogeneousLine2dTemplate(
            line_2d.getStartPoint(), line_2d.getEndPoint()) {}
  ~HomogeneousLine2dTemplate() = default;

  T getSignedDistanceToPoint(const PointType& point) const {
    return normalized_normal_vector_.x() * point.x() +
           normalized_normal_vector_.y() * point.y() +
           positive_distance_to_origin_;
  }

  bool getXFromY(const T y, T* x) const {
    CHECK_NOTNULL(x);
    if (normalized_normal_vector_.x() * normalized_normal_vector_.x() <
        std::numeric_limits<T>::epsilon()) {
      return false;
    }
    *x = -static_cast<T>(1.0) / normalized_normal_vector_.x() *
         (normalized_normal_vector_.y() * y + positive_distance_to_origin_);
    return true;
  }

  bool getYFromX(const T x, T* y) const {
    CHECK_NOTNULL(y);
    if (normalized_normal_vector_.y() * normalized_normal_vector_.y() <
        std::numeric_limits<T>::epsilon()) {
      return false;
    }
    *y = -static_cast<T>(1.0) / normalized_normal_vector_.y() *
         (normalized_normal_vector_.x() * x + positive_distance_to_origin_);
    return true;
  }

  const PointType& getNormalizedNormalVector() const {
    return normalized_normal_vector_;
  }

  T getDistanceToOrigin() const {
    return positive_distance_to_origin_;
  }

 private:
  // (n_x, n_y, d): Homogeneous line coordinates consisting of the normalized
  // normal vector n = (n_x, n_y) and the distance d from the origin.  The line
  // equation can be written as n_x * x + n_y * y + d = 0.
  PointType normalized_normal_vector_;
  T positive_distance_to_origin_;
};

using HomogeneousLine2d = HomogeneousLine2dTemplate<double>;

}  // namespace aslam

#endif  // ASLAM_LINES_HOMOGENEOUS_LINE_H_
