#ifndef ASLAM_LINES_LINE_H_
#define ASLAM_LINES_LINE_H_

#include <limits>

#include <Eigen/Dense>
#include <glog/logging.h>

#include "aslam/common/pose-types.h"

namespace aslam {

template <typename Type, int Dimension>
struct LineImpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<Type, Dimension, 1> PointType;

  LineImpl() = default;
  LineImpl(const PointType& start, const PointType& end)
      : start_point_(start), end_point_(end) {}
  virtual ~LineImpl() = default;

  bool hasStrictlyPositiveLength() const {
    return getSquaredLength() > std::numeric_limits<Type>::epsilon();
  }

  PointType getMidpoint() const {
    return 0.5 * (start_point_ + end_point_);
  }

  Type getSquaredLength() const {
    return (start_point_ - end_point_).squaredNorm();
  }

  const Eigen::Matrix<Type, Dimension, 1>& getStartPoint() const {
    return start_point_;
  }

  void setStartPoint(const Eigen::Matrix<Type, Dimension, 1>& start_point) {
    start_point_ = start_point;
  }

  Eigen::Matrix<Type, Dimension, 1>* getStartPointMutable() {
    return &start_point_;
  }

  const Eigen::Matrix<Type, Dimension, 1>& getEndPoint() const {
    return end_point_;
  }

  void setEndPoint(const Eigen::Matrix<Type, Dimension, 1>& end_point) {
    end_point_ = end_point;
  }

  Eigen::Matrix<Type, Dimension, 1>* getEndPointMutable() {
    return &end_point_;
  }

 private:
  PointType start_point_;
  PointType end_point_;
};

using Line2d = LineImpl<double, 2>;
using Lines2d = Aligned<std::vector, Line2d>;

using Line3d = LineImpl<double, 3>;
using Lines3d = Aligned<std::vector, Line3d>;

inline Line3d operator*(const Transformation& T_B_A, const Line3d& A_line) {
  return Line3d(T_B_A * A_line.getStartPoint(), T_B_A * A_line.getEndPoint());
}

}  // namespace aslam

#endif  // ASLAM_LINES_LINE_H_
