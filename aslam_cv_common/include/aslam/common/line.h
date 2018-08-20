#ifndef INCLUDE_ASLAM_COMMON_LINE_H_
#define INCLUDE_ASLAM_COMMON_LINE_H_

#include <Eigen/Dense>
#include <glog/logging.h>

#include "aslam/common/pose-types.h"

namespace aslam {

template<typename Type, int Dimension>
struct LineImpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<Type, Dimension, 1> PointType;

  LineImpl() = default;
  LineImpl(const PointType& start, const PointType& end) :
    start_point_(start), end_point_(end) {};
  virtual ~LineImpl() = default;

  PointType getReferencePoint() const {
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

struct Line : public LineImpl<double, 2> {
  Line() : angle_deg_(-1.0), angle_valid_(false) {}
  Line(double x0, double y0, double x1, double y1) {
    if (x0 < x1) {
      setStartPoint(PointType(x0, y0));
      setEndPoint(PointType(x1, y1));
    } else if (x0 > x1) {
      setStartPoint(PointType(x1, y1));
      setEndPoint(PointType(x0, y0));
    } else {
      if (y0 < y1) {
        setStartPoint(PointType(x0, y0));
        setEndPoint(PointType(x1, y1));
      } else {
        setStartPoint(PointType(x1, y1));
        setEndPoint(PointType(x0, y0));
      }
    }
    setAngleDeg();
  };

  Line(const Eigen::Vector2d& start, const Eigen::Vector2d& end) :
    Line(start(0), start(1), end(0), end(1)) {
    setAngleDeg();
  };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  double getAngleDeg() const {
    CHECK(angle_valid_);
    return angle_deg_;
  }

 private:
  void setAngleDeg() {
    const Eigen::Vector2d ref(1.0, 0.0);
    Eigen::Vector2d l;
    if (getStartPoint()(1) > getEndPoint()(1)) {
      l = Eigen::Vector2d(getStartPoint() - getEndPoint());
    } else {
      l = Eigen::Vector2d(getEndPoint() - getStartPoint());
    }
    CHECK_GT(l.norm(), 0.0);
    angle_deg_ = acos(l.dot(ref) / l.norm()) / M_PI * 180.0;

    CHECK_GE(angle_deg_, 0.0);
    CHECK_LE(angle_deg_, 180.0) << getStartPoint().transpose() << ", "
        << getEndPoint().transpose() << ", l:" << l.transpose();
    if (angle_deg_ == 180.0) {
      angle_deg_ = 0.0;
    }

    angle_valid_ = true;
  }

  double angle_deg_;
  bool angle_valid_;
};
typedef std::vector<Line> Lines;

typedef LineImpl<double, 3> Line3d;
typedef std::vector<Line3d> Lines3d;

template <typename T>
class HomogeneousLine2dTemplate {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  HomogeneousLine2dTemplate() = delete;

  HomogeneousLine2dTemplate(
      const Eigen::Matrix<T, 2, 1>& start_point,
      const Eigen::Matrix<T, 2, 1>& end_point) {
    const Eigen::Matrix<T, 2, 1> direction_vector = end_point - start_point;
    n_ << (-direction_vector(1)), direction_vector(0);
    CHECK_GT(n_.squaredNorm(), T(1e-10));
    n_.normalize();
    // Find the distance d by inserting one of the points.
    d_ = -(n_(0) * start_point(0) + n_(1) * start_point(1));
    // Make distance always positive and invert the normal vector if necessary.
    if (d_ < T(0.0)) {
      n_ = -n_;
      d_ = -d_;
    }
  }

  T getSignedDistanceToPoint(const Eigen::Matrix<T, 2, 1>& point) const {
    return n_(0) * point(0) + n_(1) * point(1) + d_;
  }

  bool getX(const T y, T* x) const {
    CHECK_NOTNULL(x);
    if (n_(0) * n_(0) < 1e-10) {
      return false;
    }
    *x = -T(1.0) / n_(0) * (n_(1) * y + d_);
    return true;
  }

  bool getY(const T x, T* y) const {
    CHECK_NOTNULL(y);
    if (n_(1) * n_(1) < 1e-10) {
      return false;
    }
    *y = -T(1.0) / n_(1) * (n_(0) * x + d_);
    return true;
  }

  const Eigen::Matrix<T, 2, 1>& getNormalVector() const {
    return n_;
  }

  T getDistance() const {
    return d_;
  }

 private:
  // (n_x, n_y, d): Homogeneous line coordinates consisting of the normalized
  // normal vector n = (n_x, n_y) and the distance d from the origin.  The line
  // equation can be written as n_x * x + n_y * y + d = 0.
  Eigen::Matrix<T, 2, 1> n_;
  T d_;
};

typedef HomogeneousLine2dTemplate<double> HomogeneousLine2d;

}  // namespace aslam

#endif /* INCLUDE_ASLAM_COMMON_LINE_H_ */
