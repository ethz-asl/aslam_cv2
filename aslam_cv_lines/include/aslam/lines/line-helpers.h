#ifndef ASLAM_LINES_LINE_HELPERS_H_
#define ASLAM_LINES_LINE_HELPERS_H_

#include <limits>

#include <Eigen/Core>
#include <aslam/cameras/camera.h>
#include <glog/logging.h>

#include "aslam/lines/line.h"

namespace aslam {

struct LineProjectionResult {
  LineProjectionResult() = delete;
  LineProjectionResult(
      const ProjectionResult& start_point_projection_result,
      const ProjectionResult& end_point_projection_result)
      : start_point_projection_result_(start_point_projection_result),
        end_point_projection_result_(end_point_projection_result) {}

  bool areStartAndEndPointDistinctAndVisible() const {
    return start_point_projection_result_.isKeypointVisible() &&
           end_point_projection_result_.isKeypointVisible();
  }

  const ProjectionResult& getStartPointProjectionResult() const {
    return start_point_projection_result_;
  }

  const ProjectionResult& getEndPointProjectionResult() const {
    return end_point_projection_result_;
  }

 private:
  ProjectionResult start_point_projection_result_;
  ProjectionResult end_point_projection_result_;
};

template <class Scalar>
LineProjectionResult projectLineIntoImagePlane(
    const LineImpl<Scalar, 3>& line_3d, const Camera& camera,
    LineImpl<Scalar, 2>* projected_line_2d) {
  CHECK_NOTNULL(projected_line_2d);

  Eigen::Matrix<Scalar, 2, 1> projected_start_point =
      Eigen::Matrix<Scalar, 2, 1>::Zero();
  const ProjectionResult start_point_projection_result =
      camera.project3(line_3d.getStartPoint(), &projected_start_point);

  Eigen::Matrix<Scalar, 2, 1> projected_end_point =
      Eigen::Matrix<Scalar, 2, 1>::Zero();
  const ProjectionResult end_point_projection_result =
      camera.project3(line_3d.getStartPoint(), &projected_end_point);

  const LineProjectionResult line_projection_result(
      start_point_projection_result, end_point_projection_result);

  *projected_line_2d =
      LineImpl<Scalar, 2>(projected_start_point, projected_end_point);

  return line_projection_result;
}

template <typename T>
inline void projectLine3dToImagePlane(
    const LineImpl<T, 3>& C_line_3d,
    const Eigen::Matrix<T, 3, 3>& camera_matrix,
    LineImpl<T, 2>* C_projected_line_2d) {
  CHECK_NOTNULL(C_projected_line_2d);

  const Eigen::Matrix<T, 3, 1>& start_point = C_line_3d.getStartPoint();
  const Eigen::Matrix<T, 3, 1>& end_point = C_line_3d.getEndPoint();
  CHECK(
      (end_point - start_point).squaredNorm() >
      std::numeric_limits<T>::epsilon());
  if (end_point(2) * end_point(2) <= std::numeric_limits<T>::epsilon()) {
    CHECK(start_point(2) * start_point(2) > std::numeric_limits<T>::epsilon());
  }
  const Eigen::Matrix<T, 3, 1> homogeneous_projected_start_point =
      camera_matrix * start_point;
  const Eigen::Matrix<T, 3, 1> homogeneous_projected_end_point =
      camera_matrix * end_point;
  Eigen::Matrix<T, 2, 1> projected_start_point = Eigen::Matrix<T, 2, 1>::Zero();
  Eigen::Matrix<T, 2, 1> projected_end_point = Eigen::Matrix<T, 2, 1>::Zero();
  if (homogeneous_projected_start_point(2) *
          homogeneous_projected_start_point(2) >
      std::numeric_limits<T>::epsilon()) {
    projected_start_point(0) = homogeneous_projected_start_point(0) /
                               homogeneous_projected_start_point(2);
    projected_start_point(1) = homogeneous_projected_start_point(1) /
                               homogeneous_projected_start_point(2);
  }
  if (homogeneous_projected_end_point(2) * homogeneous_projected_end_point(2) >
      std::numeric_limits<T>::epsilon()) {
    projected_end_point(0) =
        homogeneous_projected_end_point(0) / homogeneous_projected_end_point(2);
    projected_end_point(1) =
        homogeneous_projected_end_point(1) / homogeneous_projected_end_point(2);
  }

  *C_projected_line_2d =
      LineImpl<T, 2>(projected_start_point, projected_end_point);
}

}  // namespace aslam

#endif  // ASLAM_LINES_LINE_HELPERS_H_
