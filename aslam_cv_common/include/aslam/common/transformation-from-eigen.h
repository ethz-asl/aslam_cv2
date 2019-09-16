#ifndef ASLAM_TRANSFORMATION_FROM_EIGEN_H_
#define ASLAM_TRANSFORMATION_FROM_EIGEN_H_

#include <Eigen/Dense>
#include <aslam/common/pose-types.h>

namespace aslam {
namespace common {

  aslam::Transformation transformationFromApproximateMatrix(
    const Eigen::Matrix4d& approximate_transformation);

}  // namespace common
}  // namespace aslam
#endif  // ASLAM_TRANSFORMATION_FROM_EIGEN_H_
