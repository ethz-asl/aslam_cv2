#include <aslam/common/transformation-from-eigen.h>

namespace aslam {
namespace common {

aslam::Transformation transformationFromApproximateMatrix(
  const Eigen::Matrix4d& approximate_transformation){
    kindr::minimal::RotationQuaternion q = Quaternion::fromApproximateRotationMatrix(approximate_transformation.block<3,3>(0,0).eval());
    return kindr::minimal::QuatTransformation(q, approximate_transformation.block<3,1>(0,3).eval());
  }

}  // namespace common
}  // namespace aslam
