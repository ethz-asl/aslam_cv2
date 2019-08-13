#ifndef ASLAM_COVARIANCE_HELPERS_H_
#define ASLAM_COVARIANCE_HELPERS_H_

#include <aslam/common/pose-types.h>
#include <Eigen/Dense>

namespace aslam {
namespace common {

void rotateCovariance(const aslam::Transformation& T_B_A,
                      const aslam::TransformationCovariance& A_covariance,
                      aslam::TransformationCovariance* B_covariance);

}  // namespace common
}  // namespace aslam
#endif  // ASLAM_COVARIANCE_HELPERS_H_
