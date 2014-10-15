#ifndef ASLAM_COMMON_EIGEN_HELPERS_H_
#define ASLAM_COMMON_EIGEN_HELPERS_H_

#include <Eigen/Dense>

namespace aslam {
namespace common {
/// Function to create a 5d Eigen-vector in place.
template <typename ScalarType>
inline Eigen::Matrix<ScalarType, 5, 1> createVector5(const ScalarType& a, const ScalarType& b,
                                                     const ScalarType& c, const ScalarType& d,
                                                     const ScalarType& e) {
  Eigen::Matrix<ScalarType, 5, 1> vec;
  vec << a, b, c, d, e;
  return vec;
}

}  // namespace common
}  // namespace aslam
#endif  // ASLAM_COMMON_EIGEN_HELPERS_H_
