#ifndef ASLAM_COMMON_EIGEN_HELPERS_H_
#define ASLAM_COMMON_EIGEN_HELPERS_H_

#include <cmath>
#include <string>

#include <gtest/gtest.h>

namespace aslam {
namespace common {
template<typename LeftMat, typename RightMat>
::testing::AssertionResult MatricesEqual(const LeftMat& A,
                                         const RightMat& B,
                                         double threshold = 1e-8) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    return ::testing::AssertionFailure()
    << "Matrix size mismatch: "
    << A.rows() << "x" << A.cols() << " != "
    << B.rows() << "x" << B.cols();
  }

  bool success = true;
  std::string message;
  for (int i = 0; i < A.rows(); ++i) {
    for (int j = 0; j < A.cols(); ++j) {
      double Aij = A(i, j);
      double Bij = B(i, j);
      if (std::abs(Aij - Bij) > threshold) {
        success = false;
        message +=
            "\n  Mismatch at " + std::to_string(i) + "," + std::to_string(j) +
            " : " + std::to_string(Aij) + " != " + std::to_string(Bij);
      }
    }
  }

  return success ?
      ::testing::AssertionSuccess() :
       ::testing::AssertionFailure() << message << "\n";
}
}  // namespace common
}  // namespace aslam
#endif  // ASLAM_COMMON_EIGEN_HELPERS_H_
