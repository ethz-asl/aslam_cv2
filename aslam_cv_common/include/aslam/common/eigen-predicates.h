#ifndef GTEST_EIGEN_PREDICATE_H_
#define GTEST_EIGEN_PREDICATE_H_

#include <cmath>
#include <sstream>

// Deliberately not including Eigen here, to avoid the dependency.
#include <gtest/gtest.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#define EXPECT_NEAR_EIGEN(matrix_A, matrix_B, precision) \
  EXPECT_TRUE(gtest_catkin::MatricesEqual(matrix_A, matrix_B, precision))

#define ASSERT_NEAR_EIGEN(matrix_A, matrix_B, precision) \
  ASSERT_TRUE(gtest_catkin::MatricesEqual(matrix_A, matrix_B, precision))

namespace gtest_catkin {
// Default tolerance for MatricesEqual.
static const double kDefaultTolerance = 1e-8;

// Compare two eigen matrices by checking whether the absolute difference
// between any two elements exceeds a threshold. On failure, a helpful error
// message is returned. Use like this:
//   EXPECT_TRUE(MatricesEqual(first_matrix, second_matrix, tolerance));
template<typename LeftMatrix, typename RightMatrix>
testing::AssertionResult MatricesEqual(const LeftMatrix& A,
                                       const RightMatrix& B,
                                       double tolerance) {
  if (A.rows() != B.rows() || A.cols() != B.cols()) {
    return testing::AssertionFailure()
      << "Matrix size mismatch: "
      << A.rows() << "x" << A.cols() << " != "
      << B.rows() << "x" << B.cols();
  }

  if (A.isApprox(B, tolerance)) {
    return testing::AssertionSuccess();
  } else {
    std::stringstream spy_difference;
    testing::AssertionResult failure_reason(false);
    for (int i = 0; i < A.rows(); ++i) {
      for (int j = 0; j < A.cols(); ++j) {
        double Aij = A(i, j);
        double Bij = B(i, j);
        if (!std::isfinite(Aij) ||
            !std::isfinite(Bij) ||
            !Eigen::internal::isApprox(Aij, Bij, tolerance)) {
          spy_difference << "x";
          if (A.rows() == 1) {
            failure_reason <<
                "\nMismatch at position " << j << ": " << Aij << " != " << Bij;
          } else if (A.cols() == 1) {
            failure_reason <<
                "\nMismatch at position " << i << ": " << Aij << " != " << Bij;
          } else {
            failure_reason << "\nMismatch at "
                << i << "," << j << ": " << Aij << " != " << Bij;
          }
        } else {
          spy_difference << " ";
        }
      }
      spy_difference << "\n";
    }
    failure_reason << "\n";

    // If we have a matrix and it is not very small then print the sparsity
    // pattern of the difference between A and B.
    if (A.rows() > 1 && A.cols() > 1 && A.rows() * A.cols() > 12) {
      failure_reason << "Sparsity pattern of difference:\n"
          << spy_difference.str();
    }

    return failure_reason;
  }
}

// Compare two eigen matrices by checking whether the absolute difference
// between any two elements exceeds a threshold. On failure, a helpful error
// message is returned. Use like this:
//   EXPECT_TRUE(MatricesEqual(first_matrix, second_matrix));
template<typename LeftMatrix, typename RightMatrix>
testing::AssertionResult MatricesEqual(const LeftMatrix& A,
                                       const RightMatrix& B) {
  return MatricesEqual(A, B, kDefaultTolerance);
}
}  // namespace gtest_catkin

#endif  // GTEST_EIGEN_PREDICATE_H_
