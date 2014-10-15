#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/common/eigen-helpers.h>
#include <aslam/common/eigen-predicates.h>


TEST(TestCameraPinhole, ManualProjectionWithoutDistortion) {

  bool is_equal = false;
  const double precision = 1e-8;
  Eigen::Matrix3d matrix_A;
  Eigen::Matrix3d matrix_B;

  // Test different matrices
  matrix_A.setRandom();
  matrix_B = (matrix_B.array() + 2 * precision).matrix();

  is_equal = gtest_catkin::MatricesEqual(matrix_A, matrix_B, precision);
  EXPECT_FALSE(is_equal);

  // Test exactly equal matrices
  is_equal = gtest_catkin::MatricesEqual(matrix_A, matrix_A, precision);
  EXPECT_TRUE(is_equal);

  // Test equal matrices within precision
  matrix_B = (matrix_B.array() + 0.5 * precision).matrix();
  is_equal = gtest_catkin::MatricesEqual(matrix_A, matrix_B, precision);
  EXPECT_TRUE(is_equal);
}
