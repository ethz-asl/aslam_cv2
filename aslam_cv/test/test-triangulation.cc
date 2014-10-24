#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/triangulation/triangulation-toolbox.h>
#include <eigen-checks/gtest.h>

// TODO(burrimi): Implement triangulation test.
const double kDoubleTolerance = 1e-9;
const Eigen::Vector3d kGPoint(0, 0, 5);
const int n_observations = 20;

namespace aslam {
void FillObservations(
    int n_observations,
    const aslam::Transformation& T_I_C,
    Aligned<std::vector, Eigen::Vector2d>::type* measurements,
    Aligned<std::vector, aslam::Transformation>::type* T_G_I) {
  CHECK_NOTNULL(measurements);
  CHECK_NOTNULL(T_G_I);

  Eigen::Vector3d position_start(-2,-2,-1);
  Eigen::Vector3d position_end(2,2,1);

  // move along line from position_start to position_end
  for(int i = 0; i < n_observations; ++i) {
    Eigen::Vector3d test_pos;
    test_pos = position_start + i / (n_observations - 1) * (position_end - position_start);
    aslam::Transformation T_G_I_current(test_pos,Eigen::Quaterniond::Identity());
    T_G_I->push_back(T_G_I_current);

    aslam::Transformation T_C_G = T_I_C.inverted() * T_G_I_current.inverted();

    Eigen::Vector3d landmark_C = T_C_G.transform(kGPoint);
    Eigen::Vector2d measurement = landmark_C.head<2>() / landmark_C[2];
    measurements->push_back(measurement);
  }
}
} // namespace aslam

class TriangulationTest : public testing::Test {
 protected:
  virtual void SetUp() {
    T_I_C_.setIdentity();
  }

  aslam::Transformation T_I_C_;
};

TEST_F(TriangulationTest, LinearTriangulateFromNViews) {
  aslam::Aligned<std::vector, Eigen::Vector2d>::type measurements;
  aslam::Aligned<std::vector, aslam::Transformation>::type T_G_I;
  Eigen::Vector3d G_point;

  aslam::FillObservations(n_observations, T_I_C_, &measurements, &T_G_I);
  aslam::LinearTriangulateFromNViews(measurements, T_G_I, T_I_C_, &G_point);

  EXPECT_TRUE(EIGEN_MATRIX_NEAR(kGPoint, G_point, kDoubleTolerance));
}


ASLAM_UNITTEST_ENTRYPOINT
