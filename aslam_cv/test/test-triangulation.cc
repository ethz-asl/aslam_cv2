#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/pose-types.h>
#include <aslam/triangulation/triangulation-toolbox.h>
#include <eigen-checks/gtest.h>

// TODO(burrimi): Implement triangulation test.
const double kDoubleTolerance = 1e-9;
const Eigen::Vector3d kGPoint(0, 0, 5);
namespace aslam {
void FillObservations(
    int n_observations,
    Aligned<std::vector, Eigen::Vector2d>::type* measurements,
    Aligned<std::vector, aslam::Transformation>::type* T_G_I) {
  CHECK_NOTNULL(measurements);
  CHECK_NOTNULL(T_G_I);
  // TODO(burrimi): FillObservations.
}

class TriangulationTest : public testing::Test {
 protected:
  virtual void SetUp() {
    T_I_C_.setIdentity();
  }

  aslam::Transformation T_I_C_;
};

TEST_F(TriangulationTest, LinearTriangulateFromNViews) {
  // https://code.google.com/p/googletest/wiki/Primer
  // https://github.com/ethz-asl/eigen_checks#gtest-1

  // TODO(burrimi): Call FillObservations.
  // TODO(burrimi): Call triangulation method.
  // TODO(burrimi): Expect equality.
  // TODO(burrimi): Repeat for N different values.
}
