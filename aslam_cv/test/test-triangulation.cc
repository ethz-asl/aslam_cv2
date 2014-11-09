#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/triangulation/triangulation.h>
#include <eigen-checks/gtest.h>

const double kDoubleTolerance = 1e-9;
const Eigen::Vector3d kGPoint(0, 0, 5);
const size_t kNumObservations = 20;

void fillObservations(
    size_t n_observations,
    const aslam::Transformation& T_B_C,
    aslam::Aligned<std::vector, Eigen::Vector2d>::type* measurements,
    aslam::Aligned<std::vector, aslam::Transformation>::type* T_W_B) {
  CHECK_NOTNULL(measurements);
  CHECK_NOTNULL(T_W_B);

  Eigen::Vector3d position_start(-2,-2,-1);
  Eigen::Vector3d position_end(2,2,1);

  Eigen::Vector3d position_step((position_end - position_start) / (n_observations - 1));

  // Move along line from position_start to position_end.
  for(size_t i = 0; i < n_observations; ++i) {
    Eigen::Vector3d test_pos(position_start + i * position_step);
    aslam::Transformation T_W_B_current(test_pos, Eigen::Quaterniond::Identity());
    T_W_B->push_back(T_W_B_current);

    aslam::Transformation T_C_W = T_B_C.inverted() * T_W_B_current.inverted();

    Eigen::Vector3d C_landmark = T_C_W.transform(kGPoint);
    Eigen::Vector2d measurement = C_landmark.head<2>() / C_landmark[2];
    measurements->push_back(measurement);
  }
}

class TriangulationTest : public testing::Test {
 protected:
  virtual void SetUp() {
    T_B_C_.setIdentity();
  }

  aslam::Transformation T_B_C_;
};

TEST_F(TriangulationTest, LinearTriangulateFromNViews) {
  aslam::Aligned<std::vector, Eigen::Vector2d>::type measurements;
  aslam::Aligned<std::vector, aslam::Transformation>::type T_W_B;
  Eigen::Vector3d W_point;

  fillObservations(kNumObservations, T_B_C_, &measurements, &T_W_B);
  aslam::linearTriangulateFromNViews(measurements, T_W_B, T_B_C_, &W_point);

  EXPECT_TRUE(EIGEN_MATRIX_NEAR(kGPoint, W_point, kDoubleTolerance));
}

TEST(GeometricVisionRotatedTwoViewParamTest, LinearTriangulateFromNViewsRandomPoses) {
  constexpr size_t kNumCameraPoses = 5;

  // Create a landmark.
  const double depth = 5.0;
  Eigen::Vector3d W_landmark(1.0, 1.0, depth);

  // Generate some random camera poses and project the landmark into it.
  constexpr double kRandomTranslationNorm = 0.1;
  constexpr double kRandomRotationAngleRad = 20 / 180.0 * M_PI;

  aslam::TransformationVector T_W_C(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector3d>::type Ck_landmark(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector2d>::type keypoint_measurements(kNumCameraPoses);

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    T_W_C[pose_idx].setRandom(kRandomTranslationNorm, kRandomRotationAngleRad);
    Ck_landmark[pose_idx] = T_W_C[pose_idx].inverted().transform(W_landmark);
    keypoint_measurements[pose_idx] = Ck_landmark[pose_idx].head<2>() / Ck_landmark[pose_idx][2];
  }

  // Triangulate.
  Eigen::Vector3d W_landmark_triangulated;
  bool success = aslam::linearTriangulateFromNViews(keypoint_measurements, T_W_C,
                                                    aslam::Transformation(),
                                                    &W_landmark_triangulated);

  EXPECT_TRUE(success);
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(W_landmark, W_landmark_triangulated, 1e-10));
}

ASLAM_UNITTEST_ENTRYPOINT
