#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/test/triangulation-fixture.h>

constexpr size_t kNumObservations = 20;


TYPED_TEST(TriangulationFixture, LinearTriangulateFromNViews) {
  this->setLandmark(kGPoint);
  this->fillMeasurements(kNumObservations);
  this->expectSuccess();
}

class TriangulationMultiviewTest : public TriangulationFixture<Vector2dList> {};

TEST_F(TriangulationMultiviewTest, linearTriangulateFromNViewsMultiCam) {
  aslam::Aligned<std::vector, Eigen::Vector2d>::type measurements;
  aslam::Aligned<std::vector, aslam::Transformation>::type T_W_B;
  aslam::Aligned<std::vector, aslam::Transformation>::type T_B_C;
  std::vector<int> measurement_camera_indices;
  Eigen::Vector3d W_point;

  // To make the test simple to write, create 2 cameras and fill observations
  // for both, then simply append the vectors together.
  const unsigned int num_cameras = 2;
  T_B_C.resize(num_cameras);

  for (size_t i = 0; i < T_B_C.size(); ++i) {
    T_B_C[i].setRandom(0.2, 0.1);
    aslam::Aligned<std::vector, Eigen::Vector2d>::type cam_measurements;
    aslam::Aligned<std::vector, aslam::Transformation>::type cam_T_W_B;
    fillObservations(kNumObservations, T_B_C[i], &cam_measurements, &cam_T_W_B);

    // Append to the end of the vectors.
    measurements.insert(measurements.end(), cam_measurements.begin(),
                        cam_measurements.end());
    T_W_B.insert(T_W_B.end(), cam_T_W_B.begin(), cam_T_W_B.end());

    // Fill in the correct size of camera indices also.
    measurement_camera_indices.resize(measurements.size(), i);
  }

  aslam::linearTriangulateFromNViewsMultiCam(measurements,
      measurement_camera_indices, T_W_B, T_B_C, &W_point);

  EXPECT_TRUE(EIGEN_MATRIX_NEAR(kGPoint, W_point, kDoubleTolerance));
}

TYPED_TEST(TriangulationFixture, RandomPoses) {
  constexpr size_t kNumCameraPoses = 5;
  this->setNMeasurements(kNumCameraPoses);

  // Create a landmark.
  const double depth = 5.0;
  this->setLandmark(Eigen::Vector3d(1.0, 1.0, depth));

  // Generate some random camera poses and project the landmark into it.
  constexpr double kRandomTranslationNorm = 0.1;
  constexpr double kRandomRotationAngleRad = 20 / 180.0 * M_PI;

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    this->T_W_B_[pose_idx].setRandom(kRandomTranslationNorm, kRandomRotationAngleRad);
  }

  this->inferMeasurements();
  this->expectSuccess();
}

TYPED_TEST(TriangulationFixture, TwoParallelRays) {
  constexpr size_t kNumCameraPoses = 3;
  this->setNMeasurements(kNumCameraPoses);

  // Create a landmark.
  const double depth = 5.0;
  this->setLandmark(Eigen::Vector3d(1.0, 1.0, depth));

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    this->T_W_B_[pose_idx].setIdentity();
  }

  this->inferMeasurements();
  this->expectFailue();
}

TYPED_TEST(TriangulationFixture, TwoNearParallelRays) {
  constexpr size_t kNumCameraPoses = 2;
  this->setNMeasurements(kNumCameraPoses);

  // Create a landmark.
  const double depth = 5.0;
  this->setLandmark(Eigen::Vector3d(1.0, 1.0, depth));

  // Create near parallel rays.
  aslam::Transformation noise;
  const double disparity_angle_rad = 0.1 / 180.0 * M_PI;
  const double camrea_shift = std::atan(disparity_angle_rad) * depth;
  noise.setRandom(camrea_shift, 0.0);
  this->T_W_B_[1] = this->T_W_B_[1] * noise;

  this->inferMeasurements();
  this->expectFailue();
}

TYPED_TEST(TriangulationFixture, CombinedParallelAndGoodRays) {
  constexpr size_t kNumCameraPoses = 3;
  this->setNMeasurements(kNumCameraPoses);

  // Create near parallel rays.
  aslam::Transformation noise;
  noise.setRandom(0.01, 0.1);
  this->T_W_B_[1] = this->T_W_B_[1] * noise;

  this->T_W_B_[2].setRandom(0.5, 0.2);

  // Create a landmark.
  const double depth = 5.0;
  this->setLandmark(Eigen::Vector3d(1.0, 1.0, depth));

  this->inferMeasurements();
  this->expectSuccess();
}

ASLAM_UNITTEST_ENTRYPOINT
