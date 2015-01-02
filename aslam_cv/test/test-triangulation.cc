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

TEST_F(TriangulationTest, linearTriangulateFromNViewsMultiCam) {
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

TEST(LinearTriangulateFromNViews, RandomPoses) {
  constexpr size_t kNumCameraPoses = 5;

  // Create a landmark.
  const double depth = 5.0;
  Eigen::Vector3d W_landmark(1.0, 1.0, depth);

  // Generate some random camera poses and project the landmark into it.
  constexpr double kRandomTranslationNorm = 0.1;
  constexpr double kRandomRotationAngleRad = 20 / 180.0 * M_PI;

  aslam::TransformationVector T_W_C(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector2d>::type keypoint_measurements(kNumCameraPoses);

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    T_W_C[pose_idx].setRandom(kRandomTranslationNorm, kRandomRotationAngleRad);
    const Eigen::Vector3d Ck_landmark = T_W_C[pose_idx].inverted().transform(W_landmark);
    keypoint_measurements[pose_idx] = Ck_landmark.head<2>() / Ck_landmark[2];
  }

  // Triangulate.
  Eigen::Vector3d W_landmark_triangulated;
  aslam::TriangulationResult triangulation_result = aslam::linearTriangulateFromNViews(
                                                      keypoint_measurements,
                                                      T_W_C,
                                                      aslam::Transformation(),
                                                      &W_landmark_triangulated);

  EXPECT_TRUE(triangulation_result.wasTriangulationSuccessful());
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(W_landmark, W_landmark_triangulated, 1e-10));
}

TEST(LinearTriangulateFromNViews, TwoParallelRays) {
  constexpr size_t kNumCameraPoses = 3;
  aslam::TransformationVector T_W_C(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector3d>::type Ck_landmark(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector2d>::type keypoint_measurements(kNumCameraPoses);

  // Create a landmark.
  const double depth = 5.0;
  Eigen::Vector3d W_landmark(1.0, 1.0, depth);

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    T_W_C[pose_idx].setIdentity();
    const Eigen::Vector3d Ck_landmark = T_W_C[pose_idx].inverted().transform(W_landmark);
    keypoint_measurements[pose_idx] = Ck_landmark.head<2>() / Ck_landmark[2];
  }

  // Triangulate.
  Eigen::Vector3d W_landmark_triangulated;
  W_landmark_triangulated.setZero();
  aslam::TriangulationResult triangulation_result = aslam::linearTriangulateFromNViews(keypoint_measurements, T_W_C,
                                                    aslam::Transformation(),
                                                    &W_landmark_triangulated);
  EXPECT_FALSE(triangulation_result.wasTriangulationSuccessful());
}

TEST(LinearTriangulateFromNViews, TwoNearParallelRays) {
  constexpr size_t kNumCameraPoses = 2;
  aslam::TransformationVector T_W_C(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector3d>::type Ck_landmark(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector2d>::type keypoint_measurements(kNumCameraPoses);

  // Create a landmark.
  const double depth = 5.0;
  Eigen::Vector3d W_landmark(1.0, 1.0, depth);

  // Create near parallel rays.
  aslam::Transformation noise;
  const double disparity_angle_rad = 0.1 / 180.0 * M_PI;
  const double camrea_shift = std::atan(disparity_angle_rad) * depth;
  noise.setRandom(camrea_shift, 0.0);
  T_W_C[1] = T_W_C[1] * noise;

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    const Eigen::Vector3d Ck_landmark = T_W_C[pose_idx].inverted().transform(W_landmark);
    keypoint_measurements[pose_idx] = Ck_landmark.head<2>() / Ck_landmark[2];
  }

  // Triangulate.
  Eigen::Vector3d W_landmark_triangulated;
  W_landmark_triangulated.setZero();
  aslam::TriangulationResult triangulation_result = aslam::linearTriangulateFromNViews(keypoint_measurements, T_W_C,
                                                    aslam::Transformation(),
                                                    &W_landmark_triangulated);
  EXPECT_FALSE(triangulation_result.wasTriangulationSuccessful());
}

TEST(LinearTriangulateFromNViews, CombinedParallelAndGoodRays) {
  constexpr size_t kNumCameraPoses = 3;
  aslam::TransformationVector T_W_C(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector3d>::type Ck_landmark(kNumCameraPoses);
  aslam::Aligned<std::vector, Eigen::Vector2d>::type keypoint_measurements(kNumCameraPoses);
  T_W_C[2].setRandom(0.5, 0.2);

  // Create near parallel rays.
  aslam::Transformation noise;
  noise.setRandom(0.01, 0.1);
  T_W_C[1] = T_W_C[1] * noise;

  // Create a landmark.
  const double depth = 5.0;
  Eigen::Vector3d W_landmark(1.0, 1.0, depth);

  for (size_t pose_idx = 0; pose_idx < kNumCameraPoses; ++pose_idx) {
    const Eigen::Vector3d Ck_landmark = T_W_C[pose_idx].inverted().transform(W_landmark);
    keypoint_measurements[pose_idx] = Ck_landmark.head<2>() / Ck_landmark[2];
  }

  // Triangulate.
  Eigen::Vector3d W_landmark_triangulated;
  W_landmark_triangulated.setZero();
  aslam::TriangulationResult triangulation_result = aslam::linearTriangulateFromNViews(keypoint_measurements, T_W_C,
                                                    aslam::Transformation(),
                                                    &W_landmark_triangulated);
  EXPECT_TRUE(triangulation_result.wasTriangulationSuccessful());
  EXPECT_TRUE(EIGEN_MATRIX_NEAR(W_landmark, W_landmark_triangulated, 1e-10));
}

ASLAM_UNITTEST_ENTRYPOINT
