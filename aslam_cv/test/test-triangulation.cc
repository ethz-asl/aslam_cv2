#include <Eigen/Eigen>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/triangulation/triangulation.h>
#include <eigen-checks/gtest.h>

constexpr double kDoubleTolerance = 1e-9;
const Eigen::Vector3d kGPoint(0, 0, 5);
constexpr size_t kNumObservations = 20;

typedef aslam::Aligned<std::vector, Eigen::Vector2d>::type Vector2dList;

void fillObservations(
    size_t n_observations,
    const aslam::Transformation& T_B_C,
    Eigen::Matrix3Xd* bearing_vectors,
    aslam::Aligned<std::vector, aslam::Transformation>::type* T_W_B) {
  CHECK_NOTNULL(bearing_vectors);
  CHECK_NOTNULL(T_W_B);

  Eigen::Vector3d position_start(-2,-2,-1);
  Eigen::Vector3d position_end(2,2,1);

  Eigen::Vector3d position_step((position_end - position_start) / (n_observations - 1));

  bearing_vectors->resize(3, n_observations);
  // Move along line from position_start to position_end.
  for(size_t i = 0; i < n_observations; ++i) {
    Eigen::Vector3d test_pos(position_start + i * position_step);
    aslam::Transformation T_W_B_current(test_pos, Eigen::Quaterniond::Identity());
    T_W_B->push_back(T_W_B_current);

    aslam::Transformation T_C_W = T_B_C.inverted() * T_W_B_current.inverted();

    bearing_vectors->block<3, 1>(0, i) = T_C_W.transform(kGPoint);
  }
}

void fillObservations(
    size_t n_observations,
    const aslam::Transformation& T_B_C,
    aslam::Aligned<std::vector, Eigen::Vector2d>::type* measurements,
    aslam::Aligned<std::vector, aslam::Transformation>::type* T_W_B) {
  CHECK_NOTNULL(measurements);
  CHECK_NOTNULL(T_W_B);
  Eigen::Matrix3Xd bearing_vectors;
  fillObservations(n_observations, T_B_C, &bearing_vectors, T_W_B);
  measurements->resize(bearing_vectors.cols());
  for (int i = 0; i < bearing_vectors.cols(); ++i) {
    (*measurements)[i] = bearing_vectors.block<2, 1>(0, i) / bearing_vectors(2, i);
  }
}

template <typename MeasurementsType>
class TriangulationTest : public testing::Test {
 protected:
  virtual void SetUp() {
    T_B_C_.setIdentity();
  }

  void fillMeasurements(
      size_t n_observations) {
    fillObservations(n_observations, T_B_C_, &measurements_, &T_W_B_);
  }

  aslam::TriangulationResult triangulate(Eigen::Vector3d* result) const;

  void setNMeasurements(const size_t n) {
    C_bearing_measurements_.resize(3, n);
    T_W_B_.resize(n);
  }

  void setLandmark(const Eigen::Vector3d& p_W_L) {
    p_W_L_ = p_W_L;
  }

  void inferMeasurements() {
    for (size_t i = 0; i < T_W_B_.size(); ++i) {
      // Ignoring IMU to camera transformation (set to identity in SetUp()).
      C_bearing_measurements_.block<3, 1>(0, i) =
          T_W_B_[i].inverted().transform(p_W_L_);
    }
    setMeasurements(C_bearing_measurements_);
  }

  void setMeasurements(const Eigen::Matrix3Xd& measurements);

  void expectSuccess() {
    Eigen::Vector3d p_W_L_estimate;
    EXPECT_TRUE(triangulate(&p_W_L_estimate).wasTriangulationSuccessful());
    EXPECT_TRUE(EIGEN_MATRIX_NEAR(p_W_L_, p_W_L_estimate, kDoubleTolerance));
  }

  void expectFailue() {
    Eigen::Vector3d p_W_L_estimate;
    EXPECT_FALSE(triangulate(&p_W_L_estimate).wasTriangulationSuccessful());
  }

  aslam::Transformation T_B_C_;
  MeasurementsType measurements_;
  Eigen::Matrix3Xd C_bearing_measurements_;
  aslam::Aligned<std::vector, aslam::Transformation>::type T_W_B_;
  Eigen::Vector3d p_W_L_;
};

template<>
aslam::TriangulationResult TriangulationTest<Vector2dList>::triangulate(
    Eigen::Vector3d* result) const {
  return aslam::linearTriangulateFromNViews(measurements_, T_W_B_, T_B_C_, result);
}

template<>
aslam::TriangulationResult TriangulationTest<Eigen::Matrix3Xd>::triangulate(
    Eigen::Vector3d* result) const {
  Eigen::Matrix3Xd G_measurements(3, measurements_.cols()),
      p_G_C(3, measurements_.cols());
  for (int i = 0; i < measurements_.cols(); ++i) {
    G_measurements.block<3, 1>(0, i) = T_W_B_[i].getRotationMatrix() *
        T_B_C_.getRotationMatrix() * measurements_.block<3, 1>(0, i);
    p_G_C.block<3, 1>(0, i) = T_W_B_[i].getPosition();
  }
  return aslam::linearTriangulateFromNViews(G_measurements, p_G_C, result);
}

template <>
void TriangulationTest<Vector2dList>::setMeasurements(const Eigen::Matrix3Xd& measurements) {
  measurements_.resize(measurements.cols());
  for (int i = 0; i < measurements.cols(); ++i) {
    measurements_[i] = measurements.block<2, 1>(0, i) / measurements(2, i);
  }
}

template <>
void TriangulationTest<Eigen::Matrix3Xd>::setMeasurements(const Eigen::Matrix3Xd& measurements) {
  measurements_ = measurements;
}

typedef ::testing::Types<Vector2dList, Eigen::Matrix3Xd>
MyTypes;
TYPED_TEST_CASE(TriangulationTest, MyTypes);

TYPED_TEST(TriangulationTest, LinearTriangulateFromNViews) {
  this->setLandmark(kGPoint);
  this->fillMeasurements(kNumObservations);
  this->expectSuccess();
}

class TriangulationMultiviewTest : public TriangulationTest<Vector2dList> {};

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

TYPED_TEST(TriangulationTest, RandomPoses) {
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

TYPED_TEST(TriangulationTest, TwoParallelRays) {
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

TYPED_TEST(TriangulationTest, TwoNearParallelRays) {
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

TYPED_TEST(TriangulationTest, CombinedParallelAndGoodRays) {
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
