#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <aslam/common/eigen-predicates.h>
#include <aslam/common/memory.h>
#include <aslam/cameras/distortion.h>
#include <aslam/cameras/distortion-fisheye.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/distortion-equidistant.h>

///////////////////////////////////////////////
// Distortion factories
///////////////////////////////////////////////
template <class T>
aslam::Distortion::Ptr CreateDistortion();

template <>
aslam::Distortion::Ptr CreateDistortion<aslam::RadTanDistortion>() {
  Eigen::VectorXd params(4);
  params << 0.8, 0.01, 0.2, 0.003;
  return aslam::aligned_shared<aslam::RadTanDistortion>(params);
}

template <>
aslam::Distortion::Ptr CreateDistortion<aslam::FisheyeDistortion>() {
  Eigen::VectorXd params(1);
  params[0] = 1.3;
  return aslam::aligned_shared<aslam::FisheyeDistortion>(params);
}

template <>
aslam::Distortion::Ptr CreateDistortion<aslam::EquidistantDistortion>() {
  Eigen::VectorXd params(4);
  params << 0.2, 0.1, 0.2, 0.003;
  return aslam::aligned_shared<aslam::EquidistantDistortion>(params);
}

///////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////
template <class T>
class TestDistortions : public testing::Test {
 protected:
  TestDistortions() : distortion_(CreateDistortion<T>()) {};
  virtual ~TestDistortions() {};
  aslam::Distortion::Ptr distortion_;
};

using testing::Types;
typedef Types<aslam::RadTanDistortion,
              aslam::FisheyeDistortion,
              aslam::EquidistantDistortion> Implementations;
TYPED_TEST_CASE(TestDistortions, Implementations);


///////////////////////////////////////////////
// Test cases
///////////////////////////////////////////////
TYPED_TEST(TestDistortions, DistortAndUndistortUsingInternalParameters) {
  Eigen::Vector2d keypoint(0.5, -0.4);
  Eigen::Vector2d keypoint2 = keypoint;
  this->distortion_->distort(&keypoint2);
  this->distortion_->undistort(&keypoint2);

  EXPECT_NEAR_EIGEN(keypoint2, keypoint, 1e-12);
}

TYPED_TEST(TestDistortions, DistortAndUndistortUsingExternalParameters) {
  Eigen::Vector2d keypoint(0.8, -0.2);
  Eigen::Vector2d keypoint2 = keypoint;

  // Set new parameters.
  Eigen::VectorXd dist_coeff = this->distortion_->getParameters();
  this->distortion_->setParameters(dist_coeff / 2.0);

  this->distortion_->distortUsingExternalCoefficients(dist_coeff, &keypoint2, nullptr);
  this->distortion_->undistortUsingExternalCoefficients(dist_coeff, &keypoint2);

  EXPECT_NEAR_EIGEN(keypoint2, keypoint, 1e-12);
}

TYPED_TEST(TestDistortions, DistortAndUndistortImageCenter) {
  Eigen::Vector2d keypoint(0.0, 0.0);

  Eigen::Vector2d keypoint2 = keypoint;
  this->distortion_->undistort(&keypoint2);
  this->distortion_->distort(&keypoint2);
  EXPECT_NEAR_EIGEN(keypoint2, keypoint, 1e-12);

  keypoint2 = keypoint;
  this->distortion_->distort(&keypoint2);
  this->distortion_->undistort(&keypoint2);
  EXPECT_NEAR_EIGEN(keypoint2, keypoint, 1e-12);
}
