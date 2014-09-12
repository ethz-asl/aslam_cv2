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
// Types to test
///////////////////////////////////////////////
using testing::Types;
typedef Types<aslam::RadTanDistortion,
              aslam::FisheyeDistortion,
              aslam::EquidistantDistortion> Implementations;

///////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////
template <class DistortionType>
class TestDistortions : public testing::Test {
 protected:
  TestDistortions() : distortion_(DistortionType::createTestDistortion()) {};
  virtual ~TestDistortions() {};
  typename DistortionType::Ptr distortion_;
};

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
