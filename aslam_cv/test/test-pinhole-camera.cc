#include <memory>

#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/eigen-predicates.h>

class PinholeCameraTest : public ::testing::Test {
 protected:
  typedef aslam::RadTanDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;

  virtual void SetUp() {
    //random intrinsics
    fu_ = 300,
    fv_ = 320,
    cu_ = 340,
    cv_ = 220;
    res_u_ = 640,
    res_v_ = 480;

    //random radtan distortion parameters
    distortion_param_.resize(4);
    distortion_param_ << 0.8, 0.01, 0.2, 0.02;
  }

  void constructCamera() {
    distortion_.reset(new DistortionType(distortion_param_));
    camera_.reset(new CameraType(fu_, fv_, cu_, cv_, res_u_, res_v_, distortion_));
  }

  std::shared_ptr<CameraType> camera_;
  std::shared_ptr<DistortionType> distortion_;

  double fu_, fv_;
  double cu_, cv_;
  double res_u_, res_v_;
  Eigen::VectorXd distortion_param_;
};

TEST_F(PinholeCameraTest, CameraTest_EuclideanToOnAxisKeypoint) {
  constructCamera();

  Eigen::Vector3d euclidean(0, 0, 1);
  Eigen::Vector2d keypoint;
  camera_->project3(euclidean, &keypoint);

  Eigen::Vector2d image_center(camera_->cu(), camera_->cv());
  EXPECT_NEAR_EIGEN(image_center, keypoint, 1e-15);
}

TEST_F(PinholeCameraTest, CameraTest_isVisible) {
  constructCamera();

  Eigen::Vector2d keypoint1(0, 0);
  EXPECT_TRUE(camera_->isKeypointVisible(keypoint1)) << "Keypoint1: " << keypoint1;

  Eigen::Vector2d keypoint2(res_u_ - 1, res_v_ - 1);
  EXPECT_TRUE(camera_->isKeypointVisible(keypoint2)) << "Keypoint2: " << keypoint2;

  Eigen::Vector2d keypoint3(camera_->cu(), camera_->cv());
  EXPECT_TRUE(camera_->isKeypointVisible(keypoint3)) << "Keypoint3: " << keypoint3;

  Eigen::Vector2d keypoint4(-1, 0);
  EXPECT_FALSE(camera_->isKeypointVisible(keypoint4)) << "Keypoint4: " << keypoint4;

  Eigen::Vector2d keypoint5(-1, -1);
  EXPECT_FALSE(camera_->isKeypointVisible(keypoint5)) << "Keypoint5: " << keypoint5;

  Eigen::Vector2d keypoint6(res_u_, res_v_);
  EXPECT_FALSE(camera_->isKeypointVisible(keypoint6)) << "Keypoint6: " << keypoint6;
}

TEST_F(PinholeCameraTest, CameraTest_IsVisible) {
  constructCamera();

  EXPECT_TRUE(camera_->isProjectable3(Eigen::Vector3d(0, 0, 1)));       // Center.
  EXPECT_FALSE(camera_->isProjectable3(Eigen::Vector3d(5, -5, 1)));     // In front of cam.
  EXPECT_FALSE(camera_->isProjectable3(Eigen::Vector3d(5000, -5, 1)));  // In front of cam, outside range.
  EXPECT_FALSE(camera_->isProjectable3(Eigen::Vector3d(-10, -10, -1))); // Behind cam.
  EXPECT_FALSE(camera_->isProjectable3(Eigen::Vector3d(0, 0, -1)));     // Behind, center.
}

TEST_F(PinholeCameraTest, CameraTest_OffAxisProjectionWithoutDistortion) {
  constructCamera();

  //disable distortion
  distortion_->setParameters(Eigen::Vector4d::Zero());

  double kx = 1;
  double ky = 1.2;
  double kz = 10;

  Eigen::Vector3d euclidean(kx, ky, kz);
  Eigen::Vector2d keypoint;
  camera_->project3(euclidean, &keypoint);

  EXPECT_NEAR_EIGEN(Eigen::Vector2d(fu_*(kx/kz)+cu_, fv_*(ky/kz)+cv_), keypoint, 1e-15);
}

TEST_F(PinholeCameraTest, CameraTest_ProjectionState) {
  constructCamera();
  Eigen::Vector2d keypoint;
  aslam::ProjectionState ret;

  // In front of cam -> visible.
  ret = camera_->project3(Eigen::Vector3d(0, 0, 10), &keypoint);
  EXPECT_EQ(ret.getDetailedStatus(), aslam::ProjectionState::Status_t::KEYPOINT_VISIBLE);
  EXPECT_TRUE(static_cast<bool>(ret));

  // Behind cam -> not visible.
  ret = camera_->project3(Eigen::Vector3d(0, 0, -10), &keypoint);
  EXPECT_EQ(ret.getDetailedStatus(), aslam::ProjectionState::Status_t::POINT_BEHIND_CAMERA);
  EXPECT_FALSE(static_cast<bool>(ret));

  // In front of cam, but outside of image box. -> not visible.
  ret = camera_->project3(Eigen::Vector3d(50, 50, 10), &keypoint);
  EXPECT_EQ(ret.getDetailedStatus(), aslam::ProjectionState::Status_t::KEYPOINT_OUTSIDE_IMAGE_BOX);
  EXPECT_FALSE(static_cast<bool>(ret));

  // Invalid projection (z<min_z) -> not visible/projectable.
  ret = camera_->project3(Eigen::Vector3d(1, 1.2, 1e-15), &keypoint);
  EXPECT_EQ(ret.getDetailedStatus(), aslam::ProjectionState::Status_t::PROJECTION_INVALID);
  EXPECT_FALSE(static_cast<bool>(ret));
}

TEST(CameraComparison, TestEquality) {
  using namespace aslam;
  Eigen::VectorXd dvec(4);

  dvec << 0.5, 0.3, 0.2, 0.01;
  Distortion::Ptr distortion_A = std::make_shared<RadTanDistortion>(dvec);
  Camera::Ptr camera_A = std::make_shared<PinholeCamera>(240, 480, 100, 200, 500, 500, distortion_A);

  dvec << 0.0, 0.3, 0.2, 0.01;
  Distortion::Ptr distortion_B = std::make_shared<RadTanDistortion>(dvec);
  Camera::Ptr camera_B = std::make_shared<PinholeCamera>(240, 480, 100, 200, 500, 500, distortion_B);

  dvec << 0.5, 0.3, 0.2, 0.01;
  Distortion::Ptr distortion_C = std::make_shared<RadTanDistortion>(dvec);
  Camera::Ptr camera_C = std::make_shared<PinholeCamera>(11111, 480, 100, 200, 500, 500, distortion_C);

  EXPECT_TRUE( *camera_A == *camera_A );  // Same camera, should be equal.
  EXPECT_FALSE( *camera_A == *camera_B ); // Different distortion, should be different.
  EXPECT_FALSE( *camera_A == *camera_C ); // Different intrinsics, should be different.
}
