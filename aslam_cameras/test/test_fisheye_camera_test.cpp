#include <memory>

#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/cameras/fisheye-distortion.h>
#include <aslam/cameras/pinhole-camera.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/eigen-predicates.h>

class PosegraphErrorTerms : public ::testing::Test {
 protected:
  typedef aslam::FisheyeDistortion DistortionType;
  typedef aslam::PinholeCamera CameraType;

  virtual void SetUp() {
    distortion_param_ = 0;

    fu_ = 1;
    fv_ = 1;

    res_u_ = 640;
    res_v_ = 480;

    cu_ = res_u_ / 2.0;
    cv_ = res_v_ / 2.0;
  }

  void constructCamera() {
    distortion_ = std::shared_ptr<DistortionType>(new DistortionType(
        distortion_param_));

    camera_ = std::shared_ptr<CameraType>(new CameraType(
        fu_, fv_, cu_, cv_, res_u_, res_v_, distortion_));
  }

  std::shared_ptr<CameraType> camera_;
  std::shared_ptr<DistortionType> distortion_;

  double distortion_param_;

  double fu_, fv_;
  double cu_, cv_;
  double res_u_, res_v_;
};

class FisheyeParam : public PosegraphErrorTerms,
    public ::testing::WithParamInterface<double> {
};


TEST_F(PosegraphErrorTerms, CameraTest_EuclideanToOnAxisKeypoint) {
  constructCamera();

  Eigen::Vector3d euclidean(0, 0, 1);
  Eigen::Vector2d keypoint;
  camera_->euclideanToKeypoint(euclidean, &keypoint);

  EXPECT_NEAR_EIGEN(Eigen::Vector2d(cu_, cv_), keypoint, 1e-15);
}

TEST_F(PosegraphErrorTerms, CameraTest_EuclideanToOnAxisKeypointDistorted) {
  distortion_param_ = 5;
  constructCamera();

  Eigen::Vector3d euclidean(0, 0, 1);
  Eigen::Vector2d keypoint;
  camera_->euclideanToKeypoint(euclidean, &keypoint);

  EXPECT_NEAR_EIGEN(Eigen::Vector2d(cu_, cv_), keypoint, 1e-15);
}

TEST_F(PosegraphErrorTerms, CameraTest_IsValid) {
  constructCamera();

  Eigen::Vector2d keypoint1(0, 0);
  EXPECT_TRUE(camera_->isValid(keypoint1));

  Eigen::Vector2d keypoint2(res_u_ - 1, res_v_ - 1);
  EXPECT_TRUE(camera_->isValid(keypoint2));

  Eigen::Vector2d keypoint3(cu_, cv_);
  EXPECT_TRUE(camera_->isValid(keypoint3));

  Eigen::Vector2d keypoint4(-1, 0);
  EXPECT_FALSE(camera_->isValid(keypoint4));

  Eigen::Vector2d keypoint5(-1, -1);
  EXPECT_FALSE(camera_->isValid(keypoint5));

  Eigen::Vector2d keypoint6(res_u_, res_v_);
  EXPECT_FALSE(camera_->isValid(keypoint6));
}

TEST_F(PosegraphErrorTerms, CameraTest_IsVisible) {
  constructCamera();

  EXPECT_TRUE(camera_->isEuclideanVisible(Eigen::Vector3d(0, 0, 1)));
  EXPECT_TRUE(camera_->isEuclideanVisible(Eigen::Vector3d(-cu_, -cv_, 1)));
  EXPECT_FALSE(camera_->isEuclideanVisible(Eigen::Vector3d(
      -cu_ - 1, -cv_ - 1, 1)));
  EXPECT_FALSE(camera_->isEuclideanVisible(Eigen::Vector3d(
      res_u_ - cu_, res_v_ - cv_, 1)));
}

TEST_F(PosegraphErrorTerms, CameraTest_OffAxisProjectionWithoutDistortion) {
  fu_ = 2;
  fv_ = 5;
  constructCamera();

  double kx = 80;
  double ky = 100;
  double kz = 2;

  Eigen::Vector3d euclidean(kx, ky, kz);
  Eigen::Vector2d keypoint;
  camera_->euclideanToKeypoint(euclidean, &keypoint);

  EXPECT_NEAR_EIGEN(Eigen::Vector2d(
      fu_ * (kx / kz) + cu_, fv_ * (ky / kz) + cv_), keypoint, 1e-15);
}

INSTANTIATE_TEST_CASE_P(PosegraphErrorTerms,
                        FisheyeParam,
                        ::testing::Range(0.2, 2.0, 0.2));

TEST_P(FisheyeParam, DistortAndUndistort) {
  fu_ = 80;
  fv_ = 75;
  distortion_param_ = GetParam();
  constructCamera();

  Eigen::Vector2d keypoint((100 - cu_) / fu_, (300 - cv_) / fv_);
  Eigen::Vector2d keypoint2 = keypoint;
  distortion_->distort(&keypoint2);
  distortion_->undistort(&keypoint2);

  EXPECT_NEAR_EIGEN(keypoint2, keypoint, 1e-12);
}


ASLAM_UNITTEST_ENTRYPOINT
