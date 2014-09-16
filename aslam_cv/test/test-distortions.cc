#include <Eigen/Core>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <aslam/common/eigen-predicates.h>
#include <aslam/common/memory.h>
#include <aslam/cameras/distortion.h>
#include <aslam/cameras/distortion-fisheye.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/cameras/distortion-equidistant.h>
#include <aslam/common/numdiff-jacobian-tester.h>

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


/// Wrapper that brings the distortion function to the form needed by the differentiator.
struct Point3dJacobianFunctor : public aslam::common::NumDiffFunctor<2, 2> {

  Point3dJacobianFunctor(aslam::Distortion::ConstPtr distortion, const Eigen::VectorXd& dist_coeffs)
      : distortion_(distortion),
        dist_coeffs_(dist_coeffs){
    CHECK(distortion);
  };

  virtual ~Point3dJacobianFunctor() {};

  virtual void functional(const typename NumDiffFunctor::InputType& x,
                          typename NumDiffFunctor::ValueType& fvec,
                          typename NumDiffFunctor::JacobianType* Jout) const {
    Eigen::Vector2d out_keypoint = x;

    // Get function value and Jacobian
    if(Jout)
      distortion_->distortUsingExternalCoefficients(dist_coeffs_, &out_keypoint, Jout);
    else
      distortion_->distortUsingExternalCoefficients(dist_coeffs_, &out_keypoint, nullptr);
    fvec = out_keypoint;
  };

  aslam::Distortion::ConstPtr distortion_;
  const Eigen::VectorXd dist_coeffs_;
};

TYPED_TEST(TestDistortions, JacobianWrtKeypoint) {
  Eigen::Vector2d keypoint(0.3, -0.2);
  Eigen::VectorXd dist_coeffs = this->distortion_->getParameters();

  TEST_JACOBIAN_FINITE_DIFFERENCE(Point3dJacobianFunctor, keypoint, 1e-6, 1e-4,
                                  this->distortion_, dist_coeffs);
}

/// Wrapper that brings the distortion function to the form needed by the differentiator.
template<int numDistortion>
struct DistortionJacobianFunctor : public aslam::common::NumDiffFunctor<2, numDistortion> {

  DistortionJacobianFunctor(aslam::Distortion::ConstPtr distortion, const Eigen::Vector2d& keypoint)
      : distortion_(distortion),
        keypoint_(keypoint) {
    CHECK(distortion);
  };

  virtual ~DistortionJacobianFunctor() {};

  virtual void functional(
      const typename aslam::common::NumDiffFunctor<2, numDistortion>::InputType& x,
      typename aslam::common::NumDiffFunctor<2, numDistortion>::ValueType& fvec,
      typename aslam::common::NumDiffFunctor<2, numDistortion>::JacobianType* Jout) const {

    Eigen::Vector2d out_keypoint = keypoint_;
    Eigen::Matrix<double, 2, Eigen::Dynamic> JoutDynamic;
    JoutDynamic.setZero();

    // Get value
    distortion_->distortUsingExternalCoefficients(x, &out_keypoint, nullptr);
    fvec = out_keypoint;

    // Get Jacobian wrt distortion coeffs.
    if(Jout) {
      distortion_->distortParameterJacobian(x, keypoint_, &JoutDynamic);
      (*Jout) = JoutDynamic;
    }

  };

  aslam::Distortion::ConstPtr distortion_;
  const Eigen::Vector2d keypoint_;
};

TYPED_TEST(TestDistortions, JacobianWrtDistortion) {
  Eigen::Vector2d keypoint(0.3, -0.2);
  Eigen::VectorXd dist_coeffs = this->distortion_->getParameters();

  TEST_JACOBIAN_FINITE_DIFFERENCE(DistortionJacobianFunctor<TypeParam::parameterCount()>,
                                  dist_coeffs, 1e-6, 1e-4, this->distortion_, keypoint);
}
