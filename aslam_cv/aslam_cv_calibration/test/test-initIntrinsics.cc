#include <Eigen/Core>
#include <Eigen/StdVector>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/calibration/camera-initializer.h>
#include <aslam/calibration/target-aprilgrid.h>
#include <aslam/calibration/target-observation.h>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/pose-types.h>

///////////////////////////////////////////////
// Types to test
///////////////////////////////////////////////
template<typename Camera, typename Distortion>
struct CameraDistortion {
  typedef Camera CameraType;
  typedef Distortion DistortionType;
};

using testing::Types;
typedef Types<
    //CameraDistortion<aslam::PinholeCamera,aslam::FisheyeDistortion>,
    //CameraDistortion<aslam::UnifiedProjectionCamera,aslam::FisheyeDistortion>,
    CameraDistortion<aslam::PinholeCamera,aslam::EquidistantDistortion>,
    //CameraDistortion<aslam::UnifiedProjectionCamera,aslam::EquidistantDistortion>,
    CameraDistortion<aslam::PinholeCamera,aslam::RadTanDistortion>
    //CameraDistortion<aslam::UnifiedProjectionCamera,aslam::RadTanDistortion>,
    //CameraDistortion<aslam::PinholeCamera,aslam::NullDistortion>,
    //CameraDistortion<aslam::UnifiedProjectionCamera,aslam::NullDistortion>
    >
    Implementations;

///////////////////////////////////////////////
// Test fixture
///////////////////////////////////////////////
template <typename CameraDistortion>
class TestCameras : public testing::Test {
 public:
  typedef typename CameraDistortion::CameraType CameraType;
  typedef typename CameraDistortion::DistortionType DistortionType;
 protected:
  TestCameras() : camera_(CameraType::template createIntrinsicsTestCamera<DistortionType>()) {};
  virtual ~TestCameras() {};

  bool runInitialization(Eigen::VectorXd& intrinsics_vector,
                         std::vector<aslam::calibration::TargetObservation::Ptr> &observations)
    const {
          return aslam::calibration::initializeCameraIntrinsics<CameraType,DistortionType>(
              intrinsics_vector,
              observations);
  }

  typename CameraType::Ptr camera_;

};

TYPED_TEST_CASE(TestCameras, Implementations);

TYPED_TEST(TestCameras, InitializeIntrinsics) {


  // Create a target April grid.
  aslam::calibration::TargetAprilGrid::TargetConfiguration aprilgrid_config;
  aslam::calibration::TargetAprilGrid::Ptr aprilgrid(
      new aslam::calibration::TargetAprilGrid(aprilgrid_config));

  // Get all target grid points in the target frame.
  Eigen::Matrix3Xd points_target_frame = aprilgrid->points();
//  std::cout << "Target points:\n";
//  for (int i = 0; i < 5; ++i) {
//    std::cout << points_target_frame.col(i)(0) << "  " << points_target_frame.col(i)(1) << "  " << points_target_frame.col(i)(2) << "\n";
//  }

  // Create random camera poses and corresponding transformations.
  const double kDegToRad = M_PI / 180.0;
  std::vector<aslam::Transformation> T_CT;
  size_t n_cam_poses = 6;
  T_CT.reserve(n_cam_poses * n_cam_poses);
  for (size_t dist = 0; dist < n_cam_poses; ++dist) {
    for (size_t ang = 0; ang < n_cam_poses; ++ang) {
      aslam::Transformation T;
      T_CT.emplace_back(T.setRandom(0.1, 1.0* kDegToRad));
      //std::cout << "Dist= " << 0.1+0.2*dist << "m\tAng= " << 1+5*ang << "deg \n";
      //std::cout << "T_CT matrix:\n" << T.setRandom(0.1* dist, 1.0* kDegToRad) << "\n\n";
    }
  }
  //std::cout << "T_CT[0] matrix:\n" << T_CT.at(0) << "\n\n";

  // Create test projection camera (NullDistortion, given focal length and principal point).
  ASSERT_NE(this->camera_, nullptr) << "Test camera cannot be created.";

  // Create simulated target observation (one image).
  std::vector<aslam::calibration::TargetObservation::Ptr> target_observations;
  target_observations.reserve(1);

  Eigen::Matrix3Xd points_camera_frame; //(3, 144);
  Eigen::Matrix2Xd image_points; //(2, points_camera_frame.cols());
  Eigen::VectorXd calc_intrinsics;

  for (size_t i = 0; i < T_CT.size(); ++i) {
    // Transform all points in the target frame into the camera frame.
    aslam::Transformation T = T_CT.at(i);
    points_camera_frame = T.transformVectorized(points_target_frame);
//    std::cout << "Camera points:\n";
//    for (int i = 0; i < 5; ++i) {
//      std::cout << points_camera_frame.col(i)(0) << "  " << points_target_frame.col(i)(1) << "  " << points_target_frame.col(i)(2) << "\n";
//    }

    // Project points into the image plane.
    image_points(2, points_camera_frame.cols());
    std::vector<aslam::ProjectionResult> results;
    this->camera_->project3Vectorized(points_camera_frame, &image_points, &results);
//    std::cout << "Image points:\n";
//    for (int i = 0; i < 5; ++i) {
//      std::cout << image_points.col(i)(0) << "  " << points_target_frame.col(i)(1) << "\n";
//    }
    aslam::calibration::TargetObservation::Ptr target_observation (
        new aslam::calibration::TargetObservation(aprilgrid,
                                                  this->camera_->imageHeight(),
                                                  this->camera_->imageWidth(),
                                                  aprilgrid->setCornerIds(),
                                                  image_points));
    target_observations.emplace_back(target_observation);

    // Initialize the intrinsics using this observation.
    bool intrinsics_success = this->runInitialization(calc_intrinsics, target_observations);
//    bool intrinsics_success = aslam::calibration::initializeCameraIntrinsics<this->getCameraType(), aslam::RadTanDistortion>
//        (calc_intrinsics, target_observations); //<aslam::PinholeCamera, this->camera_->getDistortion()>
    EXPECT_TRUE(intrinsics_success) << "Intrinsics initialization failed.";


    // Compare the result against the simulated values.
    EXPECT_LT((((this->camera_->getParameters() - calc_intrinsics).norm()) /
        this->camera_->getParameters().norm()), 5e-3);
//    std::cout << "check [" << this->camera_->getParameters().transpose() << "]  vs.  ["
//                  << calc_intrinsics.transpose() << "] calc\n\n";
  }

}

ASLAM_UNITTEST_ENTRYPOINT
