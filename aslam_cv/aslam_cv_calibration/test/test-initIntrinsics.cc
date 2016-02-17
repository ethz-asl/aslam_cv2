#include <Eigen/Core>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/calibration/camera-initializer.h>
#include <aslam/calibration/target-aprilgrid.h>
//#include <aslam/calibration/target-observation.h>
#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-factory.h>
//#include <aslam/cameras/camera-pinhole.h>
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
template <class CameraDistortion>
class TestCameras : public testing::Test {
 public:
  typedef typename CameraDistortion::CameraType CameraType;
  typedef typename CameraDistortion::DistortionType DistortionType;
  protected:
    //TestCameras() : camera_(CameraType::template createTestCamera<DistortionType>() ) {};
    //virtual ~TestCameras() {};
    typename CameraType::Ptr camera_;
};

TYPED_TEST_CASE(TestCameras, Implementations);

TYPED_TEST(TestCameras, InitializeIntrinsics) {
  typename TestFixture::CameraType type_camera;
  typename TestFixture::DistortionType type_distortion;

  // Create a target April grid.
  aslam::calibration::TargetAprilGrid::TargetConfiguration aprilgrid_config;
  aslam::calibration::TargetAprilGrid::Ptr aprilgrid(
      new aslam::calibration::TargetAprilGrid(aprilgrid_config));

  // Get all target grid points in the target frame.
  Eigen::Matrix3Xd points_target_frame = aprilgrid->points();

  // Setup random check parameters.
  uint32_t im_width = 420;  // [pixel]
  uint32_t im_height = 380; // [pixel]
  Eigen::VectorXd test_intrinsics(4);
  // order: FU, FV, CU, CV [pixel]
  test_intrinsics << 330.0, 330.0, (im_width - 1.0) / 2.0, (im_height - 1.0) / 2.0;
  Eigen::Vector4d test_distortion(0.0, 0.0, 0.0, 0.0);

  // Create random camera poses and the corresponding transformations.
  const double kDegToRad = M_PI / 180.0;
  std::vector<aslam::Transformation> T_CT;
  size_t n_cam_poses = 3;
  size_t pose = 0;
  T_CT.resize(n_cam_poses * n_cam_poses -1);
  for (size_t dist = 0; dist < n_cam_poses; ++dist) {
    for (size_t ang = 0; ang < n_cam_poses; ++ang) {
      pose = dist * n_cam_poses + ang;
      T_CT.at(pose).setRandom(0.2 * dist, 5 * ang * kDegToRad);
    }
  }

  // Create test projection camera (NullDistortion, given focal length and principal point).
  aslam::Camera::Ptr testCamera = createCamera(
      aslam::CameraId::Random(),
      test_intrinsics,
      im_width, im_height,
      test_distortion,
      type_camera, //aslam::Camera::Type::kPinhole,
      type_distortion); //aslam::Distortion::Type::kRadTan);
  ASSERT_NE(testCamera, nullptr) << "Test camera cannot be created.";

  // Create simulated target observation (one image).
  std::vector<aslam::calibration::TargetObservation::Ptr> target_observations;
  target_observations.reserve(T_CT.size());

  Eigen::Matrix3Xd points_camera_frame(3, 144);   //TODO: can sizes be omitted?
  Eigen::Matrix2Xd image_points(2, points_camera_frame.cols());
  Eigen::VectorXd calc_intrinsics(4);
  for (size_t i = 0; i < T_CT.size(); ++i) {
    // Transform all points in the target frame T_p into the camera frame.
    points_camera_frame = T_CT.at(i).transformVectorized(points_target_frame);

    // Project points into the image plane.
    image_points(2, points_camera_frame.cols());
    //std::vector<ProjectionResult> results;  // TODO: remove
    testCamera->project3Vectorized(points_camera_frame, &image_points, &results);

    aslam::calibration::TargetObservation::Ptr target_observation (
        new aslam::calibration::TargetObservation(aprilgrid,
                                                  im_height,
                                                  im_width,
                                                  aprilgrid->setCornerIds(),
                                                  image_points));
    target_observations.emplace_back(target_observation);

    // Initialize the intrinsics using this observation.
    ASSERT_TRUE(aslam::calibration::initializeCameraIntrinsics<PinholeCamera>(
        calc_intrinsics, target_observations)) << "Intrinsics initialization failed.";

    // Compare the result against the simulated values.
    EXPECT_TRUE(fabs(double(test_intrinsics(0) - calc_intrinsics(0))) / test_intrinsics(0)  < 0.1);
  }

}

ASLAM_UNITTEST_ENTRYPOINT
