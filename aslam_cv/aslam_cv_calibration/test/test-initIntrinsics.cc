#include <Eigen/Core>
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

namespace aslam {
TEST(TestNCameraInitIntrinsics, testInitIntrinsics_pinhole) {

// Create a target April grid.
  calibration::TargetAprilGrid::TargetConfiguration aprilgrid_config;
  calibration::TargetAprilGrid::Ptr aprilgrid(
      new calibration::TargetAprilGrid(aprilgrid_config));

// Get all target grid points in the target frame.
  Eigen::Matrix3Xd points_target_frame = aprilgrid->points();

// Create a random camera pose and the corresponding transformation TODO(duboisf): inputs?
  kindr::minimal::QuatTransformationTemplate::Vector6 camera_pose_t_r(1, 2, 3, 0, 0, 0);
  Transformation T_CT = kindr::minimal::QuatTransformationTemplate(camera_pose_t_r);

// Setup random check parameters.
  uint32_t im_width = 420;
  uint32_t im_height = 380;
  Eigen::VectorXd test_intrinsics(100, 200, 350, 250);  // order: FU, FV, CU, CV
  Eigen::Vector4d null_distortion(0.0, 0.0, 0.0 , 0.0);

// Create test projection camera (NullDistortion, given focal length and principal point).
// TODO(duboisf): test different camera/distortion models
  Camera::Ptr testCamera = createCamera(
        CameraId::Random(),
        test_intrinsics,
        im_width, im_height,
        null_distortion,
        Camera::Type::kPinhole,
        Distortion::Type::kRadTan);
  ASSERT_NE(testCamera, nullptr) << "Test camera cannot be created.";

// Transform all points in the target frame T_p into the camera frame.
  Eigen::Matrix3Xd  points_camera_frame = T_CT * points_target_frame;

// Project points into the image plane.
  Eigen::Vector2d* image_points;
  testCamera->project3(points_camera_frame, image_points);

// Combine image_points with TargetAprilGrid to a simulated TargetObservation.

// Create target observation (one image). TODO
  std::vector<aslam::calibration::TargetObservation::Ptr> target_observations;
  target_observations.reserve(1);

  Eigen::VectorXi corner_ids;

  calibration::TargetObservation::Ptr target_observation;
  target_observation->TargetObservation(
      aprilgrid->TargetBase(aprilgrid->rows_, aprilgrid->cols_, aprilgrid->points_target_frame_),
      im_height,
      im_width,
      corner_ids, // TODO
      image_points);

  target_observations.emplace_back(target_observation);

// Initialize the intrinsics using this observation.
  Eigen::VectorXd calc_intrinsics;
  ASSERT_TRUE(calibration::initializeCameraIntrinsics<PinholeCamera>(calc_intrinsics,
                                                                     target_observations))
      << "Intrinsics initialization failed.";

// Compare the result against the simulated values.
  ASSERT_EQ(test_intrinsics, calc_intrinsics)
      << "Intrinsics incorrect.\n"
      << "(check) " << test_intrinsics << "  vs.  " << calc_intrinsics << " (calculated) \n\n";


}
}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
