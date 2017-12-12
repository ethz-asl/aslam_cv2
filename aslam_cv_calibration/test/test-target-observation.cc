#include <vector>

#include <Eigen/Core>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/pose-types.h>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "aslam/calibration/target-algorithms.h"
#include "aslam/calibration/target-aprilgrid.h"
#include "aslam/calibration/target-observation.h"

class TargetObservationTest : public ::testing::Test {
 public:
  virtual void SetUp() {
    LOG(INFO) << "In setup.";
    aslam::calibration::TargetAprilGrid::TargetConfiguration april_config;
    aslam::calibration::TargetAprilGrid april_grid(april_config);
    Eigen::Matrix3Xd corner_points_B = april_grid.points();
    aslam::Transformation T_B_C(
        aslam::Quaternion(0.0, 1.0, 0.0, 0.0),
        aslam::Position3D(
            0.5 * april_grid.width(), 0.5 * april_grid.height(), 1.0));
    Eigen::Matrix3Xd corner_points_C =
        T_B_C.inverse().transformVectorized(corner_points_B);
    const aslam::PinholeCamera::ConstPtr camera_ptr =
        aslam::PinholeCamera::createTestCamera();
    Eigen::Matrix2Xd corner_points_reprojected;
    std::vector<aslam::ProjectionResult> projection_results;
    camera_ptr->project3Vectorized(
        corner_points_C, &corner_points_reprojected, &projection_results);
    for (const aslam::ProjectionResult& projection_result :
         projection_results) {
      CHECK(projection_result);
    }
  }
};

TEST_F(TargetObservationTest, AprilGridDetection) {
  LOG(INFO) << "In test.";
}

ASLAM_UNITTEST_ENTRYPOINT
