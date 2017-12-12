#include <Eigen/Core>
#include <aslam/common/entrypoint.h>
#include <aslam/cameras/camera.h>
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
    aslam::calibration::TargetAprilGrid::TargetConfiguration target_config;
    grid_points_3d_ =
        aslam::calibration::TargetAprilGrid::createGridPoints(target_config);
  }

 protected:
  Eigen::Matrix3Xd grid_points_3d_;
};

TEST_F(TargetObservationTest, AprilGridDetection) {
  LOG(INFO) << "In test.";
}

ASLAM_UNITTEST_ENTRYPOINT
