#include <Eigen/Core>
#include <aslam/common/entrypoint.h>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "aslam/calibration/target-algorithms.h"
#include "aslam/calibration/target-aprilgrid.h"
#include "aslam/calibration/target-observation.h"

namespace aslam {
namespace calibration {

class TargetObservationTest : public ::testing::Test {
 public:
  virtual void SetUp() {
    LOG(INFO) << "In setup.";
    TargetAprilGrid::TargetConfiguration target_config;
    const Eigen::Matrix3Xd grid_points_3d =
        aslam::calibration::TargetAprilGrid::createGridPoints(target_config);
  }
};

TEST_F(TargetObservationTest, AprilGridDetection) {
  LOG(INFO) << "Here.";
}

}  // namespace calibration
}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
