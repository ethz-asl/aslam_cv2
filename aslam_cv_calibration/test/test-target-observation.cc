#include <aslam/common/entrypoint.h>
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
  }
};

TEST_F(TargetObservationTest, AprilGridDetection) {
  LOG(INFO) << "Here.";
}

ASLAM_UNITTEST_ENTRYPOINT
