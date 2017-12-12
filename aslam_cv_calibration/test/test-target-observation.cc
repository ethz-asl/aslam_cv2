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
 protected:
  virtual void SetUp() {
    LOG(INFO) << "In setup.";
    const aslam::calibration::TargetAprilGrid::TargetConfiguration april_config;
    april_grid = aslam::calibration::TargetAprilGrid::Ptr(
        new aslam::calibration::TargetAprilGrid(april_config));
    camera = aslam::PinholeCamera::ConstPtr(
        aslam::PinholeCamera::createTestCamera());
    setTargetObservation();
  }

  aslam::calibration::TargetAprilGrid::Ptr april_grid;
  aslam::PinholeCamera::ConstPtr camera;
  aslam::Transformation T_B_C;
  aslam::calibration::TargetObservation::ConstPtr april_grid_observation;

 private:
  void setTargetObservation() {
    CHECK(april_grid);
    CHECK(camera);
    Eigen::Matrix2Xd reprojected_corners;
    Eigen::VectorXi corner_ids;
    getReprojectedCornersAndIds(&reprojected_corners, &corner_ids);
    april_grid_observation = aslam::calibration::TargetObservation::ConstPtr(
        new aslam::calibration::TargetObservation(
            april_grid, camera->imageHeight(), camera->imageWidth(), corner_ids,
            reprojected_corners));
  }

  void getReprojectedCornersAndIds(
      Eigen::Matrix2Xd* reprojected_corners, Eigen::VectorXi* corner_ids) {
    CHECK(april_grid);
    CHECK(camera);
    CHECK_NOTNULL(reprojected_corners);
    CHECK_NOTNULL(corner_ids);
    setTargetTransformation();
    Eigen::Matrix3Xd corner_points_B = april_grid->points();
    Eigen::Matrix3Xd corner_points_C =
        T_B_C.inverse().transformVectorized(corner_points_B);
    Eigen::Matrix2Xd corner_points_reprojected;
    std::vector<aslam::ProjectionResult> projection_results;
    camera->project3Vectorized(
        corner_points_C, reprojected_corners, &projection_results);
    CHECK(isReprojectionValid(projection_results));
    *corner_ids = Eigen::VectorXi(april_grid->size());
    for (int index = 0; index < corner_ids->size(); ++index) {
      (*corner_ids)(index) = index;
    }
    CHECK_EQ(reprojected_corners->cols(), corner_ids->size());
  }

  void setTargetTransformation() {
    constexpr double kDistanceFromTargetMeters = 2.0;
    T_B_C = aslam::Transformation(
        aslam::Quaternion(0.0, 1.0, 0.0, 0.0),
        aslam::Position3D(
            0.5 * april_grid->width(), 0.5 * april_grid->height(),
            kDistanceFromTargetMeters));
  }

  bool isReprojectionValid(
      const std::vector<aslam::ProjectionResult>& projection_results) const {
    for (const aslam::ProjectionResult& projection_result :
         projection_results) {
      if (!projection_result) {
        return false;
      }
    }
    return true;
  }
};

TEST_F(TargetObservationTest, AprilGridDetection) {
  LOG(INFO) << "In test.";
  // This test should test the pnp algorithm for the target observation.
}

ASLAM_UNITTEST_ENTRYPOINT
