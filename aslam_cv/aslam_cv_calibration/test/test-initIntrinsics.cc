#include <Eigen/Core>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <geometric-vision/pnp-pose-estimator.h>

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

  // Generate target_observations
  std::vector<aslam::calibration::TargetObservation::Ptr> target_observations;
  target_observations.reserve(3);
  for (int i = 0; i < 3; ++i) {

    calibration::TargetObservation::Ptr observation(  //TODO(duboisf):fill in parameters
        calibration::TargetObservation(const calibration::TargetBase::Ptr& target,
                                       const uint32_t im_height,
                                       const uint32_t im_width,
                                       const Eigen::VectorXi& corner_ids,
                                       const Eigen::Matrix2Xd& image_corners));

    // Only keep the observation if it was successful.
    if (observation) {
      target_observations.emplace_back(observation);
    }
  }

  // Calculate intrinsics.
  Eigen::VectorXd calc_intrinsics;
  ASSERT_TRUE(calibration::initializeCameraIntrinsics<PinholeCamera>(calc_intrinsics, target_observations));

  // Create test projection camera.
  Camera::Ptr testCamera = aslam::createCamera(
        aslam::CameraId::Random(),
        calc_intrinsics,
        target_observations[0]->getImageWidth(), target_observations[0]->getImageHeight(),
        Eigen::Vector4d(0.0, 0.0, 0.0 , 0.0),
        Camera::Type::kPinhole,
        Distortion::Type::kRadTan);
  ASSERT_NE(testCamera, nullptr);

  // Setup matching.
  constexpr bool kRunNonlinearRefinement = true;
  const double kPixelSigma = 1.0;
  const int kMaxRansacIters = 200;
  geometric_vision::PnpPoseEstimator pnp(kRunNonlinearRefinement);

  size_t frame_id = 0;
  for (const aslam::calibration::TargetObservation::Ptr& obs : target_observations) {
    CHECK(obs);
    const Eigen::Matrix2Xd& keypoints_measured = obs->getObservedCorners();
    const Eigen::Matrix3Xd G_corner_positions = obs->getCorrespondingTargetPoints();

    aslam::Transformation T_G_C;
    std::vector<int> inliers;
    int num_iters = 0;
    bool pnp_success = pnp.absolutePoseRansacPinholeCam(
        keypoints_measured, G_corner_positions, kPixelSigma, kMaxRansacIters, testCamera, &T_G_C,
        &inliers, &num_iters);

    // Reproject the corners.
    Eigen::Matrix3Xd C_corner_positions =
        T_G_C.inverse().transformVectorized(G_corner_positions);

    Eigen::Matrix2Xd keypoints_reprojected;
    std::vector<aslam::ProjectionResult> projection_result;
    testCamera->project3Vectorized(C_corner_positions, &keypoints_reprojected, &projection_result);

    CHECK_EQ(obs->numObservedCorners(), static_cast<size_t>(keypoints_measured.cols()));
    for (size_t idx = 0u; idx < obs->numObservedCorners(); ++idx) {
      const bool pnp_inlier =
          (std::find(inliers.begin(), inliers.end(), static_cast<int>(idx)) != inliers.end());

      // Test reprojection error.
      EXPECT_EQ(keypoints_measured(0, idx), keypoints_reprojected(0, idx))
        << frame_id << ", " << obs->getObservedCornerIds()(idx) << ", "
        << keypoints_measured(0, idx) << ", " << keypoints_reprojected(0, idx) << "\n";
      EXPECT_EQ(keypoints_measured(1, idx), keypoints_reprojected(1, idx))
        << frame_id << ", " << obs->getObservedCornerIds()(idx) << ", "
        << keypoints_measured(1, idx) << ", " << keypoints_reprojected(1, idx) << "\n";
    }
    ++frame_id;
  }

}
}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
