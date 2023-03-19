#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/entrypoint.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/match.h>
#include <aslam/tracker/track-manager.h>
#include <eigen-checks/gtest.h>
#include <Eigen/Core>
#include <gtest/gtest.h>

TEST(TrackManagerTests, TestApplyMatcher) {
  aslam::TrackManager::resetIdProvider();

  aslam::Camera::Ptr camera = aslam::PinholeCamera::createTestCamera();
  aslam::VisualFrame::Ptr banana_frame =
      aslam::VisualFrame::createEmptyTestVisualFrame(camera, 0);
  aslam::VisualFrame::Ptr apple_frame =
      aslam::VisualFrame::createEmptyTestVisualFrame(camera, 1);

  Eigen::Matrix2Xd banana_keypoints = Eigen::Matrix2Xd::Zero(2, 5);
  Eigen::Matrix2Xd apple_keypoints = Eigen::Matrix2Xd::Zero(2, 5);

  // banana frame: -1, -1, 0, 1, -1
  // apple frame:  -1, 2, -1, -1, -1
  Eigen::VectorXi banana_tracks(5);
  banana_tracks << -1, -1, 0, 1, -1;

  Eigen::VectorXi apple_tracks(5);
  apple_tracks << -1, 2, -1, -1, -1;

  banana_frame->swapKeypointMeasurements(&banana_keypoints);
  apple_frame->swapKeypointMeasurements(&apple_keypoints);

  banana_frame->swapTrackIds(&banana_tracks);
  apple_frame->swapTrackIds(&apple_tracks);

  // matches_A_B: {(0,0), (1,1), (2,2), (3,3), (4,4)}
  aslam::FrameToFrameMatches matches_A_B;
  matches_A_B.reserve(5);

  matches_A_B.emplace_back(0, 0, 0.1);
  matches_A_B.emplace_back(1, 1, 0.2);
  matches_A_B.emplace_back(2, 2, 0.3);
  matches_A_B.emplace_back(3, 3, 0.4);
  matches_A_B.emplace_back(4, 4, 0.5);

  aslam::SimpleTrackManager track_manager;
  track_manager.applyMatchesToFrames(matches_A_B, apple_frame.get(), banana_frame.get());

  // Expected output:
  // banana frame: 3, 2, 0, 1, 4
  // apple frame:  3, 2, 0, 1, 4
  Eigen::VectorXi expected_banana_tracks(5);
  expected_banana_tracks << 0, 2, 0, 1, 1;

  Eigen::VectorXi expected_apple_tracks(5);
  expected_apple_tracks << 0, 2, 0, 1, 1;

  banana_tracks = banana_frame->getTrackIds();
  apple_tracks = apple_frame->getTrackIds();

  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_banana_tracks, banana_tracks));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_apple_tracks, apple_tracks));
}

TEST(TrackManagerTests, TestApplyMatchesEmpty) {
  aslam::TrackManager::resetIdProvider();

  aslam::Camera::Ptr camera = aslam::PinholeCamera::createTestCamera();
  aslam::VisualFrame::Ptr banana_frame =
      aslam::VisualFrame::createEmptyTestVisualFrame(camera, 0);
  aslam::VisualFrame::Ptr apple_frame =
      aslam::VisualFrame::createEmptyTestVisualFrame(camera, 1);

  Eigen::Matrix2Xd banana_keypoints = Eigen::Matrix2Xd::Zero(2, 5);
  Eigen::Matrix2Xd apple_keypoints = Eigen::Matrix2Xd::Zero(2, 5);

  Eigen::VectorXi banana_tracks = Eigen::VectorXi::Constant(5, -1);
  Eigen::VectorXi apple_tracks = Eigen::VectorXi::Constant(5, -1);

  banana_frame->swapKeypointMeasurements(&banana_keypoints);
  apple_frame->swapKeypointMeasurements(&apple_keypoints);

  banana_frame->swapTrackIds(&banana_tracks);
  apple_frame->swapTrackIds(&apple_tracks);

  aslam::FrameToFrameMatches matches_A_B;

  aslam::SimpleTrackManager track_manager;
  track_manager.applyMatchesToFrames(matches_A_B, apple_frame.get(), banana_frame.get());

  // Expected output:
  Eigen::VectorXi expected_banana_tracks = Eigen::VectorXi::Constant(5, -1);
  Eigen::VectorXi expected_apple_tracks = Eigen::VectorXi::Constant(5, -1);

  banana_tracks = banana_frame->getTrackIds();
  apple_tracks = apple_frame->getTrackIds();

  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_banana_tracks, banana_tracks));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_apple_tracks, apple_tracks));
}

ASLAM_UNITTEST_ENTRYPOINT
