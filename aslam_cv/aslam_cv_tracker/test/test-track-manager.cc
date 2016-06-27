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
  aslam::FrameToFrameMatchesWithScore matches_A_B;
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

  aslam::FrameToFrameMatchesWithScore matches_A_B;

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

TEST(TrackManagerTests, TestApplyMatchesUniformly) {
  aslam::TrackManager::resetIdProvider();

  aslam::Camera::Ptr camera = aslam::PinholeCamera::createTestCamera();
  aslam::VisualFrame::Ptr banana_frame =
      aslam::VisualFrame::createEmptyTestVisualFrame(camera, 0);
  aslam::VisualFrame::Ptr apple_frame =
      aslam::VisualFrame::createEmptyTestVisualFrame(camera, 1);

  const size_t kNumKeypoints = 100u;

  Eigen::Matrix2Xd banana_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);
  Eigen::Matrix2Xd apple_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);

  double image_width = static_cast<double>(camera->imageWidth());
  double image_height = static_cast<double>(camera->imageHeight());

  // Fills the frames as follows (numbers indicate the decreasing scores from
  // top left to bottom right and top right to bottom left):
  //
  //   ----->x
  //  | ------------------
  //  | |x1.0        0.5x|
  //  v |  x          x  |
  //  y |    x      x    |
  //    |      x  x      |
  //    |       xx       |
  //    |      x  x      |
  //    |    x      x    |
  //    |  x          x  |
  //    |x0.0        0.5x|
  //    ------------------


  size_t half_num_keypoints = kNumKeypoints / 2.0;
  for (size_t idx = 0; idx < half_num_keypoints; ++idx) {
    double x = image_width /
        static_cast<double>(half_num_keypoints) * static_cast<double>(idx);
    double y = image_height /
        static_cast<double>(half_num_keypoints) * static_cast<double>(idx);
    CHECK_GE(x, 0.0); CHECK_LE(x, image_width);
    CHECK_GE(y, 0.0); CHECK_LT(y, image_height);

    banana_keypoints(0, idx) = x;
    banana_keypoints(1, idx) = y;
    apple_keypoints(0, idx) = x;
    apple_keypoints(1, idx) = y;
  }

  for (size_t idx = 1; idx <= half_num_keypoints; ++idx) {
    double x = image_width -
        (image_width /
            static_cast<double>(half_num_keypoints) * static_cast<double>(idx));
    double y = image_height /
        static_cast<double>(half_num_keypoints + 1) * static_cast<double>(idx);
    CHECK_GE(x, 0.0); CHECK_LE(x, image_width);
    CHECK_GE(y, 0.0); CHECK_LT(y, image_height);

    banana_keypoints(0, half_num_keypoints + idx - 1) = x;
    banana_keypoints(1, half_num_keypoints + idx - 1) = y;
    apple_keypoints(0, half_num_keypoints + idx - 1) = x;
    apple_keypoints(1, half_num_keypoints + idx - 1) = y;
  }

  Eigen::VectorXi banana_tracks = Eigen::VectorXi::Constant(kNumKeypoints, -1);
  Eigen::VectorXi apple_tracks = Eigen::VectorXi::Constant(kNumKeypoints, -1);

  banana_frame->swapKeypointMeasurements(&banana_keypoints);
  apple_frame->swapKeypointMeasurements(&apple_keypoints);

  banana_frame->swapTrackIds(&banana_tracks);
  apple_frame->swapTrackIds(&apple_tracks);

  /// matches_A_B: {(0,0), (1,1), (2,2), (3,3), (4,4), ...}
  Eigen::VectorXd banana_scores = Eigen::VectorXd::Constant(kNumKeypoints, 0);
  Eigen::VectorXd apple_scores = Eigen::VectorXd::Constant(kNumKeypoints, 0);
  aslam::FrameToFrameMatchesWithScore matches_A_B;
  matches_A_B.reserve(kNumKeypoints);
  for (size_t match_idx = 0; match_idx < kNumKeypoints; ++match_idx) {
    double score = 1.0 - (static_cast<double>(match_idx) /
        static_cast<double>(kNumKeypoints));
    matches_A_B.emplace_back(match_idx, match_idx, score);

    apple_scores(match_idx) = score;
    banana_scores(match_idx) = score;
  }
  banana_frame->swapKeypointScores(&banana_scores);
  apple_frame->swapKeypointScores(&apple_scores);

  const size_t kBucketCapacity = 5u;
  const size_t kNumStrongToPush = 20u;
  const size_t kNumBucketsRoot = 2u;
  const double kScoreTresholdUnconditional = 0.8;
  size_t num_buckets = kNumBucketsRoot * kNumBucketsRoot;

  aslam::UniformTrackManager track_manager(kNumBucketsRoot,
                                           kBucketCapacity *
                                           kNumBucketsRoot * kNumBucketsRoot,
                                           kNumStrongToPush,
                                           kScoreTresholdUnconditional);
  track_manager.applyMatchesToFrames(matches_A_B,
                                     apple_frame.get(),
                                     banana_frame.get());

  // Expected output: We expect the last five keypoints in each bucket to
  // contain a valid track id.
  Eigen::VectorXi expected_banana_tracks =
      Eigen::VectorXi::Constant(kNumKeypoints, -1);
  Eigen::VectorXi expected_apple_tracks =
      Eigen::VectorXi::Constant(kNumKeypoints, -1);

  int track_id = 0;
  CHECK_LT(kNumStrongToPush, kNumKeypoints);
  // The first kNumStrongToPush matches will get assigned anyways, even though
  // it overflows buckets 0. They all live in bucket 0.
  for (size_t idx = 0; idx < kNumStrongToPush; ++idx) {
    expected_banana_tracks(idx) = track_id;
    expected_apple_tracks(idx) = track_id;
    ++track_id;
  }

  size_t kNumKeypointsPerBucket = kNumKeypoints / num_buckets;
  // All very strong new tracks to push lived in bucket 0, so we can just
  // start with bucket 1 to check for the kBucketCapacity strongest matches
  // in the remaining buckets.
  for (size_t bucket_idx = 1; bucket_idx < num_buckets; ++bucket_idx) {
    for (size_t keypoint_idx = 0; keypoint_idx < kBucketCapacity;
        ++keypoint_idx) {
      size_t index = keypoint_idx + kNumKeypointsPerBucket * bucket_idx;
      CHECK_LT(index, kNumKeypoints);
      expected_banana_tracks(index) = track_id;
      expected_apple_tracks(index) = track_id;
      ++track_id;
    }
  }

  banana_tracks = banana_frame->getTrackIds();
  apple_tracks = apple_frame->getTrackIds();

  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_banana_tracks, banana_tracks));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_apple_tracks, apple_tracks));
}

TEST(TrackManagerTests, TestApplyMatchesUniformEmpty) {
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

  Eigen::VectorXd banana_scores = Eigen::VectorXd::Constant(5, 100);
  Eigen::VectorXd apple_scores = Eigen::VectorXd::Constant(5, 100);

  banana_frame->swapKeypointMeasurements(&banana_keypoints);
  apple_frame->swapKeypointMeasurements(&apple_keypoints);

  banana_frame->swapTrackIds(&banana_tracks);
  apple_frame->swapTrackIds(&apple_tracks);

  banana_frame->swapKeypointScores(&banana_scores);
  apple_frame->swapKeypointScores(&apple_scores);

  aslam::FrameToFrameMatchesWithScore matches_A_B;

  const size_t kBucketCapacity = 5u;
  const size_t kNumStrongToPush = 20u;
  const size_t kNumBucketsRoot = 2u;
  const double kScoreTresholdUnconditional = 0.8;
  aslam::UniformTrackManager track_manager(kNumBucketsRoot,
                                           kBucketCapacity *
                                           kNumBucketsRoot * kNumBucketsRoot,
                                           kNumStrongToPush,
                                           kScoreTresholdUnconditional);

  track_manager.applyMatchesToFrames(matches_A_B,
                                     apple_frame.get(),
                                     banana_frame.get());

  // Expected output:
  Eigen::VectorXi expected_banana_tracks = Eigen::VectorXi::Constant(5, -1);
  Eigen::VectorXi expected_apple_tracks = Eigen::VectorXi::Constant(5, -1);

  banana_tracks = banana_frame->getTrackIds();
  apple_tracks = apple_frame->getTrackIds();

  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_banana_tracks, banana_tracks));
  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(expected_apple_tracks, apple_tracks));
}

ASLAM_UNITTEST_ENTRYPOINT
