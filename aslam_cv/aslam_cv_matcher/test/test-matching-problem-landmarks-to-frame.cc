#include <algorithm>
#include <cmath>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/match.h>
#include <aslam/matcher/matching-engine-non-exclusive.h>
#include <aslam/matcher/matching-problem-landmarks-to-frame.h>

class LandmarksToFrameMatcherTest : public testing::Test {
 protected:
  virtual void SetUp() {
    camera_ = aslam::PinholeCamera::createTestCamera();
    frame_ = aslam::VisualFrame::createEmptyTestVisualFrame(camera_, 0);

    image_space_distance_threshold_ = 25.0;
    hamming_distance_threshold_ = 60;
  }

  double image_space_distance_threshold_;
  int hamming_distance_threshold_;

  aslam::VisualFrame::Ptr frame_;
  aslam::LandmarkWithDescriptorList landmarks_;

  aslam::MatchingEngineNonExclusive<aslam::MatchingProblemLandmarksToFrame> matching_engine_;

  aslam::PinholeCamera::Ptr camera_;
};

TEST_F(MatcherTest, EmptyMatch) {
  aslam::MatchingEngineNonExclusive<aslam::MatchingProblemLandmarksToFrame> matching_engine;

  aslam::MatchingProblemLandmarksToFrame::Ptr matching_problem = aslam::aligned_shared<
      aslam::MatchingProblemLandmarksToFrame>(*frame_, landmarks_,
                                              image_space_distance_threshold_,
                                              hamming_distance_threshold_);
  aslam::MatchesWithScore matches_A_B;
  matching_engine_.match(matching_problem.get(), &matches_A_B);

  EXPECT_TRUE(matches_A_B.empty());
}

TEST_F(MatcherTest, MatchIdentity) {
  const Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, 1);

  Eigen::Vector3d projected_keypoint;
  camera_->backProject3(frame_keypoints.col(0), &projected_keypoint);

  Eigen::Matrix<unsigned char, 48, 1> frame_descriptors =
      Eigen::Matrix<unsigned char, 48, 1>::Zero();
  Eigen::Matrix<unsigned char, 48, 1> landmark_descriptors =
      Eigen::Matrix<unsigned char, 48, 1>::Zero();

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  landmarks_.emplace_back(projected_keypoint, landmark_descriptors.col(0));

  aslam::MatchingProblemLandmarksToFrame::Ptr matching_problem =
      aslam::aligned_shared<aslam::MatchingProblemLandmarksToFrame>(
          *frame_, landmarks_, image_space_distance_threshold_,
          hamming_distance_threshold_);

  aslam::MatchesWithScore matches_A_B;
  matching_engine_.match(matching_problem.get(), &matches_A_B);

  ASSERT_EQ(1u, matches_A_B.size());

  aslam::MatchWithScore match = matches_A_B[0];
  EXPECT_EQ(0, match.getIndexApple());
  EXPECT_EQ(0, match.getIndexBanana());
  EXPECT_DOUBLE_EQ(1.0, match.score);
}

TEST_F(MatcherTest, MatchIdentityWithScale) {
  const Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, 1);

  Eigen::Vector3d projected_keypoint;
  camera_->backProject3(frame_keypoints.col(0), &projected_keypoint);

  // Arbitrarily picked scaling factor.
  projected_keypoint *= 34.23232;

  Eigen::Matrix<unsigned char, 48, 1> frame_descriptors =
      Eigen::Matrix<unsigned char, 48, 1>::Zero();
  Eigen::Matrix<unsigned char, 48, 1> landmark_descriptors =
      Eigen::Matrix<unsigned char, 48, 1>::Zero();

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  landmarks_.emplace_back(projected_keypoint, landmark_descriptors.col(0));

  aslam::MatchingProblemLandmarksToFrame::Ptr matching_problem =
      aslam::aligned_shared<aslam::MatchingProblemLandmarksToFrame>(
          *frame_, landmarks_, image_space_distance_threshold_,
          hamming_distance_threshold_);

  aslam::MatchesWithScore matches_A_B;
  matching_engine_.match(matching_problem.get(), &matches_A_B);

  ASSERT_EQ(1u, matches_A_B.size());

  aslam::MatchWithScore match = matches_A_B[0];
  EXPECT_EQ(0, match.getIndexApple());
  EXPECT_EQ(0, match.getIndexBanana());
  EXPECT_DOUBLE_EQ(1.0, match.score);
}

TEST_F(MatcherTest, MatchRandomly) {
  const size_t kNumKeypoints = 2000u;
  Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);

  Eigen::Matrix3Xd projected_keypoints(3, kNumKeypoints);

  std::srand(23980u);
  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    double keypoint_x = static_cast<double>(std::rand() % camera_->imageWidth());
    double keypoint_y = static_cast<double>(std::rand() % camera_->imageHeight());

    frame_keypoints(0, keypoint_index) = keypoint_x;
    frame_keypoints(1, keypoint_index) = keypoint_y;

    double shift_translation_pixels = static_cast<double>(image_space_distance_threshold_ -
                                      (std::rand() % 2 * image_space_distance_threshold_));
    double shift_angle_radians = static_cast<double>(std::rand() % 360) / 180.0 * 3.14159;


    double x_shift = std::cos(shift_translation_pixels);
    double y_shift = std::sin(shift_translation_pixels);

    double shifted_keypoint_x = std::min<double>(
        std::max<double>(keypoint_x + x_shift, 0.1), camera_->imageWidth() - 0.1);
    double shifted_keypoint_y = std::min<double>(
        std::max<double>(keypoint_y + y_shift, 0.1), camera_->imageHeight() - 0.1);

    Eigen::Vector2d shifted_keypoint(shifted_keypoint_x, shifted_keypoint_y);

    Eigen::Vector3d projected_keypoint;
    CHECK(camera_->backProject3(shifted_keypoint, &projected_keypoint));
    projected_keypoints.col(keypoint_index) = projected_keypoint;
  }

  // Arbitrarily picked scaling factor.
  const double kScalingFactor = 1e5;

  Eigen::VectorXd random_scale = Eigen::VectorXd::Random(kNumKeypoints);
  random_scale = random_scale.cwiseAbs() * kScalingFactor;

  projected_keypoints = projected_keypoints * random_scale.asDiagonal();

  Eigen::Matrix<unsigned char, 48, Eigen::Dynamic> frame_descriptors =
      Eigen::Matrix<unsigned char, 48, Eigen::Dynamic>::Random(48, kNumKeypoints);

  Eigen::Matrix<unsigned char, 48, Eigen::Dynamic> landmark_descriptors = frame_descriptors;

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    landmarks_.emplace_back(projected_keypoints.col(keypoint_index),
                            landmark_descriptors.col(keypoint_index));
  }

  aslam::MatchingProblemLandmarksToFrame::Ptr matching_problem =
      aslam::aligned_shared<aslam::MatchingProblemLandmarksToFrame>(
          *frame_, landmarks_, image_space_distance_threshold_,
          hamming_distance_threshold_);

  aslam::MatchesWithScore matches_A_B;
  matching_engine_.match(matching_problem.get(), &matches_A_B);

  ASSERT_EQ(kNumKeypoints, matches_A_B.size());

  for (const aslam::MatchWithScore& match : matches_A_B) {
    EXPECT_EQ(match.getIndexApple(), match.getIndexBanana());
    EXPECT_DOUBLE_EQ(1.0, match.score);
  }
}

ASLAM_UNITTEST_ENTRYPOINT
