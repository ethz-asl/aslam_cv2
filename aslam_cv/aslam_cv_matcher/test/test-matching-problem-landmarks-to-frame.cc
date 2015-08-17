#include <algorithm>
#include <cmath>
#include <random>
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

static const size_t kDescriptorSizeBytes = 48u;

class LandmarksToFrameMatcherTest : public testing::Test {
 protected:
  virtual void SetUp() {
    camera_ = aslam::PinholeCamera::createTestCamera();
    frame_ = aslam::VisualFrame::createEmptyTestVisualFrame(camera_, 0);

    CHECK_EQ(kDescriptorSizeBytes, frame_->getDescriptorSizeBytes());

    image_space_distance_threshold_pixels_ = 25.0;
    hamming_distance_threshold_ = 1;
  }

  inline void match(aslam::MatchesWithScore* matches_A_B) {
    CHECK_NOTNULL(matches_A_B);
    aslam::MatchingProblemLandmarksToFrame::Ptr matching_problem =
        aslam::aligned_shared<aslam::MatchingProblemLandmarksToFrame>(
            *frame_, landmarks_, image_space_distance_threshold_pixels_,
            hamming_distance_threshold_);

    matching_engine_.match(matching_problem.get(), matches_A_B);
  }

  double image_space_distance_threshold_pixels_;
  int hamming_distance_threshold_;

  aslam::VisualFrame::Ptr frame_;
  aslam::LandmarkWithDescriptorList landmarks_;

  aslam::MatchingEngineNonExclusive<aslam::MatchingProblemLandmarksToFrame> matching_engine_;

  aslam::PinholeCamera::Ptr camera_;
};

TEST_F(LandmarksToFrameMatcherTest, EmptyMatch) {
  aslam::MatchingEngineNonExclusive<aslam::MatchingProblemLandmarksToFrame> matching_engine;

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  EXPECT_TRUE(matches_A_B.empty());
}

TEST_F(LandmarksToFrameMatcherTest, MatchIdentity) {
  const Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, 1);

  Eigen::Vector3d projected_keypoint;
  CHECK(camera_->backProject3(frame_keypoints.col(0), &projected_keypoint));

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1>::Zero();
  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> landmark_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1>::Zero();

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  landmarks_.emplace_back(projected_keypoint, landmark_descriptors.col(0));

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_EQ(1u, matches_A_B.size());

  aslam::MatchWithScore match = matches_A_B[0];
  EXPECT_EQ(0, match.getIndexApple());
  EXPECT_EQ(0, match.getIndexBanana());
  EXPECT_DOUBLE_EQ(1.0, match.score);
}

TEST_F(LandmarksToFrameMatcherTest, MatchIdentityWithScale) {
  const Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, 1);

  Eigen::Vector3d projected_keypoint;
  CHECK(camera_->backProject3(frame_keypoints.col(0), &projected_keypoint));

  // Arbitrarily picked scaling factor.
  projected_keypoint *= 34.23232;

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1>::Zero();
  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> landmark_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1>::Zero();

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  landmarks_.emplace_back(projected_keypoint, landmark_descriptors.col(0));

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_EQ(1u, matches_A_B.size());

  aslam::MatchWithScore match = matches_A_B[0];
  EXPECT_EQ(0, match.getIndexApple());
  EXPECT_EQ(0, match.getIndexBanana());
  EXPECT_DOUBLE_EQ(1.0, match.score);
}

TEST_F(LandmarksToFrameMatcherTest, MatchRandomly) {
  const size_t kNumKeypoints = 2000u;
  Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);

  Eigen::Matrix3Xd projected_keypoints(3, kNumKeypoints);

  std::random_device random_device;
  std::default_random_engine random_engine(random_device());

  CHECK_GT(camera_->imageWidth(), 0u);
  CHECK_GT(camera_->imageHeight(), 0u);
  std::uniform_int_distribution<int> uniform_dist_image_width(0, camera_->imageWidth() - 1);
  std::uniform_int_distribution<int> uniform_dist_image_height(0, camera_->imageHeight() - 1);
  std::uniform_real_distribution<double> uniform_dist_image_space(0, image_space_distance_threshold_pixels_);
  std::uniform_int_distribution<int> uniform_dist_angle(0, 359);


  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    const double keypoint_x = static_cast<double>(uniform_dist_image_width(random_engine));
    const double keypoint_y = static_cast<double>(uniform_dist_image_height(random_engine));

    frame_keypoints(0, keypoint_index) = keypoint_x;
    frame_keypoints(1, keypoint_index) = keypoint_y;

    const double shift_translation_pixels = uniform_dist_image_space(random_engine);
    const double shift_angle_radians =
        static_cast<double>(uniform_dist_angle(random_engine)) / 180.0 * 3.14159;

    const double x_shift = shift_translation_pixels * std::cos(shift_angle_radians);
    const double y_shift = shift_translation_pixels * std::sin(shift_angle_radians);

    const double shifted_keypoint_x = std::min<double>(
        std::max<double>(keypoint_x + x_shift, 0.1), camera_->imageWidth() - 0.1);
    const double shifted_keypoint_y = std::min<double>(
        std::max<double>(keypoint_y + y_shift, 0.1), camera_->imageHeight() - 0.1);

    const Eigen::Vector2d shifted_keypoint(shifted_keypoint_x, shifted_keypoint_y);

    Eigen::Vector3d projected_keypoint;
    CHECK(camera_->backProject3(shifted_keypoint, &projected_keypoint));
    projected_keypoints.col(keypoint_index) = projected_keypoint;
  }

  // Arbitrarily picked scaling factor.
  const double kScalingFactor = 1e5;

  Eigen::VectorXd random_scale = Eigen::VectorXd::Random(kNumKeypoints);
  random_scale = random_scale.cwiseAbs() * kScalingFactor;

  projected_keypoints = projected_keypoints * random_scale.asDiagonal();

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic>::Random(kDescriptorSizeBytes, kNumKeypoints);

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> landmark_descriptors =
      frame_descriptors;

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    landmarks_.emplace_back(projected_keypoints.col(keypoint_index),
                            landmark_descriptors.col(keypoint_index));
  }

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_EQ(kNumKeypoints, matches_A_B.size());

  for (const aslam::MatchWithScore& match : matches_A_B) {
    EXPECT_EQ(match.getIndexApple(), match.getIndexBanana());
    EXPECT_DOUBLE_EQ(1.0, match.score);
  }
}

TEST_F(LandmarksToFrameMatcherTest, MatchNoMatchesBecauseOfHammingDistance) {
  const size_t kNumKeypoints = 2000u;
  Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);

  Eigen::Matrix3Xd projected_keypoints(3, kNumKeypoints);

  std::random_device random_device;
  std::default_random_engine random_engine(random_device());

  std::uniform_int_distribution<int> uniform_dist_image_width(0, camera_->imageWidth() - 1);
  std::uniform_int_distribution<int> uniform_dist_image_height(0, camera_->imageHeight() - 1);
  std::uniform_real_distribution<double> uniform_dist_image_space(0, image_space_distance_threshold_pixels_);
  std::uniform_int_distribution<int> uniform_dist_angle(0, 359);


  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    const double keypoint_x = static_cast<double>(uniform_dist_image_width(random_engine));
    const double keypoint_y = static_cast<double>(uniform_dist_image_height(random_engine));

    frame_keypoints(0, keypoint_index) = keypoint_x;
    frame_keypoints(1, keypoint_index) = keypoint_y;

    const double shift_translation_pixels = uniform_dist_image_space(random_engine);
    const double shift_angle_radians =
        static_cast<double>(uniform_dist_angle(random_engine)) / 180.0 * 3.14159;

    const double x_shift = shift_translation_pixels * std::cos(shift_angle_radians);
    const double y_shift = shift_translation_pixels * std::sin(shift_angle_radians);

    const double shifted_keypoint_x = std::min<double>(
        std::max<double>(keypoint_x + x_shift, 0.1), camera_->imageWidth() - 0.1);
    const double shifted_keypoint_y = std::min<double>(
        std::max<double>(keypoint_y + y_shift, 0.1), camera_->imageHeight() - 0.1);

    const Eigen::Vector2d shifted_keypoint(shifted_keypoint_x, shifted_keypoint_y);

    Eigen::Vector3d projected_keypoint;
    CHECK(camera_->backProject3(shifted_keypoint, &projected_keypoint));
    projected_keypoints.col(keypoint_index) = projected_keypoint;
  }

  // Arbitrarily picked scaling factor.
  const double kScalingFactor = 1e5;

  Eigen::VectorXd random_scale = Eigen::VectorXd::Random(kNumKeypoints);
  random_scale = random_scale.cwiseAbs() * kScalingFactor;

  projected_keypoints = projected_keypoints * random_scale.asDiagonal();

  Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic>::Random(kDescriptorSizeBytes, kNumKeypoints);

  Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> landmark_descriptors =
      frame_descriptors;

  const int kNumBytesDifferent = 2;
  hamming_distance_threshold_ = kNumBytesDifferent * 8;
  CHECK_LT(kNumBytesDifferent, frame_descriptors.rows());
  CHECK_LT(kNumBytesDifferent, landmark_descriptors.rows());
  frame_descriptors.topRows(kNumBytesDifferent) =
      Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>::Zero(kNumBytesDifferent, kNumKeypoints);
  landmark_descriptors.topRows(kNumBytesDifferent) =
      Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>::Constant(kNumBytesDifferent, kNumKeypoints, 255u);;

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    landmarks_.emplace_back(projected_keypoints.col(keypoint_index),
                            landmark_descriptors.col(keypoint_index));
  }

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_TRUE(matches_A_B.empty());
}

TEST_F(LandmarksToFrameMatcherTest, MatchNoMatchBecauseOfSearchBand) {
  const size_t kNumKeypoints = 2000u;
  Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);

  Eigen::Matrix3Xd projected_keypoints(3, kNumKeypoints);

  std::random_device random_device;
  std::default_random_engine random_engine(random_device());

  std::uniform_int_distribution<int> uniform_dist_image_width(0, camera_->imageWidth() - 1);
  std::uniform_int_distribution<int> uniform_dist_image_height(0, camera_->imageHeight() - 1);
  std::uniform_int_distribution<int> uniform_dist_angle(0, 359);

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    const double keypoint_x = static_cast<double>(uniform_dist_image_width(random_engine));
    const double keypoint_y = static_cast<double>(uniform_dist_image_height(random_engine));

    frame_keypoints(0, keypoint_index) = keypoint_x;
    frame_keypoints(1, keypoint_index) = keypoint_y;

    const double shift_translation_pixels = image_space_distance_threshold_pixels_ + 0.1;
    const double shift_angle_radians =
        static_cast<double>(uniform_dist_angle(random_engine)) / 180.0 * 3.14159;

    const double x_shift = shift_translation_pixels * std::cos(shift_angle_radians);
    const double y_shift = shift_translation_pixels * std::sin(shift_angle_radians);

    // Move all projected keypoints out of the search band.
    const double shifted_keypoint_x = keypoint_x + x_shift;
    const double shifted_keypoint_y = keypoint_y + y_shift;

    const Eigen::Vector2d shifted_keypoint(shifted_keypoint_x, shifted_keypoint_y);

    Eigen::Vector3d projected_keypoint;
    CHECK(camera_->backProject3(shifted_keypoint, &projected_keypoint));
    projected_keypoints.col(keypoint_index) = projected_keypoint;
  }

  // Arbitrarily picked scaling factor.
  const double kScalingFactor = 1e5;

  Eigen::VectorXd random_scale = Eigen::VectorXd::Random(kNumKeypoints);
  random_scale = random_scale.cwiseAbs() * kScalingFactor;

  projected_keypoints = projected_keypoints * random_scale.asDiagonal();

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic>::Random(kDescriptorSizeBytes, kNumKeypoints);

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> landmark_descriptors =
      frame_descriptors;

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  hamming_distance_threshold_ = 1;

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    landmarks_.emplace_back(projected_keypoints.col(keypoint_index),
                            landmark_descriptors.col(keypoint_index));
  }

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);
  LOG(INFO) << "Got " << matches_A_B.size() << " matches.";

  ASSERT_TRUE(matches_A_B.empty());
}


TEST_F(LandmarksToFrameMatcherTest, MatchNoMatchBecauseLandmarksBehindCamera) {
  const size_t kNumKeypoints = 2000u;
  Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, kNumKeypoints);

  Eigen::Matrix3Xd projected_keypoints(3, kNumKeypoints);

  std::random_device random_device;
  std::default_random_engine random_engine(random_device());

  //const size_t kSeed = 23980u;
  //std::srand(23980u);
  std::uniform_int_distribution<int> uniform_dist_image_width(0, camera_->imageWidth() - 1);
  std::uniform_int_distribution<int> uniform_dist_image_height(0, camera_->imageHeight() - 1);
  std::uniform_int_distribution<int> uniform_dist_image_space(
      -image_space_distance_threshold_pixels_, image_space_distance_threshold_pixels_);
  std::uniform_int_distribution<int> uniform_dist_angle(0, 359);


  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    const double keypoint_x = static_cast<double>(uniform_dist_image_width(random_engine));
    const double keypoint_y = static_cast<double>(uniform_dist_image_height(random_engine));

    frame_keypoints(0, keypoint_index) = keypoint_x;
    frame_keypoints(1, keypoint_index) = keypoint_y;

    const double shift_translation_pixels = static_cast<double>(uniform_dist_image_space(random_engine));
    const double shift_angle_radians =
        static_cast<double>(uniform_dist_angle(random_engine)) / 180.0 * 3.14159;

    const double x_shift = shift_translation_pixels * std::cos(shift_angle_radians);
    const double y_shift = shift_translation_pixels * std::sin(shift_angle_radians);

    const double shifted_keypoint_x = std::min<double>(
        std::max<double>(keypoint_x + x_shift, 0.1), camera_->imageWidth() - 0.1);
    const double shifted_keypoint_y = std::min<double>(
        std::max<double>(keypoint_y + y_shift, 0.1), camera_->imageHeight() - 0.1);

    const Eigen::Vector2d shifted_keypoint(shifted_keypoint_x, shifted_keypoint_y);

    Eigen::Vector3d projected_keypoint;
    CHECK(camera_->backProject3(shifted_keypoint, &projected_keypoint));
    projected_keypoints.col(keypoint_index) = projected_keypoint;
  }

  // Arbitrarily picked negative scaling factor.
  const double kScalingFactor = -1e5;

  Eigen::VectorXd random_scale = Eigen::VectorXd::Random(kNumKeypoints);
  random_scale = random_scale.cwiseAbs() * kScalingFactor;

  projected_keypoints = projected_keypoints * random_scale.asDiagonal();

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic>::Random(kDescriptorSizeBytes, kNumKeypoints);

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> landmark_descriptors =
      frame_descriptors;

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    landmarks_.emplace_back(projected_keypoints.col(keypoint_index),
                            landmark_descriptors.col(keypoint_index));
  }

  aslam::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_TRUE(matches_A_B.empty());
}

ASLAM_UNITTEST_ENTRYPOINT
