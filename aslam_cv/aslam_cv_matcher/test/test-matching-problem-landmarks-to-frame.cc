#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/entrypoint.h>
#include <aslam/common/pose-types.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/matcher/match.h>
#include <aslam/matcher/matching-engine-non-exclusive.h>
#include <aslam/matcher/matching-problem-landmarks-to-frame.h>
#include <eigen-checks/gtest.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

static const size_t kDescriptorSizeBytes = 48u;
static const size_t kSeed = 233232u;
static const unsigned char kCharThreshold = 128;

typedef Eigen::Matrix<unsigned char, kDescriptorSizeBytes, Eigen::Dynamic> Descriptors;

class LandmarksToFrameMatcherTest : public testing::Test {
 protected:
  virtual void SetUp() {
    camera_ = aslam::PinholeCamera::createTestCamera();
    frame_ = aslam::VisualFrame::createEmptyTestVisualFrame(camera_, 0);

    CHECK_EQ(kDescriptorSizeBytes, frame_->getDescriptorSizeBytes());

    image_space_distance_threshold_pixels_ = 25.0;
    hamming_distance_threshold_ = 1;
  }

  inline void match(aslam::MatchingProblemLandmarksToFrame::MatchesWithScore* matches_A_B) {
    CHECK_NOTNULL(matches_A_B);
    aslam::MatchingProblemLandmarksToFrame::Ptr matching_problem =
        aslam::aligned_shared<aslam::MatchingProblemLandmarksToFrame>(
            *frame_, landmarks_, image_space_distance_threshold_pixels_,
            hamming_distance_threshold_);

    matching_engine_.match(matching_problem.get(), matches_A_B);
  }

  inline void createRandomKeypointsAndLandmarksWithinSearchRegion(
      size_t num_keypoints, double scaling_factor, Eigen::Matrix2Xd* frame_keypoints,
      Eigen::Matrix3Xd* p_C_landmarks) {
    CHECK_NOTNULL(frame_keypoints);
    CHECK_NOTNULL(p_C_landmarks);
    CHECK_GT(camera_->imageWidth(), 0u);
    CHECK_GT(camera_->imageHeight(), 0u);

    *frame_keypoints = Eigen::Matrix2Xd::Zero(2, num_keypoints);
    *p_C_landmarks = Eigen::Matrix3Xd(3, num_keypoints);

    std::default_random_engine random_engine(kSeed);

    std::uniform_int_distribution<int> uniform_dist_image_width(0, camera_->imageWidth() - 1);
    std::uniform_int_distribution<int> uniform_dist_image_height(0, camera_->imageHeight() - 1);
    std::uniform_real_distribution<double> uniform_dist_image_space(
        0, image_space_distance_threshold_pixels_);
    std::uniform_int_distribution<int> uniform_dist_angle(0, 359);

    for (size_t keypoint_index = 0u; keypoint_index < num_keypoints; ++keypoint_index) {
      const double keypoint_x = static_cast<double>(uniform_dist_image_width(random_engine));
      const double keypoint_y = static_cast<double>(uniform_dist_image_height(random_engine));

      *frame_keypoints(0, keypoint_index) = keypoint_x;
      *frame_keypoints(1, keypoint_index) = keypoint_y;

      const double shift_translation_pixels = uniform_dist_image_space(random_engine);

      const double shift_angle_radians =
          static_cast<double>(uniform_dist_angle(random_engine)) / 180.0 * M_PI;

      const double x_shift = shift_translation_pixels * std::cos(shift_angle_radians);
      const double y_shift = shift_translation_pixels * std::sin(shift_angle_radians);

      const double shifted_keypoint_x = std::min<double>(
          std::max<double>(keypoint_x + x_shift, 0.1), camera_->imageWidth() - 0.1);
      const double shifted_keypoint_y = std::min<double>(
          std::max<double>(keypoint_y + y_shift, 0.1), camera_->imageHeight() - 0.1);

      const Eigen::Vector2d shifted_keypoint(shifted_keypoint_x, shifted_keypoint_y);

      Eigen::Vector3d projected_keypoint;
      CHECK(camera_->backProject3(shifted_keypoint, &projected_keypoint));
      p_C_landmarks->col(keypoint_index) = projected_keypoint;
    }

    Eigen::VectorXd random_scale = Eigen::VectorXd::Random(num_keypoints);
    random_scale = random_scale.cwiseAbs() * scaling_factor;

    *p_C_landmarks = *p_C_landmarks * random_scale.asDiagonal();
  }

  inline void shuffleKeypointsAndLandmarkIndices(size_t num_keypoints,
                                                 Eigen::Matrix2Xd* frame_keypoints,
                                                 Descriptors* frame_descriptors,
                                                 Eigen::Matrix3Xd* p_C_landmarks,
                                                 Descriptors* landmark_descriptors) {
    CHECK_GT(num_keypoints, 0u);
    CHECK_NOTNULL(frame_keypoints);
    CHECK_NOTNULL(frame_descriptors);
    CHECK_NOTNULL(p_C_landmarks);
    CHECK_NOTNULL(landmark_descriptors);

    permutation_matrix_keypoints_.resize(num_keypoints);
    permutation_matrix_landmarks_.resize(num_keypoints);

    permutation_matrix_keypoints_.setIdentity();
    permutation_matrix_landmarks_.setIdentity();

    std::random_shuffle(permutation_matrix_keypoints_.indices().data(),
                        permutation_matrix_keypoints_.indices().data() +
                        permutation_matrix_keypoints_.indices().size());
    std::random_shuffle(permutation_matrix_landmarks_.indices().data(),
                        permutation_matrix_landmarks_.indices().data() +
                        permutation_matrix_landmarks_.indices().size());

    CHECK_EQ(permutation_matrix_keypoints_.cols(), frame_keypoints->cols());
    CHECK_EQ(permutation_matrix_keypoints_.rows(), frame_keypoints->cols());
    CHECK_EQ(permutation_matrix_landmarks_.cols(), p_C_landmarks->cols());
    CHECK_EQ(permutation_matrix_landmarks_.rows(), p_C_landmarks->cols());

    *frame_keypoints = *frame_keypoints * permutation_matrix_keypoints_;
    *frame_descriptors = *frame_descriptors * permutation_matrix_keypoints_;

    *p_C_landmarks = *p_C_landmarks * permutation_matrix_landmarks_;
    *landmark_descriptors = *landmark_descriptors * permutation_matrix_landmarks_;
  }

  inline void setKeypointsAndLandmarks(
      const Eigen::Matrix2Xd& frame_keypoints,
      const Descriptors& frame_descriptors,
      const Eigen::Matrix3Xd p_C_landmarks,
      const Descriptors& landmark_descriptors) {
    const size_t num_keypoints = frame_keypoints.cols();

    CHECK_EQ(num_keypoints, frame_descriptors.cols());
    CHECK_EQ(num_keypoints, p_C_landmarks.cols());
    CHECK_EQ(num_keypoints, landmark_descriptors.cols());

    frame_->setKeypointMeasurements(frame_keypoints);
    frame_->setDescriptors(frame_descriptors);

    for (size_t keypoint_index = 0u; keypoint_index < num_keypoints; ++keypoint_index) {
      landmarks_.emplace_back(p_C_landmarks.col(keypoint_index),
                              landmark_descriptors.col(keypoint_index));
    }
  }

  double image_space_distance_threshold_pixels_;
  int hamming_distance_threshold_;

  aslam::VisualFrame::Ptr frame_;
  aslam::LandmarkWithDescriptorList landmarks_;

  aslam::MatchingEngineNonExclusive<aslam::MatchingProblemLandmarksToFrame> matching_engine_;

  aslam::PinholeCamera::Ptr camera_;

  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_matrix_keypoints_;
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> permutation_matrix_landmarks_;
};

TEST_F(LandmarksToFrameMatcherTest, EmptyMatch) {
  aslam::MatchingEngineNonExclusive<aslam::MatchingProblemLandmarksToFrame> matching_engine;

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  EXPECT_TRUE(matches_A_B.empty());
}

TEST_F(LandmarksToFrameMatcherTest, MatchIdentity) {
  const Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, 1);

  Eigen::Vector3d projected_keypoint;
  CHECK(camera_->backProject3(frame_keypoints.col(0), &projected_keypoint));

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1>::Random();
  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> landmark_descriptors =
      frame_descriptors;

  frame_->setKeypointMeasurements(frame_keypoints);
  frame_->setDescriptors(frame_descriptors);

  landmarks_.emplace_back(projected_keypoint, landmark_descriptors.col(0));

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_EQ(1u, matches_A_B.size());

  aslam::MatchingProblemLandmarksToFrame::MatchWithScore match = matches_A_B[0];
  EXPECT_EQ(0, match.getKeypointIndex());
  EXPECT_EQ(0, match.getLandmarkIndex());
  EXPECT_DOUBLE_EQ(1.0, match.getScore());
}

TEST_F(LandmarksToFrameMatcherTest, MatchIdentityWithScale) {
  const Eigen::Matrix2Xd frame_keypoints = Eigen::Matrix2Xd::Zero(2, 1);

  Eigen::Vector3d projected_keypoint;
  CHECK(camera_->backProject3(frame_keypoints.col(0), &projected_keypoint));

  // Arbitrarily picked scaling factor.
  const double kArbitraryScalingFactor = 34.23232;
  projected_keypoint *= kArbitraryScalingFactor;

  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> frame_descriptors =
      Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1>::Random();
  const Eigen::Matrix<unsigned char, kDescriptorSizeBytes, 1> landmark_descriptors =
      frame_descriptors;

  setKeypointsAndLandmarks(frame_keypoints, frame_descriptors, projected_keypoint,
                           landmark_descriptors);

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_EQ(1u, matches_A_B.size());

  aslam::MatchingProblemLandmarksToFrame::MatchWithScore match = matches_A_B[0];
  EXPECT_EQ(0, match.getKeypointIndex());
  EXPECT_EQ(0, match.getLandmarkIndex());
  EXPECT_DOUBLE_EQ(1.0, match.getScore());
}

TEST_F(LandmarksToFrameMatcherTest, MatchRandomly) {
  const size_t kNumKeypoints = 2000u;
  const double kScalingFactor = 1e5;

  Eigen::Matrix2Xd frame_keypoints;
  Eigen::Matrix3Xd p_C_landmarks;

  createRandomKeypointsAndLandmarksWithinSearchRegion(kNumKeypoints, kScalingFactor,
                                                      &frame_keypoints, &p_C_landmarks);

  const Descriptors frame_descriptors = Descriptors::Random(kDescriptorSizeBytes, kNumKeypoints);
  const Descriptors landmark_descriptors = frame_descriptors;

  setKeypointsAndLandmarks(
      frame_keypoints, frame_descriptors, p_C_landmarks, landmark_descriptors);

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  ASSERT_EQ(kNumKeypoints, matches_A_B.size());

  for (const aslam::MatchingProblemLandmarksToFrame::MatchWithScore& match : matches_A_B) {
    EXPECT_EQ(match.getKeypointIndex(),
              match.getLandmarkIndex());
    EXPECT_DOUBLE_EQ(1.0, match.getScore());
  }
}

TEST_F(LandmarksToFrameMatcherTest, MatchRandomlyWithRandomOrder) {
  const size_t kNumKeypoints = 2000u;
  const double kScalingFactor = 1e5;

  Eigen::Matrix2Xd frame_keypoints;
  Eigen::Matrix3Xd p_C_landmarks;

  createRandomKeypointsAndLandmarksWithinSearchRegion(kNumKeypoints, kScalingFactor,
                                                      &frame_keypoints, &p_C_landmarks);

  Descriptors frame_descriptors = Descriptors::Random(kDescriptorSizeBytes, kNumKeypoints);
  Descriptors landmark_descriptors = frame_descriptors;

  shuffleKeypointsAndLandmarkIndices(kNumKeypoints, &frame_keypoints, &frame_descriptors,
                                     &p_C_landmarks, &landmark_descriptors);

  setKeypointsAndLandmarks(frame_keypoints, frame_descriptors, p_C_landmarks,
                           landmark_descriptors);

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);
  const size_t num_matches = matches_A_B.size();

  ASSERT_EQ(kNumKeypoints, num_matches);

  Eigen::MatrixXi result_matrix_keypoints(kNumKeypoints, kNumKeypoints);
  result_matrix_keypoints.setZero();

  Eigen::MatrixXi result_matrix_landmarks(kNumKeypoints, kNumKeypoints);
  result_matrix_landmarks.setZero();

  for (size_t match_index = 0u; match_index < num_matches; ++match_index) {
    const aslam::MatchingProblemLandmarksToFrame::MatchWithScore& match = matches_A_B[match_index];
    const int keypoint_index = match.getKeypointIndex();
    const int landmark_index = match.getLandmarkIndex();
    CHECK_LT(keypoint_index, kNumKeypoints);
    CHECK_LT(landmark_index, kNumKeypoints);

    result_matrix_keypoints(keypoint_index, match_index) = 1;
    result_matrix_landmarks(landmark_index, match_index) = 1;

    EXPECT_DOUBLE_EQ(1.0, match.getScore());
  }

  result_matrix_keypoints = permutation_matrix_keypoints_.toDenseMatrix() * result_matrix_keypoints;
  result_matrix_landmarks = permutation_matrix_landmarks_.toDenseMatrix() * result_matrix_landmarks;

  ASSERT_TRUE(EIGEN_MATRIX_EQUAL(result_matrix_keypoints, result_matrix_keypoints ));
}

TEST_F(LandmarksToFrameMatcherTest, MatchNoMatchesBecauseOfHammingDistance) {
  const size_t kNumKeypoints = 2000u;
  const double kScalingFactor = 1e5;

  Eigen::Matrix2Xd frame_keypoints;
  Eigen::Matrix3Xd projected_keypoints;

  createRandomKeypointsAndLandmarksWithinSearchRegion(kNumKeypoints, kScalingFactor,
                                                      &frame_keypoints, &projected_keypoints);

  Descriptors frame_descriptors = Descriptors::Random(kDescriptorSizeBytes, kNumKeypoints);
  Descriptors landmark_descriptors = frame_descriptors;

  Eigen::Matrix<unsigned char, kNumKeypoints, 1> selection_vector =
      Eigen::Matrix<unsigned char, kNumKeypoints, 1>::Random();

  selection_vector.setConstant(255u);
  selection_vector(1) = 0u;

  const int kNumBytesDifferent = 2;
  hamming_distance_threshold_ = kNumBytesDifferent * 8;
  CHECK_LT(kNumBytesDifferent, frame_descriptors.rows());
  CHECK_LT(kNumBytesDifferent, landmark_descriptors.rows());

  unsigned char kCharThreshold = 128;
  size_t num_made_invalid = 0u;

  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    if (selection_vector(keypoint_index) < kCharThreshold) {
      frame_descriptors.block<kNumBytesDifferent, 1>(0, keypoint_index) =
          Eigen::Matrix<unsigned char, kNumBytesDifferent, 1>::Zero();
      landmark_descriptors.block<kNumBytesDifferent, 1>(0, keypoint_index) =
          Eigen::Matrix<unsigned char, kNumBytesDifferent, 1>::Constant(255u);

      ++num_made_invalid;
    }
  }

  shuffleKeypointsAndLandmarkIndices(kNumKeypoints, &frame_keypoints, &frame_descriptors,
                                     &projected_keypoints, &landmark_descriptors);

  setKeypointsAndLandmarks(frame_keypoints, frame_descriptors, projected_keypoints,
                           landmark_descriptors);

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  const size_t num_matches = matches_A_B.size();
  ASSERT_EQ(num_matches, kNumKeypoints - num_made_invalid);

  Eigen::MatrixXi result_matrix_keypoints(kNumKeypoints, kNumKeypoints);
  result_matrix_keypoints.setZero();

  Eigen::MatrixXi result_matrix_landmarks(kNumKeypoints, kNumKeypoints);
  result_matrix_landmarks.setZero();

  for (size_t match_index = 0u; match_index < num_matches; ++match_index) {
    const aslam::MatchingProblemLandmarksToFrame::MatchWithScore& match = matches_A_B[match_index];
    const int keypoint_index = match.getKeypointIndex();
    const int landmark_index = match.getLandmarkIndex();
    CHECK_LT(keypoint_index, kNumKeypoints);
    CHECK_LT(landmark_index, kNumKeypoints);

    result_matrix_keypoints(keypoint_index, match_index) = 1;
    result_matrix_landmarks(landmark_index, match_index) = 1;

    EXPECT_DOUBLE_EQ(1.0, match.getScore());
  }

  result_matrix_keypoints = permutation_matrix_keypoints_ * result_matrix_keypoints;
  result_matrix_landmarks = permutation_matrix_landmarks_ * result_matrix_landmarks;

  ASSERT_TRUE(EIGEN_MATRIX_NEAR(result_matrix_keypoints, result_matrix_keypoints, 1e-8));
}

TEST_F(LandmarksToFrameMatcherTest, MatchNoMatchBecauseOfSearchBand) {
  const size_t kNumKeypoints = 2000u;
  const double kScalingFactor = 1e5;

  Eigen::Matrix2Xd frame_keypoints;
  Eigen::Matrix3Xd projected_keypoints;

  createRandomKeypointsAndLandmarksWithinSearchRegion(
      kNumKeypoints, kScalingFactor, &frame_keypoints, &projected_keypoints);

  Eigen::Matrix<unsigned char, kNumKeypoints, 1> selection_vector =
      Eigen::Matrix<unsigned char, kNumKeypoints, 1>::Random();

  const double shift_translation_pixels = image_space_distance_threshold_pixels_ + 0.1;

  std::default_random_engine random_engine(kSeed);
  std::uniform_int_distribution<int> uniform_dist_angle(0, 359);

  size_t num_made_invalid = 0u;
  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    if (selection_vector(keypoint_index) < kCharThreshold) {
      Eigen::Vector2d keypoint = frame_keypoints.col(keypoint_index);

      const double shift_angle_radians =
          static_cast<double>(uniform_dist_angle(random_engine)) / 180.0 * M_PI;

      const double x_shift = shift_translation_pixels * std::cos(shift_angle_radians);
      const double y_shift = shift_translation_pixels * std::sin(shift_angle_radians);

      keypoint(0) = keypoint(0) + x_shift;
      keypoint(1) = keypoint(1) + y_shift;

      Eigen::Vector3d projected_keypoint;
      CHECK(camera_->backProject3(keypoint, &projected_keypoint));

      projected_keypoints.col(keypoint_index) = projected_keypoint;

      ++num_made_invalid;
    }
  }

  Descriptors frame_descriptors = Descriptors::Random(kDescriptorSizeBytes, kNumKeypoints);
  Descriptors landmark_descriptors = frame_descriptors;

  shuffleKeypointsAndLandmarkIndices(kNumKeypoints, &frame_keypoints, &frame_descriptors,
                                     &projected_keypoints, &landmark_descriptors);

  setKeypointsAndLandmarks(frame_keypoints, frame_descriptors, projected_keypoints,
                           landmark_descriptors);

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);

  const size_t num_matches = matches_A_B.size();
  ASSERT_EQ(num_matches, kNumKeypoints - num_made_invalid);

  Eigen::MatrixXi result_matrix_keypoints(kNumKeypoints, kNumKeypoints);
  result_matrix_keypoints.setZero();

  Eigen::MatrixXi result_matrix_landmarks(kNumKeypoints, kNumKeypoints);
  result_matrix_landmarks.setZero();

  for (size_t match_index = 0u; match_index < num_matches; ++match_index) {
    const aslam::MatchingProblemLandmarksToFrame::MatchWithScore& match = matches_A_B[match_index];
    const int keypoint_index = match.getKeypointIndex();
    const int landmark_index = match.getLandmarkIndex();
    CHECK_LT(keypoint_index, kNumKeypoints);
    CHECK_LT(landmark_index, kNumKeypoints);

    result_matrix_keypoints(keypoint_index, match_index) = 1;
    result_matrix_landmarks(landmark_index, match_index) = 1;

    EXPECT_DOUBLE_EQ(1.0, match.getScore());
  }

  result_matrix_keypoints = permutation_matrix_keypoints_ * result_matrix_keypoints;
  result_matrix_landmarks = permutation_matrix_landmarks_ * result_matrix_landmarks;

  ASSERT_TRUE(EIGEN_MATRIX_NEAR(result_matrix_keypoints, result_matrix_keypoints, 1e-8));
}

TEST_F(LandmarksToFrameMatcherTest, MatchNoMatchBecauseLandmarksBehindCamera) {
  const size_t kNumKeypoints = 2000u;
  const double kScalingFactor = 1e5;

  Eigen::Matrix2Xd frame_keypoints;
  Eigen::Matrix3Xd projected_keypoints;

  createRandomKeypointsAndLandmarksWithinSearchRegion(kNumKeypoints, kScalingFactor,
                                                      &frame_keypoints, &projected_keypoints);

  Eigen::Matrix<unsigned char, kNumKeypoints, 1> selection_vector =
      Eigen::Matrix<unsigned char, kNumKeypoints, 1>::Random();

  size_t num_made_invalid = 0u;
  for (size_t keypoint_index = 0u; keypoint_index < kNumKeypoints; ++keypoint_index) {
    if (selection_vector(keypoint_index) < kCharThreshold) {
      projected_keypoints.col(keypoint_index) *= -1;
      ++num_made_invalid;
    }
  }

  Descriptors frame_descriptors = Descriptors::Random(kDescriptorSizeBytes, kNumKeypoints);
  Descriptors landmark_descriptors = frame_descriptors;

  shuffleKeypointsAndLandmarkIndices(kNumKeypoints, &frame_keypoints, &frame_descriptors,
                                    &projected_keypoints, &landmark_descriptors);

  setKeypointsAndLandmarks(frame_keypoints, frame_descriptors, projected_keypoints,
                           landmark_descriptors);

  aslam::MatchingProblemLandmarksToFrame::MatchesWithScore matches_A_B;
  match(&matches_A_B);
  const size_t num_matches = matches_A_B.size();
  ASSERT_EQ(num_matches, kNumKeypoints - num_made_invalid);

  Eigen::MatrixXi result_matrix_keypoints(kNumKeypoints, kNumKeypoints);
  result_matrix_keypoints.setZero();

  Eigen::MatrixXi result_matrix_landmarks(kNumKeypoints, kNumKeypoints);
  result_matrix_landmarks.setZero();

  for (size_t match_index = 0u; match_index < num_matches; ++match_index) {
    const aslam::MatchingProblemLandmarksToFrame::MatchWithScore& match = matches_A_B[match_index];
    const int keypoint_index = match.getKeypointIndex();
    const int landmark_index = match.getLandmarkIndex();
    CHECK_LT(keypoint_index, kNumKeypoints);
    CHECK_LT(landmark_index, kNumKeypoints);

    result_matrix_keypoints(keypoint_index, match_index) = 1;
    result_matrix_landmarks(landmark_index, match_index) = 1;

    EXPECT_DOUBLE_EQ(1.0, match.getScore());
  }

  result_matrix_keypoints = permutation_matrix_keypoints_ * result_matrix_keypoints;
  result_matrix_landmarks = permutation_matrix_landmarks_ * result_matrix_landmarks;

  ASSERT_TRUE(EIGEN_MATRIX_NEAR(result_matrix_keypoints, result_matrix_keypoints, 1e-8));
}

ASLAM_UNITTEST_ENTRYPOINT
