#include <aslam/descriptors/feature-descriptor-ref.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-frame-to-frame.h"

namespace aslam {

MatchingProblemFrameToFrame::MatchingProblemFrameToFrame(
                                         const std::shared_ptr<VisualFrame>& apple_frame,
                                         const std::shared_ptr<VisualFrame>& banana_frame,
                                         const Eigen::Matrix3d& C_A_B,
                                         double image_space_distance_threshold,
                                         int hamming_distance_threshold)
  : apple_frame_(apple_frame),
    banana_frame_(banana_frame),
    C_A_B_(C_A_B),
    image_space_distance_threshold_pixels_(image_space_distance_threshold),
    hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK(apple_frame) << "The given apple frame is NULL.";
  CHECK(banana_frame) << "The given banana frame is NULL.";

  CHECK_NEAR(std::abs(C_A_B_.determinant()), 1.0, 1e-8);

  descriptor_size_byes_ = apple_frame->getDescriptorSizeBytes();
  CHECK_EQ(descriptor_size_byes_, banana_frame->getDescriptorSizeBytes()) << "Apple and banana "
      "frames have different descriptor lengths.";

  apple_descriptors_ = apple_frame_->getDescriptors();
  banana_descriptors_ = banana_frame_->getDescriptors();

  // The vertical search band must be at least twice the image space distance.
  vertical_band_halfwidth_pixels_ = static_cast<int>(std::ceil(image_space_distance_threshold));

  CHECK(apple_frame->getCameraGeometry()) << "The iCam is NULL.";
  image_height_apple_frame_ = apple_frame->getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_apple_frame_, 0u) << "The apple frame has zero image rows.";
}

bool MatchingProblemFrameToFrame::doSetup() {
  CHECK(apple_frame_);
  CHECK(banana_frame_);
  CHECK_GT(image_height_apple_frame_, 0u) << "The apple frame has zero image rows.";
  size_t num_apple_keypoints = numApples();
  size_t num_banana_keypoints = numBananas();

  // First, create a LUT mapping y coordinates to apple keypoint indices.
  A_keypoints_apple_ = apple_frame_->getKeypointMeasurements();
  CHECK_EQ(num_apple_keypoints, A_keypoints_apple_.cols()) << "The number of apple keypoints does "
      "not match the number of columns in the apple keypoint matrix.";

  for (size_t keypoint_idx = 0; keypoint_idx < num_apple_keypoints; ++keypoint_idx) {
    size_t y_coordinate = static_cast<size_t>(std::round(A_keypoints_apple_(1, keypoint_idx)));
    CHECK_LT(y_coordinate, image_height_apple_frame_) << "The y coordinate for apple keypoint "
        << keypoint_idx << " is bigger than or equal to the number of rows in the image.";

    y_coordinate_to_apple_keypoint_index_map_.insert(std::make_pair(y_coordinate, keypoint_idx));
  }
  VLOG(20) << "Built LUT for apples.";

  // Then, project all banana keypoints into the apple frame.
  const Eigen::Matrix2Xd& banana_keypoints = banana_frame_->getKeypointMeasurements();

  CHECK_EQ(num_banana_keypoints, banana_keypoints.cols()) << "The number of banana keypoints "
      "and the number of columns in the banan keypoint matrix is different.";

  Camera::ConstPtr banana_cam = banana_frame_->getCameraGeometry();
  CHECK(banana_cam);

  Eigen::Matrix3Xd B_rays_banana = Eigen::Matrix3Xd::Zero(3, num_banana_keypoints);
  // Compute all back projections in the banana frame.
  for (size_t keypoint_idx = 0; keypoint_idx < num_banana_keypoints; ++keypoint_idx) {
    Eigen::Vector3d banana_ray_banana;
    banana_cam->backProject3(banana_keypoints.col(keypoint_idx), &banana_ray_banana);
    // Normalize the ray... because it may not be normalized.
    banana_ray_banana.normalize();
    B_rays_banana.col(keypoint_idx) = banana_ray_banana;
  }
  VLOG(20) << "Computed all back projections of bananas in the banana frame.";

  // Rotate all banana rays into the apple frame.
  Eigen::Matrix3Xd A_rays_banana = C_A_B_ * B_rays_banana;

  Camera::ConstPtr iCam = apple_frame_->getCameraGeometry();
  CHECK(iCam);

  // Project all banana rays in the apple frame to keypoints.
  A_projected_keypoints_banana_.resize(num_banana_keypoints);
  for (size_t keypoint_idx = 0; keypoint_idx < num_banana_keypoints; ++keypoint_idx) {
    Eigen::Vector2d apple_keypoint_banana;
    Eigen::Vector3d apple_ray_banana = A_rays_banana.col(keypoint_idx);
    ProjectionResult projection_result =
        iCam->project3(apple_ray_banana, &(A_projected_keypoints_banana_[keypoint_idx].keypoint));

    A_projected_keypoints_banana_[keypoint_idx].valid = projection_result.isKeypointVisible();
  }

  VLOG(30) << "Done with setup.";
  return true;
}

void MatchingProblemFrameToFrame::getAppleCandidatesForBanana(int banana_index,
                                                              Candidates* candidates) {
  // Get list of apple keypoint indices within some defined distance around the projected banana
  // keypoint and within some defined descriptor distance.
  CHECK(apple_frame_) << "The apple frame is NULL.";
  CHECK_GT(numApples(), 0) << "There are zero apple keypoints.";
  CHECK_EQ(numApples(), y_coordinate_to_apple_keypoint_index_map_.size()) << "The number of apples"
      " and the number of apples in the apple LUT differs. This can happen if 1. the apple frame "
      "was altered between calling setup() and getAppleCandidatesForBanana(...) or 2. if the "
      "setup() function did not build a valid LUT for apple keypoints.";
  CHECK_NOTNULL(candidates);
  candidates->clear();

  CHECK_LT(banana_index, banana_descriptors_.cols()) << "There is no descriptor for the given"
      " banana keypoint.";
  CHECK_LT(banana_index, A_projected_keypoints_banana_.size()) << "There is no projected banana "
      "keypoint for the given banana index.";
  CHECK_GT(image_height_apple_frame_, 0) << "The image height of the apple frame is zero.";

  // This creates a descriptor wrapper for the given descriptor and allows computing the hamming
  // distance between two descriptors. todo(mbuerki): Think of ways to generalize this for
  // different descriptor types (surf, sift, ...).
  CHECK_LT(banana_index, banana_descriptors_.cols());
  common::FeatureDescriptorConstRef banana_descriptor(
      &(banana_descriptors_.coeffRef(0, banana_index)), descriptor_size_byes_);

  const ProjectedKeypoint& A_projected_keypoint_banana =
      A_projected_keypoints_banana_[banana_index];
  if (A_projected_keypoint_banana.valid) {
    Eigen::Vector2d A_keypoint_banana = A_projected_keypoint_banana.keypoint;

    // Get the y coordinate of the projected banana keypoint in the apple frame.
    int projected_banana_keypoint_y_coordinate =
        static_cast<int>(std::round(A_keypoint_banana(1)));
    CHECK_GE(projected_banana_keypoint_y_coordinate, 0) << "The y coordinate of the projected "
        "banana keypoint is < 0.";
    CHECK_LT(projected_banana_keypoint_y_coordinate, image_height_apple_frame_) << "The y "
        "coordinate of the projected banana keypoint lies beyond the image boundary of the apple "
        "frame.";

    // Compute the lower and upper bound of the vertical search window, making it respect the
    // image dimension of the apple frame.
    int y_lower_int = projected_banana_keypoint_y_coordinate - vertical_band_halfwidth_pixels_;
    size_t y_lower = 0;
    if (y_lower_int > 0) y_lower = static_cast<size_t>(y_lower_int);
    CHECK_LT(y_lower, image_height_apple_frame_);

    int y_upper_int = projected_banana_keypoint_y_coordinate + vertical_band_halfwidth_pixels_;
    size_t y_upper = image_height_apple_frame_;
    if (y_upper_int < static_cast<int>(image_height_apple_frame_)) y_upper = y_upper_int;
    CHECK_GE(y_upper, 0);
    CHECK_GT(y_upper, y_lower);

    auto it_lower = y_coordinate_to_apple_keypoint_index_map_.lower_bound(y_lower);
    if (it_lower == y_coordinate_to_apple_keypoint_index_map_.end()) {
      // All keypoints in the apple frame lie below the image band -> zero candiates!
      return;
    }

    auto it_upper = y_coordinate_to_apple_keypoint_index_map_.lower_bound(y_upper);
    if (it_upper != y_coordinate_to_apple_keypoint_index_map_.end()) {
      // Pointing to a valid keypoint -> need to increment this because this needs to go one
      // beyond the border (i.e. == end() in normal for loop over a vector).
      ++it_upper;
    }

    for (auto it = it_lower; it != it_upper; ++it) {
      // Go over all the apple keyponts and compute image space distance to the projected banana
      // keypoint.
      size_t apple_index = it->second;
      CHECK_LT(apple_index, A_keypoints_apple_.cols());
      Eigen::Vector2d apple_keypoint = A_keypoints_apple_.col(apple_index);

      double image_space_distance = (apple_keypoint - A_keypoint_banana).norm();

      if (image_space_distance < image_space_distance_threshold_pixels_) {
        // This one is within the radius. Compute the descriptor distance.
        CHECK_LT(apple_index, apple_descriptors_.cols());
        common::FeatureDescriptorConstRef apple_descriptor(
            &(apple_descriptors_.coeffRef(0, apple_index)), descriptor_size_byes_);
        int hamming_distance = common::GetNumBitsDifferent(banana_descriptor, apple_descriptor);

        if (hamming_distance < hamming_distance_threshold_) {
          CHECK_GE(hamming_distance, 0);
          candidates->emplace_back(apple_index,
                                   computeMatchScore(image_space_distance, hamming_distance));
        }
      }
    }
  } else {
    LOG(WARNING) << "Banana " << banana_index << " is not valid in the apple tree.";
  }

}

double MatchingProblemFrameToFrame::computeScore(int apple_index, int banana_index) {
  CHECK_LT(apple_index, A_keypoints_apple_.cols()) << "apple index out of bounds.";
  CHECK_LT(apple_index, A_projected_keypoints_banana_.size()) << "banana index out of bounds.";

  double score = 0.0;

  const ProjectedKeypoint& projected_banana_keypoint = A_projected_keypoints_banana_[banana_index];
  if (projected_banana_keypoint.valid) {
    const Eigen::Vector2d& apple_keypoint = A_keypoints_apple_.col(apple_index);
    const Eigen::Vector2d& banana_keypoint = projected_banana_keypoint.keypoint;

    double image_space_distance = (apple_keypoint - banana_keypoint).norm();

    if (image_space_distance < image_space_distance_threshold_pixels_) {
      // This one is within the radius. Compute the descriptor distance.

      // This creates a descriptor wrapper for the given descriptor and allows computing the hamming
      // distance between two descriptors. todo(mbuerki): Think of ways to generalize this for
      // different descriptor types (surf, sift, ...).
      CHECK_LT(banana_index, banana_descriptors_.cols());
      common::FeatureDescriptorConstRef banana_descriptor(
          &(banana_descriptors_.coeffRef(0, banana_index)), descriptor_size_byes_);

      CHECK_LT(apple_index, apple_descriptors_.cols());
      common::FeatureDescriptorConstRef apple_descriptor(
          &(apple_descriptors_.coeffRef(0, apple_index)), descriptor_size_byes_);

      int hamming_distance = common::GetNumBitsDifferent(banana_descriptor, apple_descriptor);

      if (hamming_distance < hamming_distance_threshold_) {
        CHECK_GE(hamming_distance, 0);
        score = computeMatchScore(image_space_distance, hamming_distance);
      }
    }
  }
  return score;
}

void MatchingProblemFrameToFrame::setBestMatches(const Matches& bestMatches) {
  LOG(FATAL) << "Not implemented.";
}

size_t MatchingProblemFrameToFrame::numApples() const {
  CHECK(apple_frame_);
  return static_cast<size_t>(apple_frame_->getNumKeypointMeasurements());
}

size_t MatchingProblemFrameToFrame::numBananas() const {
  CHECK(banana_frame_);
  return static_cast<size_t>(banana_frame_->getNumKeypointMeasurements());
}

}  // namespace aslam
