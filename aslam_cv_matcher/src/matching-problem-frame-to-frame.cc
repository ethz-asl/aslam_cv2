#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-frame-to-frame.h"

namespace aslam {

MatchingProblemFrameToFrame::MatchingProblemFrameToFrame(const VisualFrame& apple_frame,
                                                         const VisualFrame& banana_frame,
                                                         const aslam::Quaternion& q_A_B,
                                                         double image_space_distance_threshold,
                                                         int hamming_distance_threshold)
  : apple_frame_(apple_frame),
    banana_frame_(banana_frame),
    q_A_B_(q_A_B),
    squared_image_space_distance_threshold_px_sq_(image_space_distance_threshold *
                                                  image_space_distance_threshold),
    hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK_GE(hamming_distance_threshold, 0) << "Descriptor distance needs to be positive.";
  CHECK_GE(image_space_distance_threshold, 0.0) << "Image space distance needs to be positive.";

  descriptor_size_bytes_ = apple_frame.getDescriptorSizeBytes();
  CHECK_EQ(descriptor_size_bytes_, banana_frame.getDescriptorSizeBytes()) << "Apple and banana "
      << "frames have different descriptor lengths.";

  // The vertical search band must be at least twice the image space distance.
  vertical_band_halfwidth_pixels_ = static_cast<int>(std::ceil(image_space_distance_threshold));

  CHECK(apple_frame.getCameraGeometry()) << "The iCam is NULL.";
  image_height_apple_frame_ = apple_frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_apple_frame_, 0u) << "The apple frame has zero image rows.";
}

bool MatchingProblemFrameToFrame::doSetup() {
  CHECK_GT(image_height_apple_frame_, 0u) << "The apple frame has zero image rows.";

  size_t num_apple_keypoints = numApples();
  size_t num_banana_keypoints = numBananas();
  valid_apples_.resize(num_apple_keypoints, false);
  valid_bananas_.resize(num_banana_keypoints, false);

  // First, create descriptor wrappers for all descriptors.
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& apple_descriptors =
      apple_frame_.getDescriptors();

  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& banana_descriptors =
      banana_frame_.getDescriptors();

  size_t num_apple_descriptors = static_cast<size_t>(apple_descriptors.cols());
  size_t num_banana_descriptors = static_cast<size_t>(banana_descriptors.cols());
  CHECK_EQ(num_apple_descriptors, num_apple_keypoints) << "Mismatch between the number of apple"
      << " descriptors and the number of apple keypoints.";
  CHECK_EQ(num_banana_descriptors, num_banana_keypoints) << "Mismatch between the number of banana"
      << " descriptors and the number of banana keypoints.";

  apple_descriptors_.clear();
  banana_descriptors_.clear();
  apple_descriptors_.reserve(num_apple_descriptors);
  banana_descriptors_.reserve(num_banana_descriptors);

  // This creates a descriptor wrapper for the given descriptor and allows computing the hamming
  // distance between two descriptors. todo(mbuerki): Think of ways to generalize this for
  // different descriptor types (surf, sift, ...).
  for (size_t apple_descriptor_idx = 0; apple_descriptor_idx < num_apple_descriptors;
      ++apple_descriptor_idx) {
    apple_descriptors_.emplace_back(
        &(apple_descriptors.coeffRef(0, apple_descriptor_idx)), descriptor_size_bytes_);
  }

  for (size_t banana_descriptor_idx = 0; banana_descriptor_idx < num_banana_descriptors;
      ++banana_descriptor_idx) {
    banana_descriptors_.emplace_back(
        &(banana_descriptors.coeffRef(0, banana_descriptor_idx)), descriptor_size_bytes_);
  }

  // Then, create a LUT mapping y coordinates to apple keypoint indices.
  const Eigen::Matrix2Xd& A_keypoints_apple = apple_frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_apple_keypoints), A_keypoints_apple.cols())
    << "The number of apple keypoints does not match the number of columns in the "
    << "apple keypoint matrix.";

  Camera::ConstPtr apple_camera = apple_frame_.getCameraGeometry();
  CHECK(apple_camera);

  for (size_t apple_idx = 0; apple_idx < num_apple_keypoints; ++apple_idx) {
    // Check if the apple is valid.
    const Eigen::Vector2d& apple_keypoint = A_keypoints_apple.col(apple_idx);
    if (apple_camera->isMasked(apple_keypoint)) {
      // This keypoint is masked out and hence not valid.
      valid_apples_[apple_idx] = false;
    } else {
      size_t y_coordinate = static_cast<size_t>(std::floor(apple_keypoint(1)));
      CHECK_LT(y_coordinate, image_height_apple_frame_) << "The y coordinate for apple keypoint "
          << apple_idx << " is bigger than or equal to the number of rows in the image.";

      y_coordinate_to_apple_keypoint_index_map_.insert(std::make_pair(y_coordinate, apple_idx));
      valid_apples_[apple_idx] = true;
    }
  }
  VLOG(20) << "Built LUT for valid apples.";

  // Then, project all banana keypoints into the apple frame.
  const Eigen::Matrix2Xd& banana_keypoints = banana_frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_banana_keypoints), banana_keypoints.cols()) << "The number of "
      << "banana keypoints  and the number of columns in the banana keypoint matrix is different.";

  Camera::ConstPtr banana_camera = banana_frame_.getCameraGeometry();
  CHECK(banana_camera);

  Eigen::Matrix3Xd B_rays_banana = Eigen::Matrix3Xd::Zero(3, num_banana_keypoints);
  // Compute all back projections in the banana frame.
  for (size_t banana_idx = 0; banana_idx < num_banana_keypoints; ++banana_idx) {
    const Eigen::Vector2d& banana_keypoint = banana_keypoints.col(banana_idx);
    if (banana_camera->isMasked(banana_keypoint)) {
      valid_bananas_[banana_idx] = false;
    } else {
      Eigen::Vector3d B_ray_banana;
      banana_camera->backProject3(banana_keypoint, &B_ray_banana);
      B_rays_banana.col(banana_idx) = B_ray_banana;
      valid_bananas_[banana_idx] = true;
    }
  }
  VLOG(20) << "Computed all back projections of bananas in the banana frame.";

  // Rotate all banana rays into the apple frame.
  Eigen::Matrix3Xd A_rays_banana = q_A_B_.getRotationMatrix() * B_rays_banana;

  // Project all banana rays in the apple frame to keypoints.
  A_projected_keypoints_banana_.resize(num_banana_keypoints);

  for (size_t banana_idx = 0; banana_idx < num_banana_keypoints; ++banana_idx) {
    if (valid_bananas_[banana_idx]) {
      Eigen::Vector2d A_keypoint_banana;
      Eigen::Vector3d A_ray_banana = A_rays_banana.col(banana_idx);

      ProjectionResult projection_result =
          apple_camera->project3(A_ray_banana, &A_keypoint_banana);

      // If the banana keypoint projects into the image plane of the apple frame, we accept it.
      if (projection_result.isKeypointVisible()) {
        A_projected_keypoints_banana_[banana_idx] = A_keypoint_banana;
      } else {
        valid_bananas_[banana_idx] = false;
      }
    }
  }

  VLOG(30) << "Done with setup.";
  return true;
}

void MatchingProblemFrameToFrame::getAppleCandidatesForBanana(int banana_index,
                                                              Candidates* candidates) {
  // Get list of apple keypoint indices within some defined distance around the projected banana
  // keypoint and within some defined descriptor distance.
  CHECK_EQ(numApples(), y_coordinate_to_apple_keypoint_index_map_.size()) << "The number of apples"
    << " and the number of apples in the apple LUT differs. This can happen if 1. the apple frame "
    << "was altered between calling setup() and getAppleCandidatesForBanana(...) or 2. if the "
    << "setup() function did not build a valid LUT for apple keypoints.";
  CHECK_LT(banana_index, static_cast<int>(valid_bananas_.size()))
    << "No valid flag for this banana.";
  CHECK_LT(banana_index, static_cast<int>(A_projected_keypoints_banana_.size()))
    << "No projected keypoint for this banana.";

  CHECK_NOTNULL(candidates);
  candidates->clear();
  if (numApples() == 0) {
    LOG(WARNING) << "There are zero apple keypoints.";
    return;
  }

  CHECK_LT(banana_index, static_cast<int>(A_projected_keypoints_banana_.size()))
    << "There is no projected banana keypoint for the given banana index.";
  CHECK_GT(image_height_apple_frame_, 0u) << "The image height of the apple frame is zero.";

  const Eigen::Matrix2Xd& A_keypoints_apple = apple_frame_.getKeypointMeasurements();
  const Eigen::VectorXi* apple_track_ids;
  if (apple_frame_.hasTrackIds()) {
    apple_track_ids = &apple_frame_.getTrackIds();
  } else {
    apple_track_ids = nullptr;
  }

  if (valid_bananas_[banana_index]) {
    const Eigen::Vector2d& A_keypoint_banana = A_projected_keypoints_banana_[banana_index];

    // Get the y coordinate of the projected banana keypoint in the apple frame.
    int projected_banana_keypoint_y_coordinate = static_cast<int>(std::round(A_keypoint_banana(1)));

    // Compute the lower and upper bound of the vertical search window, making it respect the
    // image dimension of the apple frame.
    int y_lower_int = projected_banana_keypoint_y_coordinate - vertical_band_halfwidth_pixels_;
    size_t y_lower = 0;
    if (y_lower_int > 0) y_lower = static_cast<size_t>(y_lower_int);
    CHECK_LT(y_lower, image_height_apple_frame_);

    int y_upper_int = projected_banana_keypoint_y_coordinate + vertical_band_halfwidth_pixels_;
    size_t y_upper = image_height_apple_frame_;
    if (y_upper_int < static_cast<int>(image_height_apple_frame_)) y_upper = y_upper_int;
    CHECK_GE(y_upper, 0u);
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
      CHECK_LT(static_cast<int>(apple_index), A_keypoints_apple.cols());
      const Eigen::Vector2d& apple_keypoint = A_keypoints_apple.col(apple_index);

      double squared_image_space_distance = (apple_keypoint - A_keypoint_banana).squaredNorm();

      if (squared_image_space_distance < squared_image_space_distance_threshold_px_sq_) {
        // This one is within the radius. Compute the descriptor distance.
        int hamming_distance = computeHammingDistance(banana_index, apple_index);

        if (hamming_distance < hamming_distance_threshold_) {
          CHECK_GE(hamming_distance, 0);
          int priority = 0;
          if (apple_track_ids != nullptr) {
            CHECK_LT(static_cast<int>(apple_index), apple_track_ids->rows());
            if ((*apple_track_ids)(apple_index) >= 0) priority = 1;
          }
          candidates->emplace_back(apple_index,
                                   banana_index,
                                   computeMatchScore(hamming_distance),
                                   priority);
        }
      }
    }
  } else {
    LOG(WARNING) << "Banana " << banana_index << " is not valid.";
  }
}

size_t MatchingProblemFrameToFrame::numApples() const {
  return static_cast<size_t>(apple_frame_.getNumKeypointMeasurements());
}

size_t MatchingProblemFrameToFrame::numBananas() const {
  return static_cast<size_t>(banana_frame_.getNumKeypointMeasurements());
}

}  // namespace aslam
