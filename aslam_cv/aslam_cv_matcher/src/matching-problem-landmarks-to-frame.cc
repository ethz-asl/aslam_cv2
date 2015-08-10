#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-landmarks-to-frame.h"

namespace aslam {

MatchingProblemLandmarksToFrame::MatchingProblemLandmarksToFrame(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : landmarks_(landmarks), frame_(frame),
    squared_image_space_distance_threshold_pixels_squared_(image_space_distance_threshold_pixels *
                                                           image_space_distance_threshold_pixels),
    hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK_GT(hamming_distance_threshold, 0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0) << "Image space distance needs to be positive.";

  descriptor_size_byes_ = frame.getDescriptorSizeBytes();
  descriptor_size_bits_ = static_cast<int>(frame.getDescriptorSizeBytes() * 8u);

  // The vertical search band must be at least twice the image space distance.
  vertical_band_halfwidth_pixels_ = static_cast<int>(std::ceil(image_space_distance_threshold_pixels));

  CHECK(frame.getCameraGeometry()) << "The camer of the visual frame is NULL.";
  image_height_frame_ = frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_frame_, 0u) << "The visual frame has zero image rows.";
}

bool MatchingProblemLandmarksToFrame::doSetup() {
  CHECK_GT(image_height_frame_, 0u) << "The visual frame has zero image rows.";

  const size_t num_frame_keypoints = numApples();
  const size_t num_landmarks = numBananas();
  valid_frame_keypoints_.resize(num_frame_keypoints, false);
  valid_landmarks_.resize(num_landmarks, false);

  // First, create descriptor wrappers for all descriptors.
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& frame_descriptors =
      frame_.getDescriptors();

  const size_t num_frame_descriptors = static_cast<size_t>(frame_descriptors.cols());
  CHECK_EQ(num_frame_descriptors, num_frame_keypoints) << "Mismatch between the number of "
      << "descriptors and the number of keypoints in the visual frame.";

  frame_descriptors_.clear();
  landmark_descriptors_.clear();
  frame_descriptors_.reserve(num_frame_descriptors);
  landmark_descriptors_.reserve(num_landmarks);

  // This creates a descriptor wrapper for the given descriptor and allows computing the hamming
  // distance between two descriptors. todo(mbuerki): Think of ways to generalize this for
  // different descriptor types (surf, sift, ...).
  for (size_t frame_descriptor_idx = 0u; frame_descriptor_idx < num_frame_descriptors;
      ++frame_descriptor_idx) {
    frame_descriptors_.emplace_back(
        &(frame_descriptors.coeffRef(0, frame_descriptor_idx)), descriptor_size_byes_);
  }

  for (size_t landmark_descriptor_idx = 0u; landmark_descriptor_idx < num_landmarks;
      ++landmark_descriptor_idx) {
    landmark_descriptors_.emplace_back(
        landmarks_[landmark_descriptor_idx].getDescriptor().data(), descriptor_size_byes_);
  }

  // Then, create a LUT mapping y coordinates to frame keypoint indices.
  const Eigen::Matrix2Xd& C_keypoints_frame = frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_frame_keypoints), C_keypoints_frame.cols())
    << "The number of keypoints in the visual frame does not match the number "
    << "of columns in the keypoint matrix.";

  Camera::ConstPtr camera = frame_.getCameraGeometry();
  CHECK(camera);

  for (size_t keypoint_idx = 0u; keypoint_idx < num_frame_keypoints; ++keypoint_idx) {
    // Check if the keypoint is valid.
    const Eigen::Vector2d& keypoint = C_keypoints_frame.col(keypoint_idx);
    if (camera->isMasked(keypoint)) {
      // This keypoint is masked out and hence not valid.
      valid_frame_keypoints_[keypoint_idx] = false;
    } else {
      size_t y_coordinate = static_cast<size_t>(std::floor(keypoint(1)));
      CHECK_LT(y_coordinate, image_height_frame_) << "The y coordinate for keypoint "
          << keypoint_idx << " is bigger than or equal to the number of rows in the image.";

      y_coordinate_to_keypoint_index_map_.insert(std::make_pair(y_coordinate, keypoint_idx));
      valid_frame_keypoints_[keypoint_idx] = true;
    }
  }
  VLOG(3) << "Built LUT for valid keypoints.";

  C_projected_landmark_keypoints_.resize(num_landmarks);

  // Then, project all landmarks into the visual frame.
  for (size_t landmark_idx = 0u; landmark_idx < num_landmarks; ++landmark_idx) {
    aslam::ProjectionResult projection_result = camera->project3(
        landmarks_[landmark_idx].get_t_G_C(),
        &C_projected_landmark_keypoints_[landmark_idx]);

    if (projection_result.isKeypointVisible()) {
      valid_landmarks_[landmark_idx] = true;
    } else {
      C_projected_landmark_keypoints_[landmark_idx].setZero();
      valid_landmarks_[landmark_idx] = false;
    }
  }

  VLOG(3) << "Computed all projections of landmarks into the visual frame.";
  return true;
}

void MatchingProblemLandmarksToFrame::getAppleCandidatesForBanana(
    int landmark_index, Candidates* candidates) {
  // Get list of keypoint indices within some defined distance around the projected landmark
  // keypoint and within some defined descriptor distance.
  CHECK_EQ(numApples(), y_coordinate_to_keypoint_index_map_.size()) << "The number of apples"
      " and the number of apples in the apple LUT differs. This can happen if 1. the visual frame "
      "was altered between calling setup() and getAppleCandidatesForBanana(...) or 2. if the "
      "setup() function did not build a valid LUT for the visual frame keypoints.";
  CHECK_LT(landmark_index, static_cast<int>(valid_landmarks_.size()))
    << "No valid flag for this landmark.";
  CHECK_LT(landmark_index, static_cast<int>(C_projected_landmark_keypoints_.size()))
    << "No projected keypoint for this landmark.";

  CHECK_NOTNULL(candidates);
  candidates->clear();
  if (numApples() == 0) {
    LOG(WARNING) << "There are zero visual frame keypoints.";
    return;
  }

  CHECK_GT(image_height_frame_, 0u) << "The image height of the visual frame is zero.";

  const Eigen::Matrix2Xd& C_keypoints_frame = frame_.getKeypointMeasurements();

  if (valid_landmarks_[landmark_index]) {
    const Eigen::Vector2d& C_keypoint_landmark = C_projected_landmark_keypoints_[landmark_index];

    // Get the y coordinate of the projected landmark keypoint in the visual frame.
    int projected_landmark_keypoint_y_coordinate = static_cast<int>(std::round(C_keypoint_landmark(1)));

    // Compute the lower and upper bound of the vertical search window, making it respect the
    // image dimension of the visual frame.
    int y_lower_int = projected_landmark_keypoint_y_coordinate - vertical_band_halfwidth_pixels_;
    size_t y_lower = 0u;
    if (y_lower_int > 0) {
      y_lower = static_cast<size_t>(y_lower_int);
    }
    CHECK_LT(y_lower, image_height_frame_) << "The y coordinate of the lower search band bound "
        << "is bigger than or equal to the number of rows in the image.";

    int y_upper_int = projected_landmark_keypoint_y_coordinate + vertical_band_halfwidth_pixels_;
    size_t y_upper = image_height_frame_;
    if (y_upper_int < static_cast<int>(image_height_frame_)) {
      y_upper = y_upper_int;
    }
    CHECK_GE(y_upper, 0u) << "The y coordinate of the upper search band bound "
        << "is negative.";
    CHECK_GT(y_upper, y_lower) << "The y coordinate of the upper search band bound "
        << " is bigger than or equal to the number of rows in the image.";

    std::multimap<size_t, size_t>::iterator it_lower =
        y_coordinate_to_keypoint_index_map_.lower_bound(y_lower);
    if (it_lower == y_coordinate_to_keypoint_index_map_.end()) {
      // All keypoints in the visual frame lie below the image band -> zero candidates!
      return;
    }

    std::multimap<size_t, size_t>::iterator it_upper =
        y_coordinate_to_keypoint_index_map_.lower_bound(y_upper);
    if (it_upper != y_coordinate_to_keypoint_index_map_.end()) {
      // Pointing to a valid keypoint -> need to increment this because this needs to go one
      // beyond the border (i.e. == end() in normal for loop over a vector).
      ++it_upper;
    }

    for (std::multimap<size_t, size_t>::iterator it = it_lower; it != it_upper; ++it) {
      // Go over all the frame keypoints and compute image space distance to the projected landmark
      // keypoint.
      size_t frame_keypoint_index = it->second;
      CHECK_LT(static_cast<int>(frame_keypoint_index), C_keypoints_frame.cols());
      const Eigen::Vector2d& frame_keypoint = C_keypoints_frame.col(frame_keypoint_index);

      double squared_image_space_distance = (frame_keypoint - C_keypoint_landmark).squaredNorm();

      if (squared_image_space_distance < squared_image_space_distance_threshold_pixels_squared_) {
        // This one is within the radius. Compute the descriptor distance.
        int hamming_distance = computeHammingDistance(landmark_index, frame_keypoint_index);

        if (hamming_distance < hamming_distance_threshold_) {
          CHECK_GE(hamming_distance, 0);
          int priority = 0;
          candidates->emplace_back(frame_keypoint_index,
                                   landmark_index,
                                   computeMatchScore(hamming_distance),
                                   priority);
        }
      }
    }
  } else {
    LOG(WARNING) << "Landmark " << landmark_index << " is not valid.";
  }
}

size_t MatchingProblemLandmarksToFrame::numApples() const {
  return frame_.getNumKeypointMeasurements();
}

size_t MatchingProblemLandmarksToFrame::numBananas() const {
  return landmarks_.size();
}

}  // namespace aslam
