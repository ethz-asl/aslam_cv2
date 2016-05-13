#include <algorithm>

#include <aslam/common/statistics/statistics.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-landmarks-to-frame-multimap.h"

namespace aslam {

MatchingProblemLandmarksToFrameMultimap::MatchingProblemLandmarksToFrameMultimap(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : MatchingProblemLandmarksToFrame(
      frame, landmarks, image_space_distance_threshold_pixels, hamming_distance_threshold) {
  CHECK_GT(hamming_distance_threshold, 0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0)
    << "Image space distance needs to be positive.";

  // The vertical search band must be at least twice the image space distance.
  vertical_band_halfwidth_pixels_ = static_cast<int>(
      std::ceil(image_space_distance_threshold_pixels));

  CHECK(frame.getCameraGeometry()) << "The camera of the visual frame is NULL.";
  image_height_frame_ = frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_frame_, 0u) << "The visual frame has zero image rows.";
  CHECK_GT(descriptor_size_bytes_, 0);
  CHECK_GT(descriptor_size_bits_, 0);
}

bool MatchingProblemLandmarksToFrameMultimap::doSetup() {
  CHECK_GT(image_height_frame_, 0u) << "The visual frame has zero image rows.";

  const size_t num_frame_keypoints = numApples();
  const size_t num_landmarks = numBananas();
  is_frame_keypoint_valid_.resize(num_frame_keypoints, false);
  is_landmark_valid_.resize(num_landmarks, false);

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
  // distance between two descriptors.
  for (size_t frame_descriptor_idx = 0u; frame_descriptor_idx < num_frame_descriptors;
      ++frame_descriptor_idx) {
    frame_descriptors_.emplace_back(
        &(frame_descriptors.coeffRef(0, frame_descriptor_idx)), descriptor_size_bytes_);
  }

  for (size_t landmark_descriptor_idx = 0u; landmark_descriptor_idx < num_landmarks;
      ++landmark_descriptor_idx) {
    landmark_descriptors_.emplace_back(
        landmarks_[landmark_descriptor_idx].getDescriptor().data(), descriptor_size_bytes_);
  }

  // Then, create a LUT mapping y coordinates to frame keypoint indices.
  const Eigen::Matrix2Xd& C_keypoints_frame = frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_frame_keypoints), C_keypoints_frame.cols())
    << "The number of keypoints in the visual frame does not match the number "
    << "of columns in the keypoint matrix.";

  Camera::ConstPtr camera = frame_.getCameraGeometry();
  CHECK(camera);

  VLOG(3) << "Adding " << num_frame_keypoints << " keypoints to the LUT.";
  size_t num_added = 0u;
  for (size_t keypoint_idx = 0u; keypoint_idx < num_frame_keypoints; ++keypoint_idx) {
    // Check if the keypoint is valid.
    const Eigen::Vector2d& keypoint = C_keypoints_frame.col(keypoint_idx);
    if (camera->isMasked(keypoint)) {
      // This keypoint is masked out and hence not valid.
      is_frame_keypoint_valid_[keypoint_idx] = false;
    } else {
      size_t y_coordinate = static_cast<size_t>(std::floor(keypoint(1)));
      CHECK_LT(y_coordinate, image_height_frame_) << "The y coordinate for keypoint "
          << keypoint_idx << " is bigger than or equal to the number of rows in the image.";

      //y_coordinate_to_keypoint_index_map_.insert(std::make_pair(y_coordinate, keypoint_idx));
      y_coordinate_to_keypoint_index_map_.emplace(y_coordinate, keypoint_idx);

      is_frame_keypoint_valid_[keypoint_idx] = true;
      ++num_added;
    }
  }
  VLOG(3) << "Built LUT for valid keypoints (" << num_added << ").";

  projected_landmark_keypoints_.resize(num_landmarks);

  // Then, project all landmarks into the visual frame.
  size_t num_valid = 0u;
  size_t num_invalid = 0u;
  VLOG(3) << "Projecting " << num_landmarks << " into the visual frame.";
  for (size_t landmark_idx = 0u; landmark_idx < num_landmarks; ++landmark_idx) {
    aslam::ProjectionResult projection_result = camera->project3(
        landmarks_[landmark_idx].get_p_C_landmark(),
        &projected_landmark_keypoints_[landmark_idx]);

    if (projection_result.isKeypointVisible()) {
      is_landmark_valid_[landmark_idx] = true;
      ++num_valid;
    } else {
      VLOG(5) << "Projection of landmark " << landmark_idx << " is invalid. "
          << std::endl << projection_result;
      projected_landmark_keypoints_[landmark_idx].setZero();
      is_landmark_valid_[landmark_idx] = false;
      ++num_invalid;
    }
  }

  VLOG(3) << "Computed all projections of landmarks into the visual frame. (valid/invalid) ("
      << num_valid << "/" << num_invalid << ")";
  return true;
}

void MatchingProblemLandmarksToFrameMultimap::getAppleCandidatesForBanana(
    int landmark_index, Candidates* candidates) {
  // Get list of keypoint indices within some defined distance around the projected landmark
  // keypoint and within some defined descriptor distance.
  CHECK_NOTNULL(candidates)->clear();
  CHECK_EQ(numApples(), y_coordinate_to_keypoint_index_map_.size()) << "The number of apples"
      << " and the number of apples in the apple LUT differs. This can happen if 1. the visual "
      << "frame was altered between calling setup() and getAppleCandidatesForBanana(...) or 2. "
      << "if the setup() function did not build a valid LUT for the visual frame keypoints.";
  CHECK_LT(landmark_index, static_cast<int>(is_landmark_valid_.size()))
    << "No valid flag for the landmark with index " << landmark_index << ".";
  CHECK_LT(landmark_index, static_cast<int>(projected_landmark_keypoints_.size()))
    << "No projected keypoint for the landmark with index " << landmark_index << ".";

  if (numApples() == 0) {
    LOG(WARNING) << "There are zero visual frame keypoints.";
    return;
  }

  CHECK_GT(image_height_frame_, 0u) << "The image height of the visual frame is zero.";

  const Eigen::Matrix2Xd& keypoints_frame = frame_.getKeypointMeasurements();

  if (is_landmark_valid_[landmark_index]) {
    const Eigen::Vector2d& keypoint_landmark = projected_landmark_keypoints_[landmark_index];

    // Get the y coordinate of the projected landmark keypoint in the visual frame.
    const int projected_landmark_keypoint_y_coordinate =
        static_cast<int>(std::round(keypoint_landmark(1)));

    // Compute the lower and upper bound of the vertical search window, making it respect the
    // image dimension of the visual frame.
    const size_t y_lower = static_cast<size_t>(std::max(
        projected_landmark_keypoint_y_coordinate - vertical_band_halfwidth_pixels_, 0));
    CHECK_LT(y_lower, image_height_frame_) << "The y coordinate of the lower search band bound "
        << "is bigger than or equal to the number of rows in the image.";

    // +1 because we will retrieve the lower_bound on this y-coordinate.
    // So let's say there are 10 keypoints on y-coordinate 420 and 10
    // keypoints on y-coordinate 421. If the search should go up to and including
    // row 420, it_upper needs to point at the first keypoint at row 421, such
    // that we iterate over all keypoints of row 420 (note: it_upper acts
    // like a .end() here and needs to point one element beyond the last one
    // we want to have included in our loop).
    // Note: It's ok if y_upper > image_height because then we will just
    // get it_upper = .end() which is what we want in this case.
    const size_t y_upper = static_cast<size_t>(projected_landmark_keypoint_y_coordinate +
                                         vertical_band_halfwidth_pixels_ + 1);

    std::multimap<size_t, size_t>::iterator it_lower =
        y_coordinate_to_keypoint_index_map_.lower_bound(y_lower);
    if (it_lower == y_coordinate_to_keypoint_index_map_.end()) {
      // All keypoints in the visual frame lie below the image band -> zero candidates!
      return;
    }

    std::multimap<size_t, size_t>::iterator it_upper =
        y_coordinate_to_keypoint_index_map_.lower_bound(y_upper);

    std::unordered_set<size_t> keypoints_indices_looked_at;

    if (it_lower == it_upper) {
      aslam::statistics::StatsCollector zero_in_search_band(
          "aslam::MatchingProblemLandmarksToFrameMultimap: 0 keypoints in search band");
      zero_in_search_band.IncrementOne();

      VLOG(5) << "Got 0 keypoints to in the search band of landmark " << landmark_index;
    }
    for (std::multimap<size_t, size_t>::iterator it = it_lower; it != it_upper; ++it) {
      // Go over all the frame keypoints and compute image space distance to the projected landmark
      // keypoint.
      const size_t frame_keypoint_index = it->second;
      keypoints_indices_looked_at.insert(frame_keypoint_index);

      CHECK_LT(static_cast<int>(frame_keypoint_index), keypoints_frame.cols());
      const Eigen::Vector2d& frame_keypoint = keypoints_frame.col(frame_keypoint_index);

      const double squared_image_space_distance =
          (frame_keypoint - keypoint_landmark).squaredNorm();

      if (squared_image_space_distance < squared_image_space_distance_threshold_pixels_squared_) {
        // This one is within the radius. Compute the descriptor distance.
        const int hamming_distance = computeHammingDistance(landmark_index, frame_keypoint_index);

        if (hamming_distance < hamming_distance_threshold_) {
          CHECK_GE(hamming_distance, 0);
          int priority = 0;
          candidates->emplace_back(
              frame_keypoint_index, landmark_index, computeMatchScore(hamming_distance), priority);
        } else {
          aslam::statistics::StatsCollector outside_hamming(
              "aslam::MatchingProblemLandmarksToFrameMultimap: Hamming distance too big");
          outside_hamming.IncrementOne();
        }
      } else {
        aslam::statistics::StatsCollector keypoint_outside_search_box(
            "aslam::MatchingProblemLandmarksToFrameMultimap: Keypoint outside search box");
        keypoint_outside_search_box.IncrementOne();
      }
    }
  } else {
    VLOG(5) << "Landmark " << landmark_index << " is not valid.";
  }
}

}  // namespace aslam
