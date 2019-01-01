#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-landmarks-to-frame.h"

DECLARE_bool(matcher_store_all_tested_pairs);

namespace aslam {

/*
MatchingProblemLandmarksToFrame::MatchingProblemLandmarksToFrame(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : landmarks_(landmarks), frame_(frame),
    descriptor_num_elements_(frame.getDescriptorSizeBytes()),
    descriptor_size_bits_(static_cast<int>(frame.getDescriptorSizeBytes() * 8u)),
    squared_image_space_distance_threshold_px_sq_(
        image_space_distance_threshold_pixels * image_space_distance_threshold_pixels),
    hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK_GT(hamming_distance_threshold, 0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0)
    << "Image space distance needs to be positive.";
  CHECK(frame.getCameraGeometry()) << "The camera of the visual frame is NULL.";

  image_height_ = frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_, 0u) << "The visual frame has zero image rows.";
  CHECK_GT(descriptor_num_elements_, 0);
  CHECK_GT(descriptor_size_bits_, 0);
}

size_t MatchingProblemLandmarksToFrame::numApples() const {
  return frame_.getNumKeypointMeasurements();
}

size_t MatchingProblemLandmarksToFrame::numBananas() const {
  return landmarks_.size();
}*/

template <>
MatchingProblemLandmarksToFrame<unsigned char>::MatchingProblemLandmarksToFrame(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList<unsigned char>& landmarks,
    double image_space_distance_threshold_pixels,
    double descriptor_distance_threshold)
  : landmarks_(landmarks), frame_(frame),
    descriptor_num_elements_(frame.getDescriptorSizeBytes()),
    squared_image_space_distance_threshold_px_sq_(
        image_space_distance_threshold_pixels * image_space_distance_threshold_pixels),
    descriptor_distance_threshold_(descriptor_distance_threshold) {
    //hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK_GT(descriptor_distance_threshold, 0.0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0)
    << "Image space distance needs to be positive.";
  CHECK(frame.getCameraGeometry()) << "The camera of the visual frame is NULL.";

  image_height_ = frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_, 0u) << "The visual frame has zero image rows.";
  CHECK_GT(descriptor_num_elements_, 0);
}


template <>
MatchingProblemLandmarksToFrame<float>::MatchingProblemLandmarksToFrame(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList<float>& landmarks,
    double image_space_distance_threshold_pixels,
    double descriptor_distance_threshold)
  : landmarks_(landmarks), frame_(frame),
    descriptor_num_elements_(frame.getFloatDescriptorNumElements()),
    squared_image_space_distance_threshold_px_sq_(
        image_space_distance_threshold_pixels * image_space_distance_threshold_pixels),
    descriptor_distance_threshold_(descriptor_distance_threshold) {
    //hamming_distance_threshold_(hamming_distance_threshold) {
  CHECK_GT(descriptor_distance_threshold, 0.0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0)
    << "Image space distance needs to be positive.";
  CHECK(frame.getCameraGeometry()) << "The camera of the visual frame is NULL.";

  image_height_ = frame.getCameraGeometry()->imageHeight();
  CHECK_GT(image_height_, 0u) << "The visual frame has zero image rows.";
  CHECK_GT(descriptor_num_elements_, 0);
}

template <>
void MatchingProblemLandmarksToFrame<unsigned char>::setupValidVectorsAndDescriptors() {
  CHECK_GT(image_height_, 0u) << "The visual frame has zero image rows.";

  const size_t num_keypoints = numApples();
  const size_t num_landmarks = numBananas();
  is_frame_keypoint_valid_.resize(num_keypoints, false);
  is_landmark_valid_.resize(num_landmarks, false);

  if (FLAGS_matcher_store_all_tested_pairs) {
    all_tested_pairs_.resize(num_landmarks);
  }

  // First, create descriptor wrappers for all descriptors.
  const Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& frame_descriptors =
      frame_.getDescriptors();
  CHECK_EQ(frame_descriptors.rows(), descriptor_num_elements_);

  const size_t num_frame_descriptors = static_cast<size_t>(frame_descriptors.cols());
  CHECK_EQ(num_frame_descriptors, num_keypoints) << "Mismatch between the number of "
      << "descriptors and the number of keypoints in the visual frame.";

  frame_descriptors_.clear();
  landmark_descriptors_.clear();
  frame_float_descriptors_.clear();
  landmark_float_descriptors_.clear();
  frame_descriptors_.reserve(num_frame_descriptors);
  landmark_descriptors_.reserve(num_landmarks);

  // This creates a descriptor wrapper for the given descriptor and allows computing the Hamming
  // distance between two descriptors.
  for (size_t frame_descriptor_idx = 0u; frame_descriptor_idx < num_frame_descriptors;
      ++frame_descriptor_idx) {
    frame_descriptors_.emplace_back(
        &(frame_descriptors.coeffRef(0, frame_descriptor_idx)), descriptor_num_elements_);
  }

  for (size_t landmark_descriptor_idx = 0u; landmark_descriptor_idx < num_landmarks;
      ++landmark_descriptor_idx) {
    CHECK_EQ(landmarks_[landmark_descriptor_idx].getDescriptor().rows(), descriptor_num_elements_)
        << "Mismatch between the descriptor size of landmark " << landmark_descriptor_idx << "("
        << landmarks_[landmark_descriptor_idx].getDescriptor().rows() << " bytes vs. "
        << descriptor_num_elements_ << " bytes for keypoints).";
    landmark_descriptors_.emplace_back(
        landmarks_[landmark_descriptor_idx].getDescriptor().data(), descriptor_num_elements_);
  }
}


template <>
void MatchingProblemLandmarksToFrame<float>::setupValidVectorsAndDescriptors() {
  CHECK_GT(image_height_, 0u) << "The visual frame has zero image rows.";

  const size_t num_keypoints = numApples();
  const size_t num_landmarks = numBananas();
  is_frame_keypoint_valid_.resize(num_keypoints, false);
  is_landmark_valid_.resize(num_landmarks, false);

  if (FLAGS_matcher_store_all_tested_pairs) {
    all_tested_pairs_.resize(num_landmarks);
  }

  // First, create descriptor wrappers for all descriptors.
  const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& frame_descriptors =
      frame_.getFloatDescriptors();
  CHECK_EQ(frame_descriptors.rows(), descriptor_num_elements_);

  const size_t num_frame_descriptors = static_cast<size_t>(frame_descriptors.cols());
  CHECK_EQ(num_frame_descriptors, num_keypoints) << "Mismatch between the number of "
      << "descriptors and the number of keypoints in the visual frame.";

  frame_descriptors_.clear();
  landmark_descriptors_.clear();
  frame_float_descriptors_.clear();
  landmark_float_descriptors_.clear();
  frame_float_descriptors_.reserve(num_frame_descriptors);
  landmark_float_descriptors_.reserve(num_landmarks);

  // This creates a descriptor wrapper for the given descriptor and allows computing the Hamming
  // distance between two descriptors.
  for (size_t frame_descriptor_idx = 0u; frame_descriptor_idx < num_frame_descriptors;
      ++frame_descriptor_idx) {
    frame_float_descriptors_.emplace_back(frame_descriptors.col(frame_descriptor_idx));
  }

  for (size_t landmark_descriptor_idx = 0u; landmark_descriptor_idx < num_landmarks;
      ++landmark_descriptor_idx) {
    CHECK_EQ(landmarks_[landmark_descriptor_idx].getDescriptor().rows(), descriptor_num_elements_)
        << "Mismatch between the descriptor size of landmark " << landmark_descriptor_idx << "("
        << landmarks_[landmark_descriptor_idx].getDescriptor().rows() << " bytes vs. "
        << descriptor_num_elements_ << " bytes for keypoints).";
    landmark_float_descriptors_.emplace_back(landmarks_[landmark_descriptor_idx].getDescriptor());
  }
}

}  // namespace aslam
