#ifndef MATCHING_PROBLEM_LANDMARKS_TO_FRAME_INL_H_
#define MATCHING_PROBLEM_LANDMARKS_TO_FRAME_INL_H_

#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>

DECLARE_bool(matcher_store_all_tested_pairs);

namespace aslam {

template <class Scalar>
size_t MatchingProblemLandmarksToFrame<Scalar>::numApples() const {
  return frame_.getNumKeypointMeasurements();
}

template <class Scalar>
size_t MatchingProblemLandmarksToFrame<Scalar>::numBananas() const {
  return landmarks_.size();
}

/*
template <class Scalar>
void MatchingProblemLandmarksToFrame<Scalar>::setupValidVectorsAndDescriptors() {
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
  CHECK_EQ(frame_descriptors.rows(), descriptor_size_bytes_);

  const size_t num_frame_descriptors = static_cast<size_t>(frame_descriptors.cols());
  CHECK_EQ(num_frame_descriptors, num_keypoints) << "Mismatch between the number of "
      << "descriptors and the number of keypoints in the visual frame.";

  frame_descriptors_.clear();
  landmark_descriptors_.clear();
  frame_descriptors_.reserve(num_frame_descriptors);
  landmark_descriptors_.reserve(num_landmarks);

  // This creates a descriptor wrapper for the given descriptor and allows computing the Hamming
  // distance between two descriptors.
  for (size_t frame_descriptor_idx = 0u; frame_descriptor_idx < num_frame_descriptors;
      ++frame_descriptor_idx) {
    frame_descriptors_.emplace_back(
        &(frame_descriptors.coeffRef(0, frame_descriptor_idx)), descriptor_size_bytes_);
  }

  for (size_t landmark_descriptor_idx = 0u; landmark_descriptor_idx < num_landmarks;
      ++landmark_descriptor_idx) {
    CHECK_EQ(landmarks_[landmark_descriptor_idx].getDescriptor().rows(), descriptor_size_bytes_)
        << "Mismatch between the descriptor size of landmark " << landmark_descriptor_idx << "("
        << landmarks_[landmark_descriptor_idx].getDescriptor().rows() << " bytes vs. "
        << descriptor_size_bytes_ << " bytes for keypoints).";
    landmark_descriptors_.emplace_back(
        landmarks_[landmark_descriptor_idx].getDescriptor().data(), descriptor_size_bytes_);
  }
}
*/
}

#endif /* MATCHING_PROBLEM_LANDMARKS_TO_FRAME_INL_H_ */
