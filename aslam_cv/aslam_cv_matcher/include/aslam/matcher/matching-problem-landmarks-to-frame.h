#ifndef ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_H_
#define ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <map>
#include <memory>
#include <vector>

#include <aslam/common/macros.h>
#include <aslam/common/pose-types.h>
#include <aslam/common-private/feature-descriptor-ref.h>
#include <Eigen/Core>

#include "aslam/matcher/match.h"
#include "aslam/matcher/matching-problem.h"
#include "aslam/matcher/matching-problem-types.h"

namespace aslam {
class VisualFrame;

/// \class MatchingProblem
/// \brief Defines the specifics of a matching problem.
/// The problem is assumed to have a list of landmarks (3D points + descriptors)
/// and a visual frame with keypoints and descriptors. The landmarks are then
/// matched against the keypoints in the frame based on image space distance
/// and descriptor distance. The landmarks are expected to be expressed in the
/// camera frame of the visual frame.
///
/// Coordinate Frames:
///   C: camera frame of the visual frame.
class MatchingProblemLandmarksToFrame : public MatchingProblem {
public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemLandmarksToFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemLandmarksToFrame);
  friend class LandmarksToFrame;

  MatchingProblemLandmarksToFrame() = delete;

  /// \brief Constructor for a landmarks-to-frame matching problem.
  ///
  /// @param[in]  frame                                       Visual frame.
  /// @param[in]  landmarks                                   List of landmarks with descriptors.
  /// @param[in]  image_space_distance_threshold_pixels       Max image space distance threshold for
  ///                                                         a keypoint and a projected landmark
  ///                                                         to become match candidates.
  /// @param[in]  hamming_distance_threshold                  Max hamming distance for a keypoint
  ///                                                         and a projected landmark
  ///                                                         to become candidates.
  MatchingProblemLandmarksToFrame(const VisualFrame& frame,
                                  const LandmarkWithDescriptorList& landmarks,
                                  double image_space_distance_threshold_pixels,
                                  int hamming_distance_threshold);
  virtual ~MatchingProblemLandmarksToFrame() {};

  static inline int getKeypointIndex(const MatchWithScore& match) {
    return match.getIndexApple();
  }

  static inline int getLandmarkIndex(const MatchWithScore& match) {
    return match.getIndexBanana();
  }

  virtual size_t numApples() const;
  virtual size_t numBananas() const;

  /// Get a short list of candidates in list a for index b.
  ///
  /// \param[in]  landmark_index  The index of b (landmark index) queried for candidates.
  /// \param[out] candidates      Candidates from the frame keypoint list that could
  ///                             potentially match the given landmark.
  void getAppleCandidatesForBanana(int landmark_index, Candidates* candidates);

  /// Retrieves match candidates for each landmark.
  virtual void getCandidates(CandidatesList* candidates_for_landmarks) {
    CHECK_NOTNULL(candidates_for_landmarks)->clear();
    const size_t num_bananas = numBananas();
    candidates_for_landmarks->resize(num_bananas);
    for (size_t banana_idx = 0u; banana_idx < num_bananas; ++banana_idx) {
      getAppleCandidatesForBanana(banana_idx, &(*candidates_for_landmarks)[banana_idx]);
    }
  }

  inline double computeMatchScore(int hamming_distance) {
    return static_cast<double>(descriptor_size_bits_ - hamming_distance) /
        static_cast<double>(descriptor_size_bits_);
  }

  inline int computeHammingDistance(int landmark_index, int frame_keypoint_index) {
    CHECK_LT(frame_keypoint_index, static_cast<int>(frame_descriptors_.size()))
        << "No descriptor for keypoint with index " << frame_keypoint_index << ".";
    CHECK_LT(landmark_index, static_cast<int>(landmark_descriptors_.size()))
        << "No descriptor for landmark with index " << landmark_index << ".";
    CHECK_LT(frame_keypoint_index, static_cast<int>(valid_frame_keypoints_.size()))
        << "No valid flag for keypoint with index " << frame_keypoint_index << ".";
    CHECK_LT(landmark_index, static_cast<int>(valid_landmarks_.size()))
        << "No valid flag for landmark with index " << landmark_index << ".";
    CHECK(valid_frame_keypoints_[frame_keypoint_index])
        << "The given frame keypoint with index " << frame_keypoint_index
        << " is not valid.";
    CHECK(valid_landmarks_[landmark_index]) << "The given landmark with index "
        << landmark_index << " is not valid.";

    const common::FeatureDescriptorConstRef& apple_descriptor =
        frame_descriptors_[frame_keypoint_index];
    const common::FeatureDescriptorConstRef& banana_descriptor =
        landmark_descriptors_[landmark_index];

    CHECK_NOTNULL(apple_descriptor.data());
    CHECK_NOTNULL(banana_descriptor.data());

    return common::GetNumBitsDifferent(banana_descriptor, apple_descriptor);
  }

  /// \brief Gets called at the beginning of the matching problem.
  /// Creates a y-coordinate LUT for all frame keypoints and projects all landmark into the
  /// frame.
  virtual bool doSetup();

protected:
  /// The landmarks to be matched.
  LandmarkWithDescriptorList landmarks_;
  /// The visual frame the landmarks are matched against.
  const VisualFrame& frame_;

  /// Index marking frame keypoints as valid or invalid.
  std::vector<unsigned char> valid_frame_keypoints_;
  /// Index marking landmarks as valid or invalid.
  std::vector<unsigned char> valid_landmarks_;

  /// The frame keypoint descriptors.
  std::vector<common::FeatureDescriptorConstRef> frame_descriptors_;

  /// The landmark descriptors.
  std::vector<common::FeatureDescriptorConstRef> landmark_descriptors_;

  /// Descriptor size in bits and bytes.
  const size_t descriptor_size_bytes_;
  const int descriptor_size_bits_;

  /// Half width of the vertical band used for match lookup in pixels.
  int vertical_band_halfwidth_pixels_;

  /// The height of the visual frame.
  size_t image_height_frame_;

  /// Pairs with image space distance >= image_space_distance_threshold_pixels_ are
  /// excluded from matches.
  double image_space_distance_threshold_pixels_;

  /// Pairs with descriptor distance >= hamming_distance_threshold_ are
  /// excluded from matches.
  int hamming_distance_threshold_;
private:
  /// The landmarks projected into the visual frame.
  aslam::Aligned<std::vector, Eigen::Vector2d>::type projected_landmark_keypoints_;
  /// Map mapping y coordinates in the image plane onto keypoint indices of the
  /// visual frame keypoints.
  std::multimap<size_t, size_t> y_coordinate_to_keypoint_index_map_;
};
}  // namespace aslam
#endif  //ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_H_
