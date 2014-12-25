#ifndef ASLAM_CV_MATCHING_PROBLEM_FRAME_TO_FRAME_H_
#define ASLAM_CV_MATCHING_PROBLEM_FRAME_TO_FRAME_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <Eigen/Core>
#include <map>
#include <memory>
#include <vector>

#include <aslam/common/macros.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/feature-descriptor-ref.h>

#include "aslam/matcher/matching-problem.h"
#include "match.h"

namespace aslam {
class VisualFrame;

/// \class MatchingProblem
///
/// \brief Defines the specifics of a matching problem.
///
/// The problem is assumed to have two visual frames (apple_frame and banana_frame) filled with
/// keypoints and binary descriptors and a rotation matrix taking vectors from the banana frame into
/// the apple frame. The problem matches banana features against apple features.
///
/// Coordinate Frames:
///   A:  apple frame
///   B:  banana frame
class MatchingProblemFrameToFrame : public MatchingProblem {
public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemFrameToFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemFrameToFrame);
  friend class MatcherTest;

  MatchingProblemFrameToFrame() = delete;

  /// \brief Constructor for a frame-to-frame matching problem.
  ///
  /// @param[in]  apple_frame                                 Shared ptr to the apple frame.
  /// @param[in]  banana_frame                                Shared ptr to the banana frame.
  /// @param[in]  q_A_B                                       Quaternion taking vectors from
  ///                                                         the banana frame into the
  ///                                                         apple frame.
  /// @param[in]  image_space_distance_threshold_pixels       Max image space distance threshold for
  ///                                                         two pairs to become match candidates.
  /// @param[in]  hamming_distance_threshold                  Max hamming distance for two pairs
  ///                                                         to become candidates.
  MatchingProblemFrameToFrame(const std::shared_ptr<VisualFrame>& apple_frame,
                              const std::shared_ptr<VisualFrame>& banana_frame,
                              const aslam::Quaternion& q_A_B,
                              double image_space_distance_threshold_pixels,
                              int hamming_distance_threshold);
  virtual ~MatchingProblemFrameToFrame() {};

  virtual size_t numApples() const;
  virtual size_t numBananas() const;

  /// Get a short list of candidates in list a for index b
  ///
  /// Return all indices of list a for n^2 matching; or use something
  /// smarter like nabo to get nearest neighbors.  Can also be used to
  /// mask out invalid elements in the lists, or an invalid b, by
  /// returning and empty candidate list.  
  ///
  /// The score for each candidate is a rough score that can be used
  /// for sorting, pre-filtering, and will be explicitly recomputed
  /// using the computeScore function.
  ///
  /// \param[in] b            The index of b queried for candidates.
  /// \param[out] candidates  Candidates from the Apples-list that could potentially match this
  ///                         element of Bananas.
  virtual void getAppleCandidatesForBanana(int banana_index, Candidates* candidates);

  inline double computeMatchScore(int hamming_distance) {
    return static_cast<double>(384 - hamming_distance) / 384.0;
  }

  inline int computeHammingDistance(int banana_index, int apple_index) {
    CHECK_LT(apple_index, apple_descriptors_.size()) << "No descriptor for this apple.";
    CHECK_LT(banana_index, banana_descriptors_.size()) << "No descriptor for this banana.";
    CHECK_LT(apple_index, valid_apples_.size()) << "No valid flag for this apple.";
    CHECK_LT(banana_index, valid_bananas_.size()) << "No valid flag for this apple.";
    CHECK(valid_apples_[apple_index]) << "The given apple is not valid.";
    CHECK(valid_bananas_[banana_index]) << "The given banana is not valid.";

    const common::FeatureDescriptorConstRef& apple_descriptor = apple_descriptors_[apple_index];
    const common::FeatureDescriptorConstRef& banana_descriptor = banana_descriptors_[banana_index];

    CHECK_NOTNULL(apple_descriptor.data());
    CHECK_NOTNULL(banana_descriptor.data());

    return common::GetNumBitsDifferent(banana_descriptor, apple_descriptor);
  }

  /// \brief Gets called at the beginning of the matching problem.
  /// Creates a y-coordinate LUT for all apple keypoints and projects all banana keypoints into the
  /// apple frame.
  virtual bool doSetup();

private:
  /// \brief The apple frame.
  std::shared_ptr<VisualFrame> apple_frame_;
  /// \brief The banana frame.
  std::shared_ptr<VisualFrame> banana_frame_;
  /// \brief Rotation matrix taking vectors from the banana frame into the apple frame.
  aslam::Quaternion q_A_B_;
  /// \breif Map mapping y coordinates in the image plane onto keypoint indices of apple keypoints.
  std::multimap<size_t, size_t> y_coordinate_to_apple_keypoint_index_map_;

  /// \brief Index marking apples as valid or invalid.
  std::vector<bool> valid_apples_;
  /// \brief Index marking bananas as valid or invalid.
  std::vector<bool> valid_bananas_;

  /// \brief The apple keypoints expressed in the apple frame.
  Eigen::Matrix2Xd* A_keypoints_apple_;

  ///
  Eigen::VectorXi* apple_track_ids_;

  /// \brief The banana keypoints projected into the apple frame, expressed in the apple frame.
  aslam::Aligned<std::vector, Eigen::Vector2d>::type A_projected_keypoints_banana_;

  /// \brief The apple descriptors.
  std::vector<common::FeatureDescriptorConstRef> apple_descriptors_;

  /// \brief The banana descriptors.
  std::vector<common::FeatureDescriptorConstRef> banana_descriptors_;

  /// \brief Descriptor size in bytes.
  size_t descriptor_size_byes_;

  /// \brief Half width of the vertical band used for match lookup in pixels.
  int vertical_band_halfwidth_pixels_;

  /// \brief Pairs with image space distance >= image_space_distance_threshold_pixels_ are
  ///        excluded from matches.
  double squared_image_space_distance_threshold_pixels_squared_;

  /// \brief Pairs with descriptor distance >= hamming_distance_threshold_ are
  ///        excluded from matches.
  int hamming_distance_threshold_;

  /// \brief The heigh of the apple frame.
  size_t image_height_apple_frame_;
};
}
#endif //ASLAM_CV_MATCHING_PROBLEM_FRAME_TO_FRAME_H_
