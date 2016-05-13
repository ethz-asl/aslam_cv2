#ifndef ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_MULTIMAP_H_
#define ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_MULTIMAP_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <map>
#include <memory>
#include <vector>

#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/common-private/feature-descriptor-ref.h>
#include <Eigen/Core>

#include "aslam/matcher/match.h"
#include "aslam/matcher/matching-problem-landmarks-to-frame.h"
#include "aslam/matcher/matching-problem-types.h"

namespace aslam {
class VisualFrame;

/// \class MatchingProblem
/// \brief Defines the specifics of a matching problem.
/// The problem is assumed to have a list of landmarks (3D points + descriptors)
/// and a visual frame with keypoints and descriptors. The landmarks are then
/// matched against the keypoints in the frame based on image space distance
/// and descriptor distance using a multimap on the keypoint y-coordinate.
/// The landmarks are expected to be expressed in the camera frame of the visual frame.
///
/// Coordinate Frames:
///   C: Camera frame of the visual frame.
class MatchingProblemLandmarksToFrameMultimap : public MatchingProblemLandmarksToFrame {
public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemLandmarksToFrameMultimap);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemLandmarksToFrameMultimap);
  ASLAM_ADD_MATCH_TYPEDEFS_WITH_ALIASES(
      MatchingProblemLandmarksToFrameMultimap, getKeypointIndex, getLandmarkIndex);

  MatchingProblemLandmarksToFrameMultimap() = delete;

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
  MatchingProblemLandmarksToFrameMultimap(const VisualFrame& frame,
                                  const LandmarkWithDescriptorList& landmarks,
                                  double image_space_distance_threshold_pixels,
                                  int hamming_distance_threshold);
  virtual ~MatchingProblemLandmarksToFrameMultimap() {};

  /// Get a short list of keypoint candidates for a given landmark index.
  ///
  /// \param[in]  landmark_index  The landmark index queried for candidates.
  /// \param[out] candidates      Candidates from the frame keypoint list that could
  ///                             potentially match the given landmark.
  virtual void getAppleCandidatesForBanana(int landmark_index, Candidates* candidates);

  /// \brief Gets called at the beginning of the matching problem.
  /// Creates a y-coordinate LUT for all frame keypoints and projects all landmark into the
  /// frame.
  virtual bool doSetup();

private:
  /// The landmarks projected into the visual frame.
  aslam::Aligned<std::vector, Eigen::Vector2d>::type projected_landmark_keypoints_;
  /// Map mapping y coordinates in the image plane onto keypoint indices of the
  /// visual frame keypoints.
  std::multimap<size_t, size_t> y_coordinate_to_keypoint_index_map_;
  /// Half width of the vertical band used for match lookup in pixels.
  int vertical_band_halfwidth_pixels_;
};
}  // namespace aslam
#endif  // ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_MULTIMAP_H_
