#ifndef ASLAM_CV_MATCHING_PROBLEM_FRAME_TO_FRAME_H_
#define ASLAM_CV_MATCHING_PROBLEM_FRAME_TO_FRAME_H_

/// \addtogroup Matching
/// @{
///
/// @}

#include <Eigen/Core>
#include <map>
#include <vector>

#include "aslam/common/macros.h"
#include "aslam/matcher/matching-problem.h"
#include "match.h"

namespace aslam {
class VisualFrame;

/// \class MatchingProblem
///
/// \brief defines the specifics of a matching problem
///
/// The problem is assumed to have two visual frames (apple_frame and banana_frame) filled with
/// keypoints and descriptors and a rotation matrix taking vectors from the banana frame into the
/// apple frame. The problem matches banana features against apple features.
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

  /// \brief Constructor for a a frame-to-frame matching problem.
  ///
  /// @param[in]  apple_frame                                 shared ptr to the apple frame.
  /// @param[in]  banana_frame                                shared ptr to the banana frame.
  /// @param[in]  C_A_B                                       Rotation matrix taking vectors from
  ///                                                         the banana frame into the
  ///                                                         apple frame.
  /// @param[in]  image_space_distance_threshold_pixels       Max image space distance threshold for
  ///                                                         two pairs to become match candidates.
  /// @param[in]  hamming_distance_threshold                  Max hamming distance for two pairs
  ///                                                         to become candidates.
  MatchingProblemFrameToFrame(const std::shared_ptr<VisualFrame>& apple_frame,
                              const std::shared_ptr<VisualFrame>& banana_frame,
                              const Eigen::Matrix3d& C_A_B,
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

  /// \brief compute the match score between items referenced by a and b.
  /// Note: this can be called multiple times from different threads.
  /// Warning: these are scores and *not* distances, higher values are better
  virtual double computeScore(int a, int b);

  inline double computeMatchScore(double image_space_distance, int hamming_distance) {
    return static_cast<double>(384 - hamming_distance) / 384.0;
  }

  /// \brief Gets called at the beginning of the matching problem.
  /// Creates a y-coordinate LUT for all apple keypoints and projects all banana keypoints into the
  /// apple frame.
  virtual bool doSetup();

  /// Called at the end of the matching process to set the output. 
  virtual void setBestMatches(const Matches& bestMatches);

private:
  /// \brief The apple frame.
  std::shared_ptr<VisualFrame> apple_frame_;
  /// \brief The banana frame.
  std::shared_ptr<VisualFrame> banana_frame_;
  /// \brief Rotation matrix taking vectors from the banana frame into the apple frame.
  Eigen::Matrix3d C_A_B_;
  /// \breif Map mapping y coordinates in the image plane onto keypoint indices of apple keypoints.
  std::multimap<size_t, size_t> y_coordinate_to_apple_keypoint_index_map_;

  /// \brief The apple keypoints expressed in the apple frame.
  Eigen::Matrix2Xd A_keypoints_apple_;

  struct ProjectedKeypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectedKeypoint() : keypoint(0.0, 0.0), valid(false) {}
    virtual ~ProjectedKeypoint() {};

    Eigen::Vector2d keypoint;
    bool valid;
  };
  /// \brief The bana keypoints projected into the apple frame, expressed in the apple frame.
  aslam::Aligned<std::vector, ProjectedKeypoint>::type A_projected_keypoints_banana_;

  /// \brief The apple descriptors.
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> apple_descriptors_;
  /// \breif The banana descriptors.
  Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> banana_descriptors_;

  /// \brief Descriptor size in bytes.
  size_t descriptor_size_byes_;

  /// \brief Half width of the vertical band used for match lookup in pixels.
  int vertical_band_halfwidth_pixels_;

  /// \brief Pairs with image space distance >= image_space_distance_threshold_pixels_ are
  ///        excluded from matches.
  double image_space_distance_threshold_pixels_;

  /// \brief Pairs with descriptor distance >= hamming_distance_threshold_ are
  ///        excluded from matches.
  int hamming_distance_threshold_;

  /// \brief The heigh of the apple frame.
  size_t image_height_apple_frame_;
};
}
#endif //ASLAM_CV_MATCHING_PROBLEM_FRAME_TO_FRAME_H_
