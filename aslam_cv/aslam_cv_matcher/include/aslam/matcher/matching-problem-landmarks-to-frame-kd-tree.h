#ifndef ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_KD_TREE_H_
#define ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_KD_TREE_H_

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
#include <nabo/nabo.h>

#include "aslam/matcher/match.h"
#include "aslam/matcher/matching-problem-landmarks-to-frame.h"
#include "aslam/matcher/matching-problem-types.h"

namespace aslam {
class VisualFrame;

// 2D grid that computes the maximum number of elements in adjacent cells on the fly.
class NeighborCellCountingGrid {
 public:
  ASLAM_POINTER_TYPEDEFS(NeighborCellCountingGrid);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(NeighborCellCountingGrid);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  NeighborCellCountingGrid() = delete;

  // min_x, max_x, min_y, max_y define the borders of the grid, while num_bins_x
  // and num_bins_y define into how many cells the grid is divided.
  NeighborCellCountingGrid(
      double min_x, double max_x, double min_y, double max_y, size_t num_bins_x, size_t num_bins_y);
  virtual ~NeighborCellCountingGrid() = default;

  void addElementToGrid(const Eigen::Vector2d& element);
  void addElementToGrid(double x, double y);

  // Returns the maximum number of elements in any cell and its directly adjacent
  // adjacent cells.
  inline int getMaxNeighborhoodCellCount() const {
    return max_neighbor_count_;
  }

 private:
  typedef std::pair<size_t, size_t> Coordinate;

  Coordinate elementToCoordinate(double x, double y) const;

  void incrementCount(const Coordinate& coordinate);

  Eigen::MatrixXi grid_count_;
  Eigen::MatrixXi grid_neighboring_cell_count_;

  const double min_x_, max_x_, min_y_, max_y_;
  const size_t num_bins_x_, num_bins_y_;
  double interval_x_, interval_y_;

  int max_neighbor_count_;
};

typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> Descriptor;

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
class MatchingProblemLandmarksToFrameKDTree : public MatchingProblemLandmarksToFrame {
public:
  ASLAM_POINTER_TYPEDEFS(MatchingProblemLandmarksToFrameKDTree);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(MatchingProblemLandmarksToFrameKDTree);
  friend class LandmarksToFrame;

  MatchingProblemLandmarksToFrameKDTree() = delete;

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
  MatchingProblemLandmarksToFrameKDTree(
      const VisualFrame& frame, const LandmarkWithDescriptorList& landmarks,
      double image_space_distance_threshold_pixels, int hamming_distance_threshold);

  virtual ~MatchingProblemLandmarksToFrameKDTree() {};

  /// Retrieves match candidates for each landmarks.
  ///
  /// \param[out] candidates_for_landmarks
  ///         Candidates from the frame keypoint list that could
  ///         potentially match for each of the landmarks.
  virtual void getCandidates(CandidatesList* candidates_for_landmarks);


  /// \brief Gets called at the beginning of the matching problem.
  /// Creates a y-coordinate LUT for all frame keypoints and projects all landmark into the
  /// frame.
  virtual bool doSetup();

private:
  Eigen::MatrixXd valid_keypoints_;
  std::vector<size_t> valid_keypoint_index_to_keypoint_index_map_;

  Eigen::Matrix2Xd C_valid_projected_landmarks_;
  std::vector<size_t> valid_landmark_index_to_landmark_index_map_;

  NeighborCellCountingGrid::UniquePtr image_space_counting_grid_;

  std::shared_ptr<Nabo::NNSearchD> nn_index_;
};
}  // namespace aslam
#endif  //ASLAM_CV_MATCHING_PROBLEM_LANDMARKS_TO_FRAME_KD_TREE_H_
