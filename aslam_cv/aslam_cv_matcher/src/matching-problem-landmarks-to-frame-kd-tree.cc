#include <algorithm>

#include <aslam/common/statistics/statistics.h>
#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/frames/visual-nframe.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-landmarks-to-frame-kd-tree.h"

namespace aslam {

MatchingProblemLandmarksToFrameKDTree::MatchingProblemLandmarksToFrameKDTree(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : MatchingProblemLandmarksToFrame(
      frame, landmarks, image_space_distance_threshold_pixels, hamming_distance_threshold) {}

bool MatchingProblemLandmarksToFrameKDTree::doSetup() {
  aslam::timing::Timer method_timer("MatchingProblemLandmarksToFrameKDTree::doSetup()");
  CHECK_GT(image_height_frame_, 0u) << "The visual frame has zero image rows.";

  const size_t num_keypoints = numApples();
  const size_t num_landmarks = numBananas();
  valid_frame_keypoints_.resize(num_keypoints, false);
  valid_landmarks_.resize(num_landmarks, false);

  if (store_tested_pairs_) {
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

  // This creates a descriptor wrapper for the given descriptor and allows computing the hamming
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

  // Then, create a LUT mapping y coordinates to frame keypoint indices.
  const Eigen::Matrix2Xd& keypoints = frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_keypoints), keypoints.cols())
    << "The number of keypoints in the visual frame does not match the number "
    << "of columns in the keypoint matrix.";

  Camera::ConstPtr camera = frame_.getCameraGeometry();
  CHECK(camera);

  VLOG(3) << "Adding " << num_keypoints << " keypoints to the LUT.";
  size_t valid_keypoint_index = 0u;
  valid_keypoints_ = Eigen::MatrixXd::Zero(2, num_keypoints);

  const double image_height = static_cast<double>(camera->imageHeight());
  const double image_width = static_cast<double>(camera->imageWidth());
  const size_t num_bins_x =
      static_cast<size_t>(std::floor(image_width / image_space_distance_threshold_pixels_));
  const size_t num_bins_y =
      static_cast<size_t>(std::floor(image_height / image_space_distance_threshold_pixels_));

  image_space_counting_grid_.reset(new NeighborCellCountingGrid(
      0.0, image_width, 0.0, image_height, num_bins_x, num_bins_y));

  valid_keypoint_index_to_keypoint_index_map_.reserve(num_keypoints);
  for (size_t keypoint_idx = 0u; keypoint_idx < num_keypoints; ++keypoint_idx) {
    // Check if the keypoint is valid.
    const Eigen::Vector2d& keypoint = keypoints.col(keypoint_idx);
    if (camera->isMasked(keypoint)) {
      // This keypoint is masked out and hence not valid.
      valid_frame_keypoints_[keypoint_idx] = false;
    } else {
      valid_keypoints_.col(valid_keypoint_index) = keypoint;
      valid_keypoint_index_to_keypoint_index_map_.emplace_back(keypoint_idx);
      valid_frame_keypoints_[keypoint_idx] = true;
      image_space_counting_grid_->addElementToGrid(keypoint);
      ++valid_keypoint_index;
    }
  }
  CHECK_EQ(valid_keypoint_index, valid_keypoint_index_to_keypoint_index_map_.size());
  valid_keypoints_.conservativeResize(2, valid_keypoint_index);
  VLOG(3) << "Built LUT for valid keypoints (" << valid_keypoint_index << ").";

  C_valid_projected_landmarks_ = Eigen::MatrixXd(2, num_landmarks);

  // Then, project all landmarks into the visual frame.
  size_t valid_landmark_index = 0u;
  VLOG(3) << "Projecting " << num_landmarks << " into the visual frame.";
  valid_landmark_index_to_landmark_index_map_.reserve(num_landmarks);
  for (size_t landmark_idx = 0u; landmark_idx < num_landmarks; ++landmark_idx) {
    Eigen::Vector2d C_projected_landmark;
    const ProjectionResult projection_result = camera->project3(
        landmarks_[landmark_idx].get_p_C_landmark(), &C_projected_landmark);

    if (projection_result.isKeypointVisible()) {
      C_valid_projected_landmarks_.col(valid_landmark_index) = C_projected_landmark;
      valid_landmark_index_to_landmark_index_map_.emplace_back(landmark_idx);
      valid_landmarks_[landmark_idx] = true;
      ++valid_landmark_index;
    } else {
      valid_landmarks_[landmark_idx] = false;
    }
  }
  CHECK_EQ(valid_landmark_index, valid_landmark_index_to_landmark_index_map_.size());
  C_valid_projected_landmarks_.conservativeResize(2, valid_landmark_index);
  VLOG(3) << "Computed all projections of landmarks into the visual frame. (valid/invalid) ("
      << valid_landmark_index << "/" << (num_landmarks - valid_landmark_index) << ")";

  // Only create the Nabo index if we have more than zero valid keypoints.
  if (valid_keypoint_index > 0u) {
    const int kDimVectors = 2;
    // Switch touch statistics (NNSearch::TOUCH_STATISTICS) off for performance.
    const int kCollectTouchStatistics = 0;
    nn_index_.reset(Nabo::NNSearchD::createKDTreeLinearHeap(
        valid_keypoints_, kDimVectors, kCollectTouchStatistics));
  }

  method_timer.Stop();
  return true;
}

void MatchingProblemLandmarksToFrameKDTree::getCandidates(
    CandidatesList* candidates_for_landmarks) {
  aslam::timing::Timer method_timer("MatchingProblemLandmarksToFrameKDTree::getCandidates");
  CHECK_NOTNULL(candidates_for_landmarks)->clear();
  candidates_for_landmarks->resize(numBananas());

  if (!nn_index_) {
    // No valid keypoints available - meaning no matches can be formed.
    return;
  }

  const int num_valid_landmarks = C_valid_projected_landmarks_.cols();

  size_t num_matches = 0u;

  CHECK(image_space_counting_grid_);
  const int num_neighbors = image_space_counting_grid_->getMaxNeighborCellCount();
  CHECK_GT(num_neighbors, 0);
  VLOG(3) << "Querying for " << num_neighbors << " num neighbors.";
  Eigen::MatrixXi indices = Eigen::MatrixXi::Constant(num_neighbors, num_valid_landmarks, -1);
  Eigen::MatrixXd distances = Eigen::MatrixXd::Constant(
      num_neighbors, num_valid_landmarks, std::numeric_limits<double>::infinity());
  const double kSearchNNEpsilon = 0.0;
  const double kSearchRadius = image_space_distance_threshold_pixels_;
  const unsigned kOptionFlags = Nabo::NNSearchD::ALLOW_SELF_MATCH;

  aslam::timing::Timer knn_timer(
      "MatchingProblemLandmarksToFrameKDTree::getCandidates - knn search");
  CHECK(nn_index_);
  nn_index_->knn(
      C_valid_projected_landmarks_, indices, distances,
      num_neighbors, kSearchNNEpsilon, kOptionFlags, kSearchRadius);
  knn_timer.Stop();

  aslam::timing::Timer knn_post_processing_timer(
      "MatchingProblemLandmarksToFrameKDTree::getCandidates - post-process knn search.");
  for (int knn_landmark_idx = 0; knn_landmark_idx < num_valid_landmarks; ++knn_landmark_idx) {
    CHECK_LT(static_cast<size_t>(knn_landmark_idx),
             valid_landmark_index_to_landmark_index_map_.size());
    const size_t landmark_index = valid_landmark_index_to_landmark_index_map_[knn_landmark_idx];
    CHECK_LT(landmark_index, numBananas());

    for (int nn_idx = 0; nn_idx < num_neighbors; ++nn_idx) {
      const int knn_keypoint_index = indices(nn_idx, knn_landmark_idx);
      const double distance_image_space_pixels = distances(nn_idx, knn_landmark_idx);

      if (knn_keypoint_index == -1 ||
          distance_image_space_pixels == std::numeric_limits<double>::infinity()) {
        break;  // No more results.
      }

      CHECK_GE(knn_keypoint_index, 0);

      CHECK_LT(knn_keypoint_index, valid_keypoint_index_to_keypoint_index_map_.size());
      const size_t keypoint_index =
          valid_keypoint_index_to_keypoint_index_map_[knn_keypoint_index];
      CHECK_LT(keypoint_index, numApples());

      const int hamming_distance = computeHammingDistance(landmark_index, keypoint_index);
      CHECK_GE(hamming_distance, 0);

      const int kPriority = 0;

      if (store_tested_pairs_) {
        CHECK_LT(landmark_index, all_tested_pairs_.size());
        all_tested_pairs_[landmark_index].emplace_back(
            keypoint_index, landmark_index, computeMatchScore(hamming_distance), kPriority);
      }

      if (hamming_distance < hamming_distance_threshold_) {
        CHECK_LT(landmark_index, candidates_for_landmarks->size());
        (*candidates_for_landmarks)[landmark_index].emplace_back(
            keypoint_index, landmark_index, computeMatchScore(hamming_distance), kPriority);
        ++num_matches;
      }
    }
  }
  knn_post_processing_timer.Stop();
  VLOG(3) << "Got " << num_matches << " matches.";
  method_timer.Stop();
}

NeighborCellCountingGrid::NeighborCellCountingGrid(
    double min_x, double max_x, double min_y, double max_y,
    size_t num_bins_x, size_t num_bins_y)
    : min_x_(min_x), max_x_(max_x), min_y_(min_y), max_y_(max_y),
      num_bins_x_(num_bins_x), num_bins_y_(num_bins_y),
      max_neighbor_count_(0) {
  CHECK_GT(max_x_, min_x_);
  CHECK_GT(max_y_, min_y_);

  interval_x_ = (max_x_ - min_x_) / static_cast<double>(num_bins_x);
  CHECK_GT(interval_x_, 0.0);

  interval_y_ = (max_y_ - min_y_) / static_cast<double>(num_bins_y);
  CHECK_GT(interval_y_, 0.0);

  grid_count_ = Eigen::MatrixXi::Zero(num_bins_y, num_bins_x);
  grid_neighboring_cell_count_ = Eigen::MatrixXi::Zero(num_bins_y, num_bins_x);
}

void NeighborCellCountingGrid::addElementToGrid(const Eigen::Vector2d& element) {
  addElementToGrid(element(0), element(1));
}
void NeighborCellCountingGrid::addElementToGrid(double x, double y) {
  Coordinate coordinate = elementToCoordinate(x, y);
  incrementCount(coordinate);
}

NeighborCellCountingGrid::Coordinate NeighborCellCountingGrid::elementToCoordinate(
    double x, double y) const {
  CHECK_GE(x, min_x_);
  CHECK_LE(x, max_x_);
  CHECK_GE(y, min_y_);
  CHECK_LE(y, max_y_);

  const double coordinate_x = x - min_x_;
  const size_t bin_index_x = static_cast<size_t>(std::floor(coordinate_x / interval_x_));
  CHECK_GE(bin_index_x, 0);
  CHECK_LT(bin_index_x, static_cast<int>(num_bins_x_));

  const double coordinate_y = y - min_y_;
  const size_t bin_index_y = static_cast<size_t>(std::floor(coordinate_y / interval_y_));
  CHECK_GE(bin_index_y, 0);
  CHECK_LT(bin_index_y, static_cast<int>(num_bins_y_));

  return std::make_pair(bin_index_x, bin_index_y);
}

void NeighborCellCountingGrid::incrementCount(const Coordinate& coordinate) {
  CHECK_LT(coordinate.first, static_cast<size_t>(grid_count_.cols()));
  CHECK_LT(coordinate.second, static_cast<size_t>(grid_count_.rows()));
  ++grid_count_(coordinate.second, coordinate.first);
  for (int x_shift = -1; x_shift <= 1; ++x_shift) {
    const int x_coordinate_shifted = static_cast<int>(coordinate.first) + x_shift;
    if (x_coordinate_shifted < 0 || x_coordinate_shifted >= grid_count_.cols()) {
      continue;
    }

    for (int y_shift = -1; y_shift <= 1; ++y_shift) {
      const int y_coordinate_shifted = static_cast<int>(coordinate.second) + y_shift;
      if (y_coordinate_shifted < 0 || y_coordinate_shifted >= grid_count_.rows()) {
        continue;
      }
      CHECK_GE(x_coordinate_shifted, 0);
      CHECK_LT(x_coordinate_shifted, grid_count_.cols());
      CHECK_GE(y_coordinate_shifted, 0);
      CHECK_LT(y_coordinate_shifted, grid_count_.rows());
      const Coordinate neighbor_coordinate = std::make_pair(
          static_cast<size_t>(x_coordinate_shifted), static_cast<size_t>(y_coordinate_shifted));

      const int new_neighbor_cell_count =
          ++grid_neighboring_cell_count_(neighbor_coordinate.second, neighbor_coordinate.first);
      max_neighbor_count_ = std::max(max_neighbor_count_, new_neighbor_cell_count);
    }
  }
}

}  // namespace aslam
