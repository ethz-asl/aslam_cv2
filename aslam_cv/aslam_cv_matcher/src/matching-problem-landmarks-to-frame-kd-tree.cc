#include <aslam/common/timer.h>
#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>

#include "aslam/matcher/matching-problem-landmarks-to-frame-kd-tree.h"

DECLARE_bool(matcher_store_all_tested_pairs);

namespace aslam {

/*
MatchingProblemLandmarksToFrameKDTree::MatchingProblemLandmarksToFrameKDTree(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : MatchingProblemLandmarksToFrame(
      frame, landmarks, image_space_distance_threshold_pixels, hamming_distance_threshold),
      search_radius_px_(image_space_distance_threshold_pixels) {
  CHECK_GT(hamming_distance_threshold, 0) << "Descriptor distance needs to be positive.";
  CHECK_GT(image_space_distance_threshold_pixels, 0.0)
    << "Image space distance needs to be positive.";
  CHECK(frame.getCameraGeometry()) << "The camera of the visual frame is NULL.";
  CHECK_GT(image_height_, 0u) << "The visual frame has zero image rows.";
  CHECK_GT(descriptor_size_bytes_, 0);
  CHECK_GT(descriptor_size_bits_, 0);
}*/

template <>
bool MatchingProblemLandmarksToFrameKDTree<unsigned char>::doSetup() {
  aslam::timing::Timer method_timer("MatchingProblemLandmarksToFrameKDTree::doSetup()");

  setupValidVectorsAndDescriptors();

  const size_t num_keypoints = numApples();
  const size_t num_landmarks = numBananas();

  all_tested_pairs_.resize(num_landmarks);

  const Eigen::Matrix2Xd& keypoints = frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_keypoints), keypoints.cols())
    << "The number of keypoints in the visual frame does not match the number "
    << "of columns in the keypoint matrix.";

  Camera::ConstPtr camera = frame_.getCameraGeometry();
  CHECK(camera);

  VLOG(3) << "Adding " << num_keypoints << " keypoints to the KD-tree.";
  valid_keypoints_ = Eigen::MatrixXd::Zero(2, num_keypoints);

  const double image_height = static_cast<double>(camera->imageHeight());
  CHECK_GT(image_height, 0.0);
  const double image_width = static_cast<double>(camera->imageWidth());
  CHECK_GT(image_width, 0.0);

  CHECK_GT(squared_image_space_distance_threshold_px_sq_, 0.0);
  const double image_space_distance_threshold_pixels =
      std::sqrt(squared_image_space_distance_threshold_px_sq_);

  const size_t num_bins_x =
      static_cast<size_t>(std::floor(image_width / image_space_distance_threshold_pixels));
  const size_t num_bins_y =
      static_cast<size_t>(std::floor(image_height / image_space_distance_threshold_pixels));

  const double kImageRangeBeginX = 0.0;
  const double kImageRangeBeginY= 0.0;
  image_space_counting_grid_.reset(new NeighborCellCountingGrid(
      kImageRangeBeginX, image_width, kImageRangeBeginY, image_height, num_bins_x, num_bins_y));

  valid_keypoint_index_to_keypoint_index_.reserve(num_keypoints);
  size_t valid_keypoint_index = 0u;
  for (size_t keypoint_idx = 0u; keypoint_idx < num_keypoints; ++keypoint_idx) {
    // Check if the keypoint is valid.
    const Eigen::Vector2d& keypoint = keypoints.col(keypoint_idx);
    if (camera->isMasked(keypoint)) {
      // This keypoint is masked out and hence not valid.
      is_frame_keypoint_valid_[keypoint_idx] = false;
    } else {
      valid_keypoints_.col(valid_keypoint_index) = keypoint;
      valid_keypoint_index_to_keypoint_index_.emplace_back(keypoint_idx);
      is_frame_keypoint_valid_[keypoint_idx] = true;
      image_space_counting_grid_->addElementToGrid(keypoint);
      ++valid_keypoint_index;
    }
  }
  CHECK_EQ(valid_keypoint_index, valid_keypoint_index_to_keypoint_index_.size());
  valid_keypoints_.conservativeResize(2, valid_keypoint_index);
  VLOG(3) << "Num valid keypoints: " << valid_keypoint_index;

  valid_projected_landmarks_ = Eigen::MatrixXd(2, num_landmarks);

  // Then, project all landmarks into the visual frame.
  size_t valid_landmark_index = 0u;
  VLOG(3) << "Projecting " << num_landmarks << " into the visual frame.";
  valid_landmark_index_to_landmark_index_.reserve(num_landmarks);
  for (size_t landmark_idx = 0u; landmark_idx < num_landmarks; ++landmark_idx) {
    Eigen::Vector2d projected_landmark;
    if (camera->project3(landmarks_[landmark_idx].get_p_C_landmark(), &projected_landmark)) {
      valid_projected_landmarks_.col(valid_landmark_index) = projected_landmark;
      valid_landmark_index_to_landmark_index_.emplace_back(landmark_idx);
      is_landmark_valid_[landmark_idx] = true;
      ++valid_landmark_index;
    } else {
      is_landmark_valid_[landmark_idx] = false;
    }
  }
  CHECK_EQ(valid_landmark_index, valid_landmark_index_to_landmark_index_.size());
  valid_projected_landmarks_.conservativeResize(2, valid_landmark_index);
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


template <>
bool MatchingProblemLandmarksToFrameKDTree<float>::doSetup() {
  aslam::timing::Timer method_timer("MatchingProblemLandmarksToFrameKDTree::doSetup()");

  setupValidVectorsAndDescriptors();

  const size_t num_keypoints = numApples();
  const size_t num_landmarks = numBananas();

  all_tested_pairs_.resize(num_landmarks);

  const Eigen::Matrix2Xd& keypoints = frame_.getKeypointMeasurements();
  CHECK_EQ(static_cast<int>(num_keypoints), keypoints.cols())
    << "The number of keypoints in the visual frame does not match the number "
    << "of columns in the keypoint matrix.";

  Camera::ConstPtr camera = frame_.getCameraGeometry();
  CHECK(camera);

  VLOG(3) << "Adding " << num_keypoints << " keypoints to the KD-tree.";
  valid_keypoints_ = Eigen::MatrixXd::Zero(2, num_keypoints);

  const double image_height = static_cast<double>(camera->imageHeight());
  CHECK_GT(image_height, 0.0);
  const double image_width = static_cast<double>(camera->imageWidth());
  CHECK_GT(image_width, 0.0);

  CHECK_GT(squared_image_space_distance_threshold_px_sq_, 0.0);
  const double image_space_distance_threshold_pixels =
      std::sqrt(squared_image_space_distance_threshold_px_sq_);

  const size_t num_bins_x =
      static_cast<size_t>(std::floor(image_width / image_space_distance_threshold_pixels));
  const size_t num_bins_y =
      static_cast<size_t>(std::floor(image_height / image_space_distance_threshold_pixels));

  const double kImageRangeBeginX = 0.0;
  const double kImageRangeBeginY= 0.0;
  image_space_counting_grid_.reset(new NeighborCellCountingGrid(
      kImageRangeBeginX, image_width, kImageRangeBeginY, image_height, num_bins_x, num_bins_y));

  valid_keypoint_index_to_keypoint_index_.reserve(num_keypoints);
  size_t valid_keypoint_index = 0u;
  for (size_t keypoint_idx = 0u; keypoint_idx < num_keypoints; ++keypoint_idx) {
    // Check if the keypoint is valid.
    const Eigen::Vector2d& keypoint = keypoints.col(keypoint_idx);
    if (camera->isMasked(keypoint)) {
      // This keypoint is masked out and hence not valid.
      is_frame_keypoint_valid_[keypoint_idx] = false;
    } else {
      valid_keypoints_.col(valid_keypoint_index) = keypoint;
      valid_keypoint_index_to_keypoint_index_.emplace_back(keypoint_idx);
      is_frame_keypoint_valid_[keypoint_idx] = true;
      image_space_counting_grid_->addElementToGrid(keypoint);
      ++valid_keypoint_index;
    }
  }
  CHECK_EQ(valid_keypoint_index, valid_keypoint_index_to_keypoint_index_.size());
  valid_keypoints_.conservativeResize(2, valid_keypoint_index);
  VLOG(3) << "Num valid keypoints: " << valid_keypoint_index;

  valid_projected_landmarks_ = Eigen::MatrixXd(2, num_landmarks);

  // Then, project all landmarks into the visual frame.
  size_t valid_landmark_index = 0u;
  VLOG(3) << "Projecting " << num_landmarks << " into the visual frame.";
  valid_landmark_index_to_landmark_index_.reserve(num_landmarks);
  for (size_t landmark_idx = 0u; landmark_idx < num_landmarks; ++landmark_idx) {
    Eigen::Vector2d projected_landmark;
    if (camera->project3(landmarks_[landmark_idx].get_p_C_landmark(), &projected_landmark)) {
      valid_projected_landmarks_.col(valid_landmark_index) = projected_landmark;
      valid_landmark_index_to_landmark_index_.emplace_back(landmark_idx);
      is_landmark_valid_[landmark_idx] = true;
      ++valid_landmark_index;
    } else {
      is_landmark_valid_[landmark_idx] = false;
    }
  }
  CHECK_EQ(valid_landmark_index, valid_landmark_index_to_landmark_index_.size());
  valid_projected_landmarks_.conservativeResize(2, valid_landmark_index);
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
/*
void MatchingProblemLandmarksToFrameKDTree::getCandidates(
    CandidatesList* candidates_for_landmarks) {
  aslam::timing::Timer method_timer("MatchingProblemLandmarksToFrameKDTree::getCandidates");
  CHECK_NOTNULL(candidates_for_landmarks)->clear();
  candidates_for_landmarks->resize(numBananas());

  if (!nn_index_) {
    LOG(WARNING) << "No valid keypoints available in the KD-tree index. No matches can be formed.";
    return;
  }

  const int num_valid_landmarks = valid_projected_landmarks_.cols();

  size_t num_matches = 0u;

  CHECK(image_space_counting_grid_);
  const int num_neighbors = image_space_counting_grid_->getMaxNeighborhoodCellCount();
  CHECK_GT(num_neighbors, 0);
  VLOG(5) << "Querying for " << num_neighbors << " num neighbors.";
  Eigen::MatrixXi indices = Eigen::MatrixXi::Constant(num_neighbors, num_valid_landmarks, -1);
  Eigen::MatrixXd distances_squared = Eigen::MatrixXd::Constant(
      num_neighbors, num_valid_landmarks, std::numeric_limits<double>::infinity());
  const double kSearchNNEpsilon = 0.0;
  CHECK_GT(search_radius_px_, 0.0);
  const unsigned kOptionFlags = Nabo::NNSearchD::ALLOW_SELF_MATCH;

  aslam::timing::Timer knn_timer(
      "MatchingProblemLandmarksToFrameKDTree::getCandidates - knn search");
  CHECK(nn_index_);
  nn_index_->knn(
      valid_projected_landmarks_, indices, distances_squared, num_neighbors, kSearchNNEpsilon,
      kOptionFlags, search_radius_px_);
  knn_timer.Stop();

  aslam::timing::Timer knn_post_processing_timer(
      "MatchingProblemLandmarksToFrameKDTree::getCandidates - post-process knn search.");
  for (int knn_landmark_idx = 0; knn_landmark_idx < num_valid_landmarks; ++knn_landmark_idx) {
    CHECK_LT(static_cast<size_t>(knn_landmark_idx),
             valid_landmark_index_to_landmark_index_.size());
    const size_t landmark_index = valid_landmark_index_to_landmark_index_[knn_landmark_idx];
    CHECK_LT(landmark_index, numBananas());

    Candidates all_tested_pairs_of_this_landmark;

    for (int nearest_neighbor_idx = 0; nearest_neighbor_idx < num_neighbors;
        ++nearest_neighbor_idx) {
      const int knn_keypoint_index = indices(nearest_neighbor_idx, knn_landmark_idx);
      const double distance_squared_image_space_pixels_squared =
          distances_squared(nearest_neighbor_idx, knn_landmark_idx);

      if (knn_keypoint_index == -1) {
        CHECK_EQ(
            distance_squared_image_space_pixels_squared, std::numeric_limits<double>::infinity());
        break;  // No more results.
      }
      CHECK_GE(knn_keypoint_index, 0);

      if (distance_squared_image_space_pixels_squared <
          squared_image_space_distance_threshold_px_sq_) {
        CHECK_LT(knn_keypoint_index, valid_keypoint_index_to_keypoint_index_.size());
        const size_t keypoint_index =
            valid_keypoint_index_to_keypoint_index_[knn_keypoint_index];
        CHECK_LT(keypoint_index, numApples());

        const int hamming_distance = computeHammingDistance(landmark_index, keypoint_index);
        CHECK_GE(hamming_distance, 0);
        CHECK_LE(hamming_distance, descriptor_size_bits_);

        const int kPriority = 0;

        if (FLAGS_matcher_store_all_tested_pairs) {
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
  }
  knn_post_processing_timer.Stop();
  VLOG(3) << "Got " << num_matches << " matches.";
  method_timer.Stop();
}*/

NeighborCellCountingGrid::NeighborCellCountingGrid(
    double min_x, double max_x, double min_y, double max_y,
    size_t num_bins_x, size_t num_bins_y)
    : min_x_(min_x), max_x_(max_x), min_y_(min_y), max_y_(max_y), num_bins_x_(num_bins_x),
      num_bins_y_(num_bins_y), max_neighbor_count_(0) {
  CHECK_GT(max_x_, min_x_);
  CHECK_GT(max_y_, min_y_);
  CHECK_GT(num_bins_x_, 0u);
  CHECK_GT(num_bins_y_, 0u);

  interval_x_ = (max_x_ - min_x_) / static_cast<double>(num_bins_x_);
  CHECK_GT(interval_x_, 0.0);

  interval_y_ = (max_y_ - min_y_) / static_cast<double>(num_bins_y_);
  CHECK_GT(interval_y_, 0.0);

  grid_neighboring_cell_count_ = Eigen::MatrixXi::Zero(num_bins_y_, num_bins_x_);
}

void NeighborCellCountingGrid::addElementToGrid(const Eigen::Vector2d& element) {
  addElementToGrid(element(0), element(1));
}

void NeighborCellCountingGrid::addElementToGrid(double x, double y) {
  Coordinate coordinate = elementToGridCoordinate(x, y);
  incrementCellCount(coordinate);
}

NeighborCellCountingGrid::Coordinate NeighborCellCountingGrid::elementToGridCoordinate(
    double x_position, double y_position) const {
  CHECK_GE(x_position, min_x_);
  CHECK_LE(x_position, max_x_);
  CHECK_GE(y_position, min_y_);
  CHECK_LE(y_position, max_y_);

  const double coordinate_x = x_position - min_x_;
  CHECK_GT(interval_x_, 0.0);
  const int bin_index_x = static_cast<int>(std::floor(coordinate_x / interval_x_));
  CHECK_GE(bin_index_x, 0);
  CHECK_LT(bin_index_x, static_cast<int>(num_bins_x_));

  const double coordinate_y = y_position - min_y_;
  CHECK_GT(interval_y_, 0.0);
  const int bin_index_y = static_cast<int>(std::floor(coordinate_y / interval_y_));
  CHECK_GE(bin_index_y, 0);
  CHECK_LT(bin_index_y, static_cast<int>(num_bins_y_));

  return Eigen::Vector2i(bin_index_x, bin_index_y);
}

void NeighborCellCountingGrid::incrementCellCount(const Coordinate& coordinate) {
  const int num_cols = grid_neighboring_cell_count_.cols();
  const int num_rows = grid_neighboring_cell_count_.rows();
  const int coordinate_x = coordinate(0);
  const int coordinate_y = coordinate(1);
  CHECK_LT(coordinate_x, num_cols);
  CHECK_LT(coordinate_y, num_rows);

  for (int x_shift = -1; x_shift <= 1; ++x_shift) {
    const int x_coordinate_shifted = coordinate_x + x_shift;
    if (x_coordinate_shifted < 0 || x_coordinate_shifted >= num_cols) {
      // The x-coordinate is out of the grid boundaries.
      continue;
    }

    for (int y_shift = -1; y_shift <= 1; ++y_shift) {
      const int y_coordinate_shifted = coordinate_y + y_shift;
      if (y_coordinate_shifted < 0 || y_coordinate_shifted >= num_rows) {
        // The y-coordinate is out of the grid boundaries.
        continue;
      }
      CHECK_GE(x_coordinate_shifted, 0);
      CHECK_LT(x_coordinate_shifted, num_cols);
      CHECK_GE(y_coordinate_shifted, 0);
      CHECK_LT(y_coordinate_shifted, num_rows);
      const Coordinate neighbor_coordinate =
          Eigen::Vector2i(x_coordinate_shifted, y_coordinate_shifted);

      const int new_neighbor_cell_count =
          ++grid_neighboring_cell_count_(neighbor_coordinate(1), neighbor_coordinate(0));
      max_neighbor_count_ = std::max(max_neighbor_count_, new_neighbor_cell_count);
    }
  }
}

}  // namespace aslam
