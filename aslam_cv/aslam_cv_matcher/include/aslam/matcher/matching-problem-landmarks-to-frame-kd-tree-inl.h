#ifndef MATCHING_PROBLEM_LANDMARKS_TO_FRAME_KD_TREE_INL_H_
#define MATCHING_PROBLEM_LANDMARKS_TO_FRAME_KD_TREE_INL_H_

namespace aslam {

/*
template <class Scalar>
MatchingProblemLandmarksToFrameKDTree<Scalar>::MatchingProblemLandmarksToFrameKDTree(
    const VisualFrame& frame,
    const LandmarkWithDescriptorList<Scalar>& landmarks,
    double image_space_distance_threshold_pixels,
    int hamming_distance_threshold)
  : MatchingProblemLandmarksToFrame<Scalar>(
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


template <class Scalar>
void MatchingProblemLandmarksToFrameKDTree<Scalar>::getCandidates(
    MatchingProblem::CandidatesList* candidates_for_landmarks) {
  aslam::timing::Timer method_timer("MatchingProblemLandmarksToFrameKDTree::getCandidates");
  CHECK_NOTNULL(candidates_for_landmarks)->clear();
  candidates_for_landmarks->resize(MatchingProblemLandmarksToFrame<Scalar>::numBananas());

  LOG(INFO) << "MatchingProblemLandmarksToFrameKDTree<Scalar>::getCandidates";

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

  double min_descriptor_distance = std::numeric_limits<double>::max();
  double max_descriptor_distance = std::numeric_limits<double>::min();

  aslam::timing::Timer knn_post_processing_timer(
      "MatchingProblemLandmarksToFrameKDTree::getCandidates - post-process knn search.");
  for (int knn_landmark_idx = 0; knn_landmark_idx < num_valid_landmarks; ++knn_landmark_idx) {
    CHECK_LT(static_cast<size_t>(knn_landmark_idx),
             valid_landmark_index_to_landmark_index_.size());
    const size_t landmark_index = valid_landmark_index_to_landmark_index_[knn_landmark_idx];
    CHECK_LT(landmark_index, MatchingProblemLandmarksToFrame<Scalar>::numBananas());

    MatchingProblem::Candidates all_tested_pairs_of_this_landmark;

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
          MatchingProblemLandmarksToFrame<Scalar>::squared_image_space_distance_threshold_px_sq_) {
        CHECK_LT(knn_keypoint_index, valid_keypoint_index_to_keypoint_index_.size());
        const size_t keypoint_index =
            valid_keypoint_index_to_keypoint_index_[knn_keypoint_index];
        CHECK_LT(keypoint_index, MatchingProblemLandmarksToFrame<Scalar>::numApples());

        const double descriptor_distance =
            MatchingProblemLandmarksToFrame<Scalar>::computeDescriptorDistance(
                landmark_index, keypoint_index);
        min_descriptor_distance = std::min(min_descriptor_distance, descriptor_distance);
        max_descriptor_distance = std::max(max_descriptor_distance, descriptor_distance);

        //LOG(INFO) << "descriptor_distance: " << descriptor_distance;
        //const int hamming_distance = computeHammingDistance(landmark_index, keypoint_index);
        //CHECK_GE(hamming_distance, 0);
        //CHECK_LE(hamming_distance, descriptor_size_bits_);

        const int kPriority = 0;

        if (FLAGS_matcher_store_all_tested_pairs) {
          CHECK_LT(landmark_index, MatchingProblemLandmarksToFrame<Scalar>::all_tested_pairs_.size());
          MatchingProblemLandmarksToFrame<Scalar>::all_tested_pairs_[landmark_index].emplace_back(
              keypoint_index, landmark_index, MatchingProblemLandmarksToFrame<Scalar>::computeMatchScore(descriptor_distance), kPriority);
              //keypoint_index, landmark_index, computeMatchScore(hamming_distance), kPriority);
        }

        if (descriptor_distance < MatchingProblemLandmarksToFrame<Scalar>::descriptor_distance_threshold_) {
        ///if (hamming_distance < hamming_distance_threshold_) {
          CHECK_LT(landmark_index, candidates_for_landmarks->size());
          const double score =
              MatchingProblemLandmarksToFrame<Scalar>::computeMatchScore(descriptor_distance);
          (*candidates_for_landmarks)[landmark_index].emplace_back(
              keypoint_index, landmark_index, score, kPriority);
              //keypoint_index, landmark_index, computeMatchScore(hamming_distance), kPriority);
          ++num_matches;
        }
      }
    }
  }
  knn_post_processing_timer.Stop();
  VLOG(1) << "Got " << num_matches << " matches, min descriptor distance: "
      << min_descriptor_distance << ", max: " << max_descriptor_distance;
  method_timer.Stop();
}


}

#endif /* MATCHING_PROBLEM_LANDMARKS_TO_FRAME_KD_TREE_INL_H_ */
