#include <cmath>
#include <limits>
#include <random>

#include <aslam/common/entrypoint.h>
#include <aslam/matcher/matching-problem-landmarks-to-frame-kd-tree.h>
#include <gtest/gtest.h>
#include <nabo/nabo.h>

namespace aslam {

TEST(Bla, TestNeighborhoodGrid) {
  const size_t kSeed = 342u;
  std::default_random_engine random_engine(kSeed);

  const double kMinMaxLimit = 2000.0;
  const double kMaxWidth = 5000.0;
  const double kMinWidth = 0.1;
  const int kMinNumIntervals = 1;
  const int kMaxNumIntervals = 5000;

  std::uniform_real_distribution<double> uniform_dist_grid_border_min(-kMinMaxLimit, kMinMaxLimit);
  std::uniform_real_distribution<double> uniform_dist_grid_width(kMinWidth, kMaxWidth);

  const double min_x = uniform_dist_grid_border_min(random_engine);
  const double min_y = uniform_dist_grid_border_min(random_engine);
  const double width = uniform_dist_grid_width(random_engine);
  const double max_x = min_x + width;
  const double max_y = min_y + width;
  CHECK_GT(max_x, min_x);
  CHECK_GT(max_y, min_y);

  std::uniform_int_distribution<int> uniform_dist_num_intervals(kMinNumIntervals, kMaxNumIntervals);

  const size_t num_intervals_x = static_cast<size_t>(uniform_dist_num_intervals(random_engine));
  CHECK_LT(num_intervals_x, static_cast<size_t>(kMaxNumIntervals));
  const size_t num_intervals_y = static_cast<size_t>(uniform_dist_num_intervals(random_engine));
  CHECK_LT(num_intervals_y, static_cast<size_t>(kMaxNumIntervals));

  NeighborCellCountingGrid grid(min_x, max_x, min_y, max_y, num_intervals_x, num_intervals_y);

  std::uniform_real_distribution<double> uniform_dist_x(min_x, max_x);
  std::uniform_real_distribution<double> uniform_dist_y(min_y, max_y);
  const size_t kNumElementsToAdd = 3000u;
  Eigen::MatrixXd elements = Eigen::MatrixXd::Zero(2, kNumElementsToAdd);
  for (size_t element_idx = 0u; element_idx < kNumElementsToAdd; ++element_idx) {
    const double position_x = uniform_dist_x(random_engine);
    const double position_y = uniform_dist_y(random_engine);
    grid.addElementToGrid(position_x, position_y);
    elements(0, element_idx) = position_x;
    elements(1, element_idx) = position_y;
  }

  const int kDimVectors = 2;
  const int kCollectTouchStatistics = 0;
  std::shared_ptr<Nabo::NNSearchD> knn_index;
  knn_index.reset(Nabo::NNSearchD::createKDTreeLinearHeap(
      elements, kDimVectors, kCollectTouchStatistics));

  const int kNumNeighborsToSearch = static_cast<int>(kNumElementsToAdd);
  Eigen::MatrixXi indices = Eigen::MatrixXi::Zero(kNumNeighborsToSearch, kNumElementsToAdd);
  Eigen::MatrixXd distances = Eigen::MatrixXd::Zero(kNumNeighborsToSearch, kNumElementsToAdd);
  const double kSearchNNEpsilon = 0.0;
  const unsigned kOptionFlags = Nabo::NNSearchD::ALLOW_SELF_MATCH;
  const double kRadius = width;
  LOG(INFO) << "kRadius: " << kRadius;

  CHECK(knn_index);
  knn_index->knn(
      elements, indices, distances, kNumNeighborsToSearch, kSearchNNEpsilon, kOptionFlags, kRadius);

  size_t max_num_neighbors = 0u;
  for (int element_idx = 0; element_idx < kNumElementsToAdd; ++element_idx) {
    size_t num_neigbors_of_this_elements = 0u;
    for (int nearest_neighbor_idx = 0; nearest_neighbor_idx < kNumNeighborsToSearch;
        ++nearest_neighbor_idx) {
      const int nn_index = indices(nearest_neighbor_idx, element_idx);
      const double distance = distances(nearest_neighbor_idx, element_idx);

      if (nn_index == -1) {
        CHECK_EQ(distance,  std::numeric_limits<double>::infinity());
        break;  // No more results.
      }

      if (distance <= kRadius) {
        ++num_neigbors_of_this_elements;
      }
    }

    max_num_neighbors = std::max(num_neigbors_of_this_elements, max_num_neighbors);
  }

  LOG(INFO) << "Max num neighbors: " << max_num_neighbors;
  LOG(INFO) << "grid: " << grid.getMaxNeighborhoodCellCount();
}


}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
