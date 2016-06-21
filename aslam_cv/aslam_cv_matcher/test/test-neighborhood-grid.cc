#include <cmath>
#include <limits>
#include <random>

#include <aslam/common/entrypoint.h>
#include <aslam/matcher/matching-problem-landmarks-to-frame-kd-tree.h>
#include <eigen-checks/gtest.h>
#include <gtest/gtest.h>
#include <nabo/nabo.h>

namespace aslam {

std::unique_ptr<NeighborCellCountingGrid> createRandomGrid(
    double* min_x, double* max_x, double* min_y, double* max_y,
    size_t* num_bins_x,  size_t* num_bins_y) {
  CHECK_NOTNULL(min_x);
  CHECK_NOTNULL(max_x);
  CHECK_NOTNULL(min_y);
  CHECK_NOTNULL(max_y);
  CHECK_NOTNULL(num_bins_x);
  CHECK_NOTNULL(num_bins_y);

  const size_t kSeed = 342u;
  std::default_random_engine random_engine(kSeed);

  const double kMinMaxLimit = 2000.0;
  const double kMaxWidth = 5000.0;
  const double kMinWidth = 0.1;
  const int kMinNumBins = 1;
  const int kMaxNumBins = 5000;

  std::uniform_real_distribution<double> uniform_dist_grid_border_min(-kMinMaxLimit, kMinMaxLimit);
  std::uniform_real_distribution<double> uniform_dist_grid_width(kMinWidth, kMaxWidth);

  *min_x = uniform_dist_grid_border_min(random_engine);
  *min_y = uniform_dist_grid_border_min(random_engine);
  const double width = uniform_dist_grid_width(random_engine);
  *max_x = *min_x + width;
  *max_y = *min_y + width;
  CHECK_GT(*max_x, *min_x);
  CHECK_GT(*max_y, *min_y);

  std::uniform_int_distribution<int> uniform_dist_num_intervals(kMinNumBins, kMaxNumBins);

  *num_bins_x = static_cast<size_t>(uniform_dist_num_intervals(random_engine));
  CHECK_GE(*num_bins_x, static_cast<size_t>(kMinNumBins));
  CHECK_LE(*num_bins_x, static_cast<size_t>(kMaxNumBins));
  *num_bins_y = static_cast<size_t>(uniform_dist_num_intervals(random_engine));
  CHECK_GE(*num_bins_y, static_cast<size_t>(kMinNumBins));
  CHECK_LE(*num_bins_y, static_cast<size_t>(kMaxNumBins));

  std::unique_ptr<NeighborCellCountingGrid> grid(
      new NeighborCellCountingGrid(*min_x, *max_x, *min_y, *max_y, *num_bins_x, *num_bins_y));

  return grid;
}

TEST(TestNeighborhoodGrid, TestSimpleHandcrafted) {
  const double min_x = 0.0;
  const double max_x = 10.0;
  const double min_y = 0.0;
  const double max_y = 10.0;
  const size_t num_bins_x = 10u;
  const size_t num_bins_y = 10u;

  NeighborCellCountingGrid grid(min_x, max_x, min_y, max_y, num_bins_x, num_bins_y);

  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  // 0 0 0 0 0 0 0 0 0 0
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 0);

  grid.addElementToGrid(Eigen::Vector2d(5.5, 5.5));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 0 0 0 0 0 0 0 0 0 0
  // 1 | 0 0 0 0 0 0 0 0 0 0
  // 2 | 0 0 0 0 0 0 0 0 0 0
  // 3 | 0 0 0 0 0 0 0 0 0 0
  // 4 | 0 0 0 0 1 1 1 0 0 0
  // 5 | 0 0 0 0 1 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 0 0 0 0
  // 9 | 0 0 0 0 0 0 0 0 0 0
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 1);

  grid.addElementToGrid(Eigen::Vector2d(0.0, 0.0));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 0 0 0 0
  // 1 | 1 1 0 0 0 0 0 0 0 0
  // 2 | 0 0 0 0 0 0 0 0 0 0
  // 3 | 0 0 0 0 0 0 0 0 0 0
  // 4 | 0 0 0 0 1 1 1 0 0 0
  // 5 | 0 0 0 0 1 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 0 0 0 0
  // 9 | 0 0 0 0 0 0 0 0 0 0
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 1);

  grid.addElementToGrid(3.0, 4.0);
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 0 0 0 0
  // 1 | 1 1 0 0 0 0 0 0 0 0
  // 2 | 0 0 0 0 0 0 0 0 0 0
  // 3 | 0 0 1 1 1 0 0 0 0 0
  // 4 | 0 0 1 1 2 1 1 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 0 0 0 0
  // 9 | 0 0 0 0 0 0 0 0 0 0
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 2);


  grid.addElementToGrid(Eigen::Vector2d(5.99, 3.3));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 0 0 0 0
  // 1 | 1 1 0 0 0 0 0 0 0 0
  // 2 | 0 0 0 0 1 1 1 0 0 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 0 0 0 0
  // 9 | 0 0 0 0 0 0 0 0 0 0
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 3);

  grid.addElementToGrid(Eigen::Vector2d(7.8, 1.999));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 1 0
  // 1 | 1 1 0 0 0 0 1 1 1 0
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 0 0 0 0
  // 9 | 0 0 0 0 0 0 0 0 0 0
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 3);

  grid.addElementToGrid(9.999, 9.999);
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 1 0
  // 1 | 1 1 0 0 0 0 1 1 1 0
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 0 0 1 1
  // 9 | 0 0 0 0 0 0 0 0 1 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 3);

  grid.addElementToGrid(Eigen::Vector2d(7.4, 9.999));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 1 0
  // 1 | 1 1 0 0 0 0 1 1 1 0
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 1 0 0 0
  // 7 | 0 0 0 0 0 0 0 0 0 0
  // 8 | 0 0 0 0 0 0 1 1 2 1
  // 9 | 0 0 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 3);

  grid.addElementToGrid(7.01, 7.999);
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 1 0
  // 1 | 1 1 0 0 0 0 1 1 1 0
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 1 0
  // 7 | 0 0 0 0 0 0 1 1 1 0
  // 8 | 0 0 0 0 0 0 2 2 3 1
  // 9 | 0 0 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 3);

  grid.addElementToGrid(Eigen::Vector2d(9.0, 7.999));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 1 0
  // 1 | 1 1 0 0 0 0 1 1 1 0
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 0 0 0 0 0 0 2 2 4 2
  // 9 | 0 0 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 4);

  grid.addElementToGrid(9.999, 0.999);
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 2 1
  // 1 | 1 1 0 0 0 0 1 1 2 1
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 0 0 0 0 0 0 2 2 4 2
  // 9 | 0 0 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 4);

  grid.addElementToGrid(Eigen::Vector2d(0.999, 9.999));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 2 1
  // 1 | 1 1 0 0 0 0 1 1 2 1
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 1 2 1 1 0 0 0
  // 4 | 0 0 1 1 3 2 2 0 0 0
  // 5 | 0 0 1 1 2 1 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 1 1 0 0 0 0 2 2 4 2
  // 9 | 1 1 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 4);

  grid.addElementToGrid(4.3, 4.8);
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 2 1
  // 1 | 1 1 0 0 0 0 1 1 2 1
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 2 3 2 1 0 0 0
  // 4 | 0 0 1 2 4 3 2 0 0 0
  // 5 | 0 0 1 2 3 2 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 1 1 0 0 0 0 2 2 4 2
  // 9 | 1 1 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 4);

  grid.addElementToGrid(Eigen::Vector2d(4.7, 4.01));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 2 1
  // 1 | 1 1 0 0 0 0 1 1 2 1
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 1 3 4 3 1 0 0 0
  // 4 | 0 0 1 3 5 4 2 0 0 0
  // 5 | 0 0 1 3 4 3 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 1 1 0 0 0 0 2 2 4 2
  // 9 | 1 1 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 5);

  grid.addElementToGrid(3.1, 4.01);
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 2 1
  // 1 | 1 1 0 0 0 0 1 1 2 1
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 2 4 5 3 1 0 0 0
  // 4 | 0 0 2 4 6 4 2 0 0 0
  // 5 | 0 0 2 4 5 3 1 0 0 0
  // 6 | 0 0 0 0 1 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 1 1 0 0 0 0 2 2 4 2
  // 9 | 1 1 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 6);


  grid.addElementToGrid(Eigen::Vector2d(3.1, 5.01));
  //   | 0 1 2 3 4 5 6 7 8 9
  //----------------------
  // 0 | 1 1 0 0 0 0 1 1 2 1
  // 1 | 1 1 0 0 0 0 1 1 2 1
  // 2 | 0 0 0 0 1 1 2 1 1 0
  // 3 | 0 0 2 4 5 3 1 0 0 0
  // 4 | 0 0 3 5 7 4 2 0 0 0
  // 5 | 0 0 3 5 6 3 1 0 0 0
  // 6 | 0 0 1 1 2 1 2 1 2 1
  // 7 | 0 0 0 0 0 0 1 1 2 1
  // 8 | 1 1 0 0 0 0 2 2 4 2
  // 9 | 1 1 0 0 0 0 1 1 2 1
  EXPECT_EQ(grid.getMaxNeighborhoodCellCount(), 7);

  Eigen::MatrixXi ground_truth_neighbor_cell_count = Eigen::MatrixXi::Zero(10, 10);
  ground_truth_neighbor_cell_count << 1, 1, 0, 0, 0, 0, 1, 1, 2, 1,
                                      1, 1, 0, 0, 0, 0, 1, 1, 2, 1,
                                      0, 0, 0, 0, 1, 1, 2, 1, 1, 0,
                                      0, 0, 2, 4, 5, 3, 1, 0, 0, 0,
                                      0, 0, 3, 5, 7, 4, 2, 0, 0, 0,
                                      0, 0, 3, 5, 6, 3, 1, 0, 0, 0,
                                      0, 0, 1, 1, 2, 1, 2, 1, 2, 1,
                                      0, 0, 0, 0, 0, 0, 1, 1, 2, 1,
                                      1, 1, 0, 0, 0, 0, 2, 2, 4, 2,
                                      1, 1, 0, 0, 0, 0, 1, 1, 2, 1;

  EXPECT_TRUE(EIGEN_MATRIX_EQUAL(grid.grid_neighboring_cell_count_,
                                 ground_truth_neighbor_cell_count));
}

TEST(TestNeighborhoodGrid, TestCornerCases) {
  double min_x, max_x, min_y, max_y = 0.0;
  size_t num_bins_x, num_bins_y = 0u;
  std::unique_ptr<NeighborCellCountingGrid> grid = createRandomGrid(
      &min_x, &max_x, &min_y, &max_y, &num_bins_x, &num_bins_y);
  CHECK_GT(max_x, min_x);
  CHECK_GT(max_y, min_y);
  CHECK(grid);

  const double kSmallFloat = 0.00001;
  EXPECT_DEATH(grid->addElementToGrid(Eigen::Vector2d(min_x - kSmallFloat, min_y)), "");
  EXPECT_DEATH(grid->addElementToGrid(Eigen::Vector2d(min_x, min_y - kSmallFloat)), "");
  EXPECT_DEATH(grid->addElementToGrid(Eigen::Vector2d(max_x + kSmallFloat, max_y)), "");
  EXPECT_DEATH(grid->addElementToGrid(Eigen::Vector2d(max_x, max_y + kSmallFloat)), "");
  EXPECT_DEATH(grid->addElementToGrid(min_x - kSmallFloat, min_y), "");
  EXPECT_DEATH(grid->addElementToGrid(min_x, min_y - kSmallFloat), "");
  EXPECT_DEATH(grid->addElementToGrid(max_x + kSmallFloat, max_y), "");
  EXPECT_DEATH(grid->addElementToGrid(max_x, max_y + kSmallFloat), "");
}

TEST(TestNeighborhoodGrid, TestLargeRandom) {
  const size_t kSeed = 34322u;
  std::default_random_engine random_engine(kSeed);

  const size_t kNumTrials = 500u;

  for (size_t trial_idx = 0u; trial_idx < kNumTrials; ++trial_idx) {
    double min_x, max_x, min_y, max_y = 0.0;
    size_t num_bins_x, num_bins_y = 0u;
    std::unique_ptr<NeighborCellCountingGrid> grid = createRandomGrid(
        &min_x, &max_x, &min_y, &max_y, &num_bins_x, &num_bins_y);
    CHECK_GT(max_x, min_x);
    CHECK_GT(max_y, min_y);
    CHECK(grid);

    std::uniform_real_distribution<double> uniform_dist_x(min_x, max_x);
    std::uniform_real_distribution<double> uniform_dist_y(min_y, max_y);
    const size_t kNumElementsToAdd = 3000u;
    Eigen::MatrixXd elements = Eigen::MatrixXd::Zero(2, kNumElementsToAdd);
    for (size_t element_idx = 0u; element_idx < kNumElementsToAdd; ++element_idx) {
      const double position_x = uniform_dist_x(random_engine);
      const double position_y = uniform_dist_y(random_engine);
      grid->addElementToGrid(position_x, position_y);
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

    const double interval_x = (max_x - min_x) / static_cast<double>(num_bins_x);
    const double interval_y = (max_y - min_y) / static_cast<double>(num_bins_y);
    const double kRadius = std::min(interval_x, interval_y);

    CHECK(knn_index);
    knn_index->knn(
        elements, indices, distances, kNumNeighborsToSearch, kSearchNNEpsilon, kOptionFlags,
        kRadius);

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

    EXPECT_LE(max_num_neighbors, grid->getMaxNeighborhoodCellCount());
  }
}
}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
