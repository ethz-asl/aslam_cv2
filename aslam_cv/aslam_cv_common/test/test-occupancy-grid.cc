#include <aslam/common/entrypoint.h>
#include <aslam/common/occupancy-grid.h>
#include <eigen-checks/gtest.h>
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <opencv2/highgui/highgui.hpp>

typedef aslam::WeightedKeypoint<double, double, size_t> Point;
typedef aslam::WeightedOccupancyGrid<Point> WeightedOccupancyGrid;

TEST(OccupancyGrid, DISABLED_addPointOrReplaceWeakestIfCellFull) {
  WeightedOccupancyGrid grid(2.0, 2.0, 1.0, 1.0);

  static constexpr size_t kMaxNumPointsPerCell = 2u;
  EXPECT_TRUE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 0.0, 0.9, 0), kMaxNumPointsPerCell));
  EXPECT_TRUE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 0.0, 1.0, 1), kMaxNumPointsPerCell));
  EXPECT_FALSE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 0.0, 0.8, 2), kMaxNumPointsPerCell));
  // Higher weight replaces point.
  EXPECT_TRUE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 0.0, 1.1, 3), kMaxNumPointsPerCell));
  // The cell should now contain the second and last added points with a weight of 1.1 and 1.0.
  const WeightedOccupancyGrid::PointList& cell = grid.getGridCell(0.5, 0.0);
  ASSERT_EQ(cell.size(), 2u);
  EXPECT_EQ(cell[0].weight, 1.1);
  EXPECT_EQ(cell[1].weight, 1.0);
  EXPECT_EQ(cell[0].id, 3u);
  EXPECT_EQ(cell[1].id, 1u);

  ASSERT_TRUE(grid.getGridCell(0.0, 1.5).empty());
  ASSERT_TRUE(grid.getGridCell(1.5, 1.5).empty());
  ASSERT_TRUE(grid.getGridCell(1.5, 0.0).empty());

  // Test on second cell.
  EXPECT_TRUE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 1.0, 1.0, 4), kMaxNumPointsPerCell));
  EXPECT_TRUE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 1.5, 1.0, 5), kMaxNumPointsPerCell));
  EXPECT_FALSE(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.5, 1.5, 1.0, 6), kMaxNumPointsPerCell));
}

TEST(OccupancyGrid, DISABLED_removeWeightedPointsFromOverFullCells) {
  WeightedOccupancyGrid grid(2.0, 2.0, 1.0, 1.0);

  grid.addPointUnconditional(Point(0.5, 0.0, 0.3, 0));
  grid.addPointUnconditional(Point(0.5, 0.0, 0.2, 1));
  grid.addPointUnconditional(Point(0.5, 0.0, 0.4, 2));

  ASSERT_EQ(grid.getNumPoints(), 3u);
  ASSERT_TRUE(grid.getGridCell(0.0, 1.5).empty());
  ASSERT_TRUE(grid.getGridCell(1.5, 1.5).empty());
  ASSERT_TRUE(grid.getGridCell(1.5, 0.0).empty());

  // Remove points from full cells.
  static constexpr size_t kMaxNumPointsPerCell = 2u;
  EXPECT_EQ(grid.removeWeightedPointsFromOverFullCells(kMaxNumPointsPerCell), 1u);

  const WeightedOccupancyGrid::PointList& cell = grid.getGridCell(0.5, 0.5);
  ASSERT_EQ(cell.size(), kMaxNumPointsPerCell);

  // The points should be sorted by descending score.
  EXPECT_EQ(cell[0].id, 2u);
  EXPECT_EQ(cell[1].id, 0u);
}

TEST(OccupancyGrid, DISABLED_CellIndexing) {
  WeightedOccupancyGrid grid(2.0, 2.0, 1.0, 1.0);

  // Add N points to cell: 2:00 1:10 0:01 1:11
  grid.addPointUnconditional(Point(0.5, 0.5, 1.0, 0));
  grid.addPointUnconditional(Point(0.5, 0.5, 1.0, 1));
  grid.addPointUnconditional(Point(0.5, 1.5, 1.0, 0));
  grid.addPointUnconditional(Point(1.5, 1.5, 1.0, 0));

  EXPECT_EQ(grid.getGridCell(0.5, 0.5).size(), 2u);
  EXPECT_EQ(grid.getGridCell(1.5, 0.5).size(), 0u);
  EXPECT_EQ(grid.getGridCell(0.5, 1.5).size(), 1u);
  EXPECT_EQ(grid.getGridCell(1.5, 1.5).size(), 1u);

  // Test corner cases.
  grid.reset();
  grid.addPointUnconditional(Point(0.5, 0.5, 1.0, 0)); // Cell 0,0

  grid.addPointUnconditional(Point(0.5, 1.0, 1.0, 1)); // Cell 0,1
  grid.addPointUnconditional(Point(0.5, 1.0, 1.0, 1)); // Cell 0,1

  grid.addPointUnconditional(Point(1.0, 1.0, 1.0, 2)); // Cell 1,1
  grid.addPointUnconditional(Point(1.0, 1.0, 1.0, 2)); // Cell 1,1
  grid.addPointUnconditional(Point(1.0, 1.0, 1.0, 2)); // Cell 1,1

  EXPECT_EQ(grid.getGridCell(0.5, 0.5).size(), 1u);
  EXPECT_EQ(grid.getGridCell(1.5, 0.5).size(), 0u);
  EXPECT_EQ(grid.getGridCell(0.5, 1.5).size(), 2u);
  EXPECT_EQ(grid.getGridCell(1.5, 1.5).size(), 3u);
}

TEST(OccupancyGrid, DISABLED_AddInvalidPointCoordinates) {
  const double kGridSize = 2.0;
  WeightedOccupancyGrid grid(kGridSize, kGridSize, 1.0, 1.0);

  EXPECT_DEATH(grid.addPointUnconditional(Point(-0.1, 0.1, 1.0, 0)), "^");
  EXPECT_DEATH(grid.addPointUnconditional(Point(-0.1, -0.1, 1.0, 1)), "^");
  EXPECT_DEATH(grid.addPointUnconditional(Point(0.1, -0.1, 1.0, 2)), "^");
  EXPECT_DEATH(grid.addPointUnconditional(Point(0.1, kGridSize, 1.0, 3)), "^");
  EXPECT_DEATH(grid.addPointUnconditional(Point(kGridSize, 0.1, 1.0, 4)), "^");
  EXPECT_DEATH(grid.addPointUnconditional(Point(kGridSize, kGridSize, 1.0, 5)), "^");

  static constexpr size_t kMaxNumPointsPerCell = 2u;
  EXPECT_DEATH(
      grid.addPointOrReplaceWeakestIfCellFull(Point(-0.1, 0.1, 1.0, 0),
                                              kMaxNumPointsPerCell), "^");
  EXPECT_DEATH(
      grid.addPointOrReplaceWeakestIfCellFull(Point(-0.1, -0.1, 1.0, 1),
                                              kMaxNumPointsPerCell), "^");
  EXPECT_DEATH(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.1, -0.1, 1.0, 2),
                                              kMaxNumPointsPerCell), "^");
  EXPECT_DEATH(
      grid.addPointOrReplaceWeakestIfCellFull(Point(0.1, kGridSize, 1.0, 3),
                                              kMaxNumPointsPerCell), "^");
  EXPECT_DEATH(
      grid.addPointOrReplaceWeakestIfCellFull(Point(kGridSize, 0.1, 1.0, 4),
                                              kMaxNumPointsPerCell), "^");
  EXPECT_DEATH(
      grid.addPointOrReplaceWeakestIfCellFull(Point(kGridSize, kGridSize, 1.0, 5),
                                              kMaxNumPointsPerCell), "^");
}

TEST(OccupancyGrid, AddPointOrReplaceWeakestNearestPoints) {
  const double kGridSize = 50.0;
  const double kMinDistanceBetweenPoints = 3.0;
  WeightedOccupancyGrid grid(kGridSize, kGridSize,
                             kMinDistanceBetweenPoints, kMinDistanceBetweenPoints);

  // Test rejection of the same point that violates the min. distance.
  grid.addPointOrReplaceWeakestNearestPoints(Point(0.1, 0.1, 1.0, 0), kMinDistanceBetweenPoints);
  grid.addPointOrReplaceWeakestNearestPoints(Point(0.1, 0.1, 3.0, 1), kMinDistanceBetweenPoints);
  grid.addPointOrReplaceWeakestNearestPoints(Point(0.1, 0.1, 2.0, 2), kMinDistanceBetweenPoints);
  ASSERT_EQ(grid.getNumPoints(), 1u);
  EXPECT_EQ(grid.getGridCell(0.1, 0.1).size(), 1u);
  EXPECT_EQ(grid.getGridCell(0.1, 0.1)[0].id, 1u);  // Point with id 1 has highest score of 3.0

  // TODO(schneith): Add more tests.
}

TEST(OccupancyGrid, DISABLED_GetOccupancyMask) {
  // Create a grid with some points.
  const double kGridSize = 100.0;
  const double kCellSize = 25.0;
  WeightedOccupancyGrid grid(kGridSize, kGridSize, kCellSize, kCellSize);
  grid.addPointUnconditional(Point(37, 37, 1.0, 0));
  grid.addPointUnconditional(Point(37, 37, 1.0, 0));
  grid.addPointUnconditional(Point(75, 75, 1.0, 2));
  ASSERT_EQ(grid.getNumPoints(), 3u);

  // Get the mask.
  const WeightedOccupancyGrid::CoordinatesType kMaskRadiusAroundPointsPx = 10.0;
  const size_t kMaxPointsPerCell = 2u;
  cv::Mat mask = grid.getOccupancyMask(kMaskRadiusAroundPointsPx, kMaxPointsPerCell);

  // Cell 0,0 should be masked completly as it is full. Check that the cell is completely masked.
  cv::Mat cell_00(mask, cv::Rect(25, 25, kMaskRadiusAroundPointsPx, kMaskRadiusAroundPointsPx));
  EXPECT_EQ(cv::countNonZero(cell_00), 0);

  cv::Mat cell_00_oversize(mask, cv::Rect(24, 24, kCellSize + 2, kCellSize + 2));
  EXPECT_EQ(cv::countNonZero(cell_00_oversize), 4 * 25 + 4);

  // Border cases for cell mask.
  EXPECT_EQ(mask.at<unsigned char>(25, 25), 0);
  EXPECT_EQ(mask.at<unsigned char>(25, 49), 0);
  EXPECT_EQ(mask.at<unsigned char>(49, 25), 0);
  EXPECT_EQ(mask.at<unsigned char>(49, 49), 0);

  EXPECT_EQ(mask.at<unsigned char>(24, 24), 255);
  EXPECT_EQ(mask.at<unsigned char>(24, 50), 255);
  EXPECT_EQ(mask.at<unsigned char>(50, 24), 255);
  EXPECT_EQ(mask.at<unsigned char>(50, 50), 255);

  // Check that the neighborhood around the points is masked.
  EXPECT_EQ(mask.at<unsigned char>(75, 75), 0);
  EXPECT_EQ(mask.at<unsigned char>(75 + kMaskRadiusAroundPointsPx, 75), 0);
  EXPECT_EQ(mask.at<unsigned char>(75 - kMaskRadiusAroundPointsPx, 75), 0);
  EXPECT_EQ(mask.at<unsigned char>(75, 75 + kMaskRadiusAroundPointsPx), 0);
  EXPECT_EQ(mask.at<unsigned char>(75, 75 - kMaskRadiusAroundPointsPx), 0);

  // Further away from the point the image shouldn't be masked.
  EXPECT_EQ(mask.at<unsigned char>(75 + kMaskRadiusAroundPointsPx + 1, 75), 255);
  EXPECT_EQ(mask.at<unsigned char>(75 - kMaskRadiusAroundPointsPx - 1, 75), 255);
  EXPECT_EQ(mask.at<unsigned char>(75, 75 + kMaskRadiusAroundPointsPx + 1), 255);
  EXPECT_EQ(mask.at<unsigned char>(75, 75 - kMaskRadiusAroundPointsPx - 1), 255);
}

TEST(OccupancyGrid, DISABLED_InvalidGridParameters) {
  // Zero sized grid.
  EXPECT_DEATH(WeightedOccupancyGrid(0.0, 1.0, 1.0, 1.0), "^");
  EXPECT_DEATH(WeightedOccupancyGrid(1.0, 0.0, 1.0, 1.0), "^");
  EXPECT_DEATH(WeightedOccupancyGrid(0.0, 0.0, 1.0, 1.0), "^");

  // Cell size bigger than grid.
  EXPECT_DEATH(WeightedOccupancyGrid(1.0, 1.0, 2.0, 0.5), "^");
  EXPECT_DEATH(WeightedOccupancyGrid(1.0, 1.0, 0.5, 2.0), "^");
  EXPECT_DEATH(WeightedOccupancyGrid(1.0, 1.0, 2.0, 2.0), "^");
}

ASLAM_UNITTEST_ENTRYPOINT
