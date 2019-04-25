#include <random>

#include <aslam/cameras/camera-pinhole.h>
#include <aslam/common/entrypoint.h>
#include <gtest/gtest.h>

#include "aslam/lines/homogeneous-line-helpers.h"
#include "aslam/lines/homogeneous-line.h"
#include "aslam/lines/line-2d-with-angle-helpers.h"
#include "aslam/lines/line-2d-with-angle.h"

namespace aslam {

template <class PointType>
class RandomPointGenerator final {
 public:
  static constexpr size_t kSeed = 42u;
  static constexpr double kMaxLineCoordinate = 1e3;
  RandomPointGenerator()
      : random_engine_(kSeed),
        uniform_line_coordinate_(-kMaxLineCoordinate, kMaxLineCoordinate) {}
  ~RandomPointGenerator() = default;

  PointType generateRandomPoint() {
    PointType point;
    point.setRandom();
    const double scale = uniform_line_coordinate_(random_engine_);
    point *= scale;
    return point;
  }

 private:
  std::default_random_engine random_engine_;
  std::uniform_real_distribution<double> uniform_line_coordinate_;
};

template <class LineType>
LineType generateRandomLine(
    RandomPointGenerator<typename LineType::PointType>*
        random_point_generator) {
  CHECK_NOTNULL(random_point_generator);
  while (true) {
    const typename LineType::PointType point_a =
        random_point_generator->generateRandomPoint();
    const typename LineType::PointType point_b =
        random_point_generator->generateRandomPoint();

    if ((point_a - point_b).squaredNorm() <
        std::numeric_limits<double>::epsilon()) {
      continue;
    }

    return LineType(point_a, point_b);
  }
}

TEST(LineTests, Test2dLineWithAngle) {
  Line2dWithAngle::PointType kPointZeroZero(0.0, 0.0);
  Line2dWithAngle::PointType kPointOneZero(1.0, 0.0);
  Line2dWithAngle::PointType kPointZeroOne(0.0, 1.0);

  // Points are not lines!
  EXPECT_DEATH(Line2dWithAngle(kPointZeroZero, kPointZeroZero), "");

  // Horizontal line.
  //  s ---- e
  //
  const Line2dWithAngle kHorizontalLineA(kPointZeroZero, kPointOneZero);
  EXPECT_DOUBLE_EQ(kHorizontalLineA.getAngleWrtXAxisRad(), 0.0);

  // Horizontal line.
  //  e ---- s
  //
  const Line2dWithAngle kHorizontalLineB(kPointOneZero, kPointZeroZero);
  EXPECT_DOUBLE_EQ(kHorizontalLineB.getAngleWrtXAxisRad(), 0.0);

  // Vertical line.
  //  s
  //  |
  //  e
  const Line2dWithAngle kVerticalLineA(kPointZeroZero, kPointZeroOne);
  EXPECT_DOUBLE_EQ(kVerticalLineA.getAngleWrtXAxisRad(), M_PI_2);

  // Vertical line.
  //  e
  //  |
  //  s
  const Line2dWithAngle kVerticalLineB(kPointZeroOne, kPointZeroZero);
  EXPECT_DOUBLE_EQ(kVerticalLineB.getAngleWrtXAxisRad(), M_PI_2);

  // Do multiple 360 degree circles with arbitrary line length and verify the
  // angle.
  constexpr size_t kSeed = 42u;
  std::default_random_engine random_engine(kSeed);
  std::uniform_int_distribution<size_t> uniform_dist_binary(0u, 1u);
  constexpr double kMaxLineLength = 1e10;
  constexpr double kMinLineLength = std::numeric_limits<double>::epsilon();
  std::uniform_real_distribution<double> uniform_line_length(
      kMinLineLength, kMaxLineLength);
  double angle_rad = 0.0;
  constexpr double kAngleIncrementDeg = 0.1;
  while (angle_rad < 1000.0) {
    const double line_length = uniform_line_length(random_engine);
    const Line2dWithAngle line =
        uniform_dist_binary(random_engine)
            ? Line2dWithAngle(
                  kPointZeroZero,
                  line_length * Line2dWithAngle::PointType(
                                    std::cos(angle_rad), std::sin(angle_rad)))
            : Line2dWithAngle(
                  line_length * Line2dWithAngle::PointType(
                                    std::cos(angle_rad), std::sin(angle_rad)),
                  kPointZeroZero);

    const double angle_rad_mod_2pi = std::fmod(angle_rad, 2.0 * M_PI);
    const double ground_truth_angle_rad =
        angle_rad_mod_2pi < M_PI ? angle_rad_mod_2pi : angle_rad_mod_2pi - M_PI;

    const double line_angle_rad = line.getAngleWrtXAxisRad();

    // 179.9999deg matches 0.0deg. Therefore we need to check for both cases.
    const double angle_diff_vs_gt_rad =
        std::abs(line_angle_rad - ground_truth_angle_rad);
    const double angle_diff_vs_gt_rad_mod_pi =
        std::abs(line_angle_rad - ground_truth_angle_rad + M_PI);
    EXPECT_TRUE(
        angle_diff_vs_gt_rad < 1e-4 || angle_diff_vs_gt_rad_mod_pi < 1e-4);

    angle_rad += kAngleIncrementDeg;
  }
}

TEST(LineTests, Test2dLineWithAngleRelativeAngles) {
  const Line2dWithAngle::PointType kPointZeroZero(0.0, 0.0);
  const Line2dWithAngle::PointType kPointOneZero(1.0, 0.0);
  const Line2dWithAngle::PointType kPointZeroOne(0.0, 1.0);

  // Horizontal line.
  //  s ---- e
  //
  const Line2dWithAngle kHorizontalLineA(kPointZeroZero, kPointOneZero);

  // Horizontal line.
  //  e ---- s
  //
  const Line2dWithAngle kHorizontalLineB(kPointOneZero, kPointZeroZero);

  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kHorizontalLineB), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kHorizontalLineA), 0.0);

  // Vertical line.
  //  s
  //  |
  //  e
  const Line2dWithAngle kVerticalLineA(kPointZeroZero, kPointZeroOne);

  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kVerticalLineA),
      M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kHorizontalLineA),
      M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kVerticalLineA),
      M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kHorizontalLineB),
      M_PI_2);

  // Vertical line.
  //  e
  //  |
  //  s
  const Line2dWithAngle kVerticalLineB(kPointZeroOne, kPointZeroZero);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kVerticalLineB), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kVerticalLineA), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kVerticalLineB),
      M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kHorizontalLineA),
      M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kVerticalLineB),
      M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kHorizontalLineB),
      M_PI_2);

  // Diagonal line.
  // s
  //  \
  //   e
  constexpr double kLineLength = 10.0;
  constexpr double kAngleRad = M_PI / 4.0;
  const Line2dWithAngle kDiagonalLineA(
      kPointZeroZero,
      kLineLength *
          Line2dWithAngle::PointType(std::cos(kAngleRad), std::sin(kAngleRad)));

  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kDiagonalLineA), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kHorizontalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kDiagonalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kHorizontalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kDiagonalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kVerticalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kDiagonalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kVerticalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kDiagonalLineA),
      M_PI / 4.0);

  // Diagonal line.
  // e
  //  \
  //   s
  const Line2dWithAngle kDiagonalLineB(
      kPointZeroZero,
      kLineLength * Line2dWithAngle::PointType(
                        -std::cos(kAngleRad), -std::sin(kAngleRad)));
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kDiagonalLineB), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kDiagonalLineA), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kDiagonalLineB), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kHorizontalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kDiagonalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kHorizontalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kDiagonalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kVerticalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kDiagonalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kVerticalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kDiagonalLineB),
      M_PI / 4.0);

  // Diagonal line.
  //   e
  //  /
  // s
  const Line2dWithAngle kDiagonalLineC(
      kPointZeroZero,
      kLineLength * Line2dWithAngle::PointType(
                        std::cos(kAngleRad), -std::sin(kAngleRad)));
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kDiagonalLineC), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kDiagonalLineA), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kDiagonalLineC), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kDiagonalLineB), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kDiagonalLineC), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kHorizontalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kDiagonalLineC),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kHorizontalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kDiagonalLineC),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kVerticalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kDiagonalLineC),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kVerticalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kDiagonalLineC),
      M_PI / 4.0);

  // Diagonal line.
  //   s
  //  /
  // e
  const Line2dWithAngle kDiagonalLineD(
      kPointZeroZero,
      kLineLength * Line2dWithAngle::PointType(
                        -std::cos(kAngleRad), std::sin(kAngleRad)));
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineD, kDiagonalLineD), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineD, kDiagonalLineC), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kDiagonalLineD), 0.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kDiagonalLineA), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineA, kDiagonalLineC), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kDiagonalLineB), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineB, kDiagonalLineC), M_PI_2);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kHorizontalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineA, kDiagonalLineC),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kHorizontalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kHorizontalLineB, kDiagonalLineC),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kVerticalLineA),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineA, kDiagonalLineC),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kDiagonalLineC, kVerticalLineB),
      M_PI / 4.0);
  EXPECT_DOUBLE_EQ(
      getAngleInRadiansBetweenLines2d(kVerticalLineB, kDiagonalLineC),
      M_PI / 4.0);
}

TEST(LineTests, Test2dLineIntersectionPoint) {
  const Line2d::PointType kPointZeroZero(0.0, 0.0);
  const Line2d::PointType kPointOneZero(1.0, 0.0);
  const Line2d::PointType kPointZeroOne(0.0, 1.0);
  const Line2d::PointType kPointOneOne(1.0, 1.0);

  // Horizontal line.
  //  s ---- e
  //
  const Line2d kHorizontalLineA(kPointZeroZero, kPointOneZero);
  Line2d::PointType intersecting_point;
  EXPECT_FALSE(
      getIntersectingPoint(
          kHorizontalLineA, kHorizontalLineA, &intersecting_point));

  // Horizontal line.
  //  e ---- s
  //
  const Line2d kHorizontalLineB(kPointOneZero, kPointZeroZero);
  EXPECT_FALSE(
      getIntersectingPoint(
          kHorizontalLineB, kHorizontalLineA, &intersecting_point));
  EXPECT_FALSE(
      getIntersectingPoint(
          kHorizontalLineA, kHorizontalLineB, &intersecting_point));

  // Vertical line.
  //  s
  //  |
  //  e
  const Line2d kVerticalLineA(kPointZeroZero, kPointZeroOne);
  EXPECT_TRUE(
      getIntersectingPoint(
          kVerticalLineA, kHorizontalLineA, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);
  EXPECT_TRUE(
      getIntersectingPoint(
          kHorizontalLineA, kVerticalLineA, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);
  EXPECT_TRUE(
      getIntersectingPoint(
          kVerticalLineA, kHorizontalLineB, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);
  EXPECT_TRUE(
      getIntersectingPoint(
          kHorizontalLineB, kVerticalLineA, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);

  // Vertical line.
  //  e
  //  |
  //  s
  const Line2d kVerticalLineB(kPointZeroOne, kPointZeroZero);
  EXPECT_FALSE(
      getIntersectingPoint(
          kVerticalLineB, kVerticalLineA, &intersecting_point));
  EXPECT_FALSE(
      getIntersectingPoint(
          kVerticalLineA, kVerticalLineB, &intersecting_point));
  EXPECT_TRUE(
      getIntersectingPoint(
          kVerticalLineB, kHorizontalLineA, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);
  EXPECT_TRUE(
      getIntersectingPoint(
          kHorizontalLineA, kVerticalLineB, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);
  EXPECT_TRUE(
      getIntersectingPoint(
          kVerticalLineB, kHorizontalLineB, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);
  EXPECT_TRUE(
      getIntersectingPoint(
          kHorizontalLineB, kVerticalLineB, &intersecting_point));
  EXPECT_EQ(intersecting_point, kPointZeroZero);

  RandomPointGenerator<Line2d::PointType> random_point_generator;

  constexpr size_t kNumTrials = 1000u;
  for (size_t idx = 0u; idx < kNumTrials; ++idx) {
    const Line2d line_a = generateRandomLine<Line2d>(&random_point_generator);
    const Line2d line_b = generateRandomLine<Line2d>(&random_point_generator);

    const Line2d::PointType& line_a_start_point = line_a.getStartPoint();
    const Line2d::PointType line_a_direction =
        line_a.getEndPoint() - line_a.getStartPoint();
    CHECK_GT(line_a_direction.squaredNorm(), 0.0);

    const Line2d::PointType& line_b_start_point = line_b.getStartPoint();
    const Line2d::PointType line_b_direction =
        line_b.getEndPoint() - line_b.getStartPoint();
    CHECK_GT(line_b_direction.squaredNorm(), 0.0);

    const Line2d::PointType b = line_b_start_point - line_a_start_point;
    Eigen::Matrix2d J;
    J << line_a_direction(0), -line_b_direction(0), line_a_direction(1),
        -line_b_direction(1);

    if (std::abs(J.determinant()) < std::numeric_limits<double>::epsilon()) {
      EXPECT_FALSE(getIntersectingPoint(line_a, line_b, &intersecting_point));
      EXPECT_FALSE(getIntersectingPoint(line_b, line_a, &intersecting_point));

    } else {
      const Line2d::PointType lambda = J.householderQr().solve(b);
      const Line2d::PointType ground_truth_intersection_point =
          line_a_start_point + lambda(0) * line_a_direction;

      constexpr double kTolerance = 1e-1;
      EXPECT_TRUE(getIntersectingPoint(line_a, line_b, &intersecting_point));
      EXPECT_NEAR(
          intersecting_point(0), ground_truth_intersection_point(0),
          kTolerance);
      EXPECT_NEAR(
          intersecting_point(1), ground_truth_intersection_point(1),
          kTolerance);
      EXPECT_TRUE(getIntersectingPoint(line_b, line_a, &intersecting_point));
      EXPECT_NEAR(
          intersecting_point(0), ground_truth_intersection_point(0),
          kTolerance);
      EXPECT_NEAR(
          intersecting_point(1), ground_truth_intersection_point(1),
          kTolerance);
    }
  }
}

TEST(LineTests, Test2dLineAverageLateralDistance) {
  const Line2d::PointType kPointZeroZero(0.0, 0.0);
  const Line2d::PointType kPointOneZero(1.0, 0.0);
  const Line2d::PointType kPointZeroOne(0.0, 1.0);
  const Line2d::PointType kPointOneOne(1.0, 1.0);

  // Horizontal line.
  //  s ---- e
  //
  const Line2d kHorizontalLineA(kPointZeroZero, kPointOneZero);
  double average_later_distance = -1.0;
  EXPECT_TRUE(
      getAverageLateralDistance(
          kHorizontalLineA, kHorizontalLineA, &average_later_distance));
  EXPECT_DOUBLE_EQ(average_later_distance, 0.0);

  // Horizontal line.
  //  e ---- s
  //
  const Line2d kHorizontalLineB(kPointOneZero, kPointZeroZero);
  EXPECT_TRUE(
      getAverageLateralDistance(
          kHorizontalLineA, kHorizontalLineB, &average_later_distance));
  EXPECT_DOUBLE_EQ(average_later_distance, 0.0);
  EXPECT_TRUE(
      getAverageLateralDistance(
          kHorizontalLineB, kHorizontalLineA, &average_later_distance));
  EXPECT_DOUBLE_EQ(average_later_distance, 0.0);

  // Vertical line.
  //  s
  //  |
  //  e
  const Line2d kVerticalLineA(kPointZeroZero, kPointZeroOne);
  EXPECT_FALSE(
      getAverageLateralDistance(
          kVerticalLineA, kHorizontalLineA, &average_later_distance));
  EXPECT_FALSE(
      getAverageLateralDistance(
          kHorizontalLineA, kVerticalLineA, &average_later_distance));

  RandomPointGenerator<Line2d::PointType> random_point_generator;

  constexpr size_t kNumTrials = 1000u;
  for (size_t idx = 0u; idx < kNumTrials; ++idx) {
    const Line2d line_a = generateRandomLine<Line2d>(&random_point_generator);
    const Line2d line_b = generateRandomLine<Line2d>(&random_point_generator);

    const Line2d::PointType line_a_vector_in_line_direction =
        line_a.getEndPoint() - line_a.getStartPoint();
    const Line2d::PointType line_b_vector_in_line_direction =
        line_b.getEndPoint() - line_b.getStartPoint();

    // If they are orthogonal, no lateral distance can be computed.
    if (std::abs(
            line_a_vector_in_line_direction.dot(
                line_b_vector_in_line_direction)) <
        std::numeric_limits<double>::epsilon()) {
      EXPECT_FALSE(
          getAverageLateralDistance(line_a, line_b, &average_later_distance));
      EXPECT_FALSE(
          getAverageLateralDistance(line_b, line_a, &average_later_distance));
    } else {
      const Line2d::PointType line_a_orthogonal_vector(
          -line_a_vector_in_line_direction(1),
          line_a_vector_in_line_direction(0));
      const Line2d line_orthogonal_to_line_a(
          line_a.getMidpoint(),
          line_a.getMidpoint() + line_a_orthogonal_vector);
      Line2d::PointType intersecting_point_line_b;
      CHECK(
          getIntersectingPoint(
              line_orthogonal_to_line_a, line_b, &intersecting_point_line_b));
      const double lateral_distance_a =
          (line_a.getMidpoint() - intersecting_point_line_b).norm();

      const Line2d::PointType line_b_orthogonal_vector(
          -line_b_vector_in_line_direction(1),
          line_b_vector_in_line_direction(0));
      const Line2d line_orthogonal_to_line_b(
          line_b.getMidpoint(),
          line_b.getMidpoint() + line_b_orthogonal_vector);
      Line2d::PointType intersecting_point_line_a;
      CHECK(
          getIntersectingPoint(
              line_orthogonal_to_line_b, line_a, &intersecting_point_line_a));
      const double lateral_distance_b =
          (line_b.getMidpoint() - intersecting_point_line_a).norm();

      const double ground_truth_avg_lateral_distance =
          0.5 * (lateral_distance_a + lateral_distance_b);

      constexpr double kTolerance = 1e-1;
      EXPECT_TRUE(
          getAverageLateralDistance(line_a, line_b, &average_later_distance));
      EXPECT_NEAR(
          average_later_distance, ground_truth_avg_lateral_distance,
          kTolerance);
      EXPECT_TRUE(
          getAverageLateralDistance(line_b, line_a, &average_later_distance));
      EXPECT_NEAR(
          average_later_distance, ground_truth_avg_lateral_distance,
          kTolerance);
    }
  }
}

TEST(HomogeneousLineTests, TestBasicHomogeneousLineFeatures) {
  const HomogeneousLine2d::PointType kPointZeroZero(0.0, 0.0);
  const Line2d::PointType kPointOneZero(1.0, 0.0);
  const Line2d::PointType kPointZeroOne(0.0, 1.0);

  // Points are not lines!
  EXPECT_DEATH(
      const HomogeneousLine2d kZeroLengthLine(kPointZeroZero, kPointZeroZero),
      "");

  // Horizontal line.
  const HomogeneousLine2d kHorizontalLineA(kPointZeroZero, kPointOneZero);
  const HomogeneousLine2d kHorizontalLineB(kPointOneZero, kPointZeroZero);

  constexpr size_t kSeed = 42u;
  std::default_random_engine random_engine(kSeed);
  constexpr double kMaxDelta = 1e3;
  std::uniform_real_distribution<double> uniform_delta(-kMaxDelta, kMaxDelta);

  constexpr size_t kNumTrials = 1000u;
  for (size_t idx = 0u; idx < kNumTrials; ++idx) {
    const double v = uniform_delta(random_engine);
    double x = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(kHorizontalLineA.getXFromY(v, &x));
    x = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(kHorizontalLineB.getXFromY(v, &x));
    double y = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(kHorizontalLineA.getYFromX(v, &y));
    EXPECT_DOUBLE_EQ(y, 0.0);
    y = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(kHorizontalLineB.getYFromX(v, &y));
    EXPECT_DOUBLE_EQ(y, 0.0);
  }

  // Vertical line.
  const HomogeneousLine2d kVerticalLineA(kPointZeroZero, kPointZeroOne);
  const HomogeneousLine2d kVerticalLineB(kPointZeroOne, kPointZeroZero);
  for (size_t idx = 0u; idx < kNumTrials; ++idx) {
    const double v = uniform_delta(random_engine);
    double y = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(kVerticalLineA.getYFromX(v, &y));
    y = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(kVerticalLineB.getYFromX(v, &y));
    double x = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(kVerticalLineA.getXFromY(v, &x));
    EXPECT_DOUBLE_EQ(x, 0.0);
    x = std::numeric_limits<double>::infinity();
    EXPECT_TRUE(kVerticalLineB.getXFromY(v, &x));
    EXPECT_DOUBLE_EQ(x, 0.0);
  }

  RandomPointGenerator<HomogeneousLine2d::PointType> random_point_generator;

  for (size_t idx = 0u; idx < kNumTrials; ++idx) {
    const HomogeneousLine2d::PointType point_a =
        random_point_generator.generateRandomPoint();
    const HomogeneousLine2d::PointType point_b =
        random_point_generator.generateRandomPoint();

    if ((point_a - point_b).squaredNorm() <
        std::numeric_limits<double>::epsilon()) {
      continue;
    }

    const HomogeneousLine2d line_a(point_a, point_b);
    const HomogeneousLine2d line_b(point_b, point_a);

    const HomogeneousLine2d::PointType vector_in_line_direction =
        point_b - point_a;
    ASSERT_TRUE(
        vector_in_line_direction.squaredNorm() >
        std::numeric_limits<double>::epsilon());

    constexpr size_t kNumPointsToSample = 1000u;
    for (size_t idx = 0u; idx < kNumTrials; ++idx) {
      const double delta = uniform_delta(random_engine);
      const HomogeneousLine2d::PointType point =
          point_a + delta * vector_in_line_direction;

      double y = 0.0;
      double x = 0.0;
      constexpr double kTolerance = 1e-2;
      EXPECT_TRUE(line_a.getXFromY(point(1), &x));
      EXPECT_NEAR(point(0), x, kTolerance);
      EXPECT_TRUE(line_b.getXFromY(point(1), &x));
      EXPECT_NEAR(point(0), x, kTolerance);

      EXPECT_TRUE(line_a.getYFromX(point(0), &y));
      EXPECT_NEAR(point(1), y, kTolerance);
      EXPECT_TRUE(line_b.getYFromX(point(0), &y));
      EXPECT_NEAR(point(1), y, kTolerance);
    }

    HomogeneousLine2d::PointType vector_orthogonal_to_line(
        -vector_in_line_direction(1), vector_in_line_direction(0));
    vector_orthogonal_to_line.normalize();

    constexpr double kTolerance = 1e-3;
    EXPECT_TRUE(
        (std::abs(
             vector_orthogonal_to_line(0) -
             line_a.getNormalizedNormalVector()(0)) < kTolerance &&
         std::abs(
             vector_orthogonal_to_line(1) -
             line_a.getNormalizedNormalVector()(1)) < kTolerance) ||
        (std::abs(
             -vector_orthogonal_to_line(0) -
             line_a.getNormalizedNormalVector()(0)) < kTolerance &&
         std::abs(
             -vector_orthogonal_to_line(1) -
             line_a.getNormalizedNormalVector()(1)) < kTolerance));

    const Line2d orthogonal_line(kPointZeroZero, vector_orthogonal_to_line);
    const Line2d line_2d(point_a, point_b);

    Line2d::PointType intersecting_point;
    EXPECT_TRUE(
        getIntersectingPoint(line_2d, orthogonal_line, &intersecting_point));

    const double ground_truth_distance_from_origin = intersecting_point.norm();
    EXPECT_NEAR(
        ground_truth_distance_from_origin, line_a.getDistanceToOrigin(),
        kTolerance);
    EXPECT_NEAR(
        ground_truth_distance_from_origin, line_b.getDistanceToOrigin(),
        kTolerance);
  }
}

TEST(HomogeneousLineTests, TestSignedDistance) {
  RandomPointGenerator<HomogeneousLine2d::PointType> random_point_generator;

  constexpr size_t kNumTrials = 1000u;
  for (size_t idx = 0u; idx < kNumTrials; ++idx) {
    const HomogeneousLine2d::PointType start_point =
        random_point_generator.generateRandomPoint();
    const HomogeneousLine2d::PointType end_point =
        random_point_generator.generateRandomPoint();
    const HomogeneousLine2d line(start_point, end_point);
    const Line2d line_2d(start_point, end_point);

    const HomogeneousLine2d::PointType direction_vector =
        end_point - start_point;
    HomogeneousLine2d::PointType orthogonal_vector(
        -direction_vector(1), direction_vector(0));
    orthogonal_vector.normalize();
    // Find the distance d by inserting one of the points.
    const double distance_from_origin =
        -(orthogonal_vector(0) * start_point(0) +
          orthogonal_vector(1) * start_point(1));
    // Make distance always positive and invert the normal vector if necessary.
    const double sign_switch = distance_from_origin < 0.0 ? -1.0 : 1.0;

    constexpr size_t kNumPointSamples = 100u;
    for (size_t point_idx = 0u; point_idx < kNumPointSamples; ++point_idx) {
      const HomogeneousLine2d::PointType point =
          random_point_generator.generateRandomPoint();
      const Line2d orthogonal_line_intersecting_point(
          point, point + line.getNormalizedNormalVector());

      Line2d::PointType intersecting_point;
      EXPECT_TRUE(
          getIntersectingPoint(
              line_2d, orthogonal_line_intersecting_point,
              &intersecting_point));
      const double distance = sign_switch * (point - intersecting_point).norm();

      const double side_determinant =
          (point(0) - start_point(0)) * (end_point(1) - start_point(1)) -
          (point(1) - start_point(1)) * (end_point(0) - start_point(0));

      constexpr double kTolerance = 1e-8;
      ASSERT_NEAR(
          side_determinant < 0.0 ? distance : -distance,
          line.getSignedDistanceToPoint(point), kTolerance);
    }
  }
}

TEST(HomogeneousLineTests, TestHomogeneousLinesRectangleIntersection) {
  constexpr double kXLowerLimit = -10.0;
  constexpr double kXUpperLimit = 50.0;
  constexpr double kYLowerLimit = 20.0;
  constexpr double kYUpperLimit = 100.0;
  const Eigen::Vector2d x_limits(kXLowerLimit, kXUpperLimit);
  const Eigen::Vector2d y_limits(kYLowerLimit, kYUpperLimit);

  const HomogeneousLine2d kLineHorizontalOutsideA(
      HomogeneousLine2d::PointType(0.0, 0.0),
      HomogeneousLine2d::PointType(1.0, 0.0));
  Vector2dListTemplate<double> intersections;
  getLineIntersectionWithRectangle(
      kLineHorizontalOutsideA, x_limits, y_limits, &intersections);
  EXPECT_TRUE(intersections.empty());

  const HomogeneousLine2d kLineHorizontalOutsideB(
      HomogeneousLine2d::PointType(0.0, 200.0),
      HomogeneousLine2d::PointType(1.0, 200.0));
  getLineIntersectionWithRectangle(
      kLineHorizontalOutsideB, x_limits, y_limits, &intersections);
  EXPECT_TRUE(intersections.empty());

  const HomogeneousLine2d kLineVerticalOutsideA(
      HomogeneousLine2d::PointType(-30.0, 0.0),
      HomogeneousLine2d::PointType(-30.0, 1.0));

  getLineIntersectionWithRectangle(
      kLineVerticalOutsideA, x_limits, y_limits, &intersections);
  EXPECT_TRUE(intersections.empty());

  const HomogeneousLine2d kLineVerticalOutsideB(
      HomogeneousLine2d::PointType(200.0, 0.0),
      HomogeneousLine2d::PointType(200.0, 1.0));
  getLineIntersectionWithRectangle(
      kLineVerticalOutsideB, x_limits, y_limits, &intersections);
  EXPECT_TRUE(intersections.empty());

  constexpr double kYValue = 50.0;
  const HomogeneousLine2d kLineHorizontalIntersecting(
      HomogeneousLine2d::PointType(0.0, kYValue),
      HomogeneousLine2d::PointType(1.0, kYValue));
  getLineIntersectionWithRectangle(
      kLineHorizontalIntersecting, x_limits, y_limits, &intersections);
  ASSERT_EQ(intersections.size(), 2u);
  EXPECT_DOUBLE_EQ(intersections[0u](0), kXLowerLimit);
  EXPECT_DOUBLE_EQ(intersections[0u](1), kYValue);
  EXPECT_DOUBLE_EQ(intersections[1u](0), kXUpperLimit);
  EXPECT_DOUBLE_EQ(intersections[1u](1), kYValue);
  intersections.clear();

  constexpr double kXValue = 40.0;
  const HomogeneousLine2d kLineVerticallIntersecting(
      HomogeneousLine2d::PointType(kXValue, 0.0),
      HomogeneousLine2d::PointType(kXValue, 1.0));
  getLineIntersectionWithRectangle(
      kLineVerticallIntersecting, x_limits, y_limits, &intersections);
  ASSERT_EQ(intersections.size(), 2u);
  EXPECT_DOUBLE_EQ(intersections[0u](0), kXValue);
  EXPECT_DOUBLE_EQ(intersections[0u](1), kYLowerLimit);
  EXPECT_DOUBLE_EQ(intersections[1u](0), kXValue);
  EXPECT_DOUBLE_EQ(intersections[1u](1), kYUpperLimit);
  intersections.clear();

  const HomogeneousLine2d kLineHorizontalIntersectingOnLowerLimits(
      HomogeneousLine2d::PointType(0.0, kYLowerLimit),
      HomogeneousLine2d::PointType(1.0, kYLowerLimit));
  getLineIntersectionWithRectangle(
      kLineHorizontalIntersectingOnLowerLimits, x_limits, y_limits,
      &intersections);
  EXPECT_TRUE(intersections.empty());
  intersections.clear();

  const HomogeneousLine2d kLineHorizontalIntersectingOnUpperLimits(
      HomogeneousLine2d::PointType(0.0, kYUpperLimit),
      HomogeneousLine2d::PointType(1.0, kYUpperLimit));
  getLineIntersectionWithRectangle(
      kLineHorizontalIntersectingOnUpperLimits, x_limits, y_limits,
      &intersections);
  EXPECT_TRUE(intersections.empty());
  intersections.clear();

  const HomogeneousLine2d kLineVerticalIntersectingOnLowerLimits(
      HomogeneousLine2d::PointType(kXLowerLimit, 0.0),
      HomogeneousLine2d::PointType(kXLowerLimit, 1.0));
  getLineIntersectionWithRectangle(
      kLineVerticalIntersectingOnLowerLimits, x_limits, y_limits,
      &intersections);
  EXPECT_TRUE(intersections.empty());
  intersections.clear();

  const HomogeneousLine2d kLineVerticalIntersectingOnUpperLimits(
      HomogeneousLine2d::PointType(kXUpperLimit, 0.0),
      HomogeneousLine2d::PointType(kXUpperLimit, 1.0));
  getLineIntersectionWithRectangle(
      kLineVerticalIntersectingOnUpperLimits, x_limits, y_limits,
      &intersections);
  EXPECT_TRUE(intersections.empty());
}

}  // namespace aslam

ASLAM_UNITTEST_ENTRYPOINT
