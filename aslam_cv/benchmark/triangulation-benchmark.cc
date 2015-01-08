#include <aslam/common/entrypoint.h>
#include <aslam/test/triangulation-fixture.h>
#include <sm/timing/Timer.hpp>

void sampleXYPlaneSine(const double x_min, const double x_max,
                       const size_t num_samples, Eigen::Matrix3Xd* result) {
  CHECK_NOTNULL(result)->resize(3, num_samples);
  CHECK_GT(num_samples, 1);
  for (size_t i = 0; i < num_samples; ++i) {
    double x = x_min + i * (x_max - x_min) / (num_samples - 1);
    result->block<3, 1>(0, i) << x, sin(x), 0;
  }
}

template <typename MeasurementType>
size_t handleOffset() {
  CHECK(false);
  return 0u;
}

template <>
size_t handleOffset<Vector2dList>() { return 1000u; }

template <>
size_t handleOffset<Eigen::Matrix3Xd>() { return 2000u; }

template <typename MeasurementType>
void plotIfDone() {}

constexpr size_t kMaxNumMeasurements = 50;

template <>
void plotIfDone<Eigen::Matrix3Xd>() {
  FILE* plot = popen("gnuplot --persist", "w");
  fprintf(plot, "set logscale y\n");
  fprintf(plot, "plot '-' w l, '-' w l\n");
  for (size_t i = 2; i <= kMaxNumMeasurements; ++i) {
    fprintf(plot, "%u %lf\n", i, sm::timing::Timing::getMeanSeconds(
        std::to_string(handleOffset<Vector2dList>() + i)));
  }
  fprintf(plot, "e\n");
  for (size_t i = 2; i <= kMaxNumMeasurements; ++i) {
    fprintf(plot, "%u %lf\n", i, sm::timing::Timing::getMeanSeconds(
        std::to_string(handleOffset<Eigen::Matrix3Xd>() + i)));
  }
  fprintf(plot, "e\n");
  fflush(plot);
}

TYPED_TEST(TriangulationFixture, Performance) {
  constexpr double kDepth = 5;
  constexpr size_t num_samples = 10;
  this->p_W_L_ << 1, 1, kDepth;
  CHECK_GT(kMaxNumMeasurements, 1);
  for (size_t num_measurements = 2; num_measurements <= kMaxNumMeasurements;
      ++num_measurements) {
    this->setNMeasurements(num_measurements);
    Eigen::Matrix3Xd p_W_B;
    sampleXYPlaneSine(-100., 100., num_measurements, &p_W_B);
    for (size_t i = 0; i < num_measurements; ++i) {
      this->T_W_B_[i].getPosition() = p_W_B.block<3, 1>(0, i);
    }
    this->inferMeasurements();
    for (size_t i = 0; i < num_samples; ++i) {
      sm::timing::Timer timer(std::to_string(
          handleOffset<TypeParam>() + num_measurements));
      this->expectSuccess();
      timer.stop();
    }
  }
  plotIfDone<TypeParam>();
}

ASLAM_UNITTEST_ENTRYPOINT
