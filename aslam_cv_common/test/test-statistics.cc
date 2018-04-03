#include <thread>
#include <vector>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <aslam/common/entrypoint.h>
#include <aslam/common/statistics/statistics.h>

TEST(StatisticsCollector, Multithread_Adds) {
  const std::string kStatisticsName("value_name");
  constexpr size_t kNumThreads = 100u;
  constexpr size_t kNumValues = 10000u;

  auto AddStatisticValues = [&]() {
    for (size_t i = 0u; i < kNumValues; ++i) {
      statistics::StatsCollectorImpl collector(kStatisticsName);
      collector.IncrementOne();
    }
  };

  std::vector<std::thread> threads;
  for (size_t thread_idx = 0u; thread_idx < kNumThreads; ++thread_idx) {
    threads.push_back(std::thread(AddStatisticValues));
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  EXPECT_EQ(statistics::Statistics::GetNumSamples(kStatisticsName),
            kNumThreads * kNumValues);
  EXPECT_EQ(statistics::Statistics::GetMean(kStatisticsName), 1u);
  EXPECT_EQ(statistics::Statistics::GetMax(kStatisticsName),  1u);
  EXPECT_EQ(statistics::Statistics::GetMin(kStatisticsName),  1u);
}

ASLAM_UNITTEST_ENTRYPOINT
