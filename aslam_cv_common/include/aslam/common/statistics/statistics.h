#ifndef ASLAM_COMMON_STATISTICS_STATISTICS_H_
#define ASLAM_COMMON_STATISTICS_STATISTICS_H_

#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "aslam/common/statistics/accumulator.h"

///
// Example usage:
//
// #define ENABLE_STATISTICS 1 // Turn on/off the statistics calculation
// #include <aslam/common/statistics/statistics.h>
//
// double my_distance = measureDistance();
// statistics::DebugStatsCollector distance_stat("Distance measurement");
// distance_stat.addSample(my_distance);
//
// std::cout << statistics::Statistics::print();
///

namespace statistics {

const double kNumSecondsPerNanosecond = 1.e-9;

template <class DataType>
struct StatisticsMapValue {
  static const size_t kWindowSize = 100u;

  inline StatisticsMapValue() {
    time_last_called_ = std::chrono::system_clock::now();
  }

  inline void addValue(const DataType sample) {
    std::chrono::time_point<std::chrono::system_clock> now =
        std::chrono::system_clock::now();
    const int64_t dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              now - time_last_called_)
                              .count();
    time_last_called_ = now;

    values_.add(sample);
    time_deltas_.add(dt_ns);
  }

  inline int64_t getLastDeltaTimeNanoseconds() const {
    if (time_deltas_.getTotalNumSamples() > 0u) {
      return time_deltas_.getMostRecent();
    } else {
      return -1;
    }
  }

  inline DataType getLastValue() const {
    if (values_.getTotalNumSamples()) {
      return values_.getMostRecent();
    } else {
      return static_cast<DataType>(0);
    }
  }

  inline DataType getSum() const {
    return values_.getSum();
  }

  size_t getTotalNumSamples() const {
    return values_.getTotalNumSamples();
  }

  double getMean() const {
    return values_.getMean();
  }

  double getRollingMean() const {
    return values_.getRollingMean();
  }

  DataType getMax() const {
    return values_.getMax();
  }

  DataType getMin() const {
    return values_.getMin();
  }

  double getLazyVariance() const {
    return values_.getLazyVariance();
  }

  double getMeanNumCallsPerSecond() const {
    const double mean_dt_ns = time_deltas_.getMean();
    CHECK_GE(mean_dt_ns, 0.0);
    if (mean_dt_ns > std::numeric_limits<double>::epsilon()) {
      return 1e9 / mean_dt_ns;
    } else {
      return -1.0;
    }
  }

  double getMeanDeltaTimeNanoseconds() const {
    return time_deltas_.getMean();
  }

  double getRollingMeanDeltaTimeNanoseconds() const {
    return time_deltas_.getRollingMean();
  }

  int64_t getMaxDeltaTimeNanoseconds() const {
    return time_deltas_.getMax();
  }

  int64_t getMinDeltaTimeNanoseconds() const {
    return time_deltas_.getMin();
  }

  double getLazyVarianceDeltaTimeNanoseconds() const {
    return time_deltas_.getLazyVariance();
  }

 private:
  // Create an accumulator with specified window size.
  Accumulator<DataType, DataType, kWindowSize> values_;
  Accumulator<int64_t, int64_t, kWindowSize> time_deltas_;
  std::chrono::time_point<std::chrono::system_clock> time_last_called_;
};

// A class that has the statistics interface but does nothing. Swapping this in
// in place of the Statistics class (say with a typedef) eliminates the function
// calls.
class DummyStatsCollector {
 public:
  explicit DummyStatsCollector(const size_t /*handle*/) {}
  explicit DummyStatsCollector(const std::string& /*tag*/) {}
  void addSample(const double /*sample*/) const {}
  void incrementOne() const {}
  size_t getHandle() const {
    return 0u;
  }
};

class StatsCollectorImpl {
 public:
  explicit StatsCollectorImpl(const size_t handle);
  explicit StatsCollectorImpl(const std::string& tag);
  ~StatsCollectorImpl() = default;

  void addSample(const double sample) const;
  void incrementOne() const;
  size_t getHandle() const;

 private:
  size_t handle_;
};

class Statistics {
 public:
  typedef std::map<std::string, size_t> map_t;
  friend class StatsCollectorImpl;
  // Definition of static functions to query the stats.
  static size_t getHandle(const std::string& tag);
  static bool hasHandle(const std::string& tag);
  static std::string getTag(const size_t handle);
  static double getLastValue(const size_t handle);
  static double getLastValue(const std::string& tag);
  static double getSum(const size_t handle);
  static double getSum(const std::string& tag);
  static double getMean(const size_t handle);
  static double getMean(const std::string& tag);
  static size_t getNumSamples(const size_t handle);
  static size_t getNumSamples(const std::string& tag);
  static double getVariance(const size_t handle);
  static double getVariance(const std::string& tag);
  static double getMin(const size_t handle);
  static double getMin(const std::string& tag);
  static double getMax(const size_t handle);
  static double getMax(const std::string& tag);
  static double getHz(const size_t handle);
  static double getHz(const std::string& tag);

  static double getMeanDeltaTimeNanoseconds(const std::string& tag);
  static double getMeanDeltaTimeNanoseconds(const size_t handle);
  static int64_t getMaxDeltaTimeNanoseconds(const std::string& tag);
  static int64_t getMaxDeltaTimeNanoseconds(const size_t handle);
  static int64_t getMinDeltaTimeNanoseconds(const std::string& tag);
  static int64_t getMinDeltaTimeNanoseconds(const size_t handle);
  static int64_t getLastDeltaTimeNanoseconds(const std::string& tag);
  static int64_t getLastDeltaTimeNanoseconds(const size_t handle);
  static double getVarianceDeltaTimeNanoseconds(const std::string& tag);
  static double getVarianceDeltaTimeNanoseconds(const size_t handle);

  static void writeToYamlFile(const std::string& path);
  static void print(std::ostream& out);  // NOLINT
  static std::string print();
  static void reset();
  static const map_t& getStatsCollectors() {
    return Instance().tag_map_;
  }

 private:
  void addSample(const size_t handle, const double sample);

  static Statistics& Instance();

  Statistics();
  ~Statistics() = default;

  using list_t = std::vector<statistics::StatisticsMapValue<double>>;

  list_t stats_collectors_;
  map_t tag_map_;
  size_t max_tag_length_;
  std::mutex mutex_;
};

#if ENABLE_STATISTICS
using StatsCollector = StatsCollectorImpl;
#else
using StatsCollector = DummyStatsCollector;
#endif

inline std::string secondsToTimeString(const double seconds) {
  double secs = fmod(seconds, 60);
  int minutes = (seconds / 60);
  int hours = (seconds / 3600);
  minutes = minutes - (hours * 60);

  char buffer[256];
  snprintf(
      buffer, sizeof(buffer),
#ifdef SM_TIMING_SHOW_HOURS
      "%02d:"
#endif
#ifdef SM_TIMING_SHOW_MINUTES
      "%02d:"
#endif
      "%09.6f",
#ifdef SM_TIMING_SHOW_HOURS
      hours,
#endif
#ifdef SM_TIMING_SHOW_MINUTES
      minutes,
#endif
      secs);
  return buffer;
}

}  // namespace statistics

#endif  // ASLAM_COMMON_STATISTICS_STATISTICS_H_
