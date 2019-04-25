#ifndef ASLAM_COMMON_STATISTICS_ACCUMULATOR_H_
#define ASLAM_COMMON_STATISTICS_ACCUMULATOR_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <glog/logging.h>

namespace statistics {
using IndexType = size_t;
static constexpr IndexType kInfiniteWindowSize =
    std::numeric_limits<IndexType>::max();
// If the window size is set to -1, the vector will grow infinitely, otherwise,
// the vector has a fixed size.
template <typename SampleType, typename SumType, IndexType WindowSize>
class Accumulator {
 public:
  Accumulator()
      : sample_index_(0),
        total_samples_(0),
        sum_(0),
        window_sum_(0),
        min_(std::numeric_limits<SampleType>::max()),
        max_(std::numeric_limits<SampleType>::lowest()),
        most_recent_(0) {
    CHECK_GT(WindowSize, 0u);
    if (WindowSize < kInfiniteWindowSize) {
      samples_.reserve(WindowSize);
    }
  }

  void add(const SampleType sample) {
    most_recent_ = sample;
    if (sample_index_ < WindowSize) {
      samples_.push_back(sample);
      window_sum_ += sample;
      ++sample_index_;
    } else {
      SampleType& oldest = samples_.at(sample_index_++ % WindowSize);
      window_sum_ += sample - oldest;
      oldest = sample;
    }
    sum_ += sample;
    ++total_samples_;
    if (sample > max_) {
      max_ = sample;
    }
    if (sample < min_) {
      min_ = sample;
    }
  }

  size_t getTotalNumSamples() const {
    return total_samples_;
  }

  SumType getSum() const {
    return sum_;
  }

  double getMean() const {
    return (total_samples_ == 0u) ? 0.0
                                  : static_cast<double>(sum_) /
                                        static_cast<double>(total_samples_);
  }

  // Rolling mean is only used for fixed sized data for now. We don't need this
  // function for our infinite accumulator at this point.
  double getRollingMean() const {
    if (WindowSize < kInfiniteWindowSize) {
      return static_cast<double>(window_sum_) /
             static_cast<double>(std::min(sample_index_, WindowSize));
    } else {
      return getMean();
    }
  }

  SampleType getMostRecent() const {
    return most_recent_;
  }

  SumType getMax() const {
    return max_;
  }

  SumType getMin() const {
    return min_;
  }

  double getLazyVariance() const {
    if (samples_.size() < 2u) {
      return 0.0;
    }

    double var = 0.0;
    double mean = getRollingMean();

    for (size_t i = 0u; i < samples_.size(); ++i) {
      var += (static_cast<double>(samples_[i]) - mean) *
             (static_cast<double>(samples_[i]) - mean);
    }

    var /= static_cast<double>(samples_.size() - 1u);
    return var;
  }

  double getStandardDeviation() const {
    return std::sqrt(getLazyVariance());
  }

  const std::vector<SampleType>& getSamples() const {
    return samples_;
  }

 private:
  std::vector<SampleType> samples_;
  IndexType sample_index_;
  IndexType total_samples_;
  SumType sum_;
  SumType window_sum_;
  SampleType min_;
  SampleType max_;
  SampleType most_recent_;
};

typedef Accumulator<double, double, kInfiniteWindowSize> Accumulatord;

}  // namespace statistics

#endif  // ASLAM_COMMON_STATISTICS_ACCUMULATOR_H_
