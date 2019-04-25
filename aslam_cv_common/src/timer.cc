#include "aslam/common/timer.h"

#include <algorithm>
#include <fstream>  // NOLINT
#include <math.h>
#include <ostream>  //NOLINT
#include <sstream>
#include <stdio.h>
#include <string>

#include "aslam/common/time.h"

namespace timing {

const double kNumNanosecondsPerNanosecond = 1.e-9;

Timing& Timing::Instance() {
  static Timing t;
  return t;
}

Timing::Timing() : max_tag_length_(0u) {}

Timing::~Timing() {}

// Static functions to query the timers:
size_t Timing::getHandle(const std::string& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  // Search for an existing tag.
  map_t::iterator tag_iterator = Instance().tag_map_.find(tag);
  if (tag_iterator == Instance().tag_map_.end()) {
    // If it is not there, create a tag.
    size_t handle = Instance().timers_.size();
    Instance().tag_map_[tag] = handle;
    Instance().timers_.push_back(statistics::StatisticsMapValue<int64_t>());
    // Track the maximum tag length to help printing a table of timing values
    // later.
    Instance().max_tag_length_ =
        std::max(Instance().max_tag_length_, tag.size());
    return handle;
  } else {
    return tag_iterator->second;
  }
}

std::string Timing::getTag(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  std::string tag;

  // Perform a linear search for the tag.
  for (const typename map_t::value_type& current_tag : Instance().tag_map_) {
    if (current_tag.second == handle) {
      return current_tag.first;
    }
  }
  return tag;
}

// Class functions used for timing.
TimerImpl::TimerImpl(const std::string& tag, bool construct_stopped)
    : is_timing_(false), handle_(Timing::getHandle(tag)), tag_(tag) {
  if (!construct_stopped) {
    start();
  }
}

TimerImpl::~TimerImpl() {
  if (is_timing_) {
    stop();
  }
}

void TimerImpl::start() {
  is_timing_ = true;
  time_ = std::chrono::system_clock::now();
}

int64_t TimerImpl::stop() {
  if (is_timing_) {
    std::chrono::time_point<std::chrono::system_clock> now =
        std::chrono::system_clock::now();
    const int64_t dt_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - time_)
            .count();
    Timing::Instance().addTime(handle_, dt_ns);
    is_timing_ = false;
    return dt_ns;
  }
  return 0.0;
}

void TimerImpl::discard() {
  is_timing_ = false;
}

size_t TimerImpl::getHandle() const {
  return handle_;
}

void Timing::addTime(const size_t handle, const int64_t time_nanoseconds) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  timers_[handle].addValue(time_nanoseconds);
}

int64_t Timing::getTotalNumNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].getSum();
}

int64_t Timing::getTotalNumNanoseconds(const std::string& tag) {
  return getTotalNumNanoseconds(getHandle(tag));
}

double Timing::getMeanNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].getMean();
}

double Timing::getMeanNanoseconds(const std::string& tag) {
  return getMeanNanoseconds(getHandle(tag));
}

size_t Timing::getNumSamples(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].getTotalNumSamples();
}

size_t Timing::getNumSamples(const std::string& tag) {
  return getNumSamples(getHandle(tag));
}

double Timing::getVarianceNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].getLazyVariance();
}

double Timing::getVarianceNanoseconds(const std::string& tag) {
  return getVarianceNanoseconds(getHandle(tag));
}

int64_t Timing::getMinNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].getMin();
}

int64_t Timing::getMinNanoseconds(const std::string& tag) {
  return getMinNanoseconds(getHandle(tag));
}

int64_t Timing::getMaxNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].getMax();
}

int64_t Timing::getMaxNanoseconds(const std::string& tag) {
  return getMaxNanoseconds(getHandle(tag));
}

double Timing::getHz(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return 1.0 / Instance().timers_[handle].getRollingMean();
}

double Timing::getHz(const std::string& tag) {
  return getHz(getHandle(tag));
}

std::string millisecondsToTimeString(const double milliseconds) {
  constexpr double kMillisecondsToSeconds = 1e-3;
  const double seconds = milliseconds * kMillisecondsToSeconds;
  double secs = fmod(seconds, 60);
  int minutes = (seconds / 60);
  int hours = (seconds / 3600);
  minutes = minutes - (hours * 60);

  char buffer[256];
  snprintf(
      buffer, sizeof(buffer),
      "%02d:"
      "%02d:"
      "%09.6f",
      hours, minutes, secs);
  return buffer;
}

void Timing::writeToYamlFile(const std::string& path) {
  const map_t& tag_map = Instance().tag_map_;

  if (tag_map.empty()) {
    return;
  }

  std::ofstream output_file(path);

  if (!output_file) {
    LOG(ERROR) << "Could not write timing: Unable to open file: " << path;
    return;
  }

  VLOG(1) << "Writing timing to file: " << path;
  for (const map_t::value_type& tag : tag_map) {
    const size_t index = tag.second;

    if (getNumSamples(index) > 0) {
      std::string label = tag.first;

      // We do not want colons or hashes in a label, as they might interfere
      // with reading the yaml later.
      std::replace(label.begin(), label.end(), ':', '_');
      std::replace(label.begin(), label.end(), '#', '_');

      output_file << label << ":"
                  << "\n";
      output_file << "  num_samples: " << getNumSamples(index) << "\n";
      output_file << "  sum [ms]: "
                  << aslam::time::to_milliseconds(getTotalNumNanoseconds(index))
                  << "\n";
      output_file << "  mean [ms]: "
                  << aslam::time::to_milliseconds(getMeanNanoseconds(index))
                  << "\n";
      output_file << "  std_dev [ms]: "
                  << aslam::time::to_milliseconds(
                         sqrt(getVarianceNanoseconds(index)))
                  << "\n";
      output_file << "  min [ms]: "
                  << aslam::time::to_milliseconds(getMinNanoseconds(index))
                  << "\n";
      output_file << "  max [ms]: "
                  << aslam::time::to_milliseconds(getMaxNanoseconds(index))
                  << "\n";
    }
    output_file << "\n";
  }
}

void Timing::print(std::ostream& out) {  // NOLINT
  map_t tagMap;
  {
    std::lock_guard<std::mutex> lock(Instance().mutex_);
    tagMap = Instance().tag_map_;
  }

  if (tagMap.empty()) {
    return;
  }

  out << "Timings in [ms]\n";
  out << "-----------\n";
  for (typename map_t::value_type t : tagMap) {
    size_t time_i = t.second;
    out.width((std::streamsize)Instance().max_tag_length_);
    out.setf(std::ios::left, std::ios::adjustfield);
    out << t.first << "\t";
    out.width(7);

    out.setf(std::ios::right, std::ios::adjustfield);
    out << getNumSamples(time_i) << "\t";
    if (getNumSamples(time_i) > 0u) {
      out << aslam::time::timeNanosecondsToString(
                 getTotalNumNanoseconds(time_i))
          << "\t";
      const double mean_ms =
          aslam::time::to_milliseconds(getMeanNanoseconds(time_i));
      const double stddev_ms =
          aslam::time::to_milliseconds(sqrt(getVarianceNanoseconds(time_i)));
      out << "(" << millisecondsToTimeString(mean_ms) << " +- ";
      out << millisecondsToTimeString(stddev_ms) << ")\t";

      const double min_ms =
          aslam::time::to_milliseconds(getMinNanoseconds(time_i));
      const double max_ms =
          aslam::time::to_milliseconds(getMaxNanoseconds(time_i));
      // The min or max are out of bounds.
      out << "[" << millisecondsToTimeString(min_ms) << ","
          << millisecondsToTimeString(max_ms) << "]";
    }
    out << std::endl;
  }
}

std::string Timing::print() {
  std::stringstream ss;
  print(ss);
  return ss.str();
}

void Timing::reset() {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  Instance().tag_map_.clear();
}

}  // namespace timing
