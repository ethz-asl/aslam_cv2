#include "aslam/common/statistics/statistics.h"

#include <cmath>
#include <fstream>  // NOLINT
#include <ostream>  // NOLINT
#include <sstream>

namespace statistics {

Statistics& Statistics::Instance() {
  static Statistics instance;
  return instance;
}

Statistics::Statistics() : max_tag_length_(0) {}

// Static functions to query the stats collectors:
size_t Statistics::getHandle(const std::string& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  // Search for an existing tag.
  map_t::iterator i = Instance().tag_map_.find(tag);
  if (i == Instance().tag_map_.end()) {
    // If it is not there, create a tag.
    size_t handle = Instance().stats_collectors_.size();
    Instance().tag_map_[tag] = handle;
    Instance().stats_collectors_.emplace_back(StatisticsMapValue<double>());
    // Track the maximum tag length to help printing a table of values later.
    Instance().max_tag_length_ =
        std::max(Instance().max_tag_length_, tag.size());
    return handle;
  } else {
    return i->second;
  }
}

// Return true if a handle has been initialized for a specific tag.
// In contrast to GetHandle(), this allows testing for existence without
// modifying the tag/handle map.
bool Statistics::hasHandle(const std::string& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  bool tag_found = Instance().tag_map_.count(tag);
  return tag_found;
}

std::string Statistics::getTag(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  std::string tag;
  // Perform a linear search for the tag.
  for (typename map_t::value_type current_tag : Instance().tag_map_) {
    if (current_tag.second == handle) {
      return current_tag.first;
    }
  }
  return tag;
}

StatsCollectorImpl::StatsCollectorImpl(const size_t handle) : handle_(handle) {}

StatsCollectorImpl::StatsCollectorImpl(const std::string& tag)
    : handle_(Statistics::getHandle(tag)) {}

size_t StatsCollectorImpl::getHandle() const {
  return handle_;
}
void StatsCollectorImpl::addSample(double sample) const {
  Statistics::Instance().addSample(handle_, sample);
}
void StatsCollectorImpl::incrementOne() const {
  Statistics::Instance().addSample(handle_, 1.0);
}

void Statistics::addSample(const size_t handle, const double sample) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  stats_collectors_[handle].addValue(sample);
}

double Statistics::getLastValue(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getLastValue();
}

double Statistics::getLastValue(const std::string& tag) {
  return getLastValue(getHandle(tag));
}

double Statistics::getSum(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getSum();
}

double Statistics::getSum(const std::string& tag) {
  return getSum(getHandle(tag));
}

double Statistics::getMean(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMean();
}

double Statistics::getMean(const std::string& tag) {
  return getMean(getHandle(tag));
}

size_t Statistics::getNumSamples(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getTotalNumSamples();
}

size_t Statistics::getNumSamples(const std::string& tag) {
  return getNumSamples(getHandle(tag));
}

double Statistics::getVariance(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getLazyVariance();
}

double Statistics::getVariance(const std::string& tag) {
  return getVariance(getHandle(tag));
}

double Statistics::getMin(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMin();
}

double Statistics::getMin(const std::string& tag) {
  return getMin(getHandle(tag));
}

double Statistics::getMax(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMax();
}

double Statistics::getMax(const std::string& tag) {
  return getMax(getHandle(tag));
}

double Statistics::getHz(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMeanNumCallsPerSecond();
}

double Statistics::getHz(const std::string& tag) {
  return getHz(getHandle(tag));
}

// Delta time statistics.
double Statistics::getMeanDeltaTimeNanoseconds(const std::string& tag) {
  return getMeanDeltaTimeNanoseconds(getHandle(tag));
}

double Statistics::getMeanDeltaTimeNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMeanDeltaTimeNanoseconds();
}

int64_t Statistics::getMaxDeltaTimeNanoseconds(const std::string& tag) {
  return getMaxDeltaTimeNanoseconds(getHandle(tag));
}

int64_t Statistics::getMaxDeltaTimeNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMaxDeltaTimeNanoseconds();
}

int64_t Statistics::getMinDeltaTimeNanoseconds(const std::string& tag) {
  return getMinDeltaTimeNanoseconds(getHandle(tag));
}

int64_t Statistics::getMinDeltaTimeNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getMinDeltaTimeNanoseconds();
}

int64_t Statistics::getLastDeltaTimeNanoseconds(const std::string& tag) {
  return getLastDeltaTimeNanoseconds(getHandle(tag));
}

int64_t Statistics::getLastDeltaTimeNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().stats_collectors_[handle].getLastDeltaTimeNanoseconds();
}

double Statistics::getVarianceDeltaTimeNanoseconds(const std::string& tag) {
  return getVarianceDeltaTimeNanoseconds(getHandle(tag));
}

double Statistics::getVarianceDeltaTimeNanoseconds(const size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance()
      .stats_collectors_[handle]
      .getLazyVarianceDeltaTimeNanoseconds();
}

void Statistics::print(std::ostream& out) {  // NOLINT
  const map_t& tag_map = Instance().tag_map_;

  if (tag_map.empty()) {
    return;
  }

  out << "Statistics\n";

  out.width((std::streamsize)Instance().max_tag_length_);
  out.setf(std::ios::left, std::ios::adjustfield);
  out << "-----------";
  out.width(7);
  out.setf(std::ios::right, std::ios::adjustfield);
  out << "#\t";
  out << "Hz\t";
  out << "(avg     +- std    )\t";
  out << "[min,max]\n";

  for (const typename map_t::value_type& t : tag_map) {
    size_t i = t.second;
    out.width((std::streamsize)Instance().max_tag_length_);
    out.setf(std::ios::left, std::ios::adjustfield);
    out << t.first << "\t";
    out.width(7);

    out.setf(std::ios::right, std::ios::adjustfield);
    out << getNumSamples(i) << "\t";
    if (getNumSamples(i) > 0) {
      out << getHz(i) << "\t";
      double mean = getMean(i);
      double stddev = sqrt(getVariance(i));
      out << "(" << mean << " +- ";
      out << stddev << ")\t";

      double min_value = getMin(i);
      double max_value = getMax(i);

      out << "[" << min_value << "," << max_value << "]";
    }
    out << std::endl;
  }
}

void Statistics::writeToYamlFile(const std::string& path) {
  const map_t& tag_map = Instance().tag_map_;

  if (tag_map.empty()) {
    return;
  }

  std::ofstream output_file(path);

  if (!output_file) {
    LOG(ERROR) << "Could not write statistics: Unable to open file: " << path;
    return;
  }

  VLOG(1) << "Writing statistics to file: " << path;
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
      output_file << "  samples: " << getNumSamples(index) << "\n";
      output_file << "  mean: " << getMean(index) << "\n";
      output_file << "  stddev: " << sqrt(getVariance(index)) << "\n";
      output_file << "  min: " << getMin(index) << "\n";
      output_file << "  max: " << getMax(index) << "\n";
    }
    output_file << "\n";
  }
}

std::string Statistics::print() {
  std::stringstream ss;
  print(ss);
  return ss.str();
}

void Statistics::reset() {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  Instance().tag_map_.clear();
}

}  // namespace statistics
