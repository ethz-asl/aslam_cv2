#include <algorithm>
#include <math.h>
#include <ostream>  //NOLINT
#include <sstream>
#include <stdio.h>
#include <string>

#include <aslam/common/timer.h>

namespace aslam {
namespace timing {

const double kNumSecondsPerNanosecond = 1.e-9;

Timing& Timing::Instance() {
  static Timing t;
  return t;
}

Timing::Timing() : maxTagLength_(0) { }

Timing::~Timing() { }

// Static functions to query the timers:
size_t Timing::GetHandle(std::string const& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  // Search for an existing tag.
  map_t::iterator i = Instance().tagMap_.find(tag);
  if (i == Instance().tagMap_.end()) {
    // If it is not there, create a tag.
    size_t handle = Instance().timers_.size();
    Instance().tagMap_[tag] = handle;
    Instance().timers_.push_back(statistics::StatisticsMapValue());
    // Track the maximum tag length to help printing a table of timing values
    // later.
    Instance().maxTagLength_ = std::max(Instance().maxTagLength_, tag.size());
    return handle;
  } else {
    return i->second;
  }
}

std::string Timing::GetTag(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  std::string tag;

  // Perform a linear search for the tag.
  for (typename map_t::value_type current_tag : Instance().tagMap_) {
    if (current_tag.second == handle) {
      return current_tag.first;
    }
  }
  return tag;
}

// Class functions used for timing.
Timer::Timer(std::string const& tag, bool construct_stopped)
    : timing_(false), handle_(Timing::GetHandle(tag)), tag_(tag) {
  if (!construct_stopped)
    Start();
}

Timer::~Timer() {
  if (IsTiming())
    Stop();
}

void Timer::Start() {
  timing_ = true;
  time_ = std::chrono::system_clock::now();
}

double Timer::Stop() {
  if (timing_) {
    std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
    double dt = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        now - time_).count()) * kNumSecondsPerNanosecond;
    Timing::Instance().AddTime(handle_, dt);
    timing_ = false;
    return dt;
  }
  return 0.0;
}

void Timer::Discard() {
  timing_ = false;
}

bool Timer::IsTiming() const {
  return timing_;
}

size_t Timer::GetHandle() const {
  return handle_;
}

void Timing::AddTime(size_t handle, double seconds) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  timers_[handle].AddValue(seconds);
}

double Timing::GetTotalSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].Sum();
}

double Timing::GetTotalSeconds(std::string const& tag) {
  return GetTotalSeconds(GetHandle(tag));
}

double Timing::GetMeanSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].Mean();
}

double Timing::GetMeanSeconds(std::string const& tag) {
  return GetMeanSeconds(GetHandle(tag));
}

size_t Timing::GetNumSamples(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].TotalSamples();
}

size_t Timing::GetNumSamples(std::string const& tag) {
  return GetNumSamples(GetHandle(tag));
}

double Timing::GetVarianceSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].LazyVariance();
}

double Timing::GetVarianceSeconds(std::string const& tag) {
  return GetVarianceSeconds(GetHandle(tag));
}

double Timing::GetMinSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].Min();
}

double Timing::GetMinSeconds(std::string const& tag) {
  return GetMinSeconds(GetHandle(tag));
}

double Timing::GetMaxSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].Max();
}

double Timing::GetMaxSeconds(std::string const& tag) {
  return GetMaxSeconds(GetHandle(tag));
}

double Timing::GetHz(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return 1.0 / Instance().timers_[handle].RollingMean();
}

double Timing::GetHz(std::string const& tag) {
  return GetHz(GetHandle(tag));
}

std::string Timing::SecondsToTimeString(double seconds) {
  double secs = fmod(seconds, 60);
  int minutes = (seconds / 60);
  int hours = (seconds / 3600);
  minutes = minutes - (hours * 60);

  char buffer[256];
  snprintf(buffer, sizeof(buffer), "%02d:" "%02d:" "%09.6f", hours, minutes, secs);
  return buffer;
}

void Timing::Print(std::ostream& out) {  //NOLINT
  map_t tagMap;
  {
    std::lock_guard<std::mutex> lock(Instance().mutex_);
    tagMap = Instance().tagMap_;
  }

  if (tagMap.empty()) {
    return;
  }

  out << "SM Timing\n";
  out << "-----------\n";
  for (typename map_t::value_type t : tagMap) {
    size_t i = t.second;
    out.width((std::streamsize) Instance().maxTagLength_);
    out.setf(std::ios::left, std::ios::adjustfield);
    out << t.first << "\t";
    out.width(7);

    out.setf(std::ios::right, std::ios::adjustfield);
    out << GetNumSamples(i) << "\t";
    if (GetNumSamples(i) > 0) {
      out << SecondsToTimeString(GetTotalSeconds(i)) << "\t";
      double meansec = GetMeanSeconds(i);
      double stddev = sqrt(GetVarianceSeconds(i));
      out << "(" << SecondsToTimeString(meansec) << " +- ";
      out << SecondsToTimeString(stddev) << ")\t";

      double minsec = GetMinSeconds(i);
      double maxsec = GetMaxSeconds(i);

      // The min or max are out of bounds.
      out << "[" << SecondsToTimeString(minsec) << "," << SecondsToTimeString(maxsec) << "]";
    }
    out << std::endl;
  }
}

std::string Timing::Print() {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

void Timing::Reset() {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  Instance().tagMap_.clear();
}

}  // namespace timing
}  // namespace aslam
