#ifndef ASLAM_TIMING_TIMER_H_
#define ASLAM_TIMING_TIMER_H_

#include <algorithm>
#include <chrono>
#include <limits>
#include <mutex>
#include <map>
#include <string>
#include <vector>

#include <aslam/common/statistics/statistics.h>

namespace aslam {
namespace timing {

// A class that has the timer interface but does nothing. Swapping this in in
// place of the Timer class (say with a typedef) should allow one to disable
// timing. Because all of the functions are inline, they should just disappear.
class DummyTimer {
 public:
  DummyTimer(size_t /*handle*/, bool construct_stopped = false) {
    static_cast<void>(construct_stopped);
  }
  DummyTimer(std::string const& /*tag*/, bool construct_stopped = false) {
    static_cast<void>(construct_stopped);
  }
  ~DummyTimer() {}
  void Start() {}
  double Stop() { return -1.0; }
  bool IsTiming() { return false; }
};

class Timer {
 public:
  Timer(std::string const& tag, bool construct_stopped = false);
  ~Timer();

  void Start();
  // Returns the amount of time passed between Start() and Stop().
  double Stop();
  void Discard();
  bool IsTiming() const;
  size_t GetHandle() const;

 private:
  std::chrono::time_point<std::chrono::system_clock> time_;

  bool timing_;
  size_t handle_;
  std::string tag_;
};

class Timing {
 public:
  typedef std::map<std::string, size_t> map_t;
  friend class Timer;
  // Definition of static functions to query the timers.
  static size_t GetHandle(std::string const& tag);
  static std::string GetTag(size_t handle);
  static double GetTotalSeconds(size_t handle);
  static double GetTotalSeconds(std::string const& tag);
  static double GetMeanSeconds(size_t handle);
  static double GetMeanSeconds(std::string const& tag);
  static size_t GetNumSamples(size_t handle);
  static size_t GetNumSamples(std::string const& tag);
  static double GetVarianceSeconds(size_t handle);
  static double GetVarianceSeconds(std::string const& tag);
  static double GetMinSeconds(size_t handle);
  static double GetMinSeconds(std::string const& tag);
  static double GetMaxSeconds(size_t handle);
  static double GetMaxSeconds(std::string const& tag);
  static double GetHz(size_t handle);
  static double GetHz(std::string const& tag);
  static void Print(std::ostream& out);  // NOLINT
  static std::string Print();
  static std::string SecondsToTimeString(double seconds);
  static void Reset();
  static const map_t& GetTimers() {
    return Instance().tagMap_;
  }

 private:
  void AddTime(size_t handle, double seconds);

  static Timing& Instance();

  Timing();
  ~Timing();

  typedef std::vector<statistics::StatisticsMapValue> list_t;

  list_t timers_;
  map_t tagMap_;
  size_t maxTagLength_;
  std::mutex mutex_;
};

#if ENABLE_TIMING
typedef Timer DebugTimer;
#else
typedef DummyTimer DebugTimer;
#endif

}       // namespace timing
}       // namespace aslam
#endif  // ASLAM_TIMING_TIMER_H_
