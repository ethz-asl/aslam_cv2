/*
 * Copyright (c) 2011-2013, Paul Furgale and others.
 * All rights reserved.
 *
 * This code is published under the Revised BSD (New BSD) license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the <organization> nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ASLAM_COMMON_TIMER_H_
#define ASLAM_COMMON_TIMER_H_

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "aslam/common/statistics/statistics.h"

namespace timing {

// A class that has the timer interface but does nothing. Swapping this in
// place of the Timer class (say with a typedef) should allow one to disable
// timing. Because all of the functions are inline, they should just disappear.
class DummyTimer {
 public:
  DummyTimer(size_t /*handle*/, bool construct_stopped = false) {  // NOLINT
    static_cast<void>(construct_stopped);
  }
  DummyTimer(
      const std::string& /*tag*/, bool construct_stopped = false) {  // NOLINT
    static_cast<void>(construct_stopped);
  }
  ~DummyTimer() {}
  void start() {}
  int64_t stop() {
    return -1;
  }
  void discard() {}
  bool ssTiming() const {
    return false;
  }
  size_t getHandle() const {
    return 0u;
  }
};

class TimerImpl {
 public:
  TimerImpl(const std::string& tag, bool construct_stopped = false);  // NOLINT
  ~TimerImpl();

  void start();
  // Returns the amount of time passed between start() and stop() in
  // nanoseconds.
  int64_t stop();
  void discard();
  bool ssTiming() const;
  size_t getHandle() const;

 private:
  std::chrono::time_point<std::chrono::system_clock> time_;

  bool is_timing_;
  size_t handle_;
  std::string tag_;
};

class Timing {
 public:
  typedef std::map<std::string, size_t> map_t;
  friend class TimerImpl;
  // Definition of static functions to query the timers.
  static size_t getHandle(const std::string& tag);
  static std::string getTag(const size_t handle);
  static int64_t getTotalNumNanoseconds(const size_t handle);
  static int64_t getTotalNumNanoseconds(const std::string& tag);
  static double getMeanNanoseconds(size_t handle);
  static double getMeanNanoseconds(const std::string& tag);
  static size_t getNumSamples(size_t handle);
  static size_t getNumSamples(const std::string& tag);
  static double getVarianceNanoseconds(const size_t handle);
  static double getVarianceNanoseconds(const std::string& tag);
  static int64_t getMinNanoseconds(const size_t handle);
  static int64_t getMinNanoseconds(const std::string& tag);
  static int64_t getMaxNanoseconds(const size_t handle);
  static int64_t getMaxNanoseconds(const std::string& tag);
  static double getHz(size_t handle);
  static double getHz(const std::string& tag);
  static void writeToYamlFile(const std::string& path);
  static void print(std::ostream& out);  // NOLINT
  static std::string print();
  static void reset();
  static const map_t& getTimerImpls() {
    return Instance().tag_map_;
  }

 private:
  void addTime(const size_t handle, const int64_t time_nanoseconds);

  static Timing& Instance();

  Timing();
  ~Timing();

  using list_t = std::vector<statistics::StatisticsMapValue<int64_t>>;

  list_t timers_;
  map_t tag_map_;
  size_t max_tag_length_;
  std::mutex mutex_;
};

#if ENABLE_TIMING
typedef TimerImpl Timer;
#else
typedef DummyTimer Timer;
#endif

}  // namespace timing

#endif  // ASLAM_COMMON_TIMER_H_
