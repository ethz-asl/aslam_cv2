#ifndef ASLAM_TIME_H_
#define ASLAM_TIME_H_

#include <chrono>
#include <cstdint>

namespace aslam {
uint64_t nanoSecondsSinceEpoch() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();
};

uint64_t secondsToNanoSeconds(double seconds) {
  constexpr double kSecondsToNanoSeconds = 1e9;
  return static_cast<uint64_t>(seconds * kSecondsToNanoSeconds);
}

double nanoSecondsToSeconds(uint64_t nano_seconds) {
  constexpr double kNanoSecondsToSeconds = 1e-9;
  return static_cast<double>(nano_seconds * kNanoSecondsToSeconds);
}
}  // namespace aslam

#endif  // ASLAM_TIME_H_
