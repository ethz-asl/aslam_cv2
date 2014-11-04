#ifndef ASLAM_TIME_H_
#define ASLAM_TIME_H_

#include <chrono>
#include <cstdint>

namespace aslam {

/// \brief get the current time in nanoseconds since epoch.
inline int64_t nanoSecondsSinceEpoch() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>
        (std::chrono::system_clock::now().time_since_epoch()).count();
};

/// \brief convert double seconds to integer nanoseconds since epoch.
inline int64_t secondsToNanoSeconds(double seconds) {
  constexpr double kSecondsToNanoSeconds = 1e9;
  return static_cast<int64_t>(seconds * kSecondsToNanoSeconds);
}

/// \brief convert integer nanoseconds into double seconds since epoch.
inline double nanoSecondsToSeconds(int64_t nano_seconds) {
  constexpr double kNanoSecondsToSeconds = 1e-9;
  return static_cast<double>(nano_seconds * kNanoSecondsToSeconds);
}

/// \brief return a magic number representing an invalid timestamp.
///        std::numeric_limits<int64_t>::min()
inline constexpr int64_t getInvalidTime() {
  return std::numeric_limits<int64_t>::min();
}

/// \brief Is the time valid? This uses a magic number
inline bool isValidTime(int64_t time) {
  return time != getInvalidTime();
}

}  // namespace aslam

#endif  // ASLAM_TIME_H_
