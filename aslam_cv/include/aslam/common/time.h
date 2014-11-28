#ifndef ASLAM_TIME_H_
#define ASLAM_TIME_H_

#include <chrono>
#include <cstdint>

namespace aslam {
namespace time {

namespace internal {
template<typename TimeUnit> struct time_traits;
struct sec; template<> struct time_traits<sec> { static constexpr size_t nanoseconds = 1e9; };
struct milli; template<> struct time_traits<milli> { static constexpr size_t nanoseconds = 1e6; };
struct micro; template<> struct time_traits<micro> { static constexpr size_t nanoseconds = 1e3; };
struct nano; template<> struct time_traits<nano> { static constexpr size_t nanoseconds = 1; };
template<typename TimeUnit> inline constexpr int64_t convertToNanoseconds(int64_t value) {
  return value * time_traits<TimeUnit>::nanoseconds;
}
}  // namespace internal

/// Convenience functions to convert the specified unit to the nanoseconds format.
/// Example: int64_t sampling_time = aslam::time::microseconds(10);
constexpr auto seconds = internal::convertToNanoseconds<internal::sec>;
constexpr auto milliseconds = internal::convertToNanoseconds<internal::milli>;
constexpr auto microseconds = internal::convertToNanoseconds<internal::micro>;
constexpr auto nanoseconds = internal::convertToNanoseconds<internal::nano>;

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

}  // namespace time
}  // namespace aslam

#endif  // ASLAM_TIME_H_
