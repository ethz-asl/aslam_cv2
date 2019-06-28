#ifndef ASLAM_CAMERAS_CERES_JET_MATH_H_
#define ASLAM_CAMERAS_CERES_JET_MATH_H_

#include <cmath>

namespace aslam {
namespace jet_math {

inline double sqrt(const double x) {
  return std::sqrt(x);
}

inline double tan(const double x) {
  return std::tan(x);
}

inline double atan(const double x) {
  return std::atan(x);
}

// Mathematical functions for use with the Jet type of the Ceres solver.
// Adapted from: ceres/jet.h

// sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))
template <template <typename, int> class JetType, typename T, int N>
JetType<T, N> sqrt(const JetType<T, N>& f) {
  const T tmp = sqrt(f.a);
  const T two_a_inverse = T(1.0) / (T(2.0) * tmp);
  return JetType<T, N>(tmp, f.v * two_a_inverse);
}

// tan(a + h) ~= tan(a) + (1 + tan(a)^2) h
template <template <typename, int> class JetType, typename T, int N>
inline JetType<T, N> tan(const JetType<T, N>& f) {
  const T tan_a = tan(f.a);
  const T tmp = T(1.0) + tan_a * tan_a;
  return JetType<T, N>(tan_a, tmp * f.v);
}

// atan(a + h) ~= atan(a) + 1 / (1 + a^2) h
template <template <typename, int> class JetType, typename T, int N>
inline JetType<T, N> atan(const JetType<T, N>& f) {
  const T tmp = T(1.0) / (T(1.0) + f.a * f.a);
  return JetType<T, N>(atan(f.a), tmp * f.v);
}

}  // namespace jet_math
}  // namespace aslam

#endif  // ASLAM_CAMERAS_CERES_JET_MATH_H_
