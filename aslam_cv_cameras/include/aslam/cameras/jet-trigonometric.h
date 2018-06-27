#ifndef ASLAM_CAMERAS_JET_TRIGONOMETRIC_H_
#define ASLAM_CAMERAS_JET_TRIGONOMETRIC_H_

#include <cmath>

namespace aslam {
namespace jet_trigonometric {

inline double sqrt(double x) {
  return std::sqrt(x);
}

inline double tan(double x) {
  return std::tan(x);
}

inline double atan(double x) {
  return std::atan(x);
}

template <template <typename, int> class JetType, typename T, int N>
inline JetType<T, N> sqrt(const JetType<T, N>& f) {
  JetType<T, N> g;
  g.a = sqrt(f.a);
  const T two_a_inverse = T(1.0) / (T(2.0) * g.a);
  g.v = f.v * two_a_inverse;
  return g;
}

template <template <typename, int> class JetType, typename T, int N>
inline JetType<T, N> tan(const JetType<T, N>& f) {
  const T tan_a = tan(f.a);
  const T tmp = T(1.0) + tan_a * tan_a;
  return JetType<T, N>(tan_a, tmp * f.v);
}

template <template <typename, int> class JetType, typename T, int N>
inline JetType<T, N> atan(const JetType<T, N>& f) {
  const T tmp = T(1.0) / (T(1.0) + f.a * f.a);
  return JetType<T, N>(atan(f.a), tmp * f.v);
}
}  // namespace jet_trigonometric

}  // namespace aslam

#endif  // ASLAM_CAMERAS_JET_TRIGONOMETRIC_H_
