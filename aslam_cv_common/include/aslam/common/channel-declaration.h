#ifndef ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
#include <string>

#include <Eigen/Dense>
#include <aslam/common/channel.h>
#include <aslam/common/macros.h>

// Extract the type from an expression which wraps a type inside braces. This
// is done to protect the commas in some types.
template<typename T> struct ArgumentType;
template<typename T, typename U> struct ArgumentType<T(U)> { typedef U Type; };
#define GET_TYPE(TYPE) ArgumentType<void(TYPE)>::Type

#define DECLARE_CHANNEL_IMPL(NAME, TYPE)             \
namespace aslam {                                    \
namespace channels {                                 \
struct NAME : aslam::Channel<GET_TYPE(TYPE)> {       \
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;                   \
  typedef typename GET_TYPE(TYPE) Type;              \
  static const std::string& name() { return #NAME; } \
};                                                   \
const std::string NAME##_CHANNEL = #NAME;            \
typedef GET_TYPE(TYPE) NAME##_CHANNEL_TYPE;          \
}                                                    \
}                                                    \

// Wrap types that contain commas inside braces.
#define DECLARE_CHANNEL(x, ...) DECLARE_CHANNEL_IMPL(x, (__VA_ARGS__))

#endif  // ASLAM_CV_COMMON_CHANNEL_DECLARATIONS_H_
