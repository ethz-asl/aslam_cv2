#ifndef ASLAM_COMMON_PREDICATES_H_
#define ASLAM_COMMON_PREDICATES_H_

#include <memory>

namespace aslam {

template <typename Value>
bool checkSharedEqual(
    const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
  if (lhs && rhs) {
    // if they are both nonnull, check for equality
    return (*lhs) == (*rhs);
  } else {
    // otherwise, check if they are both null
    return (!lhs) && (!rhs);
  }
}

}  // namespace aslam

#endif  // ASLAM_COMMON_PREDICATES_H_
