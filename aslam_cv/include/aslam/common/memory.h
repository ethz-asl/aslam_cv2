#ifndef ASLAM_MEMORY_HELPERS_H_
#define ASLAM_MEMORY_HELPERS_H_

#include <memory>

namespace aslam {
/// \brief Aligned allocator to be used like std::make_shared
///
/// To be used like:
///   std::shared_ptr my_ptr = aligned_shared<aslam::RadTanDistortion>(params);
template<typename Type, typename ... Arguments>
inline std::shared_ptr<Type> aligned_shared(Arguments&&... arguments) {
  typedef typename std::remove_const<Type>::type TypeNonConst;
  return std::allocate_shared<Type>(Eigen::aligned_allocator<TypeNonConst>(),
                                    std::forward<Arguments>(arguments)...);
}

} // namespace aslam

#endif  // ASLAM_MEMORY_HELPERS_H_

