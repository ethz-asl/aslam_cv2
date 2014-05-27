#ifndef ASLAM_COMMON_ALIGNED_ALLOCATION_H_
#define ASLAM_COMMON_ALIGNED_ALLOCATION_H_

#include <functional>
#include <map>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <utility>

#include <Eigen/Core>
#include <Eigen/StdVector>

// This file provides helpers type-traits to simplify the aligned allocation
// requirements for Eigen:
// See https://github.com/ethz-asl/aslam_cv2/wiki/Eigen-alignment-in-aslam_cv

template<template<typename, typename> class Container, typename Type>
struct Aligned {
  typedef Container<Type, Eigen::aligned_allocator<Type> > type;
};

template<typename KeyType, typename ValueType>
struct AlignedMap {
  typedef std::map<
      KeyType, ValueType,
      std::less<KeyType>,
      Eigen::aligned_allocator<std::pair<const KeyType, ValueType> > > type;
};

template<typename KeyType, typename ValueType>
struct AlignedUnorderedMap {
  typedef std::unordered_map<KeyType, ValueType,
      std::hash<KeyType>, std::equal_to<KeyType>,
      Eigen::aligned_allocator<std::pair<const KeyType, ValueType> > > type;
};

template<typename Type, typename ... Arguments>
inline std::shared_ptr<Type> aligned_shared(Arguments&&... arguments) {
  typedef typename std::remove_const<Type>::type TypeNonConst;
  return std::allocate_shared<Type>(Eigen::aligned_allocator<TypeNonConst>(),
                                    std::forward<Arguments>(arguments)...);
}
#endif  // ASLAM_COMMON_ALIGNED_ALLOCATION_H_
