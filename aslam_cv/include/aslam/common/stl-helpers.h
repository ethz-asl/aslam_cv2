#ifndef ASLAM_STL_HELPERS_H_
#define ASLAM_STL_HELPERS_H_

#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

#include <aslam/common/memory.h>
#include <Eigen/Core>
#include <glog/logging.h>

namespace aslam {
namespace common {

template<typename RandAccessIter>
double median(RandAccessIter begin, RandAccessIter end) {
  CHECK(begin != end);
  size_t size = end - begin;
  size_t middle_idx = size / 2;
  RandAccessIter target_high = begin + middle_idx;
  std::nth_element(begin, target_high, end);

  // Odd number of elements.
  if (size % 2 != 0) {
    return *target_high;
  }
  // Even number of elements.
  double target_high_value = *target_high;
  RandAccessIter target_low = target_high - 1;
  std::nth_element(begin, target_low, end);
  return (target_high_value + *target_low) / 2.0;
}

template<typename ElementType>
std::vector<ElementType> drawNRandomElements(size_t N, const std::vector<ElementType>& input) {
  if (input.size() <= N) {
    return input;
  }

  // Draw random indices.
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, N);

  std::unordered_set<size_t> random_indices;
  while (random_indices.size() < N) {
    random_indices.insert(distribution(generator));
  }

  // Copy to output.
  std::vector<ElementType> output;
  output.reserve(N);
  for (size_t idx : random_indices) {
    output.emplace_back(input[idx]);
  }
  return output;
}

template<int VectorDim>
inline void convertEigenToStlVector(
    const Eigen::template Matrix<double, VectorDim, Eigen::Dynamic>& input,
    typename Aligned<std::vector, Eigen::template Matrix<double, VectorDim, 1>>::type* output) {
  CHECK_NOTNULL(output);
  size_t num_cols = input.cols();
  output->clear();
  output->reserve(num_cols);

  auto inserter = std::inserter(*output, output->end());
  for (size_t idx = 0; idx < num_cols; ++idx) {
    *inserter++ = input.col(idx);
  }
}

}  // namespace common
}  // namespace aslam

#endif  // ASLAM_STL_HELPERS_H_
