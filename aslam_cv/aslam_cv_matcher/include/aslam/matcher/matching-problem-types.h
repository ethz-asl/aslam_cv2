#ifndef MATCHER_MATCHING_PROBLEM_TYPES_H_
#define MATCHER_MATCHING_PROBLEM_TYPES_H_

#include <vector>

#include <aslam/common/memory.h>
#include <Eigen/Core>

namespace aslam {

typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> Descriptor;

struct LandmarkWithDescriptor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LandmarkWithDescriptor() = delete;
  LandmarkWithDescriptor(const Eigen::Vector3d& p_C_landmark,
                         const Descriptor& descriptor)
    : p_C_landmark_(p_C_landmark), descriptor_(descriptor) {}
  virtual ~LandmarkWithDescriptor() = default;

  const Eigen::Vector3d& get_p_C_landmark() const {
    return p_C_landmark_;
  }

  const Descriptor& getDescriptor() const {
    return descriptor_;
  }

 private:
  Eigen::Vector3d p_C_landmark_;
  Descriptor descriptor_;
};

typedef Aligned<std::vector, LandmarkWithDescriptor>::type LandmarkWithDescriptorList;

}  // namespace aslam

#endif  // MATCHER_MATCHING_PROBLEM_TYPES_H_
