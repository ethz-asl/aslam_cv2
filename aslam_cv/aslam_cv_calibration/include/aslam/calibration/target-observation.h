#ifndef ASLAM_CALIBRATION_TARGET_OBSERVATION_H
#define ASLAM_CALIBRATION_TARGET_OBSERVATION_H

#include <memory>
#include <unordered_map>

#include <aslam/common/macros.h>
#include <Eigen/Core>

#include "aslam/calibration/target-base.h"

namespace aslam {
namespace calibration {

class TargetObservation  {
 public:
  ASLAM_POINTER_TYPEDEFS(TargetObservation);

  TargetObservation(const TargetBase::Ptr& target,
      const Eigen::VectorXi& corner_ids,
      const Eigen::Matrix2Xd& image_corners) :
        target_(target),
        corner_ids_(corner_ids),
        image_corners_(image_corners) {
    CHECK(target);
    CHECK_EQ(corner_ids.rows(), image_corners.cols());
    buildIndex();
  }

  virtual ~TargetObservation() {};

  TargetBase::Ptr getTarget() { return target_; };
  TargetBase::ConstPtr getTarget() const { return target_; };

 private:
  void buildIndex() {
    cornerid_to_index_map_.clear();
    for (int index = 0; index < corner_ids_.rows(); ++index) {
      cornerid_to_index_map_.emplace(corner_ids_(index), index);
    }
  };

  const TargetBase::Ptr target_;
  const Eigen::VectorXi corner_ids_;
  const Eigen::Matrix2Xd image_corners_;

  std::unordered_map<size_t, size_t> cornerid_to_index_map_;
};

}  // namespace calibration
}  // namespace aslam

#endif  // ASLAM_CALIBRATION_TARGET_OBSERVATION_H
