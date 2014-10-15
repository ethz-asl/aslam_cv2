#ifndef LINEAR_TRIANGULATION_H_
#define LINEAR_TRIANGULATION_H_

#include <vector>

#include <Eigen/Core>

#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>

namespace aslam {
namespace geometric_vision {

bool triangulateFromNormalizedTwoViews(const Eigen::Vector2d& measurement1,
                                       const aslam::Transformation& camera_pose1,
                                       const Eigen::Vector2d& measurement2,
                                       const aslam::Transformation& camera_pose2,
                                       Eigen::Vector3d* triangulated_point);

bool triangulateFromNormalizedTwoViewsHomogeneous(const Eigen::Vector2d& measurement1,
                                                  const aslam::Transformation& camera_pose1,
                                                  const Eigen::Vector2d& measurement2,
                                                  const aslam::Transformation& camera_pose2,
                                                  Eigen::Vector3d* triangulated_point);

bool triangulateFromNormalizedNViews(
    const Aligned<std::vector, Eigen::Vector2d>::type& measurements,
    const Aligned<std::vector, aslam::Transformation>::type& camera_poses,
    Eigen::Vector3d* triangulated_point);

}  // namespace geometric_vision
}  // namespace aslam

#endif /* LINEAR_TRIANGULATION_H_ */
