#ifndef ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_

#include <Eigen/Dense>
#include <aslam/common/channel-declaration.h>

// 3D keypoint coordinates in Lidar Frame
DECLARE_CHANNEL(KEYPOINT_VECTORS, Eigen::Matrix3Xd)

#endif  // ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
