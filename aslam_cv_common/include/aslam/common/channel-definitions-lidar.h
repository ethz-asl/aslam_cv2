#ifndef ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_

#include <Eigen/Dense>
#include <aslam/common/channel-declaration.h>

// 3D keypoint coordinates in Lidar Frame
DECLARE_CHANNEL(KEYPOINT_VECTORS, Eigen::Matrix3Xd)

// 2D pixel coordinates in the Lidar Image
DECLARE_CHANNEL(LIDAR_2D_MEASUREMENTS, Eigen::Matrix2Xd)

/// The keypoint descriptors. (extractor output)(cols are descriptors)
DECLARE_CHANNEL(
    LIDAR_DESCRIPTORS, Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>)

#endif  // ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
