#ifndef ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_

#include <Eigen/Dense>
#include <aslam/common/channel-declaration.h>

/// Track ID's for tracked features. (-1 if not tracked); (feature tracker
/// output)
DECLARE_CHANNEL(LIDAR_TRACK_IDS, Eigen::VectorXi)

// 3D keypoint coordinates in Lidar Frame
DECLARE_CHANNEL(LIDAR_3D_MEASUREMENTS, Eigen::Matrix3Xd)

// 2D pixel coordinates in the Lidar Image
DECLARE_CHANNEL(LIDAR_2D_MEASUREMENTS, Eigen::Matrix2Xd)

/// The keypoint descriptors. (extractor output)(cols are descriptors)
DECLARE_CHANNEL(
    LIDAR_DESCRIPTORS,
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>)

/// Keypoint coordinate uncertainties of the raw keypoints. (keypoint detector
/// output)
/// (cols are uncertainties)
DECLARE_CHANNEL(LIDAR_KEYPOINT_2D_MEASUREMENT_UNCERTAINTIES, Eigen::VectorXd)

#endif  // ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
