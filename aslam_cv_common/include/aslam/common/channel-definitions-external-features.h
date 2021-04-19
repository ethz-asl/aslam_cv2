#ifndef ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_EXTERNAL_FEATURES_H_
#define ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_EXTERNAL_FEATURES_H_

#include <Eigen/Dense>
#include <aslam/common/channel-declaration.h>

/// Coordinates of the raw keypoints. (keypoint detector output)
/// (cols are keypoints)
DECLARE_CHANNEL(EXTERNAL_KEYPOINT_MEASUREMENTS, Eigen::Matrix2Xd)

/// Keypoint coordinate uncertainties of the raw keypoints. (keypoint detector output)
/// (cols are uncertainties)
DECLARE_CHANNEL(EXTERNAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES, Eigen::VectorXd)

/// Keypoint orientation from keypoint extractor. (keypoint detector output)
/// Computed orientation of the keypoint (-1 if not applicable);
/// it's in [0,360) degrees and measured relative to image coordinate system, ie in clockwise.
DECLARE_CHANNEL(EXTERNAL_KEYPOINT_ORIENTATIONS, Eigen::VectorXd)

/// Diameter of the meaningful keypoint neighborhood. (keypoint detector output)
DECLARE_CHANNEL(EXTERNAL_KEYPOINT_SCALES, Eigen::VectorXd)

/// The score by which the most strong keypoints have been selected. Can be used for the further
/// sorting or subsampling. (keypoint detector output)
DECLARE_CHANNEL(EXTERNAL_KEYPOINT_SCORES, Eigen::VectorXd)

/// The keypoint descriptors. (extractor output)
/// (cols are descriptors)
DECLARE_CHANNEL(EXTERNAL_DESCRIPTORS,
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>)

/// Track ID's for tracked features. (-1 if not tracked); (feature tracker output)
DECLARE_CHANNEL(EXTERNAL_TRACK_IDS, Eigen::VectorXi)


#endif  // ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_EXTERNAL_FEATURES_H_
