#ifndef ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
#define ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_

#include <Eigen/Dense>
#include <aslam/common/channel-declaration.h>

/// Coordinates of the raw keypoints. (keypoint detector output)
/// (cols are keypoints)
DECLARE_CHANNEL(VISUAL_KEYPOINT_MEASUREMENTS, Eigen::Matrix2Xd)

/// Keypoint coordinate uncertainties of the raw keypoints. (keypoint detector output)
/// (cols are uncertainties)
DECLARE_CHANNEL(VISUAL_KEYPOINT_MEASUREMENT_UNCERTAINTIES, Eigen::VectorXd)

/// Keypoint orientation from keypoint extractor. (keypoint detector output)
/// Computed orientation of the keypoint (-1 if not applicable);
/// it's in [0,360) degrees and measured relative to image coordinate system, ie in clockwise.
DECLARE_CHANNEL(VISUAL_KEYPOINT_ORIENTATIONS, Eigen::VectorXd)

/// Diameter of the meaningful keypoint neighborhood. (keypoint detector output)
DECLARE_CHANNEL(VISUAL_KEYPOINT_SCALES, Eigen::VectorXd)

/// The score by which the most strong keypoints have been selected. Can be used for the further
/// sorting or subsampling. (keypoint detector output)
DECLARE_CHANNEL(VISUAL_KEYPOINT_SCORES, Eigen::VectorXd)

/// The keypoint descriptors. (extractor output)
/// (cols are descriptors)
/// This channel stores the visual keypoint binary descriptors
DECLARE_CHANNEL(DESCRIPTORS,
                Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>)

/// Track ID's for tracked features. (-1 if not tracked); (feature tracker output)
/// This channel stores the visual keypoint track ids
DECLARE_CHANNEL(TRACK_IDS, Eigen::VectorXi)

/// The semantic channels are separated from the visual keypoints, so don't try
/// to access the semantic measurements with visual keypoints index.

/// Parameters of bounding boxes from detectors in pixel coordinates
/// This channel is organized as one col per box and the values 
/// are in the order of centroid_col, centroid_row, bb width, bb height 
DECLARE_CHANNEL(SEMANTIC_OBJECT_MEASUREMENTS, Eigen::Matrix4Xd)

/// Semantic object measurement uncertainy from the detector
/// (cols are uncertainties)
DECLARE_CHANNEL(SEMANTIC_OBJECT_MEASUREMENT_UNCERTAINTIES, Eigen::VectorXd)

/// Semantic object class ids from the detector
/// (cols are ids)
DECLARE_CHANNEL(SEMANTIC_OBJECT_CLASS_IDS, Eigen::VectorXi)

/// Semantic measurements descriptors
/// (cols are descriptor)
DECLARE_CHANNEL(SEMANTIC_OBJECT_DESCRIPTORS,
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>)

/// Semantic object measurement track ids
/// (cols are object track ids)
/// The value of -1 is stored if not the measurement is not tracked 
DECLARE_CHANNEL(SEMANTIC_OBJECT_TRACK_IDS, Eigen::VectorXi)

/// The raw image.
DECLARE_CHANNEL(RAW_IMAGE, cv::Mat)

DECLARE_CHANNEL(CV_MAT, cv::Mat)

#endif  // ASLAM_CV_COMMON_CHANNEL_DEFINITIONS_H_
