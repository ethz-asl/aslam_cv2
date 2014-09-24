#ifndef ASLAM_UNDISTORT_HELPERS_H_
#define ASLAM_UNDISTORT_HELPERS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

//This file contains modified opencv routines which use aslam's distort and project functionality.
//check http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html

namespace aslam {
namespace common {

/// \brief: calculates the inner(min)/outer(max) rectangle on the undistorted image
///  @param[in] input_camera Input camera geometry
///  @param[in] undistort_to_pinhole Undistort image to a pinhole projection
///                                  (remove distortion and projection effects)
///  @param[out] inner Inscribed image rectangle (all pixels valid)
///  @param[out] outer Circumscribed image rectangle (no pixels lost)
void getUndistortRectangles(const aslam::Camera& input_camera, bool undistort_to_pinhole,
                            cv::Rect_<float>& inner, cv::Rect_<float>& outer);

/// \brief Returns the new camera matrix based on the free scaling parameter.
/// INPUT:
/// @param[in] input_camera Aslam camera geometry (distortion and intrinsics used)
/// @param[in] alpha Free scaling parameter between 0 (when all the pixels in the undistorted image
///                  will be valid) and 1 (when all the source image pixels will be retained in the
///                  undistorted image)
///  @param[in] undistort_to_pinhole Undistort image to a pinhole projection
///                                  (remove distortion and projection effects)
/// @param[out] output_size Image size after undistortion. By default it will be set to imageSize.
/// @return The output camera matrix.
Eigen::Matrix3d getOptimalNewCameraMatrix(const aslam::Camera& input_camera, double alpha,
                                          double scale, bool undistort_to_pinhole);

/// \brief Returns the new camera matrix based on the free scaling parameter.
/// @param[in] input_camera Input camera geometry
/// @param[in] output_camera_matrix Desired output camera matrix (see \ref getOptimalNewCameraMatrix)
/// @param[in] scale Output image size scaling parameter wrt. to input image size.
/// @param[in] map_type Type of the output maps. (cv::CV_32FC1, cv::CV_32FC2 or cv::CV_16SC2)
///                     Use cv::CV_16SC2 if you don't know what to choose. (fastest fixed-point)
/// @param[out] map_u Map that transforms u-coordinates from distorted to undistorted image plane.
/// @param[out] map_v Map that transforms v-coordinates from distorted to undistorted image plane.
void buildUndistortMap(const aslam::Camera& input_camera, const aslam::Camera& output_camera,
                       bool undistort_to_pinhole, int map_type, cv::OutputArray map_u,
                       cv::OutputArray map_v);

} //namespace common
} //namespace aslam

#endif // ASLAM_UNDISTORT_HELPERS_H_
