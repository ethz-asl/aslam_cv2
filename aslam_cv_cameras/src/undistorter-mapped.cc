#include "aslam/cameras/undistorter-mapped.h"

#include <aslam/cameras/camera-factory.h>
#include <aslam/cameras/convert-maps-legacy.h>
#include <aslam/common/undistort-helpers.h>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp>  // cv::remap

namespace aslam {

std::shared_ptr<MappedUndistorter> createMappedUndistorterToPinhole(
    const aslam::UnifiedProjectionCamera& unified_proj_camera, float alpha,
    float scale, aslam::InterpolationMethod interpolation_type) {
  CHECK_GE(alpha, 0.0);
  CHECK_LE(alpha, 1.0);
  CHECK_GT(scale, 0.0);

  // Create a copy of the input camera.
  UnifiedProjectionCamera::Ptr input_camera(
      dynamic_cast<UnifiedProjectionCamera*>(unified_proj_camera.clone()));
  CHECK(input_camera);

  // Create the scaled output camera with removed distortion.
  const bool kUndistortToPinhole = true;
  Eigen::Matrix3d output_camera_matrix = common::getOptimalNewCameraMatrix(
      *input_camera, alpha, scale, kUndistortToPinhole);

  Eigen::Matrix<double, PinholeCamera::parameterCount(), 1> intrinsics;
  intrinsics << output_camera_matrix(0, 0), output_camera_matrix(1, 1),
      output_camera_matrix(0, 2), output_camera_matrix(1, 2);

  const int output_width = static_cast<int>(scale * input_camera->imageWidth());
  const int output_height =
      static_cast<int>(scale * input_camera->imageHeight());
  PinholeCamera::Ptr output_camera = aslam::createCamera<aslam::PinholeCamera>(
      intrinsics, output_width, output_height);
  CHECK(output_camera);

  cv::Mat map_u, map_v;
  VLOG(1) << "Building undistorter map";
  common::buildUndistortMap(
      *input_camera, *output_camera, CV_16SC2, map_u, map_v);

  // Convert map to non-fixed point representation for easy lookup of values.
  cv::Mat map_u_float = map_u.clone();
  cv::Mat map_v_float = map_v.clone();
  aslam::convertMapsLegacy(map_u, map_v, map_u_float, map_v_float, CV_32FC1);

  return std::shared_ptr<MappedUndistorter>(
      new MappedUndistorter(
          input_camera, output_camera, map_u, map_v, map_u_float, map_v_float,
          interpolation_type));
}

MappedUndistorter::MappedUndistorter()
    : interpolation_method_(aslam::InterpolationMethod::Linear) {}

MappedUndistorter::MappedUndistorter(
    Camera::Ptr input_camera, Camera::Ptr output_camera, const cv::Mat& map_u,
    const cv::Mat& map_v, const cv::Mat& map_u_float,
    const cv::Mat& map_v_float, aslam::InterpolationMethod interpolation)
    : Undistorter(input_camera, output_camera),
      map_u_(map_u),
      map_v_(map_v),
      map_u_float_(map_u_float),
      map_v_float_(map_v_float),
      interpolation_method_(interpolation) {
  CHECK_EQ(static_cast<size_t>(map_u_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_u_.cols), output_camera->imageWidth());
  CHECK_EQ(static_cast<size_t>(map_v_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_v_.cols), output_camera->imageWidth());
  CHECK_EQ(
      static_cast<size_t>(map_u_float_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_u_float_.cols), output_camera->imageWidth());
  CHECK_EQ(
      static_cast<size_t>(map_v_float_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_v_float_.cols), output_camera->imageWidth());
}

void MappedUndistorter::processImage(
    const cv::Mat& input_image, cv::Mat* output_image) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(input_image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(input_image.rows));
  CHECK_NOTNULL(output_image);
  cv::remap(
      input_image, *output_image, map_u_, map_v_,
      static_cast<int>(interpolation_method_));
}

void MappedUndistorter::processPoint(
    const Eigen::Vector2d& input_point, Eigen::Vector2d* output_point) const {
  CHECK_NOTNULL(output_point);
  *output_point = input_point;
  processPoint(output_point);
}

void MappedUndistorter::processPoint(Eigen::Vector2d* point) const {
  VLOG(250) << "Distorted point: " << (*point)[0] << "/" << (*point)[1];
  CHECK_LE((*point)[0], map_u_.cols);
  CHECK_LE((*point)[1], map_v_.rows);
  *point = Eigen::Vector2d(
      map_u_float_.at<float>((*point)[1], (*point)[0]),
      map_v_float_.at<float>((*point)[1], (*point)[0]));

  VLOG(250) << "Undistorted point: " << (*point)[0] << "/" << (*point)[1];
}

}  // namespace aslam
