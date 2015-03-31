#ifndef ASLAM_PIPELINE_MAPPED_UNDISTORTER_INL_H_
#define ASLAM_PIPELINE_MAPPED_UNDISTORTER_INL_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/common/undistort-helpers.h>

namespace aslam {

template <typename CameraType>
std::unique_ptr<MappedUndistorter> createMappedUndistorter(
    const std::shared_ptr<CameraType>& camera_ptr, float alpha, float scale,
    aslam::InterpolationMethod interpolation_type) {
  CHECK(camera_ptr != nullptr);

  CHECK_GE(alpha, 0.0); CHECK_LE(alpha, 1.0);
  CHECK_GT(scale, 0.0);

  // Create a copy of the input camera.
  std::shared_ptr<CameraType> input_camera(
      dynamic_cast<CameraType*>(camera_ptr->clone()));
  CHECK(input_camera);

  // Create the scaled output camera with removed distortion.
  const bool kUndistortToPinhole = false;
  Eigen::Matrix3d output_camera_matrix = aslam::common::getOptimalNewCameraMatrix(
      *input_camera, alpha, scale, kUndistortToPinhole);

  Eigen::Matrix<double, camera_ptr->parameterCount(), 1> intrinsics;

  switch(camera_ptr->getType()) {
    case Camera::Type::kPinhole:
      intrinsics << output_camera_matrix(0, 0), output_camera_matrix(1, 1),
                    output_camera_matrix(0, 2), output_camera_matrix(1, 2);
      break;
    case Camera::Type::kUnifiedProjection:
      intrinsics << camera_ptr->xi(),
                    output_camera_matrix(0, 0), output_camera_matrix(1, 1),
                    output_camera_matrix(0, 2), output_camera_matrix(1, 2);
      break;
    default:
      LOG(FATAL) << "Unknown camera model: "
        << static_cast<std::underlying_type<Camera::Type>::type>(
            camera_ptr->getType());
  }

  const int output_width = static_cast<int>(scale * camera_ptr->imageWidth());
  const int output_height = static_cast<int>(scale * camera_ptr->imageHeight());

  Camera::Ptr output_camera = aslam::createCamera<CameraType>(
      intrinsics, output_width, output_height);
  CHECK(output_camera);

  cv::Mat map_u, map_v;
  aslam::common::buildUndistortMap(*input_camera, *output_camera, CV_16SC2, map_u, map_v);

  return std::unique_ptr<MappedUndistorter>(
      new MappedUndistorter(input_camera, output_camera, map_u, map_v, interpolation_type));
}

}  // namespace aslam
#endif  // ASLAM_PIPELINE_MAPPED_UNDISTORTER_INL_H_
