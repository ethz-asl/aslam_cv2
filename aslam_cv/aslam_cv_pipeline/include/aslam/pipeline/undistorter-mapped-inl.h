#ifndef ASLAM_PIPELINE_MAPPED_UNDISTORTER_INL_H_
#define ASLAM_PIPELINE_MAPPED_UNDISTORTER_INL_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-factory.h>
#include <aslam/common-private/undistort-helpers.h>

namespace aslam {

template <>
inline std::unique_ptr<MappedUndistorter> createMappedUndistorter(
    const aslam::Camera::Ptr& camera_ptr, float alpha, float scale,
    aslam::InterpolationMethod interpolation_type) {
  CHECK(camera_ptr != nullptr);

  switch(camera_ptr->getType()) {
    case Camera::Type::kUnifiedProjection: {
      aslam::UnifiedProjectionCamera::Ptr unified_projection_cam =
          std::dynamic_pointer_cast<aslam::UnifiedProjectionCamera>(camera_ptr);
      CHECK(unified_projection_cam != nullptr);
      return createMappedUndistorter(unified_projection_cam, alpha, scale,
                                     interpolation_type);
    }
    case Camera::Type::kPinhole: {
      aslam::PinholeCamera::Ptr pinhole_cam =
          std::dynamic_pointer_cast<aslam::PinholeCamera>(camera_ptr);
      CHECK(pinhole_cam != nullptr);
      return createMappedUndistorter(pinhole_cam, alpha, scale,
                                     interpolation_type);
    }
    default: {
      LOG(FATAL) << "Unknown camera model: "
        << static_cast<std::underlying_type<Camera::Type>::type>(
            camera_ptr->getType());
      return nullptr;
    }
  }
}

template <typename CameraType>
inline std::unique_ptr<MappedUndistorter> createMappedUndistorter(
    const std::shared_ptr<CameraType>& camera_ptr, float alpha, float scale,
    aslam::InterpolationMethod interpolation_type) {
  CHECK(camera_ptr != nullptr);

  CHECK_GE(alpha, 0.0); CHECK_LE(alpha, 1.0);
  CHECK_GT(scale, 0.0);

  // Create a copy of the input camera.
  aslam::Camera::Ptr input_camera(camera_ptr->clone());
  CHECK(input_camera);

  // Create the scaled output camera with removed distortion.
  const bool kUndistortToPinhole = false;
  Eigen::Matrix3d output_camera_matrix = common::getOptimalNewCameraMatrix(
      *input_camera, alpha, scale, kUndistortToPinhole);

  const int output_width = static_cast<int>(scale * camera_ptr->imageWidth());
  const int output_height = static_cast<int>(scale * camera_ptr->imageHeight());

  Camera::Ptr output_camera;
  Eigen::MatrixXd intrinsics(camera_ptr->getParameterSize(), 1);
  switch(camera_ptr->getType()) {
    case Camera::Type::kPinhole:
      intrinsics << output_camera_matrix(0, 0), output_camera_matrix(1, 1),
                    output_camera_matrix(0, 2), output_camera_matrix(1, 2);
      break;
      output_camera.reset(new PinholeCamera(intrinsics, output_width, output_height));
      CHECK(output_camera);
    case Camera::Type::kUnifiedProjection: {
      aslam::UnifiedProjectionCamera::ConstPtr unified_proj_cam_ptr =
          std::dynamic_pointer_cast<const aslam::UnifiedProjectionCamera>(camera_ptr);
      CHECK(unified_proj_cam_ptr != nullptr)
        << "Cast to unified projection camera failed.";
      intrinsics << unified_proj_cam_ptr->xi(),
                    output_camera_matrix(0, 0), output_camera_matrix(1, 1),
                    output_camera_matrix(0, 2), output_camera_matrix(1, 2);
      output_camera.reset(new UnifiedProjectionCamera(intrinsics, output_width, output_height));
      CHECK(output_camera);
    }
      break;
    default:
      LOG(FATAL) << "Unknown camera model: "
        << static_cast<std::underlying_type<Camera::Type>::type>(
            camera_ptr->getType());
  }

  cv::Mat map_u, map_v;
  common::buildUndistortMap(*input_camera, *output_camera, CV_16SC2, map_u, map_v);

  return std::unique_ptr<MappedUndistorter>(
      new MappedUndistorter(input_camera, output_camera, map_u, map_v, interpolation_type));
}

}  // namespace aslam
#endif  // ASLAM_PIPELINE_MAPPED_UNDISTORTER_INL_H_
