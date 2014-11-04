#ifndef ASLAM_CAMERAS_CAMERA_FACTORY_H_
#define ASLAM_CAMERAS_CAMERA_FACTORY_H_

#include <memory>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/camera-pinhole.h>
#include <aslam/cameras/camera-unified-projection.h>
#include <aslam/cameras/distortion.h>
#include <aslam/cameras/distortion-equidistant.h>
#include <aslam/cameras/distortion-fisheye.h>
#include <aslam/cameras/distortion-radtan.h>
#include <aslam/common/unique-id.h>

namespace aslam {

// TODO(dymczykm) Move here other factory functions from camera.h?

/// \brief A factory function to create a derived class camera
///
/// This function takes a vectors of intrinsics and distortion parameters
/// and produces a camera.
/// \param[in] id Id of the camera.
/// \param[in] intrinsics A vector of projection intrinsic parameters.
/// \param[in] image_width The width of the image associated with this camera.
/// \param[in] image_height The height of the image associated with this camera.
/// \param[in] distortion_parameters The parameters of the distortion object.
/// \param[in] camera_type The camera model.
/// \param[in] distortion_type The distortion model.
/// \returns A new camera based on the provided arguments.
Camera::Ptr createCamera(aslam::CameraId id, const Eigen::VectorXd& intrinsics,
                         uint32_t image_width, uint32_t image_height,
                         const Eigen::VectorXd& distortion_parameters,
                         Camera::CameraType camera_type,
                         Distortion::DistortionType distortion_type) {
  CHECK(id.isValid());

  Distortion::UniquePtr distortion;
  switch(distortion_type) {
    case Distortion::DistortionType::kNoDistortion:
      distortion = nullptr;
      break;
    case Distortion::DistortionType::kEquidistant:
      distortion.reset(new EquidistantDistortion(distortion_parameters));
      break;
    case Distortion::DistortionType::kFisheye:
      distortion.reset(new FisheyeDistortion(distortion_parameters));
      break;
    case Distortion::DistortionType::kRadTan:
      distortion.reset(new RadTanDistortion(distortion_parameters));
      break;
    default:
      LOG(FATAL) << "Unknown distortion model.";
  }

  Camera::Ptr camera;
  switch(camera_type) {
    case Camera::CameraType::kPinhole:
      camera.reset(new PinholeCamera(intrinsics, image_width, image_height,
                                     distortion));
      break;
    case Camera::CameraType::kUnifiedProjection:
      camera.reset(new UnifiedProjectionCamera(intrinsics, image_width,
                                               image_height, distortion));
      break;
    default:
      LOG(FATAL) << "Unknown distortion model.";
  }

  camera->setId(id);
  return camera;
}

}  // namespace aslam

#endif  // ASLAM_CAMERAS_CAMERA_FACTORY_H_
