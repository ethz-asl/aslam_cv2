#ifndef ASLAM_CAMERAS_CAMERA_FACTORY_H_
#define ASLAM_CAMERAS_CAMERA_FACTORY_H_

#include <memory>
#include <type_traits>

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
/// This function takes vectors of intrinsics and distortion parameters
/// and creates a camera.
/// \param[in] id Id of the camera.
/// \param[in] intrinsics A vector of projection intrinsic parameters.
/// \param[in] image_width Image width in pixels.
/// \param[in] image_height Image height in pixels.
/// \param[in] distortion_parameters The parameters of the distortion object.
/// \param[in] camera_type The camera model.
/// \param[in] distortion_type The distortion model.
/// \returns A new camera based on the provided arguments.
Camera::Ptr createCamera(aslam::CameraId id, const Eigen::VectorXd& intrinsics,
                         uint32_t image_width, uint32_t image_height,
                         const Eigen::VectorXd& distortion_parameters,
                         Camera::Type camera_type,
                         Distortion::Type distortion_type) {
  CHECK(id.isValid());

  Distortion::UniquePtr distortion;
  switch(distortion_type) {
    case Distortion::Type::kNoDistortion:
      distortion = nullptr;
      break;
    case Distortion::Type::kEquidistant:
      distortion.reset(new EquidistantDistortion(distortion_parameters));
      break;
    case Distortion::Type::kFisheye:
      distortion.reset(new FisheyeDistortion(distortion_parameters));
      break;
    case Distortion::Type::kRadTan:
      distortion.reset(new RadTanDistortion(distortion_parameters));
      break;
    default:
      LOG(FATAL) << "Unknown distortion model: "
        << static_cast<std::underlying_type<Distortion::Type>::type>(
            distortion_type);
  }
  if (distortion != nullptr) {
    CHECK(distortion->distortionParametersValid(distortion_parameters))
        << "Invalid distortion parameters: "
        << distortion_parameters.transpose();
  }

  Camera::Ptr camera;
  switch(camera_type) {
    case Camera::Type::kPinhole:
      camera.reset(new PinholeCamera(intrinsics, image_width, image_height,
                                     distortion));
      break;
    case Camera::Type::kUnifiedProjection:
      camera.reset(new UnifiedProjectionCamera(intrinsics, image_width,
                                               image_height, distortion));
      break;
    default:
      LOG(FATAL) << "Unknown distortion model: "
        << static_cast<std::underlying_type<Camera::Type>::type>(camera_type);
  }

  camera->setId(id);
  return camera;
}

}  // namespace aslam

#endif  // ASLAM_CAMERAS_CAMERA_FACTORY_H_
