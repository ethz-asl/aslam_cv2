#ifndef ASLAM_NCAMERA_FACTORY_H_
#define ASLAM_NCAMERA_FACTORY_H_

#include <aslam/cameras/ncamera.h>

namespace aslam {

  /// \brief Creates an artificial 4-camera rig in a plane with a camera pointing in
  ///        each direction.
  ///        This is similar to the V-Charge or JanETH camera system.
  aslam::NCamera::Ptr createPlanar4CameraRig();
  aslam::NCamera::Ptr createSingleCameraRig();
}  // namespace aslam

#endif // ASLAM_NCAMERA_FACTORY_H_
