#ifndef ASLAM_CAMERA_RIG_H_
#define ASLAM_CAMERA_RIG_H_
#include <cstdint>

#include <aslam/common/macros.h>
#include <aslam/common/types.h>
#include <Eigen/Dense>
#include <vector>
#include <aslam/cameras/Camera.h>

namespace sm {
class PropertyTree;
}

namespace aslam {

///
/// \class CameraRig
/// \brief A class representing a multi-camera system
///
/// Here is where some documentation should go
///
/// Coordinate frames involved:
/// - B  : The body frame of the camera rig
/// - Ci : A coordinate frame attached to camera i. 
///
class CameraRig
{
 public:
  ASLAM_POINTER_TYPEDEFS(CameraRig);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(CameraRig);

  typedef std::vector<TransformationPtr> TransformationVector;
  typedef std::vector<Camera::Ptr> CameraVector;

  CameraRig();
  CameraRig(const sm::PropertyTree& propertyTree);
  virtual ~CameraRig();

  /// \brief get the pose of camera i with respect to the vehicle frame
  const Transformation& getT_v_c(size_t cameraIndex) const;

  /// \brief get the pose of vehicle frame with respect to the camera i
  const Transformation& get_T_C_B(size_t cameraIndex) const;

  /// \brief get the pose of vehicle frame with respect to the camera i
  TransformationPtr get_T_C_B_mutable(size_t cameraIndex);

  /// \brief set the pose of vehicle frame with respect to the camera i
  void set_T_C_B(size_t cameraIndex, const Transformation& T_Ci_B);

  /// \brief set the pose of vehicle frame with respect to the camera i
  void set_T_C_B(size_t cameraIndex, TransformationPtr T_Ci_B);

  /// \brief get all transformations
  const TransformationVector& getTransformationVector();

  /// \brief get the geometry object for camera i
  const Camera& getCamera(size_t cameraIndex) const;

  /// \brief get the geometry object for camera i
  Camera::Ptr getCameraMutable(size_t cameraIndex);

  /// \brief get the geometry object for camera i
  void setCamera(size_t cameraIndex, Camera::Ptr camera);
  
  /// \brief get the geometry object for camera i
  void setCamera(size_t cameraIndex, const Camera& camera);

  /// \brief how many cameras does this system have?
  size_t numCameras() const;

  /// \brief get all cameras
  const CameraVector& getCameraVector();

 private:

  /// \brief The mounting transformations
  TransformationVector T_C_B_;

  /// \brief The camera geometries
  CameraVector cameras_;

  /// \brief A unique id for this camera system
  //CameraSystemId id_;

};

} // namespace aslam


#endif /* ASLAM_CAMERA_RIG_H_ */
