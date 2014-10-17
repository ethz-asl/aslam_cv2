#ifndef ASLAM_NCAMERA_H_
#define ASLAM_NCAMERA_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include <aslam/cameras/camera.h>
#include <aslam/common/macros.h>
#include <aslam/common/pose-types.h>
#include <sm/aligned_allocation.h>

namespace sm {
class PropertyTree;
}

namespace aslam {

/// \class NCameras
/// \brief A class representing a calibrated multi-camera system
///
/// Coordinate frames involved:
/// - B  : The body frame of the camera rig
/// - Ci : A coordinate frame attached to camera i. 
///
class NCamera
{
 public:
  ASLAM_POINTER_TYPEDEFS(NCamera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(NCamera);
  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Aligned<std::vector, Transformation>::type TransformationVector;

  /// \brief default constructor builds an empty camera rig
  NCamera();

  /// \brief initialize from a list of transformations and a list of cameras
  ///
  /// The two lists must be parallel arrays (same size). The transformation
  /// at T_C_B[i] corresponds to the camera at cameras[i].
  ///
  /// @param id unique id for this camera rig
  /// @param T_C_B a list of transformations that take poinst from B to Ci
  /// @param cameras a list cameras
  /// @param label a human-readable name for this camera rig
  NCamera(const NCameraId& id,
          const TransformationVector& T_C_B,
          const std::vector<Camera::Ptr>& cameras,
          const std::string& label);

  /// \brief initialize from a property tree
  NCamera(const sm::PropertyTree& propertyTree);
  
  virtual ~NCamera() {};

  /// \brief get the number of cameras
  size_t getNumCameras() const;

  /// \brief get the pose of body frame with respect to the camera i
  const Transformation& get_T_C_B(size_t cameraIndex) const;

  /// \brief get the pose of body frame with respect to the camera i
  Transformation& get_T_C_B_Mutable(size_t cameraIndex);

  /// \brief get the pose of body frame with respect to the camera with a camera id
  /// The method will assert if the camera is not in the rig!
  const Transformation& get_T_C_B(const CameraId& camera_id) const;

  /// \brief get the pose of body frame with respect to the camera with a camera id
  /// The method will assert if the camera is not in the rig!
  Transformation& get_T_C_B_Mutable(const CameraId& camera_id);

  /// \brief set the pose of body frame with respect to the camera i
  void set_T_C_B(size_t cameraIndex, const Transformation& T_Ci_B);

  /// \brief get all transformations
  const TransformationVector& getTransformationVector() const;

  /// \brief get the geometry object for camera i
  const Camera& getCamera(size_t cameraIndex) const;

  /// \brief get the geometry object for camera i
  Camera& getCameraMutable(size_t cameraIndex);

  /// \brief get the geometry object for camera i
  Camera::Ptr getCameraShared(size_t cameraIndex);

  /// \brief get the geometry object for camera i
  Camera::ConstPtr getCameraShared(size_t cameraIndex) const;

  /// \brief get the geometry object for camera i
  void setCamera(size_t cameraIndex, Camera::Ptr camera);
  
  /// \brief how many cameras does this system have?
  size_t numCameras() const;

  /// \brief get all cameras
  const std::vector<Camera::Ptr>& getCameraVector() const;

  /// \brief gets the id for the camera at index i
  const CameraId& getCameraId(size_t cameraIndex) const;
  
  /// \brief does this rig have a camera with this id
  bool hasCameraWithId(const CameraId& id) const;
  
  /// \brief get the index of the camera with the id
  /// @returns -1 if the rig doesn't have a camera with this id
  int getCameraIndex(const CameraId& id) const;
  
  /// \brief get the camera id.
  inline const aslam::NCameraId& getId() const { return id_; }
  
  /// \brief set the camera id.
  inline void setId(const aslam::NCameraId& id) { id_ = id; }

  /// \brief equality
  bool operator==(const NCamera& other) const;
  
  /// \brief get a label for the camera
  inline const std::string& getLabel() const { return label_; }

  /// \brief set a label for the camera
  inline void setLabel(const std::string& label) {label_ = label;}

 private:
  /// \brief internal consistency checks and initialization
  void initInternal();

  /// \brief A unique id for this camera system
  NCameraId id_;

  /// \brief The mounting transformations
  TransformationVector T_C_B_;

  /// \brief The camera geometries
  std::vector<Camera::Ptr> cameras_;

  /// \brief a map from camera id to index
  std::unordered_map<CameraId, size_t> idToIndex_;

  /// A label for this camera rig, a name.
  std::string label_;
};

} // namespace aslam

#endif /* ASLAM_NCAMERA_H_ */
