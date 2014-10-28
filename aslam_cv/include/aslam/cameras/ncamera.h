#ifndef ASLAM_NCAMERA_H_
#define ASLAM_NCAMERA_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <aslam/common/macros.h>
#include <aslam/common/memory.h>
#include <aslam/common/pose-types.h>
#include <aslam/common/unique-id.h>

namespace sm {
class PropertyTree;
}
namespace aslam {
class Camera;
}

namespace aslam {

/// \class NCameras
/// \brief A class representing a calibrated multi-camera system
///
/// Coordinate frames involved:
/// - B  : The body frame of the camera rig
/// - Ci : A coordinate frame attached to camera i. 
///
class NCamera {
 public:
  ASLAM_POINTER_TYPEDEFS(NCamera);
  enum {CLASS_SERIALIZATION_VERSION = 1};
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Aligned<std::vector, Transformation>::type TransformationVector;

protected:
  /// \brief default constructor builds an empty camera rig
  NCamera() = default;

public:
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
      const std::vector<std::shared_ptr<Camera>>& cameras,
      const std::string& label);

  /// \brief initialize from a property tree
  NCamera(const sm::PropertyTree& propertyTree);
  ~NCamera() = default;

  /// Copy constructor for clone.
  NCamera(const NCamera&) = default;
  void operator=(const NCamera&) = delete;

  /// Method to clone this instance (Make sure the Camera and NCamera ID's are
  /// set to your requirement after cloning!)
  NCamera* clone() const {
    return new NCamera(static_cast<NCamera const&>(*this));
  };

  /// \brief get the number of cameras
  size_t getNumCameras() const;

  /// \brief get the pose of body frame with respect to the camera i
  const Transformation& get_T_C_B(size_t camera_index) const;

  /// \brief get the pose of body frame with respect to the camera i
  Transformation& get_T_C_B_Mutable(size_t camera_index);

  /// \brief get the pose of body frame with respect to the camera with a camera id
  /// The method will assert that the camera is not in the rig!
  const Transformation& get_T_C_B(const CameraId& camera_id) const;

  /// \brief get the pose of body frame with respect to the camera with a camera id
  /// The method will assert that the camera is not in the rig!
  Transformation& get_T_C_B_Mutable(const CameraId& camera_id);

  /// \brief set the pose of body frame with respect to the camera i
  void set_T_C_B(size_t camera_index, const Transformation& T_Ci_B);

  /// \brief get all transformations
  const TransformationVector& getTransformationVector() const;

  /// \brief get the geometry object for camera i
  const Camera& getCamera(size_t camera_index) const;

  /// \brief get the geometry object for camera i
  Camera& getCameraMutable(size_t camera_index);

  /// \brief get the geometry object for camera i
  std::shared_ptr<Camera> getCameraShared(size_t camera_index);

  /// \brief get the geometry object for camera i
  std::shared_ptr<const Camera> getCameraShared(size_t camera_index) const;

  /// \brief get the geometry object for camera i
  void setCamera(size_t camera_index, std::shared_ptr<Camera> camera);

  /// \brief how many cameras does this system have?
  size_t numCameras() const;

  /// \brief get all cameras
  const std::vector<std::shared_ptr<Camera>>& getCameraVector() const;

  /// \brief gets the id for the camera at index i
  const CameraId& getCameraId(size_t camera_index) const;

  /// \brief does this rig have a camera with this id
  bool hasCameraWithId(const CameraId& id) const;

  /// \brief get the index of the camera with the id
  /// @returns -1 if the rig doesn't have a camera with this id
  int getCameraIndex(const CameraId& id) const;

  /// \brief get the camera id.
  inline const aslam::NCameraId& getId() const {return id_;}

  /// \brief set the camera id.
  inline void setId(const aslam::NCameraId& id) {id_ = id;}

  /// \brief equality
  bool operator==(const NCamera& other) const;

  /// \brief get a label for the camera
  inline const std::string& getLabel() const {return label_;}

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
  std::vector<std::shared_ptr<Camera>> cameras_;

  /// \brief a map from camera id to index
  std::unordered_map<CameraId, size_t> id_to_index_;

  /// A label for this camera rig, a name.
  std::string label_;
};

} // namespace aslam

#endif /* ASLAM_NCAMERA_H_ */
