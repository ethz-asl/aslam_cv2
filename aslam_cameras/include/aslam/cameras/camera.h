#ifndef ASLAM_CAMERAS_CAMERA_H_
#define ASLAM_CAMERAS_CAMERA_H_
#include <cstdint>

#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <Eigen/Dense>
#include <glog/logging.h>

// TODO(slynen)
//namespace sm {
//class PropertyTree;
//}

namespace aslam {
class Camera {
 public:
  ASLAM_POINTER_TYPEDEFS(Camera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Camera);

  Camera();
  // TODO(slynen)
  //explicit Camera(const sm::PropertyTree& property_tree);
  virtual ~Camera();

  virtual bool operator==(const Camera& other) const;

  virtual uint32_t imageWidth() const = 0;
  virtual uint32_t imageHeight() const = 0;

  /// Project a point expressed in euclidean coordinates to a 2d image measurement.
  virtual bool euclideanToKeypoint(const Eigen::Vector3d& point,
                                   Eigen::Matrix<double, 2, 1>* out_point) const = 0;
  /// Project a point expressed in euclidean coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool euclideanToKeypoint(const Eigen::Vector3d & point,
                                   Eigen::Matrix<double, 2, 1>* out_point,
                                   Eigen::Matrix<double, 2, 3>* out_jacobian) const = 0;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement.
  virtual bool homogeneousToKeypoint(const Eigen::Vector4d& homogeneous_point,
                                     Eigen::Matrix<double, 2, 1>* out_point) const = 0;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool homogeneousToKeypoint(const Eigen::Vector4d & homogeneous_point,
                                     Eigen::Matrix<double, 2, 1>* out_point,
                                     Eigen::Matrix<double, 2, 4>* out_jacobian) const = 0;

  /// Compute the 3d bearing vector in euclidean coordinates from the 2d image measurement.
  virtual bool keypointToEuclidean(const Eigen::Matrix<double, 2, 1>& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point) const = 0;


  /// Compute the 3d bearing vector in euclidean coordinates and the relevant jacobian
  /// from the 2d image measurement.
  virtual bool keypointToEuclidean(const Eigen::Vector2d& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point,
                                   Eigen::Matrix<double, 3, 2>* out_jacobian) const = 0;

  /// Compute the 3d bearing vector in homogenous coordinates from the 2d image measurement.
  virtual bool keypointToHomogeneous(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point) const = 0;

  /// Compute the 3d bearing vector in homogeneous coordinates and the relevant
  /// jacobian from the 2d image measurement.
  virtual bool keypointToHomogeneous(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point,
                                     Eigen::Matrix<double, 4, 2>* out_jacobian) const = 0;

  /// Compute the jacobian of the image measurement w.r.t. the intrinsics.
  virtual bool euclideanToKeypointIntrinsicsJacobian(
      const Eigen::Matrix<double, 3, 1>& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  /// Compute the jacobian of the image measurement w.r.t. the distortion.
  virtual bool euclideanToKeypointDistortionJacobian(
      const Eigen::Matrix<double, 3, 1>& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  /// Compute the jacobian of the image measurement w.r.t. the intrinsics.
  virtual bool homogeneousToKeypointIntrinsicsJacobian(
      const Eigen::Matrix<double, 4, 1>& homogeneous_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  /// Compute the jacobian of the image measurement w.r.t. the distortion.
  virtual bool homogeneousToKeypointDistortionJacobian(
      const Eigen::Matrix<double, 4, 1>& homogeneous_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  // \brief creates a random valid keypoint.
  virtual Eigen::Matrix<double, 2, 1> createRandomKeypoint() const = 0;

  // \brief creates a random visible point. Negative depth means random between
  /// 0 and 100 meters.
  virtual Eigen::Matrix<double, 3, 1> createRandomVisiblePoint(double depth) const = 0;

  virtual Eigen::VectorXd& getParametersMutable() = 0;

  virtual double* getParameterMutablePtr() = 0;

  //////////////////////////////////////////////////////////////
  // SHUTTER SUPPORT
  //////////////////////////////////////////////////////////////

  /// Return the temporal offset of a rolling shutter camera.
  uint64_t getTemporalOffsetNanoSeconds() const {
    return line_delay_nano_seconds_;
  }

  /// Set the temporal offset of a rolling shutter camera.
  void setTemporalOffsetNanoSeconds(uint64_t line_delay_nano_seconds) {
    line_delay_nano_seconds_ = line_delay_nano_seconds;
  }

  // The amount of time elapsed between the start of the image and the
  // keypoint. For a global shutter camera, this can return Duration(0).
  inline int64_t temporalOffsetNanoSeconds(
      const Eigen::Matrix<double, 2, 1>& keypoint) const {
    CHECK_GE(0, keypoint(1));
    CHECK_LT(imageWidth(), keypoint(1));
    return static_cast<int64_t>(keypoint(1)) * line_delay_nano_seconds_;
  }

  //////////////////////////////////////////////////////////////
  // VALIDITY TESTING
  //////////////////////////////////////////////////////////////

  virtual bool isValid(const Eigen::Matrix<double, 2, 1>& keypoint) const = 0;

  virtual bool isEuclideanVisible(
      const Eigen::Matrix<double, 3, 1>& point) const = 0;

  virtual bool isHomogeneousVisible(
      const Eigen::Matrix<double, 4, 1>& homogeneous_point) const = 0;

  virtual bool isProjectionInvertible() const = 0;

  /// \brief get the camera id.
  aslam::CameraId getId() const { return id_; }

  /// \brief set the camera id.
  void setId(aslam::CameraId id) { id_ = id; }

  /// \brief get a label for the camera
  const std::string& getLabel() const {return label_;}

  /// \brief set a label for the camera
  void setLabel(const std::string& label) {label_ = label;}

 private:
  /// The delay per scan line for a rolling shutter camera.
  uint64_t line_delay_nano_seconds_;
  /// A label for this camera, a name.
  std::string label_;
  /// The id of this camera.
  aslam::CameraId id_;
};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_CAMERA_H_
