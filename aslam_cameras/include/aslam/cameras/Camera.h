#ifndef ASLAM_CAMERAS_CAMERA_H_
#define ASLAM_CAMERAS_CAMERA_H_
#include <cstdint>

#include <aslam/common/macros.h>
#include <Eigen/Dense>

namespace sm {
class PropertyTree;
}

namespace aslam {
class Camera {
 public:
  ASLAM_POINTER_TYPEDEFS(Camera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Camera);

  Camera();
  explicit Camera(const sm::PropertyTree& property_tree);
  virtual ~Camera();

  virtual bool operator==(const Camera& other) const;

  // The functions below take variable sized matrices and hence they are prefixed by "vs" to avoid confusion with
  // the fixed-size matrix versions available in derived classes. The default implementation of these "vs" functions
  // will be inefficient as it will call the fixed-size derived functions and copy the data into the variable size matrices.
  // Still, this base class is useful as it allows us to maintain a heterogeneous list of camera geometries and simplify the
  // python interface.
  /// Project a point expressed in euclidean coordinates to a 2d image measurement.
  virtual bool euclideanToKeypoint(const Eigen::Vector3d& point,
                                   Eigen::Matrix<double, 2, 1>* out_point) const = 0;
  /// Project a point expressed in euclidean coordinates to a 2d image measurement and calculate the relevant jacobian.
  virtual bool euclideanToKeypoint(const Eigen::Vector3d & point,
                                   Eigen::Matrix<double, 2, 1>* out_point,
                                   Eigen::Matrix<double, 2, 3>* out_jacobian) const = 0;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement.
  virtual bool homogeneousToKeypoint(const Eigen::Vector4d& homogeneous_point,
                                     Eigen::Matrix<double, 2, 1>* out_point) const = 0;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement and calculate the relevant jacobian.
  virtual bool homogeneousToKeypoint(const Eigen::Vector4d & homogeneous_point,
                                     Eigen::Matrix<double, 2, 1>* out_point,
                                     Eigen::Matrix<double, 2, 4>* out_jacobian) const = 0;

  // If the camera model is invertible, these functions produce 3d
  // points.  otherwise they produce points where the length has no
  // meaning. In that case the points are not necessarily on the
  // unit sphere in R^3 (that might be expensive to compute).

  /// Compute the 3d bearing vector in euclidean coordinates from the 2d image measurement.
  virtual bool keypointToEuclidean(const Eigen::VectorXd& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point) const = 0;


  /// Compute the 3d bearing vector in euclidean coordinates and the relevant jacobian from the 2d image measurement.
  virtual bool keypointToEuclidean(const Eigen::Vector2d& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point,
                                   Eigen::Matrix<double, 3, 2>* out_jacobian) const = 0;

  /// Compute the 3d bearing vector in homogenous coordinates from the 2d image measurement.
  virtual bool keypointToHomogeneous(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point) const = 0;

  /// Compute the 3d bearing vector in homogeneous coordinates and the relevant jacobian from the 2d image measurement.
  virtual bool keypointToHomogeneous(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point,
                                     Eigen::Matrix<double, 4, 2>* out_jacobian) const = 0;

  /// Return the temporal offset of a rolling shutter camera.
  uint64_t getTemporalOffsetNanoSeconds() const {
    return line_delay_nano_seconds_;
  }

  /// Set the temporal offset of a rolling shutter camera.
  void setTemporalOffsetNanoSeconds(uint64_t line_delay_nano_seconds) {
    line_delay_nano_seconds_ = line_delay_nano_seconds;
  }

 private:
  /// The delay per scan line for a rolling shutter camera.
  uint64_t line_delay_nano_seconds_;
};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_CAMERA_H_
