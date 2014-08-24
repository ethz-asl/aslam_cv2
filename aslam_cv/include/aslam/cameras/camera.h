#ifndef ASLAM_CAMERAS_CAMERA_H_
#define ASLAM_CAMERAS_CAMERA_H_

#include <cstdint>

#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <Eigen/Dense>
#include <glog/logging.h>

// TODO(slynen) Enable commented out PropertyTree support
//namespace sm {
//class PropertyTree;
//}

namespace aslam {
class Camera {
 public:
  ASLAM_POINTER_TYPEDEFS(Camera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Camera);

  Camera();
  // TODO(slynen) Enable commented out PropertyTree support
  //explicit Camera(const sm::PropertyTree& property_tree);
  virtual ~Camera();

  //////////////////////////////////////////////////////////////
  /// \name Information about the camera
  /// @{

  /// \brief get the camera id.
  aslam::CameraId getId() const { return id_; }

  /// \brief set the camera id.
  void setId(aslam::CameraId id) { id_ = id; }

  /// \brief get a label for the camera
  const std::string& getLabel() const {return label_;}

  /// \brief set a label for the camera
  void setLabel(const std::string& label) {label_ = label;}

  /// \brief The width of the image
  virtual uint32_t imageWidth() const = 0;

  /// \brief The height of the image
  virtual uint32_t imageHeight() const = 0;

  /// \brief Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function. 
  void print(std::ostream& out = std::cout, const std::string& text = std::string());

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project points
  /// @{

  /// Project a point expressed in euclidean coordinates to a 2d image measurement.
  virtual bool project3(const Eigen::Vector3d& point,
                                   Eigen::Vector2d* out_keypoint) const = 0;

  /// Project a point expressed in euclidean coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool project3(const Eigen::Vector3d & point,
                                   Eigen::Vector2d* out_keypoint,
                                   Eigen::Matrix<double, 2, 3>* out_jacobian) const = 0;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement.
  virtual bool project4(const Eigen::Vector4d& point,
                        Eigen::Vector2d* out_keypoint) const = 0;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool project4(const Eigen::Vector4d & point,
                        Eigen::Vector2d* out_keypoint,
                        Eigen::Matrix<double, 2, 4>* out_jacobian) const = 0;

  /// Compute the 3d bearing vector in euclidean coordinates from the 2d image measurement.
  virtual bool backProject3(const Eigen::Vector2d& keypoint,
                            Eigen::Vector3d* out_keypoint) const = 0;


  /// Compute the 3d bearing vector in euclidean coordinates and the relevant jacobian
  /// from the 2d image measurement.
  virtual bool backProject3(const Eigen::Vector2d& keypoint,
                            Eigen::Vector3d* out_point,
                            Eigen::Matrix<double, 3, 2>* out_jacobian) const = 0;

  /// Compute the 3d bearing vector in homogenous coordinates from the 2d image measurement.
  virtual bool backProject4(Eigen::Vector2d const& keypoint,
                            Eigen::Vector4d* out_point) const = 0;

  /// Compute the 3d bearing vector in homogeneous coordinates and the relevant
  /// jacobian from the 2d image measurement.
  virtual bool backProject4(Eigen::Vector2d const& keypoint,
                            Eigen::Vector4d* out_point,
                            Eigen::Matrix<double, 4, 2>* out_jacobian) const = 0;

  /// Compute the jacobian of the image measurement w.r.t. the intrinsics.
  virtual bool project3IntrinsicsJacobian(
      const Eigen::Vector3d& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;
  
  /// Compute the jacobian of the image measurement w.r.t. the intrinsics.
  virtual bool project4IntrinsicsJacobian(
      const Eigen::Vector4d& point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian) const = 0;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Functional methods to project and back-project points
  /// @{
  
  /// This function projects a point into the image using the
  /// intrinsics parameters and distortion parameters that are
  /// passed in as arguments. If any of the Jacobians are nonnull,
  /// they should be filled in with the Jacobian with respect to
  /// small changes in the argument.
  virtual bool project3Functional(const Eigen::VectorXd& intrinsics_params,
                                  const Eigen::Vector3d& point,
                                  Eigen::Vector2d * out_point,
                                  Eigen::Matrix<double, 2, Eigen::Dynamic>* out_intrinsics_jacobian,
                                  Eigen::Matrix<double, 2, Eigen::Dynamic>* out_point_jacobian) const = 0;
                         
  /// @}
  
  //////////////////////////////////////////////////////////////
  /// \name Methods to support rolling shutter cameras
  /// @{

  /// Return the temporal offset of a rolling shutter camera.
  uint64_t getLineDelayNanoSeconds() const {
    return line_delay_nano_seconds_;
  }

  /// Set the temporal offset of a rolling shutter camera.
  void setLineDelayNanoSeconds(uint64_t line_delay_nano_seconds) {
    line_delay_nano_seconds_ = line_delay_nano_seconds;
  }

  // The amount of time elapsed between the first row of the image and the
  // keypoint. For a global shutter camera, this can return Duration(0).
  inline int64_t temporalOffsetNanoSeconds(
      const Eigen::Vector2d& keypoint) const {
    // Don't check validity. This allows points to wander in and out
    // of the frame during optimization
    return static_cast<int64_t>(keypoint(1)) * line_delay_nano_seconds_;
  }

  // The amount of time elapsed between the first row of the image and the
  // last row of the image. For a global shutter camera, this can return 0.
  inline int64_t maxTemporalOffsetNanoSeconds() const {
    return this->imageHeight() * line_delay_nano_seconds_;
  }


  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to test validity and visibility
  /// @{

  /// Is the image point within the image bounds?
  virtual bool isVisible(const Eigen::Vector2d& keypoint) const = 0;

  /// Can the projection function be run on this point?
  /// This doesn't test if the projected point is visible, only
  /// if the projection function can be run without numerical
  /// errors or singularities..
  virtual bool isProjectable3(
      const Eigen::Vector3d& point) const = 0;

  /// Can the projection function be run on this point?
  /// This doesn't test if the projected point is visible, only
  /// if the projection function can be run without numerical
  /// errors or singularities.
  virtual bool isProjectable4(
      const Eigen::Vector4d& point) const = 0;

  /// @}


  //////////////////////////////////////////////////////////////
  /// \name Methods to support unit testing.
  /// @{

  /// \brief creates a random valid keypoint.
  virtual Eigen::Vector2d createRandomKeypoint() const = 0;
  
  /// \brief creates a random visible point. Negative depth means random between
  /// 0 and 100 meters.
  virtual Eigen::Vector3d createRandomVisiblePoint(double depth) const = 0;

  /// \brief is this camera equal to another
  virtual bool operator==(const Camera& other) const;
  
  /// @}

  /// \name Methods to support optimizaiton
  /// @{

  /// Get the intrinsic parameters. 
  /// This should include distortion parameters
  virtual const Eigen::VectorXd& getParameters() const = 0;

  /// Set the intrinsic parameters.
  virtual void setParameters(const Eigen::VectorXd& params) = 0;

  /// Get the intrinsic parameters (mutable).
  virtual Eigen::VectorXd& getParametersMutable() = 0;

  /// Get the intrinsic parameters (memory pointer).
  virtual double* getParameterMutablePtr() = 0;

  /// How many intrinsic parameters are there?
  virtual size_t getParameterSize() const = 0;

  /// @}

 private:
  /// The delay per scanline for a rolling shutter camera.
  uint64_t line_delay_nano_seconds_;
  /// A label for this camera, a name.
  std::string label_;
  /// The id of this camera.
  aslam::CameraId id_;
};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_CAMERA_H_
