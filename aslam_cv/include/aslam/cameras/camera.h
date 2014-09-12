#ifndef ASLAM_CAMERAS_CAMERA_H_
#define ASLAM_CAMERAS_CAMERA_H_

#include <cstdint>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <aslam/cameras/distortion.h>

// TODO(slynen) Enable commented out PropertyTree support
//namespace sm {
//class PropertyTree;
//}

namespace aslam {

/// \brief A factory function to create a derived class camera
///
/// This function takes a vectors of intrinsics and distortion parameters
/// and produces a camera.
/// \param[in] intrinsics A vector of projection intrinsic parameters.
/// \param[in] image_width The width of the image associated with this camera.
/// \param[in] image_height The height of the image associated with this camera.
/// \param[in] distortion_parameters The parameters of the distortion object.
/// \returns A new camera based on the template types.
template <typename CameraType, typename DistortionType>
typename CameraType::Ptr createCamera(const Eigen::VectorXd& intrinsics,
                                      uint32_t image_width, uint32_t image_height,
                                      const Eigen::VectorXd& distortion_parameters)
{
  typename DistortionType::Ptr distortion(new DistortionType(distortion_parameters));
  typename CameraType::Ptr camera(new CameraType(intrinsics, image_width, image_height, distortion));
  return camera;
}

/// \brief A factory function to create a derived class camera without distortion.
///
/// This function takes a vectors of intrinsics and distortion parameters
/// and produces a camera.
/// \param[in] intrinsics A vector of projection intrinsic parameters.
/// \param[in] image_width The width of the image associated with this camera.
/// \param[in] image_height The height of the image associated with this camera.
/// \returns A new camera based on the template types.
template <typename CameraType>
typename CameraType::Ptr createCamera(const Eigen::VectorXd& intrinsics,
                                      uint32_t image_width, uint32_t image_height)
{
  typename CameraType::Ptr camera(new CameraType(intrinsics, image_width, image_height));
  return camera;
}

/// \struct ProjectionResult
/// \brief This struct is returned by the camera projection methods and holds the result state
///        of the projection operation.
struct ProjectionResult {
  /// Possible projection state.
  enum class Status {
    /// Keypoint is visible and projection was successful.
    KEYPOINT_VISIBLE,
    /// Keypoint is NOT visible but projection was successful.
    KEYPOINT_OUTSIDE_IMAGE_BOX,
    /// The projected point lies behind the camera plane.
    POINT_BEHIND_CAMERA,
    /// The projection was unsuccessful.
    PROJECTION_INVALID,
    /// Default value after construction.
    UNINITIALIZED
  };

  ProjectionResult() : status_(Status::UNINITIALIZED) {};
  ProjectionResult(Status status) : status_(status) {};

  /// \brief ProjectionResult can be typecasted to bool and is true if the projected keypoint
  ///        is visible. Simplifies the check for a successful projection.
  ///        Example usage:
  /// @code
  ///          aslam::ProjectionResult ret = camera_->project3(Eigen::Vector3d(0, 0, -10), &keypoint);
  ///          if(ret) std::cout << "Projection was successful!\n";
  /// @endcode
  explicit operator bool() const { return isKeypointVisible(); };

  /// \brief Compare objects.
  bool operator==(const ProjectionResult& other) const { return status_ == other.status_; };

  /// \brief Convenience function to print the state using streams.
  friend std::ostream& operator<< (std::ostream& out, const ProjectionResult& state)
  {
    std::string enum_str;
    switch (state.status_){
      case Status::KEYPOINT_VISIBLE:            enum_str = "KEYPOINT_VISIBLE"; break;
      case Status::KEYPOINT_OUTSIDE_IMAGE_BOX:  enum_str = "KEYPOINT_OUTSIDE_IMAGE_BOX"; break;
      case Status::POINT_BEHIND_CAMERA:         enum_str = "POINT_BEHIND_CAMERA"; break;
      case Status::PROJECTION_INVALID:          enum_str = "PROJECTION_INVALID"; break;
      default:
        case Status::UNINITIALIZED:             enum_str = "UNINITIALIZED"; break;
    }
    out << "ProjectionResult: " << enum_str << std::endl;
    return out;
  }

  /// \brief Check whether the projection was successful and the point is visible in the image.
  bool isKeypointVisible() const { return (status_ == Status::KEYPOINT_VISIBLE); };

  /// \brief Returns the exact state of the projection operation.
  ///        Example usage:
  /// @code
  ///          aslam::ProjectionResult ret = camera_->project3(Eigen::Vector3d(0, 0, -1), &keypoint);
  ///          if(ret.getDetailedStatus() == aslam::ProjectionResult::Status::KEYPOINT_OUTSIDE_IMAGE_BOX)
  ///            std::cout << "Point behind camera! Lets do something...\n";
  /// @endcode
  Status getDetailedStatus() const { return status_; };

 private:
  /// Stores the projection state.
  Status status_;
};

/// \class Camera
/// \brief The base camera class provides methods to project/backproject euclidean and
///        homogeneous points. The actual projection is implemented in the derived classes
///        for euclidean coordinates only; homogeneous coordinates are support by a conversion.
///        The intrinsic parameters are documented in the specialized camera classes.
class Camera {
 public:
  ASLAM_POINTER_TYPEDEFS(Camera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(Camera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

  // TODO(slynen) Enable commented out PropertyTree support
  //explicit Camera(const sm::PropertyTree& property_tree);
 protected:
  Camera() = delete;

  /// \brief Camera base constructor.
  /// @param[in] intrinsics Vector containing the intrinsic parameters.
  Camera(const Eigen::VectorXd& intrinsics);

 public:
  virtual ~Camera() {};

  /// \brief Compare this camera to another camera object.
  virtual bool operator==(const Camera& other) const;

  /// \brief Convenience function to print the state using streams.
  std::ostream& operator<<(std::ostream& out) {
    this->printParameters(out, std::string(""));
    return out;
  };

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Information about the camera
  /// @{

  /// \brief Get the camera id.
  const aslam::CameraId& getId() const { return id_; }

  /// \brief Set the camera id.
  void setId(const aslam::CameraId& id) { id_ = id; }

  /// \brief Get a label for the camera.
  const std::string& getLabel() const {return label_;}

  /// \brief Set a label for the camera.
  void setLabel(const std::string& label) {label_ = label;}

  /// \brief The width of the image in pixels.
  uint32_t imageWidth() const { return image_width_; }

  /// \brief The height of the image in pixels.
  uint32_t imageHeight() const { return image_height_; }

  /// \brief Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras.
  virtual void printParameters(std::ostream& out, const std::string& text) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points
  /// @{

  /// \brief Projects a euclidean point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_3d     The point in euclidean coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @return Contains information about the success of the projection. Check "struct
  ///         ProjectionResult" for more information.
  virtual const ProjectionResult project3(const Eigen::Vector3d& point_3d,
                                         Eigen::Vector2d* out_keypoint) const = 0;

  /// \brief Projects a euclidean point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_3d     The point in euclidean coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @param[out] out_jacobian The Jacobian w.r.t. to changes in the euclidean point.
  /// @return Contains information about the success of the projection. Check "struct
  ///         ProjectionResult" for more information.
  virtual const ProjectionResult project3(const Eigen::Vector3d& point_3d,
                                         Eigen::Vector2d* out_keypoint,
                                         Eigen::Matrix<double, 2, 3>* out_jacobian) const = 0;

  /// \brief Compute the 3d bearing vector in euclidean coordinates given a keypoint in
  ///        image coordinates. Uses the projection (& distortion) models.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in euclidean coordinates (with z=1 -> non-normalized).
  /// @return Was the projection successful?
  virtual bool backProject3(const Eigen::Vector2d& keypoint,
                            Eigen::Vector3d* out_point_3d) const = 0;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project homogeneous points
  /// @{

  /// \brief Projects a homogeneous point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_4d     The point in homogeneous coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @return Contains information about the success of the projection. Check "struct
  ///         ProjectionResult" for more information.
  const ProjectionResult project4(const Eigen::Vector4d& point_4d,
                                 Eigen::Vector2d* out_keypoint) const;

  /// \brief Projects a euclidean point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_4d     The point in homogeneous coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @param[out] out_jacobian The Jacobian w.r.t. to changes in the homogeneous point.
  /// @return Contains information about the success of the projection. Check "struct
  ///         ProjectionResult" for more information.
  const ProjectionResult project4(const Eigen::Vector4d& point_4d,
                                 Eigen::Vector2d* out_keypoint,
                                 Eigen::Matrix<double, 2, 4>* out_jacobian) const;

  /// \brief Compute the 3d bearing vector in homogeneous coordinates given a keypoint in
  ///        image coordinates. Uses the projection (& distortion) models.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in homogeneous coordinates.
  /// @return Was the projection successful?
  bool backProject4(const Eigen::Vector2d& keypoint,
                    Eigen::Vector4d* out_point_4d) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Functional methods to project and back-project points
  /// @{

  /// \brief This function projects a point into the image using the intrinsic parameters
  ///        that are passed in as arguments. If any of the Jacobians are nonnull, they
  ///        should be filled in with the Jacobian with respect to small changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[in]  intrinsics_external     External intrinsic parameter vector.
  /// @param[in]  distortion_coefficients_external External distortion parameter vector.
  ///                                              Parameter is ignored is no distortion is active.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @return Contains information about the success of the projection. Check "struct
  ///         ProjectionResult" for more information.
  virtual const ProjectionResult project3Functional(
      const Eigen::Vector3d& point_3d,
      const Eigen::VectorXd& intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint) const = 0;

  /// \brief This function projects a point into the image using the intrinsic parameters
  ///        that are passed in as arguments. If any of the Jacobians are nonnull, they
  ///        should be filled in with the Jacobian with respect to small changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[in]  intrinsics_external     External intrinsic parameter vector.
  /// @param[in]  distortion_coefficients_external External distortion parameter vector.
  ///                                              Parameter is ignored is no distortion is active.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @param[out] out_jacobian_point      The Jacobian w.r.t. to changes in the euclidean point.
  ///                                       nullptr: calculation is skipped.
  /// @param[out] out_jacobian_intrinsics The Jacobian w.r.t. to changes in the intrinsics.
  ///                                       nullptr: calculation is skipped.
  /// @param[out] out_jacobian_distortion The Jacobian wrt. to changes in the distortion parameters.
  ///                                       nullptr: calculation is skipped.
  /// @return Contains information about the success of the projection. Check "struct
  ///         ProjectionResult" for more information.
  virtual const ProjectionResult project3Functional(
      const Eigen::Vector3d& point_3d,
      const Eigen::VectorXd& intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const = 0;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to support rolling shutter cameras
  /// @{

  /// \brief  Return the temporal offset of a rolling shutter camera.
  ///         Global shutter cameras return zero.
  /// @return Line delay in nano seconds.
  uint64_t getLineDelayNanoSeconds() const {
    return line_delay_nano_seconds_;
  }

  /// \brief Set the temporal offset of a rolling shutter camera.
  /// @param[in] line_delay_nano_seconds Line delay in nano seconds.
  void setLineDelayNanoSeconds(uint64_t line_delay_nano_seconds) {
    line_delay_nano_seconds_ = line_delay_nano_seconds;
  }

  /// \brief The amount of time elapsed between the first row of the image and the
  ///        keypoint. For a global shutter camera, this can return Duration(0).
  /// @param[in] keypoint Keypoint to which the delay should be calculated.
  /// @return Time elapsed between the first row of the image and the
  ///         keypoint in nanoseconds.
  int64_t temporalOffsetNanoSeconds(const Eigen::Vector2d& keypoint) const {
    // Don't check validity. This allows points to wander in and out
    // of the frame during optimization
    return static_cast<int64_t>(keypoint(1)) * line_delay_nano_seconds_;
  }

  /// \brief The amount of time elapsed between the first row of the image and the
  ///        last row of the image. For a global shutter camera, this can return 0.
  int64_t maxTemporalOffsetNanoSeconds() const {
    return this->imageHeight() * line_delay_nano_seconds_;
  }

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to test validity and visibility
  /// @{

  /// \brief Can the projection function be run on this point? This doesn't test if
  ///        the projected point is visible, only if the projection function can be run
  ///        without numerical errors or singularities.
  bool isProjectable3(const Eigen::Vector3d& point) const;

  /// \brief  Can the projection function be run on this point? This doesn't test
  ///         if the projected point is visible, only if the projection function
  ///         can be run without numerical errors or singularities.
  bool isProjectable4(const Eigen::Vector4d& point) const;

  /// \brief  Check if a given keypoint is inside the imaging box of the camera.
  bool isKeypointVisible(const Eigen::Vector2d& keypoint) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to support unit testing.
  /// @{

  /// \brief Creates a random valid keypoint..
  virtual Eigen::Vector2d createRandomKeypoint() const = 0;

  /// \brief Creates a random visible point. Negative depth means random between
  ///        0 and 100 meters.
  virtual Eigen::Vector3d createRandomVisiblePoint(double depth) const = 0;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to support optimization
  /// @{

  /// \brief Returns a pointer to the underlying distortion object.
  /// @return ptr to distortion model; nullptr if none is set or not available
  ///         for the camera type
  virtual aslam::Distortion::Ptr distortion() { return nullptr; };

  /// \brief Returns a const pointer to the underlying distortion object.
  /// @return const_ptr to distortion model; nullptr if none is set or not available
  ///         for the camera type
  virtual const aslam::Distortion::Ptr distortion() const { return nullptr; };

  /// Get the intrinsic parameters.
  virtual const Eigen::VectorXd& getParameters() const { return intrinsics_; };

  /// Set the intrinsic parameters. Parameters are documented in the specialized camera classes.
  virtual void setParameters(const Eigen::VectorXd& params) = 0;

  /// @}

  /// \name Factory Methods
  /// @{

  /// \brief A factory function to create a derived class camera
  ///
  /// This function takes a vectors of intrinsics and distortion parameters
  /// and produces a camera.
  /// \param[in] intrinsics  A vector of projection intrinsic parameters.
  /// \param[in] imageWidth  The width of the image associated with this camera.
  /// \param[in] imageHeight The height of the image associated with this camera.
  /// \param[in] distortionParameters The parameters of the distortion object.
  /// \returns A new camera based on the template types.
  template<typename DerivedCamera, typename DerivedDistortion>
  static std::shared_ptr<Camera> construct(
      const Eigen::VectorXd& intrinsics,
      uint32_t imageWidth,
      uint32_t imageHeight,
      const Eigen::VectorXd& distortionParameters);

  /// @}

 protected:
  /// Set the image width. Only accessible by derived classes.
  void setImageWidth(uint32_t width){ image_width_ = width; }

  /// Set the image height. Only accessible by derived classes.
  void setImageHeight(uint32_t height){ image_height_ = height; }

 private:
  /// The delay per scanline for a rolling shutter camera in nanoseconds.
  uint64_t line_delay_nano_seconds_;
  /// A label for this camera, a name.
  std::string label_;
  /// The id of this camera.
  aslam::CameraId id_;
  /// The width of the image
  uint32_t image_width_;
  /// The height of the image
  uint32_t image_height_;

 protected:
  /// Parameter vector for the intrinsic parameters of the model.
  Eigen::VectorXd intrinsics_;
};
}  // namespace aslam
#include "camera-inl.h"
#endif  // ASLAM_CAMERAS_CAMERA_H_
