#ifndef ASLAM_CAMERAS_CAMERA_H_
#define ASLAM_CAMERAS_CAMERA_H_

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <glog/logging.h>

#include <aslam/common/macros.h>
#include <aslam/common/types.h>
#include <aslam/common/unique-id.h>
#include <aslam/cameras/distortion.h>

// TODO(slynen) Enable commented out PropertyTree support
//namespace sm {
//class PropertyTree;
//}

namespace aslam {

// Forward declarations
class MappedUndistorter;

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
                                      const Eigen::VectorXd& distortion_parameters) {
  typename aslam::Distortion::UniquePtr distortion(new DistortionType(distortion_parameters));
  typename CameraType::Ptr camera(
      new CameraType(intrinsics, image_width, image_height, distortion));
  aslam::CameraId id;
  id.randomize();
  camera->setId(id);
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
                                      uint32_t image_width, uint32_t image_height) {
  typename CameraType::Ptr camera(new CameraType(intrinsics, image_width, image_height));
  aslam::CameraId id;
  id.randomize();
  camera->setId(id);
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
  // Make the enum values accessible from the outside without the additional indirection.
  static Status KEYPOINT_VISIBLE;
  static Status KEYPOINT_OUTSIDE_IMAGE_BOX;
  static Status POINT_BEHIND_CAMERA;
  static Status PROJECTION_INVALID;
  static Status UNINITIALIZED;

  constexpr ProjectionResult() : status_(Status::UNINITIALIZED) {};
  constexpr ProjectionResult(Status status) : status_(status) {};

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

  /// \brief Compare projection status.
  bool operator==(const ProjectionResult::Status& other) const { return status_ == other; };

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
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum { CLASS_SERIALIZATION_VERSION = 1 };

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

  // TODO(slynen) Enable commented out PropertyTree support
  //explicit Camera(const sm::PropertyTree& property_tree);
 protected:
  Camera() = delete;

  /// \brief Camera base constructor with distortion.
  /// @param[in] intrinsics Vector containing the intrinsic parameters.
  /// @param[in] distortion unique_ptr to the distortion model
  /// @param[in] image_width Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  Camera(const Eigen::VectorXd& intrinsics, aslam::Distortion::UniquePtr& distortion,
         uint32_t image_width, uint32_t image_height);

  /// \brief Camera base constructor without distortion.
  /// @param[in] intrinsics Vector containing the intrinsic parameters.
  /// @param[in] image_width Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  Camera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height);

 public:
  virtual ~Camera() {};

  /// \brief Compare this camera to another camera object.
  virtual bool operator==(const Camera& other) const;

  /// \brief Convenience function to print the state using streams.
  std::ostream& operator<<(std::ostream& out) {
    this->printParameters(out, std::string(""));
    return out;
  };

  /// \brief Clones the camera instance and returns a pointer to the copy.
  virtual aslam::Camera* clone() const = 0;

 protected:
  /// Copy constructor for clone operation.
  Camera(const Camera& other) :
  line_delay_nano_seconds_(other.line_delay_nano_seconds_),
  label_(other.label_),
  id_(other.id_),
  image_width_(other.image_width_),
  image_height_(other.image_height_),
  intrinsics_(other.intrinsics_) {
    // Clone distortion if model is set.
    if (other.distortion_)
      distortion_.reset(other.distortion_->clone());
  };

  void operator=(const Camera&) = delete;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Information about the camera
  /// @{
 public:
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

  /// \brief The number of intrinsic parameters.
  virtual int getParameterSize() const  = 0;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points
  /// @{

  /// \brief Projects a euclidean point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_3d     The point in euclidean coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @return Contains information about the success of the projection. Check
  ///         \ref ProjectionResult for more information.
  const ProjectionResult project3(const Eigen::Vector3d& point_3d,
                                  Eigen::Vector2d* out_keypoint) const;

  /// \brief Projects a euclidean point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the po int.
  /// @param[in]  point_3d     The point in euclidean coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @param[out] out_jacobian The Jacobian wrt. to changes in the euclidean point.
  /// @return Contains information about the success of the projection. Check
  ///         \ref ProjectionResult for more information.
  const ProjectionResult project3(const Eigen::Vector3d& point_3d,
                                  Eigen::Vector2d* out_keypoint,
                                  Eigen::Matrix<double, 2, 3>* out_jacobian) const;

  /// \brief Projects a matrix of euclidean points to 2d image measurements. Applies the
  ///        projection (& distortion) models to the points.
  ///
  /// This vanilla version just repeatedly calls backProject3. Camera implementers
  /// are encouraged to override for efficiency.
  /// @param[in]  point_3d      The point in euclidean coordinates.
  /// @param[out] out_keypoints The keypoint in image coordinates.
  /// @param[out] out_results   Contains information about the success of the
  ///                           projections. Check \ref ProjectionResult for
  ///                           more information.
  virtual void project3Vectorized(const Eigen::Matrix3Xd& points_3d,
                                  Eigen::Matrix2Xd* out_keypoints,
                                  std::vector<ProjectionResult>* out_results) const;

  /// \brief Compute the 3d bearing vector in euclidean coordinates given a keypoint in
  ///        image coordinates. Uses the projection (& distortion) models.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in euclidean coordinates
  /// @return Was the projection successful?
  virtual bool backProject3(const Eigen::Vector2d& keypoint,
                            Eigen::Vector3d* out_point_3d) const = 0;

  /// \brief Compute the 3d bearing vectors in euclidean coordinates given a list of
  ///        keypoints in image coordinates. Uses the projection (& distortion) models.
  ///
  /// This vanilla version just repeatedly calls backProject3. Camera implementers
  /// are encouraged to override for efficiency.
  /// TODO(schneith): implement efficient backProject3Vectorized
  /// @param[in]  keypoints     Keypoints in image coordinates.
  /// @param[out] out_point_3ds Bearing vectors in euclidean coordinates (with z=1 -> non-normalized).
  /// @param[out] out_success   Were the projections successful?
  virtual void backProject3Vectorized(const Eigen::Matrix2Xd& keypoints,
                                      Eigen::Matrix3Xd* out_points_3d,
                                      std::vector<bool>* out_success) const;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project homogeneous points
  /// @{

  /// \brief Projects a homogeneous point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_4d     The point in homogeneous coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @return Contains information about the success of the projection. Check
  ///         \ref ProjectionResult for more information.
  const ProjectionResult project4(const Eigen::Vector4d& point_4d,
                                  Eigen::Vector2d* out_keypoint) const;

  /// \brief Projects a euclidean point to a 2d image measurement. Applies the
  ///        projection (& distortion) models to the point.
  /// @param[in]  point_4d     The point in homogeneous coordinates.
  /// @param[out] out_keypoint The keypoint in image coordinates.
  /// @param[out] out_jacobian The Jacobian wrt. to changes in the homogeneous point.
  /// @return Contains information about the success of the projection. Check \ref
  ///         ProjectionResult for more information.
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
  ///                                     NOTE: If nullptr, use internal intrinsic parameters.
  /// @param[in]  distortion_coefficients_external External distortion parameter vector.
  ///                                     Parameter is ignored is no distortion is active.
  ///                                     NOTE: If nullptr, use internal distortion parameters.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @return Contains information about the success of the projection. Check \ref
  ///         ProjectionResult for more information.
  const ProjectionResult project3Functional(
      const Eigen::Vector3d& point_3d,
      const Eigen::VectorXd* intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint) const;

  /// \brief This function projects a point into the image using the intrinsic parameters
  ///        that are passed in as arguments. If any of the Jacobians are nonnull, they
  ///        should be filled in with the Jacobian with respect to small changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[in]  intrinsics_external     External intrinsic parameter vector.
  ///                                     NOTE: If nullptr, use internal intrinsic parameters.
  /// @param[in]  distortion_coefficients_external External distortion parameter vector.
  ///                                     Parameter is ignored is no distortion is active.
  ///                                     NOTE: If nullptr, use internal distortion parameters.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @param[out] out_jacobian_point      The Jacobian wrt. to changes in the euclidean point.
  ///                                       nullptr: calculation is skipped.
  /// @param[out] out_jacobian_intrinsics The Jacobian wrt. to changes in the intrinsics.
  ///                                       nullptr: calculation is skipped.
  /// @param[out] out_jacobian_distortion The Jacobian wrt. to changes in the distortion parameters.
  ///                                       nullptr: calculation is skipped.
  /// @return Contains information about the success of the projection. Check \ref
  ///         ProjectionResult for more information.
  virtual const ProjectionResult project3Functional(
      const Eigen::Vector3d& point_3d,
      const Eigen::VectorXd* intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const = 0;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to create an undistorter for this camera.
  /// @{

 public:
  /// \brief Factory method to create a mapped undistorter for this camera geometry.
  ///        NOTE: The undistorter stores a copy of this camera and changes to this geometry
  ///              are not connected with the undistorter!
  /// @param[in] alpha Free scaling parameter between 0 (when all the pixels in the undistorted image
  ///                  will be valid) and 1 (when all the source image pixels will be retained in the
  ///                  undistorted image)
  /// @param[in] scale Output image size scaling parameter wrt. to input image size.
  /// @param[in] interpolation_type Check \ref InterpolationMethod to see the available types.
  /// @return Pointer to the created mapped undistorter.
  virtual std::unique_ptr<MappedUndistorter> createMappedUndistorter(float alpha, float scale,
      aslam::InterpolationMethod interpolation_type) const = 0;

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
  template<typename Scalar>
  bool isKeypointVisible(const Eigen::Matrix<Scalar, 2, 1>& keypoint) const;

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
  /// \name Methods to interface the underlying distortion model.
  /// @{

  /// \brief Returns a pointer to the underlying distortion object.
  /// @return Pointer for the distortion model;
  ///         NOTE: Returns nullptr if no model is set or not available for the camera type
  virtual aslam::Distortion* getDistortionMutable() { return distortion_.get(); };

  /// \brief Returns a const pointer to the underlying distortion object.
  /// @return ConstPointer for the distortion model;
  ///         NOTE: Returns nullptr if no model is set or not available for the camera type
  virtual const aslam::Distortion* getDistortion() const { return distortion_.get(); };

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to access the intrinsic parameters.
  /// @{

  /// Get the intrinsic parameters (const).
  inline const Eigen::VectorXd& getParameters() const { return intrinsics_; };

  /// Get the intrinsic parameters.
  inline double* getParametersMutable() { return &intrinsics_.coeffRef(0, 0); };

  /// Set the intrinsic parameters. Parameters are documented in the specialized
  /// camera classes.
  void setParameters(const Eigen::VectorXd& params) {
    CHECK_EQ(getParameterSize(), params.size());
    intrinsics_ = params;
  }

  /// Function to check wheter the given intrinic parameters are valid for this model.
  virtual bool intrinsicsValid(const Eigen::VectorXd& intrinsics) = 0;

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
  static typename DerivedCamera::Ptr construct(
      const Eigen::VectorXd& intrinsics,
      uint32_t imageWidth,
      uint32_t imageHeight,
      const Eigen::VectorXd& distortionParameters);

  /// @}

 private:
  /// The delay per scanline for a rolling shutter camera in nanoseconds.
  uint64_t line_delay_nano_seconds_;
  /// A label for this camera, a name.
  std::string label_;
  /// The id of this camera.
  aslam::CameraId id_;
  /// The width of the image
  const uint32_t image_width_;
  /// The height of the image
  const uint32_t image_height_;

 protected:
  /// Parameter vector for the intrinsic parameters of the model.
  Eigen::VectorXd intrinsics_;

  /// \brief The distortion for this camera.
  ///        NOTE: Can be nullptr if no distortion model is set.
  aslam::Distortion::UniquePtr distortion_;
};
}  // namespace aslam
#include "camera-inl.h"
#endif  // ASLAM_CAMERAS_CAMERA_H_
