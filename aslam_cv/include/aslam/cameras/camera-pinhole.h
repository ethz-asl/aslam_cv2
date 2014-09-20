#ifndef ASLAM_CAMERAS_PINHOLE_CAMERA_H_
#define ASLAM_CAMERAS_PINHOLE_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/macros.h>

namespace aslam {

/// \class PinholeCamera
/// \brief An implementation of the pinhole camera model with (optional) distortion.
///
/// The usual model of a pinhole camera follows these steps:
///    - Transformation: Transform the point into a coordinate frame associated with the camera
///    - Normalization:  Project the point onto the normalized image plane: \f$\mathbf y := \left[ x/z,y/z\right] \f$
///    - Distortion:     apply a nonlinear transformation to \f$y\f$ to account for radial and tangential distortion of the lens
///    - Projection:     Project the point into the image using a standard \f$3 \time 3\f$ projection matrix
///
///  Intrinsic parameters ordering: fu, fv, cu, cv
///  Reference: http://en.wikipedia.org/wiki/Pinhole_camera_model
class PinholeCamera : public Camera {
  enum { kNumOfParams = 4 };
 public:
  ASLAM_POINTER_TYPEDEFS(PinholeCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // TODO(slynen) Enable commented out PropertyTree support
  // PinholeCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

 protected:
  /// \brief Empty constructor for serialization interface.
  PinholeCamera();

  /// Copy constructor for clone operation.
  PinholeCamera(const PinholeCamera& other) : Camera(other), distortion_(nullptr) {
    if (other.distortion_) // Clone distortion if model is set.
      distortion_.reset(other.distortion_->clone());
  };
  void operator=(const PinholeCamera&) = delete;

 public:
  virtual aslam::Camera* clone() const
  {
    return new PinholeCamera(static_cast<const PinholeCamera&>(*this));
  }


 public:
  /// \brief Construct a PinholeCamera with distortion.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters (fu,fv,cu,cv).
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  /// @param[in] distortion   Pointer to the distortion model.
  PinholeCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height,
                aslam::Distortion::Ptr distortion);

  /// \brief Construct a PinholeCamera without distortion.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters (fu,fv,cu,cv).
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  PinholeCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height);

  /// \brief Construct a PinholeCamera with distortion.
  /// @param[in] focallength_cols Focal length in pixels; cols (width-direction).
  /// @param[in] focallength_rows Focal length in pixels; rows (height-direction).
  /// @param[in] imagecenter_cols Image center in pixels; cols (width-direction).
  /// @param[in] imagecenter_rows Image center in pixels; rows (height-direction).
  /// @param[in] image_width      Image width in pixels.
  /// @param[in] image_height     Image height in pixels.
  /// @param[in] distortion       Pointer to the distortion model.
  PinholeCamera(double focallength_cols, double focallength_rows,
                double imagecenter_cols, double imagecenter_rows,
                uint32_t image_width, uint32_t image_height,
                aslam::Distortion::Ptr distortion);

  /// \brief Construct a PinholeCamera without distortion.
  /// @param[in] focallength_cols Focal length in pixels; cols (width-direction).
  /// @param[in] focallength_rows Focal length in pixels; rows (height-direction).
  /// @param[in] imagecenter_cols Image center in pixels; cols (width-direction).
  /// @param[in] imagecenter_rows Image center in pixels; rows (height-direction).
  /// @param[in] image_width      Image width in pixels.
  /// @param[in] image_height     Image height in pixels.
  PinholeCamera(double focallength_cols, double focallength_rows,
                double imagecenter_cols, double imagecenter_rows,
                uint32_t image_width, uint32_t image_height);

  virtual ~PinholeCamera() {};

  /// \brief Compare this camera to another camera object.
  virtual bool operator==(const Camera& other) const;

  /// \brief Convenience function to print the state using streams.
  std::ostream& operator<<(std::ostream& out) {
    this->printParameters(out, std::string(""));
    return out;
  };

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points
  /// @{

  /// \brief Compute the 3d bearing vector in euclidean coordinates given a keypoint in
  ///        image coordinates. Uses the projection (& distortion) models.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in euclidean coordinates (with z=1 -> non-normalized).
  virtual bool backProject3(const Eigen::Vector2d& keypoint,
                            Eigen::Vector3d* out_point_3d) const;

  /// \brief Checks the success of a projection operation and returns the result in a
  ///        ProjectionResult object.
  /// @param[in] keypoint Keypoint in image coordinates.
  /// @param[in] point_3d Projected point in euclidean.
  /// @return The ProjectionResult object contains details about the success of the projection.
  template <typename DerivedKeyPoint, typename DerivedPoint3d>
  inline const ProjectionResult evaluateProjectionResult(
      const Eigen::MatrixBase<DerivedKeyPoint>& keypoint,
      const Eigen::MatrixBase<DerivedPoint3d>& point_3d) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Functional methods to project and back-project points
  /// @{

  // Get the overloaded non-virtual project3Functional(..) from base into scope.
  using Camera::project3Functional;

  /// \brief Template version of project3Functional.
  template <typename ScalarType, typename DistortionType,
            typename MIntrinsics, typename MDistortion>
  const ProjectionResult project3Functional(
      const Eigen::Matrix<ScalarType, 3, 1>& point_3d,
      const Eigen::MatrixBase<MIntrinsics>& intrinsics_external,
      const Eigen::MatrixBase<MDistortion>& distortion_coefficients_external,
      Eigen::Matrix<ScalarType, 2, 1>* out_keypoint) const;

  /// \brief This function projects a point into the image using the intrinsic parameters
  ///        that are passed in as arguments. If any of the Jacobians are nonnull, they
  ///        should be filled in with the Jacobian with respect to small changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[in]  intrinsics_external     External intrinsic parameter vector.
  ///                                     NOTE: If nullptr, use internal distortion parameters.
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
  /// @return Contains information about the success of the projection. Check
  ///         \ref ProjectionResult for more information.
  virtual const ProjectionResult project3Functional(
      const Eigen::Vector3d& point_3d,
      const Eigen::VectorXd* intrinsics_external,
      const Eigen::VectorXd* distortion_coefficients_external,
      Eigen::Vector2d* out_keypoint,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_distortion) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to support unit testing.
  /// @{

  /// \brief Creates a random valid keypoint..
  virtual Eigen::Vector2d createRandomKeypoint() const;

  /// \brief Creates a random visible point. Negative depth means random between
  ///        0 and 100 meters.
  virtual Eigen::Vector3d createRandomVisiblePoint(double depth) const;

  /// \brief Get a set of border rays
  void getBorderRays(Eigen::MatrixXd & rays) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to set/get distortion parameters.
  /// @{

  /// \brief Returns a pointer to the underlying distortion object.
  /// @return ptr to distortion model; nullptr if none is set or not available
  ///         for the camera type
  virtual aslam::Distortion::Ptr distortion() { return distortion_; };

  /// \brief Returns a const pointer to the underlying distortion object.
  /// @return const_ptr to distortion model; nullptr if none is set or not available
  ///         for the camera type
  virtual const aslam::Distortion::Ptr distortion() const { return distortion_; };

  /// \brief Create a test camera object for unit testing.
  template<typename DistortionType>
  static PinholeCamera::Ptr createTestCamera()   {
    return PinholeCamera::Ptr(new PinholeCamera(400, 400, 320, 240, 640, 480,
                                                DistortionType::createTestDistortion()));
  }

  /// \brief Create a test camera object for unit testing. (without distortion)
  static PinholeCamera::Ptr createTestCamera() {
    return PinholeCamera::Ptr(new PinholeCamera(400, 400, 320, 240, 640, 480));
  }

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to access intrinsics.
  /// @{

  /// \brief Returns the camera matrix for the pinhole projection.
  Eigen::Matrix3d getCameraMatrix() const {
    Eigen::Matrix3d K;
    K << fu(), 0.0,  cu(),
         0.0,  fv(), cv(),
         0.0,  0.0,  1.0;
    return K;
  }

  /// \brief The horizontal focal length in pixels.
  double fu() const { return intrinsics_[0]; };
  /// \brief The vertical focal length in pixels.
  double fv() const { return intrinsics_[1]; };
  /// \brief The horizontal image center in pixels.
  double cu() const { return intrinsics_[2]; };
  /// \brief The vertical image center in pixels.
  double cv() const { return intrinsics_[3]; };

  /// \brief Returns the number of intrinsic parameters used in this camera model.
  inline static constexpr int parameterCount() {
      return kNumOfParams;
  }

  /// \brief Returns the number of intrinsic parameters used in this camera model.
  inline virtual int getParameterSize() const {
      return kNumOfParams;
  }

  /// \brief Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  virtual void printParameters(std::ostream& out, const std::string& text) const;

  /// @}

 private:
  /// \brief The distortion of this camera.
  aslam::Distortion::Ptr distortion_;

  /// \brief Minimal depth for a valid projection.
  static constexpr double kMinimumDepth = 1e-10;
};

}  // namespace aslam

#include "aslam/cameras/camera-pinhole-inl.h"

#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_H_
