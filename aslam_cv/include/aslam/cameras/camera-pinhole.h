#ifndef ASLAM_CAMERAS_PINHOLE_CAMERA_H_
#define ASLAM_CAMERAS_PINHOLE_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/macros.h>

namespace aslam {

/// \class PinholeCamera
/// \brief An implementation of the pinhole camera model with (optional) distortion.
///        Intrinsic parameters ordering: fu, fv, cu, cv
///        Reference: http://en.wikipedia.org/wiki/Pinhole_camera_model
class PinholeCamera : public Camera {
  enum { kNumOfParams = 4 };
 public:
  ASLAM_POINTER_TYPEDEFS(PinholeCamera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(PinholeCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  // TODO(slynen) Enable commented out PropertyTree support
  // PinholeCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

 protected:
  PinholeCamera() = delete;

 public:
  /// \brief Construct a PinholeCamera with distortion.
  /// @param[in] focalLengthCols focallength; cols direction in pixels (width-direction)
  /// @param[in] focalLengthRows focallength; rows direction in pixels (width-direction)
  /// @param[in] imageCenterCols image center; cols direction in pixels (height-direction)
  /// @param[in] imageCenterRows image center; rows direction in pixels (height-direction)
  /// @param[in] imageWidth      image width in pixels
  /// @param[in] imageHeight     image height in pixels
  /// @param[in] distortion      Pointer to the distortion model.
  PinholeCamera(double focalLengthCols, double focalLengthRows,
                double imageCenterCols, double imageCenterRows,
                uint32_t imageWidth, uint32_t imageHeight,
                aslam::Distortion::Ptr distortion);

  /// \brief Construct a PinholeCamera without distortion.
  /// @param[in] focalLengthCols focallength; cols direction in pixels (width-direction)
  /// @param[in] focalLengthRows focallength; rows direction in pixels (width-direction)
  /// @param[in] imageCenterCols image center; cols direction in pixels (height-direction)
  /// @param[in] imageCenterRows image center; rows direction in pixels (height-direction)
  /// @param[in] imageWidth      image width in pixels
  /// @param[in] imageHeight     image height in pixels
  PinholeCamera(double focalLengthCols, double focalLengthRows,
                double imageCenterCols, double imageCenterRows,
                uint32_t resolutionWidth, uint32_t resolutionHeight);

  virtual ~PinholeCamera() {};

  /// \brief Compare this camera to another camera object.
  virtual bool operator==(const Camera& other) const;

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points
  /// @{

  /// \brief This function projects a point into the image using the intrinsic parameters
  ///        that are passed in as arguments. If any of the Jacobians are nonnull, they
  ///        should be filled in with the Jacobian with respect to small changes in the argument.
  /// @param[in]  point_3d                The point in euclidean coordinates.
  /// @param[out] out_keypoint            The keypoint in image coordinates.
  /// @param[out] out_jacobian_point      The Jacobian w.r.t. to changes in the euclidean point.
  ///                                     If nullptr: Jacobian calculation is skipped.
  /// @param[out] out_jacobian_intrinsics The Jacobian w.r.t. to changes in the intrinsics.
  ///                                     If nullptr: Jacobian calculation is skipped.
  /// @return The ProjectionState object contains details about the success of the projection.
  virtual const ProjectionState project3Functional(
      const Eigen::Vector3d& point_3d,
      Eigen::Vector2d* out_keypoint,
      const Eigen::VectorXd* intrinsics_external,
      Eigen::Matrix<double, 2, 3>* out_jacobian_point,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* out_jacobian_intrinsics) const;

  /// \brief Compute the 3d bearing vector in euclidean coordinates given a keypoint in
  ///        image coordinates. Uses the projection (& distortion) models.
  /// @param[in]  keypoint            Keypoint in image coordinates.
  /// @param[out] out_point_3d        Bearing vector in homogeneous coordinates.
  /// @param[in]  intrinsics_external Vector containing the intrinsic parameters.
  ///                                 If nullptr: Internal parameters will be used.
  virtual void backProject3(
      const Eigen::Vector2d& keypoint,
      Eigen::Vector3d* out_point_3d,
      const Eigen::VectorXd* intrinsics_external) const;

  /// \brief Template version of project3Functional.
  template <typename ScalarType, typename DistortionType>
  const ProjectionState project3Functional(
      const Eigen::Matrix<ScalarType, 3, 1>& point,
      const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& intrinsics,
      Eigen::Matrix<ScalarType, 2, 1>* out_point) const;

  /// \brief Checks the success of a projection operation and returns the result in a
  ///        ProjectionState object.
  /// @param[in] keypoint Keypoint in image coordinates.
  /// @param[in] point_3d Projected point in euclidean.
  /// @return The ProjectionState object contains details about the success of the projection.
  const ProjectionState evaluateProjectionState(const Eigen::Vector2d& keypoint,
                                                const Eigen::Vector3d& point_3d) const;

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

  /// \brief Set the intrinsic parameters for the pinhole model.
  /// @param[in] params Intrinsic parameters (dim=4: fu, fv, cu, cv)
  virtual void setParameters(const Eigen::VectorXd& params);

  /// \brief Returns the number of intrinsic parameters used in this camera model.
  inline static constexpr size_t parameterCount() {
      return kNumOfParams;
  }

  /// \brief Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  virtual void printParameters(std::ostream& out, const std::string& text);

  /// @}

 private:
  /// \brief The distortion of this camera.
  aslam::Distortion::Ptr distortion_;

  static constexpr double kMinimumDepth = 1e-10;
};

}  // namespace aslam

#include "aslam/cameras/camera-pinhole-inl.h"

#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_H_
