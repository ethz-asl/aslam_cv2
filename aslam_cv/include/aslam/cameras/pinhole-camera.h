#ifndef ASLAM_CAMERAS_PINHOLE_CAMERA_H_
#define ASLAM_CAMERAS_PINHOLE_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/macros.h>

namespace aslam {

template<typename Derived>
Eigen::Map<const Eigen::VectorXd> vecmap(const Eigen::MatrixBase<Derived>& vec){
  return Eigen::Map<const Eigen::VectorXd>(static_cast<const Derived&>(vec).data(),vec.size());
}

class PinholeCamera : public Camera {
  enum { kNumOfParams = 4 };
 public:
  ASLAM_POINTER_TYPEDEFS(PinholeCamera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(PinholeCamera);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PinholeCamera();
  PinholeCamera(double focalLengthU, double focalLengthV,
                double imageCenterU, double imageCenterV, int resolutionU,
                int resolutionV, aslam::Distortion::Ptr distortion,
                const Eigen::VectorXd& distortionParameters);

  PinholeCamera(double focalLengthU, double focalLengthV,
                double imageCenterU, double imageCenterV, int resolutionU,
                int resolutionV);
  // TODO(slynen) Enable commented out PropertyTree support
  // PinholeCamera(const sm::PropertyTree& config);

  virtual ~PinholeCamera();

  /// Project a point expressed in euclidean coordinates to a 2d image measurement.
  virtual bool project3(const Eigen::Vector3d& point,
                        Eigen::Matrix<double, 2, 1>* out_point) const;

  /// Project a point expressed in euclidean coordinates to a 2d image measurement,
  /// works for an arbitrary scalar type
  // TODO(dymczykm) stop templating on DistortionType after we'll able
  // to get derived type from this class member ('_distortion')
  template <typename ScalarType, typename DistortionType>
  bool project3(const Eigen::Matrix<ScalarType, 3, 1>& point,
                const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& intrinsics,
                Eigen::Matrix<ScalarType, 2, 1>* out_point) const;

  /// Project a point expressed in euclidean coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool project3(const Eigen::Vector3d & point,
                        Eigen::Matrix<double, 2, 1>* out_point,
                        Eigen::Matrix<double, 2, 3>* out_jacobian) const;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement.
  virtual bool project4(const Eigen::Vector4d& homogeneous_point,
                        Eigen::Matrix<double, 2, 1>* out_point) const;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool project4(const Eigen::Vector4d & homogeneous_point,
                        Eigen::Matrix<double, 2, 1>* out_point,
                        Eigen::Matrix<double, 2, 4>* out_jacobian) const;

  /// Compute the 3d bearing vector in euclidean coordinates from the 2d image measurement.
  virtual bool backProject3(const Eigen::Matrix<double, 2, 1>& keypoint,
                            Eigen::Matrix<double, 3, 1>* out_point) const;


  /// Compute the 3d bearing vector in euclidean coordinates and the relevant jacobian
  /// from the 2d image measurement.
  virtual bool backProject3(const Eigen::Vector2d& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point,
                                   Eigen::Matrix<double, 3, 2>* out_jacobian) const;

  /// Compute the 3d bearing vector in homogenous coordinates from the 2d image measurement.
  virtual bool backProject4(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point) const;

  /// Compute the 3d bearing vector in homogeneous coordinates and the relevant
  /// jacobian from the 2d image measurement.
  virtual bool backProject4(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point,
                                     Eigen::Matrix<double, 4, 2>* out_jacobian) const;

  virtual bool project3IntrinsicsJacobian(
      const Eigen::Matrix<double, 3, 1>& p,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const;

  virtual bool project4IntrinsicsJacobian(
      const Eigen::Matrix<double, 4, 1>& p,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const;

  virtual bool operator==(const PinholeCamera& other) const;

  /// \brief The horizontal resolution in pixels.
  virtual uint32_t imageWidth() const {
    return _ru;
  }
  /// \brief The vertical resolution in pixels.
  virtual uint32_t imageHeight() const {
    return _rv;
  }

  void setDistortion(const aslam::Distortion::Ptr& distortion, 
                     const Eigen::VectorXd& params) {
    _distortion = distortion;
    if (distortion) {
      CHECK(distortion->distortionParametersValid(vecmap(params)));
      _intrinsics.conservativeResize(kNumOfParams + distortion->getParameterSize());
      _intrinsics.tail(distortion->getParameterSize()) = params;
    }
  }

  aslam::Distortion::Ptr& distortion() {
    return _distortion;
  }

  const aslam::Distortion::Ptr distortion() const {
    return _distortion;
  }

  Eigen::Matrix3d getCameraMatrix() const {
    Eigen::Matrix3d K;
    const double& fu = _intrinsics(0);
    const double& fv = _intrinsics(1);
    const double& cu = _intrinsics(2);
    const double& cv = _intrinsics(3);

    K << fu, 0.0, cu, 0.0, fv, cv, 0.0, 0.0, 1.0;
    return K;
  }

  double focalLengthCol() const {
    return _intrinsics[0];
  }
  double focalLengthRow() const {
    return _intrinsics[1];
  }
  double opticalCenterCol() const {
    return _intrinsics[2];
  }
  double opticalCenterRow() const {
    return _intrinsics[3];
  }

  /// \brief The horizontal focal length in pixels.
  double fu() const {
    return _intrinsics[0];
  }
  /// \brief The vertical focal length in pixels.
  double fv() const {
    return _intrinsics[1];
  }
  /// \brief The horizontal image center in pixels.
  double cu() const {
    return _intrinsics[2];
  }
  /// \brief The vertical image center in pixels.
  double cv() const {
    return _intrinsics[3];
  }
  /// \brief The horizontal resolution in pixels.
  int ru() const {
    return _ru;
  }
  /// \brief The vertical resolution in pixels.
  int rv() const {
    return _rv;
  }

  // \brief creates a random valid keypoint.
  virtual Eigen::Matrix<double, 2, 1> createRandomKeypoint() const;

  // \brief creates a random visible point. Negative depth means random between 0 and 100 meters.
  virtual Eigen::Matrix<double, 3, 1> createRandomVisiblePoint(double depth) const;

  virtual bool project3Functional(const Eigen::VectorXd& intrinsics_params,
                                  const Eigen::Vector3d& point,
                                  Eigen::Vector2d * out_point,
                                  Eigen::Matrix<double, 2, Eigen::Dynamic>* out_intrinsics_jacobian,
                                  Eigen::Matrix<double, 2, Eigen::Dynamic>* out_point_jacobian) const;

  //////////////////////////////////////////////////////////////
  // VALIDITY TESTING
  //////////////////////////////////////////////////////////////
  template <typename ScalarType>
  bool isKeypointVisible(const Eigen::Matrix<ScalarType, 2, 1>& keypoint) const;

  virtual bool isVisible(const Eigen::Matrix<double, 2, 1>& keypoint) const {
    return isKeypointVisible(keypoint);
  }

  virtual bool isProjectable3(const Eigen::Matrix<double, 3, 1>& p) const;

  virtual bool isProjectable4(const Eigen::Matrix<double, 4, 1>& ph) const;

  virtual const Eigen::VectorXd& getParameters() const;

  virtual void setParameters(const Eigen::VectorXd& params);

  virtual size_t getParameterSize() const;

  virtual Eigen::VectorXd& getParametersMutable() {
    return _intrinsics;
  }

  virtual double* getParameterMutablePtr() {
    return _intrinsics.data();
  }

  /// \brief resize the intrinsics based on a scaling of the image.
  virtual void resizeIntrinsics(double scale);

  /// \brief Get a set of border rays
  virtual void getBorderRays(Eigen::MatrixXd & rays);

  // TODO(slynen): Reintegrate once we have the calibration targets.
  /*
  /// \brief initialize the intrinsics based on a list of views of a gridded calibration target
  /// \return true on success
  virtual bool initializeIntrinsics(const std::vector<GridCalibrationTargetObservation> &observations);

  /// \brief compute the reprojection error based on a checkerboard observation.
  /// \return the number of corners successfully observed and projected
  virtual size_t computeReprojectionError(
      const GridCalibrationTargetObservation & obs,
      const sm::kinematics::Transformation & T_target_camera,
      double & outErr) const;

  /// \brief estimate the transformation of the camera with respect to the calibration target
  ///        On success out_T_t_c is filled in with the transformation that takes points from
  ///        the camera frame to the target frame
  /// \return true on success
  virtual bool estimateTransformation(const GridCalibrationTargetObservation & obs,
                                      sm::kinematics::Transformation & out_T_t_c) const;
   */

 private:
  void updateTemporaries();

   /// Get the distortion parameters from the intrinsics vector
   Eigen::Map<const Eigen::VectorXd> getDistortionParameters() const;

  // Vector storing intrinsic parameters in a contiguous block of memory.
  // Ordering: fu, fv, cu, cv
  Eigen::VectorXd _intrinsics;

  /// \brief The horizontal resolution in pixels.
  int _ru;
  /// \brief The vertical resolution in pixels.
  int _rv;

  /// \brief A computed value for speeding up computation.
  double _recip_fu;
  double _recip_fv;
  double _fu_over_fv;

  /// \brief The distortion of this camera.
  aslam::Distortion::Ptr _distortion;

  static constexpr double kMinimumDepth = 1e-10;
};

}  // namespace aslam

#include "aslam/cameras/pinhole-camera-inl.h"

#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_H_
