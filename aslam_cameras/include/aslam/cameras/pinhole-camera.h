#ifndef ASLAM_CAMERAS_PINHOLE_CAMERA_H_
#define ASLAM_CAMERAS_PINHOLE_CAMERA_H_
#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/macros.h>

namespace aslam {
class PinholeCamera : public Camera {
 public:
  ASLAM_POINTER_TYPEDEFS(PinholeCamera);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(PinholeCamera);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  enum {
    IntrinsicsDimension = 4
  };
  enum {
    DesignVariableDimension = IntrinsicsDimension
  };

  PinholeCamera();
  PinholeCamera(double focalLengthU, double focalLengthV,
                double imageCenterU, double imageCenterV, int resolutionU,
                int resolutionV, aslam::Distortion::Ptr distortion);

  PinholeCamera(double focalLengthU, double focalLengthV,
                double imageCenterU, double imageCenterV, int resolutionU,
                int resolutionV);
//  PinholeCamera(const sm::PropertyTree& config);

  virtual ~PinholeCamera();

  /// Project a point expressed in euclidean coordinates to a 2d image measurement.
  virtual bool euclideanToKeypoint(const Eigen::Vector3d& point,
                                   Eigen::Matrix<double, 2, 1>* out_point) const;
  /// Project a point expressed in euclidean coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool euclideanToKeypoint(const Eigen::Vector3d & point,
                                   Eigen::Matrix<double, 2, 1>* out_point,
                                   Eigen::Matrix<double, 2, 3>* out_jacobian) const;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement.
  virtual bool homogeneousToKeypoint(const Eigen::Vector4d& homogeneous_point,
                                     Eigen::Matrix<double, 2, 1>* out_point) const;

  /// Project a point expressed in homogenous coordinates to a 2d image measurement
  /// and calculate the relevant jacobian.
  virtual bool homogeneousToKeypoint(const Eigen::Vector4d & homogeneous_point,
                                     Eigen::Matrix<double, 2, 1>* out_point,
                                     Eigen::Matrix<double, 2, 4>* out_jacobian) const;

  /// Compute the 3d bearing vector in euclidean coordinates from the 2d image measurement.
  virtual bool keypointToEuclidean(const Eigen::Matrix<double, 2, 1>& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point) const;


  /// Compute the 3d bearing vector in euclidean coordinates and the relevant jacobian
  /// from the 2d image measurement.
  virtual bool keypointToEuclidean(const Eigen::Vector2d& keypoint,
                                   Eigen::Matrix<double, 3, 1>* out_point,
                                   Eigen::Matrix<double, 3, 2>* out_jacobian) const;

  /// Compute the 3d bearing vector in homogenous coordinates from the 2d image measurement.
  virtual bool keypointToHomogeneous(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point) const;

  /// Compute the 3d bearing vector in homogeneous coordinates and the relevant
  /// jacobian from the 2d image measurement.
  virtual bool keypointToHomogeneous(Eigen::Vector2d const& keypoint,
                                     Eigen::Matrix<double, 4, 1>* out_point,
                                     Eigen::Matrix<double, 4, 2>* out_jacobian) const;

  virtual bool euclideanToKeypointIntrinsicsJacobian(
      const Eigen::Matrix<double, 3, 1>& p,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const;

  virtual bool euclideanToKeypointDistortionJacobian(
      const Eigen::Matrix<double, 3, 1>& p,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJd) const;

  virtual bool homogeneousToKeypointIntrinsicsJacobian(
      const Eigen::Matrix<double, 4, 1>& p,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJi) const;

  virtual bool homogeneousToKeypointDistortionJacobian(
      const Eigen::Matrix<double, 4, 1>& p,
      Eigen::Matrix<double, 2, Eigen::Dynamic>* outJd) const;

  virtual bool operator==(const PinholeCamera& other) const;

  /// \brief The horizontal resolution in pixels.
  virtual uint32_t imageWidth() const {
    return _ru;
  }
  /// \brief The vertical resolution in pixels.
  virtual uint32_t imageHeight() const {
    return _rv;
  }


  void setDistortion(const aslam::Distortion::Ptr& distortion) {
    _distortion = distortion;
  }

  aslam::Distortion::Ptr& distortion() {
    return _distortion;
  }

  const aslam::Distortion::Ptr distortion() const {
    return _distortion;
  }

  Eigen::Matrix3d getCameraMatrix() const {
    Eigen::Matrix3d K;
    K << _fu, 0.0, _cu, 0.0, _fv, _cv, 0.0, 0.0, 1.0;
    return K;
  }

  double focalLengthCol() const {
    return _fu;
  }
  double focalLengthRow() const {
    return _fv;
  }
  double opticalCenterCol() const {
    return _cu;
  }
  double opticalCenterRow() const {
    return _cv;
  }

  /// \brief The horizontal focal length in pixels.
  double fu() const {
    return _fu;
  }
  /// \brief The vertical focal length in pixels.
  double fv() const {
    return _fv;
  }
  /// \brief The horizontal image center in pixels.
  double cu() const {
    return _cu;
  }
  /// \brief The vertical image center in pixels.
  double cv() const {
    return _cv;
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

  //////////////////////////////////////////////////////////////
  // VALIDITY TESTING
  //////////////////////////////////////////////////////////////
  virtual bool isProjectionInvertible() const {
    return false;
  }

  virtual bool isValid(const Eigen::Matrix<double, 2, 1>& keypoint) const;

  virtual bool isEuclideanVisible(const Eigen::Matrix<double, 3, 1>& p) const;

  virtual bool isHomogeneousVisible(const Eigen::Matrix<double, 4, 1>& ph) const;

  virtual void update(const double* v);
  virtual int minimalDimensions() const;
  virtual void getParameters(Eigen::MatrixXd & P) const;
  virtual void setParameters(const Eigen::MatrixXd & P);
  virtual Eigen::Vector2i parameterSize() const;

  static constexpr int parameterCount() const;

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
  /// \brief The horizontal focal length in pixels.
  double& _fu;
  /// \brief The vertical focal length in pixels.
  double& _fv;
  /// \brief The horizontal image center in pixels.
  double& _cu;
  /// \brief The vertical image center in pixels.
  double& _cv;
  /// \brief The horizontal resolution in pixels.
  int _ru;
  /// \brief The vertical resolution in pixels.
  int _rv;

  double _intrinsics[4];

  /// \brief A computed value for speeding up computation.
  double _recip_fu;
  double _recip_fv;
  double _fu_over_fv;

  /// \brief The distortion of this camera.
  aslam::Distortion::Ptr _distortion;
};
}  // namespace aslam
#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_H_
