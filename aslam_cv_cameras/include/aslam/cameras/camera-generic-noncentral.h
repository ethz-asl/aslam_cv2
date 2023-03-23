#ifndef ASLAM_CAMERAS_GENERIC_NONCENTRAL_CAMERA_H_
#define ASLAM_CAMERAS_GENERIC_NONCENTRAL_CAMERA_H_

#include <Eigen/Geometry>

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/crtp-clone.h>
#include <aslam/common/macros.h>
#include <aslam/common/types.h>

namespace aslam {

// Forward declarations.
class MappedUndistorter;
class NCamera;

/// \class GenericNoncentralCamera
/// \brief An implementation of the generic non-central camera model.
///
///
///  Reference: 
class GenericNoncentralCamera : public aslam::Cloneable<Camera, GenericNoncentralCamera> {
  friend class NCamera;

  int kNumOfParams;

 public:
  ASLAM_POINTER_TYPEDEFS(GenericNoncentralCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum Parameters { 
    kCalibrationMinX = 0,
    kCalibrationMinY = 1,
    kCalibrationMaxX = 2,
    kCalibrationMaxY = 3,
    kGridWidth = 4,
    kGridHeight = 5,
    kGrid = 6,
  };

  // TODO(slynen) Enable commented out PropertyTree support
  // GenericNoncentralCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

 public:
  /// \brief Empty constructor for serialization interface.
  GenericNoncentralCamera();

  /// Copy constructor for clone operation.
  GenericNoncentralCamera(const GenericNoncentralCamera& other) = default;
  void operator=(const GenericNoncentralCamera&) = delete;

 public:
  /// \brief Construct a GenericNoncentralCamera while supplying distortion. Distortion is removed.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  /// @param[in] distortion   Pointer to the distortion model.
  GenericNoncentralCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height,
                aslam::Distortion::UniquePtr& distortion);
                
  /// \brief Construct a GenericNoncentralCamera without distortion.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  GenericNoncentralCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height);

  virtual ~GenericNoncentralCamera() {};

  /// \brief Convenience function to print the state using streams.
  friend std::ostream& operator<<(std::ostream& out, const GenericNoncentralCamera& camera);

  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Methods to project and back-project euclidean points
  /// @{

  /// \brief Compute the 3d bearing vector in euclidean coordinates given a keypoint in
  ///        image coordinates. Uses the projection model.
  ///        The result might be in normalized image plane for some camera implementations but not
  ///        for the general case.
  /// @param[in]  keypoint     Keypoint in image coordinates.
  /// @param[out] out_point_3d Bearing vector in euclidean coordinates
  /// @return Contains if back-projection was possible

virtual bool backProject6(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                            Eigen::ParametrizedLine<double, 3>* out_line_3d) const;


virtual bool backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                          Eigen::Vector3d* out_line_3d) const;

  bool backProject6WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::ParametrizedLine<double, 3>* out_line_3d, Eigen::Matrix<double, 6, 2>* out_jacobian_pixel) const; 
  bool backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const; 

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
      const Eigen::Ref<const Eigen::Vector3d>& point_3d,
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

  /// @}

 public:
  //////////////////////////////////////////////////////////////
  /// \name Methods to access intrinsics.
  /// @{

  /// \brief The horizontal focal length in pixels.
  double calibrationMinX() const { return intrinsics_[Parameters::kCalibrationMinX]; };
  /// \brief The vertical focal length in pixels.
  double calibrationMinY() const { return intrinsics_[Parameters::kCalibrationMinY]; };
  /// \brief The horizontal image center in pixels.
  double calibrationMaxX() const { return intrinsics_[Parameters::kCalibrationMaxX]; };
  /// \brief The vertical image center in pixels.
  double calibrationMaxY() const { return intrinsics_[Parameters::kCalibrationMaxY]; };
  /// \brief The horizontal image center in pixels.
  double gridWidth() const { return intrinsics_[Parameters::kGridWidth]; };
  /// \brief The vertical image center in pixels.
  double gridHeight() const { return intrinsics_[Parameters::kGridHeight]; };
  /// \brief The total size of the grid.
  double gridSize() const { return gridWidth() * gridHeight(); };
  /// \brief The centerpoint of the calibrated area.
  Eigen::Vector2d centerOfCalibratedArea() const {
    return Eigen::Vector2d(
      0.5 * (calibrationMinX() + calibrationMaxX() + 1.),
      0.5 * (calibrationMinY() + calibrationMaxY() + 1.)
    );
  };

  /// \brief Returns the number of intrinsic parameters used in this camera model.
  inline int parameterCount() const {
      return kNumOfParams;
  }

  /// \brief Returns the number of intrinsic parameters used in this camera model.
  inline virtual int getParameterSize() const {
      return kNumOfParams;
  }

  /// Static function that checks whether the given intrinsic parameters are valid for this model.
  static bool areParametersValid(const Eigen::VectorXd& parameters);

  /// Function to check whether the given intrinsic parameters are valid for
  /// this model.
  virtual bool intrinsicsValid(const Eigen::VectorXd& intrinsics) const;

  /// Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  virtual void printParameters(std::ostream& out, const std::string& text) const;

  /// @}

  /// \brief Create a test camera object for unit testing.
  template <typename DistortionType>
  static GenericNoncentralCamera::Ptr createTestCamera() {
    return GenericNoncentralCamera::Ptr(
        std::move(createTestCameraUnique<DistortionType>()));
  }

  /// \brief Create a test camera object for unit testing.
  template <typename DistortionType>
  static GenericNoncentralCamera::UniquePtr createTestCameraUnique() {
    Distortion::UniquePtr distortion = DistortionType::createTestDistortion();
    Eigen::Matrix< double, 22, 1 > intrinsics;
    for(int i = 0; i < 22; i++) intrinsics(i) = (i+1)*(i+2);
    GenericNoncentralCamera::UniquePtr camera(
        new GenericNoncentralCamera(intrinsics, 640, 480, distortion));
    CameraId id;
    generateId(&id);
    camera->setId(id);
    return std::move(camera);
  }

  /// \brief Create a test camera object for unit testing. (without distortion)
  static GenericNoncentralCamera::Ptr createTestCamera();

  /// \brief return the first value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> firstPointGridValue() const { return pointGridAccess(0,0); };
  /// \brief return the last value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> lastPointGridValue() const { return pointGridAccess(gridHeight()-1, gridWidth()-1); };

  /// \brief return the first value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> firstDirectionGridValue() const { return directionGridAccess(0,0); };
  /// \brief return the last value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> lastDirectionGridValue() const { return directionGridAccess(gridHeight()-1, gridWidth()-1); };

  // position of the pixel expressed in gridpoints
  Eigen::Vector2d transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const;
  Eigen::Vector2d transformGridPointToImagePixel(const Eigen::Vector2d& gridpoint) const; // <- inverse of transformImagePixelToGridPoint

  double pixelScaleToGridScaleX(double length) const;
  double pixelScaleToGridScaleY(double length) const;

  Eigen::Vector3d pointGridAccess(const int y, const int x) const;
  Eigen::Vector3d pointGridAccess(const Eigen::Vector2d gridpoint) const;
  Eigen::Vector3d directionGridAccess(const int y, const int x) const;
  Eigen::Vector3d directionGridAccess(const Eigen::Vector2d gridpoint) const;
 private:
  /// \brief Minimal depth for a valid projection.
  static const double kMinimumDepth;

  bool isValidImpl() const override;
  void setRandomImpl() override;
  bool isEqualImpl(const Sensor& other, const bool verbose) const override;


  bool isInCalibratedArea(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const;

  //void interpolateCubicBSplineSurface(Eigen::Vector2d keypoint, Eigen::Vector3d* out_point_3d) const;
  void interpolateTwoCubicBSplineSurfaces(Eigen::Vector2d keypoint, Eigen::ParametrizedLine<double, 3>* out_line_3d) const;
  void interpolateCubicBSpline(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d, double frac_y, Eigen::Vector3d* out_point_3d) const;

  bool loadFromYamlNodeImpl(const YAML::Node&) override;
  void saveToYamlNodeImpl(YAML::Node*) const override;
  Eigen::VectorXd getIntrinsics() const;
  Eigen::VectorXd getGrid() const;

  bool backProject3WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::Vector3d* out_point_3d, Eigen::Matrix<double, 3, 2>* out_jacobian_pixel) const;
  bool backProject6WithJacobian(const Eigen::Ref<const Eigen::Vector2d>& keypoint, const Eigen::Ref<const Eigen::VectorXd>& intrinsics,
                                 Eigen::ParametrizedLine<double, 3>* out_line_3d, Eigen::Matrix<double, 6, 2>* out_jacobian_pixel) const;
  const ProjectionResult project3WithInitialEstimate(const Eigen::Ref<const Eigen::Vector3d>& point_3d, const Eigen::VectorXd* intrinsics,
                                  Eigen::Vector2d* out_keypoint) const;
  void NoncentralGenericBSpline_Unproject_ComputeResidualAndJacobian(double frac_x, double frac_y, Eigen::Matrix<double, 6, 1> l[4][4], Eigen::ParametrizedLine<double, 3>* result, Eigen::Matrix<double, 6, 2>* dresult_dxy) const;

  /// Treating the given direction as a vector with unit length, pointing to a
  /// spot on the unit sphere, determines two right-angled tangent vectors for
  /// this point of the unit sphere.
  void TangentsForDirection(Eigen::Vector3d direction, Eigen::Vector3d* tangent1, Eigen::Vector3d* tangent2) const;
  /// Computes the Jacobian of ComputeTangentsForDirection() wrt. the given
  /// direction.
  void TangentsJacobianWrtLineDirection(Eigen::Vector3d direction, Eigen::Matrix<double, 6, 3>* tangentsJacobianDirection) const;
};

}  // namespace aslam

#include "aslam/cameras/camera-generic-noncentral-inl.h"

#endif  // ASLAM_CAMERAS_GENERIC_NONCENTRAL_CAMERA_H_
