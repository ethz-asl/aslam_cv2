#ifndef ASLAM_CAMERAS_GENERIC_CAMERA_H_
#define ASLAM_CAMERAS_GENERIC_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/crtp-clone.h>
#include <aslam/common/macros.h>
#include <aslam/common/types.h>

namespace aslam {

// Forward declarations.
class MappedUndistorter;
class NCamera;

/// \class GenericCamera
/// \brief An implementation of the generic camera model.
///
///
///  Reference: 
class GenericCamera : public aslam::Cloneable<Camera, GenericCamera> {
  friend class NCamera;

  enum { kNumOfParams = 6 };

 public:
  ASLAM_POINTER_TYPEDEFS(GenericCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  enum Parameters { 
    kCalibrationMinX = 0,
    kCalibrationMinY = 1,
    kCalibrationMaxX = 2,
    kCalibrationMaxY = 3,
    kGridWidth = 4,
    kGridHeight = 5,
  };

  // TODO(slynen) Enable commented out PropertyTree support
  // GenericCamera(const sm::PropertyTree& config);

  //////////////////////////////////////////////////////////////
  /// \name Constructors/destructors and operators
  /// @{

 public:
  /// \brief Empty constructor for serialization interface.
  GenericCamera();

  /// Copy constructor for clone operation.
  GenericCamera(const GenericCamera& other) = default;
  void operator=(const GenericCamera&) = delete;

 public:
  /// \brief Construct a GenericCamera while supplying distortion. Distortion is removed.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  /// @param[in] distortion   Pointer to the distortion model.
  GenericCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height,
                aslam::Distortion::UniquePtr& distortion);
                
  /// \brief Construct a GenericCamera without distortion.
  /// @param[in] intrinsics   Vector containing the intrinsic parameters.
  /// @param[in] image_width  Image width in pixels.
  /// @param[in] image_height Image height in pixels.
  GenericCamera(const Eigen::VectorXd& intrinsics, uint32_t image_width, uint32_t image_height);

  /// \brief Construct a GenericCamera while supplying distortion. Distortion is removed.
  /// @param[in] calibration_min_x
  /// @param[in] calibration_min_y
  /// @param[in] calibration_max_x
  /// @param[in] calibration_max_y
  /// @param[in] grid_width
  /// @param[in] grid_height
  /// @param[in] image_width      Image width in pixels.
  /// @param[in] image_height     Image height in pixels.
  /// @param[in] distortion       Pointer to the distortion model.
  GenericCamera(double calibration_min_x, double calibration_min_y,
                double calibration_max_x, double calibration_max_y,
                double grid_width, double grid_height,
                uint32_t image_width, uint32_t image_height,
                aslam::Distortion::UniquePtr& distortion);

  /// \brief Construct a GenericCamera without distortion.
  /// @param[in] calibration_min_x
  /// @param[in] calibration_min_y
  /// @param[in] calibration_max_x
  /// @param[in] calibration_max_y
  /// @param[in] grid_width
  /// @param[in] grid_height
  /// @param[in] image_width      Image width in pixels.
  /// @param[in] image_height     Image height in pixels.
  GenericCamera(double calibration_min_x, double calibration_min_y,
                double calibration_max_x, double calibration_max_y,
                double grid_width, double grid_height,
                uint32_t image_width, uint32_t image_height);

  virtual ~GenericCamera() {};

  /// \brief Convenience function to print the state using streams.
  friend std::ostream& operator<<(std::ostream& out, const GenericCamera& camera);

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
  virtual bool backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                            Eigen::Vector3d* out_point_3d) const;

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

  const ProjectionResult project3Functional(
    const Eigen::Ref<const Eigen::Vector3d>& point_3d,
    Eigen::Vector2d* out_keypoint,
    Eigen::Matrix<double, 2, 3>* out_jacobian_point) const;

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
  inline static constexpr int parameterCount() {
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
  static GenericCamera::Ptr createTestCamera() {
    return GenericCamera::Ptr(
        std::move(createTestCameraUnique<DistortionType>()));
  }

  /// \brief Create a test camera object for unit testing.
  template <typename DistortionType>
  static GenericCamera::UniquePtr createTestCameraUnique() {
    Distortion::UniquePtr distortion = DistortionType::createTestDistortion();
    GenericCamera::UniquePtr camera(
        new GenericCamera(15, 15, 736, 464, 16, 11, 640, 480, distortion));
    CameraId id;
    generateId(&id);
    camera->setId(id);
    return std::move(camera);
  }

  /// \brief Create a test camera object for unit testing. (without distortion)
  static GenericCamera::Ptr createTestCamera();

  /// \brief return the first value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> firstGridValue() const { return grid_[0][0]; };
  /// \brief return the last value of the grid for unit testing.
  Eigen::Matrix<double, 3, 1> lastGridValue() const { return grid_[gridHeight()-1][gridWidth()-1]; };

  // position of the pixel expressed in gridpoints
  Eigen::Vector2d transformImagePixelToGridPoint(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const;
  Eigen::Vector2d transformGridPointToImagePixel(const Eigen::Vector2d& gridpoint) const; // <- inverse of transformImagePixelToGridPoint

  double pixelScaleToGridScaleX(double length) const;
  double pixelScaleToGridScaleY(double length) const;

  // mainly for testing
  Eigen::Vector3d valueAtGridpoint(const Eigen::Vector2d gridpoint) const;

 private:
  /// \brief Minimal depth for a valid projection.
  static const double kMinimumDepth;

  bool isValidImpl() const override;
  void setRandomImpl() override;
  bool isEqualImpl(const Sensor& other, const bool verbose) const override;


  bool isInCalibratedArea(const Eigen::Ref<const Eigen::Vector2d>& keypoint) const;

  void interpolateCubicBSplineSurface(Eigen::Vector2d keypoint, Eigen::Vector3d* out_point_3d) const;
  void interpolateCubicBSpline(Eigen::Vector3d a, Eigen::Vector3d b, Eigen::Vector3d c, Eigen::Vector3d d, double frac_y, Eigen::Vector3d* out_point_3d) const;

  bool loadFromYamlNodeImpl(const YAML::Node&) override;
  void saveToYamlNodeImpl(YAML::Node*) const override;

  void CentralGenericBSpline_Unproject_ComputeResidualAndJacobian(double frac_x, double frac_y, Eigen::Matrix<double, 3, 1> p[4][4], Eigen::Matrix<double, 3, 1>* result, Eigen::Matrix<double, 3, 2>* dresult_dxy) const;

    /// Vector containing the grid of the model.
  std::vector<std::vector<Eigen::Matrix<double, 3, 1>>> grid_; // TODO(beni) double to template?
};

}  // namespace aslam

#include "aslam/cameras/camera-generic-inl.h"

#endif  // ASLAM_CAMERAS_GENERIC_CAMERA_H_
