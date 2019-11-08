#ifndef ASLAM_CAMERAS_LIDAR_CAMERA_H_
#define ASLAM_CAMERAS_LIDAR_CAMERA_H_

#include <aslam/cameras/camera.h>
#include <aslam/cameras/distortion.h>
#include <aslam/common/crtp-clone.h>
#include <aslam/common/macros.h>
#include <aslam/common/types.h>

namespace aslam {

class MappedUndistorter;
class NCamera;

class LidarCamera : public aslam::Cloneable<Camera, LidarCamera> {
  friend class NCamera;
  enum { kNumOfParams = 0 };

 public:
  ASLAM_POINTER_TYPEDEFS(LidarCamera);

  enum { CLASS_SERIALIZATION_VERSION = 1 };
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 public:
  /// \brief Empty constructor for serialization interface.
  LidarCamera();

  /// Copy constructor for clone operation.
  LidarCamera(const LidarCamera& other) = default;
  void operator=(const LidarCamera&) = delete;

 public:
  LidarCamera(uint32_t image_width, uint32_t image_height);

  virtual ~LidarCamera() = default;

  friend std::ostream& operator<<(std::ostream& out, const LidarCamera& camera);

  virtual bool backProject3(const Eigen::Ref<const Eigen::Vector2d>& keypoint,
                            Eigen::Vector3d* out_point_3d) const;

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
  /// \brief Returns the number of intrinsic parameters used in this camera
  /// model.
  inline static constexpr int parameterCount() { return kNumOfParams; }

  /// \brief Returns the number of intrinsic parameters used in this camera
  /// model.
  inline virtual int getParameterSize() const { return kNumOfParams; }

  /// Static function that checks whether the given intrinsic parameters are
  /// valid for this model.
  static bool areParametersValid(const Eigen::VectorXd& parameters);

  /// Function to check whether the given intrinsic parameters are valid for
  /// this model.
  virtual bool intrinsicsValid(const Eigen::VectorXd& intrinsics) const;

  /// Print the internal parameters of the camera in a human-readable form
  /// Print to the ostream that is passed in. The text is extra
  /// text used by the calling function to distinguish cameras
  virtual void printParameters(std::ostream& out,
                               const std::string& text) const;

  /// \brief Create a test camera object for unit testing.
  template <typename DistortionType>
  static LidarCamera::Ptr createTestCamera() {
    Distortion::UniquePtr distortion = DistortionType::createTestDistortion();
    LidarCamera::Ptr camera(new LidarCamera(320, 1024));
    CameraId id;
    generateId(&id);
    camera->setId(id);
    return camera;
  }

  /// \brief Create a test camera object for unit testing. (without distortion)
  static LidarCamera::Ptr createTestCamera();

 private:
  bool isValidImpl() const override;
  void setRandomImpl() override;
  bool isEqualImpl(const Sensor& other, const bool verbose) const override;
};

}  // namespace aslam

#endif  // ASLAM_CAMERAS_PINHOLE_CAMERA_H_
