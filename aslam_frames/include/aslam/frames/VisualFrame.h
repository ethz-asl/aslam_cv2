#ifndef ASLAM_FRAMES_VISUAL_FRAME_H_
#define ASLAM_FRAMES_VISUAL_FRAME_H_

#include <memory>
#include <unordered_map>

#include <aslam/cameras/Camera.h>
#include <aslam/common/channel.h>
#include <aslam/common/macros.h>
#include <Eigen/Dense>

namespace aslam {
class Camera;

class VisualFrame  {
 public:
  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> DescriptorsT;
  ASLAM_POINTER_TYPEDEFS(VisualFrame);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  virtual bool operator==(const VisualFrame& other) const;
  /// \brief The keypoint measurements stored in a frame.
  const Eigen::Matrix2Xd& getKeypointMeasurements() const;
  /// \brief The keypoint measurement uncertainties stored in a frame.
  const Eigen::VectorXd& getKeypointMeasurementUncertainties() const;
  /// \brief The keypoint orientations stored in a frame.
  const Eigen::VectorXd& getKeypointOrientations() const;
  /// \brief The keypoint scales stored in a frame.
  const Eigen::VectorXd& getKeypointScales() const;
  /// \brief The descriptors stored in a frame.
  const DescriptorsT& getBriskDescriptors() const;

  /// \brief A pointer to the keypoint measurements, can be used to swap in new data.
  Eigen::Matrix2Xd* getKeypointMeasurementsMutable();
  /// \brief A pointer to the keypoint measurement uncertainties, can be used to swap in new data.
  Eigen::VectorXd* getKeypointMeasurementUncertaintiesMutable();
  /// \brief A pointer to the keypoint orientations, can be used to swap in new data.
  Eigen::VectorXd* getKeypointOrientationsMutable();
  /// \brief A pointer to the keypoint scales, can be used to swap in new data.
  Eigen::VectorXd* getKeypointScalesMutable();
  /// \brief A pointer to the descriptors, can be used to swap in new data.
  DescriptorsT* getBriskDescriptorsMutable();


  /// \brief Return block expression of the keypoint measurement pointed to by index.
  const Eigen::Block<Eigen::Matrix2Xd, 2, 1> getKeypointMeasurement(
      size_t index) const;
  /// \brief Return block expression of the keypoint measurement uncertainty to
  //         by index.
  double getKeypointMeasurementUncertainty(size_t index) const;
  /// \brief Return block expression of the keypoint orientation.
  double getKeypointOrientation(size_t index) const;
  /// \brief Return block expression of the keypoint scale to by index.
  double getKeypointScale(size_t index) const;
  /// \brief Return pointer location of the descriptor pointed to by index.
  const char* getBriskDescriptor(size_t index) const;


  /// \brief Replace (copy) the internal keypoint measurements by the passed ones.
  const void setKeypointMeasurements(const Eigen::Matrix2Xd& keypoints);
  /// \brief Replace (copy) the internal keypoint measurement uncertainties
  ///        by the passed ones.
  const void setKeypointMeasurementUncertainties(
      const Eigen::VectorXd& uncertainties);
  /// \brief Replace (copy) the internal keypoint orientations by the passed ones.
  const void setKeypointOrientations(const Eigen::VectorXd& orientations);
  /// \brief Replace (copy) the internal keypoint orientations by the passed ones.
  const void setKeypointScales(const Eigen::VectorXd& scales);
  /// \brief Replace (copy) the internal descriptors by the passed ones.
  const void setBriskDescriptors(const DescriptorsT& descriptors);

  /// \brief The camera geometry.
  const Camera::ConstPtr getCameraGeometry() const;
  /// \brief Set the camera geometry.
  void setCameraGeometry(const Camera::Ptr& camera);

 private:
  aslam::channels::ChannelGroup channels_;
  Camera::Ptr camera_geometry_;
};
}  // namespace aslam
#endif  // ASLAM_FRAMES_VISUAL_FRAME_H_
