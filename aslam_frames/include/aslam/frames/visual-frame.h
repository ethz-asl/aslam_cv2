#ifndef ASLAM_FRAMES_VISUAL_FRAME_H_
#define ASLAM_FRAMES_VISUAL_FRAME_H_

#include <memory>
#include <unordered_map>

#include <aslam/cameras/camera.h>
#include <aslam/common/channel.h>
#include <aslam/common/channel-declaration.h>
#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <Eigen/Dense>
#include <cstdint>

namespace aslam {
class Camera;

class VisualFrame  {
 public:
  typedef Eigen::Matrix<char, Eigen::Dynamic, Eigen::Dynamic> DescriptorsT;
  ASLAM_POINTER_TYPEDEFS(VisualFrame);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  VisualFrame();
  virtual ~VisualFrame();

  virtual bool operator==(const VisualFrame& other) const;

  template<typename CHANNEL_DATA_TYPE>
  void addChannel(const std::string& channel) {
    aslam::channels::addChannel<CHANNEL_DATA_TYPE>(channel, &channels_);
  }

  /// \brief Are there keypoint measurements stored in this frame.
  bool hasKeypointMeasurements() const;
  /// \brief Are there keypoint measurement uncertainties stored in this frame.
  bool hasKeypointMeasurementUncertainties() const;
  /// \brief Are there keypoint orientations stored in this frame.
  bool hasKeypointOrientations() const;
  /// \brief Are there keypoint scales stored in this frame.
  bool hasKeypointScales() const;
  /// \brief Are there descriptors stored in this frame.
  bool hasBriskDescriptors() const;

  /// \brief Is a certain channel stored in this frame.
  bool hasChannel(const std::string& channel) const {
    return aslam::channels::hasChannel(channel, channels_);
  }

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

  template<typename CHANNEL_DATA_TYPE>
  const CHANNEL_DATA_TYPE& getChannelData(const std::string& channel) const {
    return aslam::channels::getChannelData<CHANNEL_DATA_TYPE>(channel, channels_);
  }

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

  template<typename CHANNEL_DATA_TYPE>
  CHANNEL_DATA_TYPE* getChannelDataMutable(const std::string& channel) const {
    CHANNEL_DATA_TYPE& data =
        aslam::channels::getChannelData<CHANNEL_DATA_TYPE>(channel,
                                                           channels_);
    return &data;
  }

  /// \brief Return block expression of the keypoint measurement pointed to by index.
  const Eigen::Block<Eigen::Matrix2Xd, 2, 1> getKeypointMeasurement(
      size_t index) const;
  /// \brief Return the keypoint measurement uncertainty index.
  double getKeypointMeasurementUncertainty(size_t index) const;
  /// \brief Return the keypoint orientation at index.
  double getKeypointOrientation(size_t index) const;
  /// \brief Return the keypoint scale at index.
  double getKeypointScale(size_t index) const;
  /// \brief Return pointer location of the descriptor pointed to by index.
  const char* getBriskDescriptor(size_t index) const;

  /// \brief Replace (copy) the internal keypoint measurements by the passed ones.
  void setKeypointMeasurements(const Eigen::Matrix2Xd& keypoints);
  /// \brief Replace (copy) the internal keypoint measurement uncertainties
  ///        by the passed ones.
  void setKeypointMeasurementUncertainties(
      const Eigen::VectorXd& uncertainties);
  /// \brief Replace (copy) the internal keypoint orientations by the passed ones.
  void setKeypointOrientations(const Eigen::VectorXd& orientations);
  /// \brief Replace (copy) the internal keypoint orientations by the passed ones.
  void setKeypointScales(const Eigen::VectorXd& scales);
  /// \brief Replace (copy) the internal descriptors by the passed ones.
  void setBriskDescriptors(const DescriptorsT& descriptors);

  template<typename CHANNEL_DATA_TYPE>
  void setChannelData(const std::string& channel,
                      const CHANNEL_DATA_TYPE& data_new) {
    if (!aslam::channels::hasChannel(channel, channels_)) {
      aslam::channels::addChannel<CHANNEL_DATA_TYPE>(channel, &channels_);
    }
    CHANNEL_DATA_TYPE& data =
        aslam::channels::getChannelData<CHANNEL_DATA_TYPE>(channel, channels_);
    data = data_new;
  }

  /// \brief The camera geometry.
  const Camera::ConstPtr getCameraGeometry() const;
  /// \brief Set the camera geometry.
  void setCameraGeometry(const Camera::Ptr& camera);

  /// \brief get the frame id.
  aslam::FrameId getId() const { return id_; }

  /// \brief set the frame id.
  void setId(aslam::FrameId id) { id_ = id; }

  /// \brief get the timestamp
  uint64_t getTimestamp() const{ return stamp_; }
  
  /// \brief set the timestamp
  void setTimestamp(uint64_t stamp){ stamp_ = stamp; }
  
 private:
  /// integer nanosecond timestamp
  uint64_t stamp_;
  aslam::FrameId id_;
  aslam::channels::ChannelGroup channels_;
  Camera::Ptr camera_geometry_;
};
}  // namespace aslam
#endif  // ASLAM_FRAMES_VISUAL_FRAME_H_
