#ifndef ASLAM_FRAMES_VISUAL_FRAME_H_
#define ASLAM_FRAMES_VISUAL_FRAME_H_

#include <memory>
#include <unordered_map>
#include <cstdint>

#include <aslam/cameras/camera.h>
#include <aslam/common/channel.h>
#include <aslam/common/channel-declaration.h>
#include <aslam/common/macros.h>
#include <aslam/common/unique-id.h>
#include <Eigen/Dense>


namespace aslam {
class Camera;

/// \class VisualFrame
/// \brief An image and keypoints from a single camera.
///
/// This class stores data from an image and keypoints taken from a single
/// camera. It stores a pointer to the camera's intrinsic calibration,
/// an id that uniquely identifies this frame, and a measurement timestamp.
///
/// The class also stores a ChannelGroup object that can be used to hold
/// keypoint data, the raw image, and other associated information.
///
/// The frame stores three timestamps. The stamp_ field stores the current
/// timestamp that is being used in processing. This can be a derived value
/// based on timestamp correction. To save the raw data, the class also stores
/// the hardware timestamp, and the system timestamp (the time the image was
/// received at the host computer).
///
/// The camera geometry stored in the frame and accessible by getCameraGeometry()
/// may refer to a transformed geometry that includes downsampling and undistortion.
/// However, we recommend to always store the raw image with the frame so that
/// no information is lost. In order to be able to plot on these raw images, we must
/// also store the original camera geometry object (accessible with
/// getRawCameraGeometry()). To plot transformed keypoints on the raw image, one
/// may use toRawImageCoordinates() or toRawImageCoordinatesVectorized() to recover
/// the raw image coordinates of any keypoint.
class VisualFrame  {
 public:
  typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> DescriptorsT;
  ASLAM_POINTER_TYPEDEFS(VisualFrame);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(VisualFrame);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VisualFrame();

  virtual ~VisualFrame() {};

  virtual bool operator==(const VisualFrame& other) const;

  template<typename CHANNEL_DATA_TYPE>
  void addChannel(const std::string& channel) {
    aslam::channels::addChannel<CHANNEL_DATA_TYPE>(channel, &channels_);
  }

  /// \brief Are there keypoint measurements stored in this frame?
  bool hasKeypointMeasurements() const;

  /// \brief Are there keypoint measurement uncertainties stored in this frame?
  bool hasKeypointMeasurementUncertainties() const;

  /// \brief Are there keypoint orientations stored in this frame?
  bool hasKeypointOrientations() const;

  /// \brief Are there keypoint scores stored in this frame?
  bool hasKeypointScores() const;

  /// \brief Are there keypoint scales stored in this frame?
  bool hasKeypointScales() const;

  /// \brief Are there descriptors stored in this frame?
  bool hasDescriptors() const;

  /// \brief Is there a raw image stored in this frame?
  bool hasRawImage() const;

  /// \brief Is a certain channel stored in this frame?
  bool hasChannel(const std::string& channel) const {
    return aslam::channels::hasChannel(channel, channels_);
  }

  /// \brief The keypoint measurements stored in a frame.
  const Eigen::Matrix2Xd& getKeypointMeasurements() const;

  /// \brief The keypoint measurement uncertainties stored in a frame.
  const Eigen::VectorXd& getKeypointMeasurementUncertainties() const;

  /// \brief The keypoint orientations stored in a frame.
  const Eigen::VectorXd& getKeypointOrientations() const;

  /// \brief The keypoint scores stored in a frame.
  const Eigen::VectorXd& getKeypointScores() const;

  /// \brief The keypoint scales stored in a frame.
  const Eigen::VectorXd& getKeypointScales() const;

  /// \brief The descriptors stored in a frame.
  const DescriptorsT& getDescriptors() const;

  /// \brief The raw image stored in a frame.
  const cv::Mat& getRawImage() const;

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

  /// \brief A pointer to the keypoint scores, can be used to swap in new data.
  Eigen::VectorXd* getKeypointScoresMutable();

  /// \brief A pointer to the keypoint scales, can be used to swap in new data.
  Eigen::VectorXd* getKeypointScalesMutable();

  /// \brief A pointer to the descriptors, can be used to swap in new data.
  DescriptorsT* getDescriptorsMutable();

  /// \brief A pointer to the raw image, can be used to swap in new data.
  cv::Mat* getRawImageMutable();

  template<typename CHANNEL_DATA_TYPE>
  CHANNEL_DATA_TYPE* getChannelDataMutable(const std::string& channel) const {
    CHANNEL_DATA_TYPE& data =
        aslam::channels::getChannelData<CHANNEL_DATA_TYPE>(channel,
                                                           channels_);
    return &data;
  }

  /// \brief Return block expression of the keypoint measurement pointed to by index.
  const Eigen::Block<Eigen::Matrix2Xd, 2, 1> getKeypointMeasurement(size_t index) const;

  /// \brief Return the keypoint measurement uncertainty index.
  double getKeypointMeasurementUncertainty(size_t index) const;

  /// \brief Return the keypoint orientation at index.
  double getKeypointOrientation(size_t index) const;

  /// \brief Return the keypoint score at index.
  double getKeypointScore(size_t index) const;

  /// \brief Return the keypoint scale at index.
  double getKeypointScale(size_t index) const;

  /// \brief Return pointer location of the descriptor pointed to by index.
  const unsigned char* getDescriptor(size_t index) const;

  /// \brief Replace (copy) the internal keypoint measurements by the passed ones.
  void setKeypointMeasurements(const Eigen::Matrix2Xd& keypoints);

  /// \brief Replace (copy) the internal keypoint measurement uncertainties
  ///        by the passed ones.
  void setKeypointMeasurementUncertainties(const Eigen::VectorXd& uncertainties);

  /// \brief Replace (copy) the internal keypoint orientations by the passed ones.
  void setKeypointOrientations(const Eigen::VectorXd& orientations);

  /// \brief Replace (copy) the internal keypoint scores by the passed ones.
  void setKeypointScores(const Eigen::VectorXd& scores);

  /// \brief Replace (copy) the internal keypoint orientations by the passed ones.
  void setKeypointScales(const Eigen::VectorXd& scales);

  /// \brief Replace (copy) the internal descriptors by the passed ones.
  void setDescriptors(const DescriptorsT& descriptors);

  /// \brief Replace (copy) the internal descriptors by the passed ones.
  void setDescriptors(const Eigen::Map<const DescriptorsT>& descriptors);

  /// \brief Replace (copy) the internal raw image by the passed ones.
  ///        This is a shallow copy by default. Please clone the image if it
  ///        should be owned by the VisualFrame.
  void setRawImage(const cv::Mat& image);

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

  /// \brief Replace (swap) the internal keypoint measurements by the passed ones.
  ///        This method creates the channel if it doesn't exist
  void swapKeypointMeasurements(Eigen::Matrix2Xd* keypoints);

  /// \brief Replace (swap) the internal keypoint measurement uncertainties
  ///        by the passed ones.
  void swapKeypointMeasurementUncertainties(Eigen::VectorXd* uncertainties);

  /// \brief Replace (swap) the internal keypoint orientations by the passed ones.
  void swapKeypointOrientations(Eigen::VectorXd* orientations);

  /// \brief Replace (swap) the internal keypoint scores by the passed ones.
  void swapKeypointScores(Eigen::VectorXd* scores);

  /// \brief Replace (swap) the internal keypoint orientations by the passed ones.
  void swapKeypointScales(Eigen::VectorXd* scales);

  /// \brief Replace (swap) the internal descriptors by the passed ones.
  void swapDescriptors(DescriptorsT* descriptors);

  /// \brief Swap channel data with the data passed in. This will only work
  ///        if the channel data type has a swap() method.
  template<typename CHANNEL_DATA_TYPE>
  void swapChannelData(const std::string& channel,
                       CHANNEL_DATA_TYPE* data_new) {
    CHECK_NOTNULL(data_new);
    if (!aslam::channels::hasChannel(channel, channels_)) {
      aslam::channels::addChannel<CHANNEL_DATA_TYPE>(channel, &channels_);
    }
    CHANNEL_DATA_TYPE& data =
        aslam::channels::getChannelData<CHANNEL_DATA_TYPE>(channel, channels_);
    data.swap(*data_new);
  }

  /// \brief The camera geometry.
  const Camera::ConstPtr getCameraGeometry() const;

  /// \brief Set the camera geometry.
  void setCameraGeometry(const Camera::ConstPtr& camera);

  /// \brief The camera geometry.
  const Camera::ConstPtr getRawCameraGeometry() const;

  /// \brief Set the camera geometry.
  void setRawCameraGeometry(const Camera::ConstPtr& camera);

  /// \brief Convert keypoint coordinates to raw image coordinates.
  ///
  /// \param[in] keypoint               The keypoint coordinates with respect to
  ///                                   the camera calibration available from
  ///                                   getCameraGeometry().
  /// \param[out] out_image_coordinates The image coordinates with respect to the
  ///                                   camera calibration available from
  ///                                   getRawCameraGeometry().
  aslam::ProjectionResult toRawImageCoordinates(const Eigen::Vector2d& keypoint,
                                                Eigen::Vector2d* out_image_coordinates);

  /// \brief Convert keypoint coordinates to raw image coordinates.
  ///
  /// \param[in] keypoints              The keypoint coordinates with respect to
  ///                                   the camera calibration available from
  ///                                   getCameraGeometry().
  /// \param[out] out_image_coordinates The image coordinates with respect to the
  ///                                   camera calibration available from
  ///                                   getRawCameraGeometry().
  /// \param[out] results               One result for each keypoint.
  void toRawImageCoordinatesVectorized(const Eigen::Matrix2Xd& keypoints,
                                       Eigen::Matrix2Xd* out_image_coordinates,
                                       std::vector<aslam::ProjectionResult>* results);

  /// \brief get the frame id.
  inline const aslam::FrameId& getId() const { return id_; }

  /// \brief set the frame id.
  inline void setId(const aslam::FrameId& id) { id_ = id; }

  /// \brief get the timestamp.
  inline int64_t getTimestamp() const { return stamp_; }
  
  /// \brief set the timestamp.
  inline void setTimestamp(int64_t stamp){ stamp_ = stamp; }
  
  /// \brief get the hardware timestamp.
  inline int64_t getHardwareTimestamp() const { return hardwareStamp_; }

  /// \brief set the hardware timestamp.
  inline void setHardwareTimestamp(int64_t stamp) { hardwareStamp_ = stamp; }

  /// \brief get the system (host computer) timestamp.
  inline int64_t getSystemTimestamp() const { return systemStamp_; }

  /// \brief set the system (host computer) timestamp.
  inline void setSystemTimestamp(int64_t stamp) { systemStamp_ = stamp; }

  /// \brief Set the size of the descriptor in bytes.
  int32_t getDescriptorSizeBytes() const { return num_bytes_descriptor_; };

  /// \brief Set the size of the descriptor in bytes.
  void setDescriptorSizeBytes(size_t num_bytes) { num_bytes_descriptor_= num_bytes; };

  /// \brief print out a human-readable version of this frame
  void print(std::ostream& out, const std::string& label) const;
 private:
  /// \brief integer nanoseconds since epoch.
  int64_t stamp_;
  /// hardware timestamp. The scale and offset will be different for every device.
  int64_t hardwareStamp_;
  /// \brief host system timestamp in integer nanoseconds since epoch.
  int64_t systemStamp_;
  /// \brief Descriptor size in bytes.
  size_t num_bytes_descriptor_;
  aslam::FrameId id_;
  aslam::channels::ChannelGroup channels_;
  Camera::ConstPtr camera_geometry_;
  Camera::ConstPtr raw_camera_geometry_;
};

inline std::ostream& operator<<(std::ostream& out, const VisualFrame& rhs) {
  rhs.print(out, "");
  return out;
}
}  // namespace aslam
#endif  // ASLAM_FRAMES_VISUAL_FRAME_H_
