///
/// \file visual-pipeline.h
///
/// \date   Sep 2, 2014
/// \author paul.furgale@gmail.com
///

#ifndef VISUAL_PIPELINE_H_
#define VISUAL_PIPELINE_H_

namespace aslam {

/// \class VisualPipeline
/// \brief An interface for pipelines that turn images into VisualNFrames
///
/// This is the abstract interface for visual pipelines that turn raw images
/// into VisualNFrame data. The underlying pipeline may include undistortion
/// or rectification, image constrast enhancement, feature detection and
/// descriptor computation, or other operations.
///
/// The class has an input NCameras calibration struct that represents the
/// intrinsic and extrinsic calibration of the raw camera system. The output
/// NCameras struct represents the calibration parameters of the images and
/// keypoints that are in the VisualNFrames struct. This is the calibration after
/// undistortion, etc.
///
/// The class should synchronize images with nearby timestamps and handle
/// out-of-order images. When all frames of a VisualNFrame are complete,
/// they are added to a list of output frames in the order that they are
/// completed. This list should be sorted by time (oldest first) and the number
/// of elements can be queried by numVisualNFramesComplete(). The getNext()
/// function retrieves the oldest complete VisualNFrames and leaves the remaining.
/// The getLatestAndClear() function gets the newest VisualNFrames and discards
/// anything older.
class VisualPipeline {
public:
  VisualPipeline();
  virtual ~VisualPipeline();

  /// \brief Add an image to the visual pipeline
  ///
  /// This function is called by a user when an image is received.
  /// The pipeline then processes the images and constructs VisualNFrames
  /// call numVisualNFramesComplete() to find out how many VisualNFrames are
  /// completed.
  ///
  /// \param[in] cameraIndex The index of the camera that this image corresponds to
  /// \param[in] image the image data
  /// \param[in] systemStamp the host time in integer nanoseconds since epoch
  /// \param[in] hardwareStamp the camera's hardware timestamp. Can be set to "invalid".
  virtual void processImage(int cameraIndex,
                            const cv::Mat& image,
                            int64_t systemStamp,
                            int64_t hardwareStamp) = 0;


  /// \brief How many completed VisualNFrames are waiting to be retrieved?
  virtual size_t numVisualNFramesComplete() const = 0;

  /// \brief  Get the next available set of processed frames
  ///
  /// This may not be the latest data, it is simply the next in a FIFO queue.
  /// If there are no VisualNFrames waiting, this returns a NULL pointer.
  virtual std::shared_ptr<VisualNFrame> getNext() = 0;

  /// \brief Get the latest available data and clear anything older.
  ///
  /// If there are no VisualNFrames waiting, this returns a NULL pointer.
  virtual std::shared_ptr<VisualNFrame> getLatestAndClear() = 0;

  /// \brief Get the input camera system that corresponds to the images
  ///        passed in to processImage().
  ///
  /// Because this pipeline may do things like image undistortion or
  /// rectification, the input and output camera systems may not be the same.
  virtual std::shared_ptr<NCameras> getInputCameraSystem() const = 0;

  /// \brief Get the output camera system that corresponds to the VisualNFrame
  ///        data that comes out.
  ///
  /// Because this pipeline may do things like image undistortion or
  /// rectification, the input and output camera systems may not be the same.
  virtual std::shared_ptr<NCameras> getOutputcameraSystem() const = 0;

private:
};

}  // namespace aslam

#endif // VISUAL_PIPELINE_H_ 
