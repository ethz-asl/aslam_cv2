#ifndef VISUAL_PROCESSOR_H
#define VISUAL_PROCESSOR_H

#include <memory>

namespace cv { class Mat; }

namespace aslam {

class VisualFrame;
class Camera;

/// \class VisualPipeline
/// \brief An interface for processors that turn images into VisualFrames
///
/// This is the abstract interface for visual processors that turn raw images
/// into VisualFrame data. The underlying pipeline may include undistortion
/// or rectification, image contrast enhancement, feature detection and
/// descriptor computation, or other operations.
///
/// The class has two Camera calibration structs that represent the
/// extrinsic calibration of the camera.
/// The "input" calibration (getInputCamera()) represents the calibration of
/// raw camera, before any image processing, resizing, or undistortion
/// has taken place. The "output" calibration (getOutputCamera())
/// represents the calibration parameters of the images and keypoints that get
/// set in the VisualFrame struct. These are the camera parameters after
/// image processing, resizing, undistortion, etc.
class VisualPipeline {
public:
  VisualPipeline();
  ~VisualPipeline();

  /// \brief Add an image to the visual processor
  ///
  /// This function is called by a user when an image is received.
  /// The processor then processes the images and constructs a VisualFrame.
  ///
  /// \param[in] Image the image data.
  /// \param[in] SystemStamp the host time in integer nanoseconds since epoch.
  /// \param[in] HardwareStamp the camera's hardware timestamp. Can be set to "invalid".
  /// \returns   The visual frame built from the image data.
  virtual std::shared_ptr<VisualFrame> processImage(const cv::Mat& image,
                                                    int64_t systemStamp,
                                                    int64_t hardwareStamp) = 0;

  /// \brief Get the input camera that corresponds to the image
  ///        passed in to processImage().
  ///
  /// Because this processor may do things like image undistortion or
  /// rectification, the input and output camera may not be the same.
  virtual std::shared_ptr<Camera> getInputCamera() const = 0;

  /// \brief Get the output camerathat corresponds to the VisualFrame
  ///        data that comes out.
  ///
  /// Because this pipeline may do things like image undistortion or
  /// rectification, the input and output camera may not be the same.
  virtual std::shared_ptr<Camera> getOutputCamera() const = 0;
};

}  // namespace aslam

#endif // VISUAL_PROCESSOR_H
