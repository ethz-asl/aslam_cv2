#ifndef NULL_VISUAL_PIPELINE_H
#define NULL_VISUAL_PIPELINE_H

#include "visual-pipeline.h"

namespace aslam {

/// \class NullVisualPipeline
/// \brief A visual pipeline that does not transform the image.
class NullVisualPipeline : public VisualPipeline {
public:
  ASLAM_POINTER_TYPEDEFS(NullVisualPipeline);
  ASLAM_DISALLOW_EVIL_CONSTRUCTORS(NullVisualPipeline);

  /// \param[in] camera The camera that produces the images.
  /// \param[in] copyImages If true, images passed in are cloned before storing
  ///                       in the frame.
  NullVisualPipeline(const std::shared_ptr<Camera>& camera, bool copyImages);

  virtual ~NullVisualPipeline();

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
                                                    int64_t hardwareStamp) const;

  /// \brief Get the input camera that corresponds to the image
  ///        passed in to processImage().
  ///
  /// Because this processor may do things like image undistortion or
  /// rectification, the input and output camera may not be the same.
  virtual const std::shared_ptr<Camera>& getInputCamera() const;

  /// \brief Get the output camera that corresponds to the VisualFrame
  ///        data that comes out.
  ///
  /// Because this pipeline may do things like image undistortion or
  /// rectification, the input and output camera may not be the same.
  virtual const std::shared_ptr<Camera>& getOutputCamera() const;

private:
  std::shared_ptr<Camera> camera_;
  bool copyImages_;
};

}  // namespace aslam

#endif // NULL_VISUAL_PIPELINE_H
