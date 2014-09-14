#include <aslam/pipeline/visual-pipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/pipeline/undistorter.h>

#include <opencv2/core/core.hpp>

namespace aslam {

VisualPipeline::VisualPipeline(const std::shared_ptr<Camera>& input_camera,
                               const std::shared_ptr<Camera>& output_camera,
                               bool copy_images)
: input_camera_(input_camera), output_camera_(output_camera),
  copy_images_(copy_images) { }

VisualPipeline::~VisualPipeline() { }

VisualPipeline::VisualPipeline(const std::shared_ptr<Undistorter>& preprocessing,
                               bool copy_images)
: preprocessing_(preprocessing), copy_images_(copy_images) {
  CHECK_NOTNULL(preprocessing.get());
  input_camera_ = preprocessing->getInputCamera();
  output_camera_ = preprocessing->getOutputCamera();
}

std::shared_ptr<VisualFrame> VisualPipeline::processImage(
    const cv::Mat& raw_image, int64_t systemStamp, int64_t hardwareStamp) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(raw_image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(raw_image.rows));

  // \TODO(PTF) Eventually we can put timestamp correction policies in here.
  std::shared_ptr<VisualFrame> frame(new VisualFrame);
  frame->setTimestamp(systemStamp);
  frame->setSystemTimestamp(systemStamp);
  frame->setHardwareTimestamp(hardwareStamp);
  frame->setRawCameraGeometry(input_camera_);
  frame->setCameraGeometry(output_camera_);
  FrameId id;
  id.randomize();
  frame->setId( id );
  if(copy_images_) {
    frame->setRawImage(raw_image.clone());
  } else {
    frame->setRawImage(raw_image);
  }

  cv::Mat image;
  if(preprocessing_) {
    preprocessing_->processImage(raw_image, &image);
  } else {
    image = raw_image;
  }
  /// Send the image to the derived class for processing
  processFrame(image, &frame);

  return frame;
}

}  // namespace aslam
