#include <aslam/pipeline/visual-pipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/frames/visual-frame.h>

#include <opencv2/core/core.hpp>

namespace aslam {

VisualPipeline::VisualPipeline(const std::shared_ptr<Camera>& input_camera,
                               const std::shared_ptr<Camera>& output_camera)
: input_camera_(input_camera), output_camera_(output_camera) { }

VisualPipeline::~VisualPipeline() { }

std::shared_ptr<VisualFrame> VisualPipeline::processImage(
    const cv::Mat& image, int64_t systemStamp, int64_t hardwareStamp) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(image.rows));

  // \TODO(PTF) Eventually we can put timestamp correction policies in here.
  std::shared_ptr<VisualFrame> frame(new VisualFrame);
  frame->setTimestamp(systemStamp);
  frame->setSystemTimestamp(systemStamp);
  frame->setHardwareTimestamp(hardwareStamp);
  frame->setCameraGeometry(output_camera_);
  FrameId id;
  id.randomize();
  frame->setId( id );

  /// Send the image to the derived class for processing
  processFrame(image, &frame);

  return frame;
}

}  // namespace aslam
