#include <aslam/pipeline/visual-pipeline-null.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/cameras/camera.h>
#include <opencv2/core/core.hpp>

namespace aslam {

NullVisualPipeline::NullVisualPipeline(const std::shared_ptr<Camera>& camera,
                                       bool copyImages) :
    camera_(camera), copyImages_(copyImages) {
  CHECK_NOTNULL(camera_.get());
}

NullVisualPipeline::~NullVisualPipeline() { }

std::shared_ptr<VisualFrame> NullVisualPipeline::processImage(
    const cv::Mat& image, int64_t systemStamp, int64_t hardwareStamp) const {
  CHECK_EQ(camera_->imageWidth(), static_cast<size_t>(image.cols));
  CHECK_EQ(camera_->imageHeight(), static_cast<size_t>(image.rows));

  std::shared_ptr<VisualFrame> frame(new VisualFrame);
  if(copyImages_){
    frame->setRawImage(image.clone());
  } else {
    frame->setRawImage(image);
  }
  frame->setTimestamp(systemStamp);
  frame->setSystemTimestamp(systemStamp);
  frame->setHardwareTimestamp(hardwareStamp);
  frame->setCameraGeometry(camera_);

  FrameId id;
  id.randomize();
  frame->setId( id );
  return frame;
}

const std::shared_ptr<Camera>& NullVisualPipeline::getInputCamera() const {
  return camera_;
}

const std::shared_ptr<Camera>& NullVisualPipeline::getOutputCamera() const {
  return camera_;
}

}  // namespace aslam
