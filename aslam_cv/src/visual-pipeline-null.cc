#include <aslam/pipeline/visual-pipeline-null.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/cameras/camera.h>
#include <opencv2/core/core.hpp>

namespace aslam {

NullVisualPipeline::NullVisualPipeline(const std::shared_ptr<Camera>& camera,
                                       bool copyImages) :
   VisualPipeline(camera, camera), copyImages_(copyImages) { }

NullVisualPipeline::~NullVisualPipeline() { }

void NullVisualPipeline::processFrame(const cv::Mat& image,
                                      std::shared_ptr<VisualFrame>* frame) const {
  CHECK_NOTNULL(frame);
  CHECK_NOTNULL(frame->get());
  if(copyImages_){
    (*frame)->setImage(image.clone());
  } else {
    (*frame)->setImage(image);
  }
}
}  // namespace aslam
