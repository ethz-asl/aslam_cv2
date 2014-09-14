#include <aslam/pipeline/visual-pipeline-null.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/cameras/camera.h>
#include <opencv2/core/core.hpp>

namespace aslam {

NullVisualPipeline::NullVisualPipeline(const std::shared_ptr<Camera>& camera,
                                       bool copy_image) :
   VisualPipeline(camera, camera, copy_image) { }

NullVisualPipeline::~NullVisualPipeline() { }

}  // namespace aslam
