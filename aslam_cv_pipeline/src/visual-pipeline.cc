#include <aslam/pipeline/visual-pipeline.h>

#include <aslam/cameras/camera.h>
#include <aslam/frames/visual-frame.h>
#include <aslam/pipeline/undistorter.h>

#include <opencv2/core/core.hpp>
#include <sensor_msgs/image_encodings.h>

DECLARE_bool(map_builder_save_color_image_as_resources);

namespace aslam {

VisualPipeline::VisualPipeline(const Camera::ConstPtr& input_camera,
                               const Camera::ConstPtr& output_camera, bool copy_images)
: input_camera_(input_camera), output_camera_(output_camera),
  copy_images_(copy_images) {
  CHECK(input_camera);
  CHECK(output_camera);
}


VisualPipeline::VisualPipeline(std::unique_ptr<Undistorter>& preprocessing, bool copy_images)
: preprocessing_(std::move(preprocessing)),
  copy_images_(copy_images) {
  CHECK_NOTNULL(preprocessing_.get());
  input_camera_ = preprocessing_->getInputCameraShared();
  output_camera_ = preprocessing_->getOutputCameraShared();
}

std::shared_ptr<VisualFrame> VisualPipeline::processImage(const cv::Mat& raw_image,
                                                          int64_t timestamp, const std::string& encoding) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(raw_image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(raw_image.rows));

  // \TODO(PTF) Eventually we can put timestamp correction policies in here.
  std::shared_ptr<VisualFrame> frame(new VisualFrame);
  frame->setTimestampNanoseconds(timestamp);
  frame->setRawCameraGeometry(input_camera_);
  frame->setCameraGeometry(output_camera_);
  FrameId id;
  generateId(&id);
  frame->setId(id);

  if(FLAGS_map_builder_save_color_image_as_resources) {
    // Check if raw_image needs to be greyscaled and if color image is saved.
    if(encoding != sensor_msgs::image_encodings::MONO8 && encoding != sensor_msgs::image_encodings::MONO16) {
      VLOG(1) << "Recieved color image with encoding " << encoding;
      cv::Mat color_image = raw_image.clone();
      cv::cvtColor(raw_image, raw_image, cv::COLOR_BGR2GRAY);
      // The setting copy_images_ has no effect as the images are always copied.
      frame->setColorImage(color_image);
    }
  } else {
    CHECK_EQ(encoding, sensor_msgs::image_encodings::MONO8);
  }
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
  processFrameImpl(image, frame.get());

  return frame;
}

}  // namespace aslam
