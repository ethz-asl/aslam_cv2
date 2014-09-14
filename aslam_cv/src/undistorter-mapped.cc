#include <aslam/pipeline/undistorter-mapped.h>
#include <aslam/cameras/camera.h>
#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp> // cv::remap

namespace aslam {

MappedUndistorter::MappedUndistorter(){ }

MappedUndistorter::MappedUndistorter(const std::shared_ptr<Camera>& input_camera,
                                     const std::shared_ptr<Camera>& output_camera,
                                     const cv::Mat& map_x, const cv::Mat& map_y,
                                     int interpolation_method)
: Undistorter(input_camera, output_camera), map_x_(map_x), map_y_(map_y),
  interpolation_method_(interpolation_method) {
  CHECK_EQ(static_cast<size_t>(map_x_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_x_.cols), output_camera->imageWidth());
  CHECK_EQ(static_cast<size_t>(map_y_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_y_.cols), output_camera->imageWidth());
}

MappedUndistorter::~MappedUndistorter() { }

void MappedUndistorter::undistortImage(const cv::Mat& input_image,
                                       cv::Mat* output_image) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(input_image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(input_image.rows));
  CHECK_NOTNULL(output_image);
  cv::remap(input_image, *output_image, map_x_, map_y_, interpolation_method_);
}

}  // namespace aslam
