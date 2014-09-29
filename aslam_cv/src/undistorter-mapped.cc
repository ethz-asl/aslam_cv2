#include <aslam/pipeline/undistorter-mapped.h>
#include <aslam/frames/visual-frame.h>
#include <glog/logging.h>
#include <opencv2/imgproc/imgproc.hpp> // cv::remap

namespace aslam {

MappedUndistorter::MappedUndistorter()
    : interpolation_method_(aslam::InterpolationMethod::Linear) {}

MappedUndistorter::MappedUndistorter(Camera::Ptr input_camera, Camera::Ptr output_camera,
                                     const cv::Mat& map_u, const cv::Mat& map_v,
                                     aslam::InterpolationMethod interpolation)
: Undistorter(input_camera, output_camera), map_u_(map_u), map_v_(map_v),
  interpolation_method_(interpolation) {
  CHECK_EQ(static_cast<size_t>(map_u_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_u_.cols), output_camera->imageWidth());
  CHECK_EQ(static_cast<size_t>(map_v_.rows), output_camera->imageHeight());
  CHECK_EQ(static_cast<size_t>(map_v_.cols), output_camera->imageWidth());
}

void MappedUndistorter::processImage(const cv::Mat& input_image, cv::Mat* output_image) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(input_image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(input_image.rows));
  CHECK_NOTNULL(output_image);
  cv::remap(input_image, *output_image, map_u_, map_v_, static_cast<int>(interpolation_method_));
}

}  // namespace aslam
