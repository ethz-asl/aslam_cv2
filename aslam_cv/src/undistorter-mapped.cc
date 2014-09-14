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
                                     int interpolation)
: Undistorter(input_camera, output_camera), map_x_(map_x), map_y_(map_y),
  interpolation_(interpolation) { }

MappedUndistorter::~MappedUndistorter() { }

void MappedUndistorter::undistortImage(const cv::Mat& input_image,
                                 cv::Mat* output_image) const {
  CHECK_EQ(input_camera_->imageWidth(), static_cast<size_t>(input_image.cols));
  CHECK_EQ(input_camera_->imageHeight(), static_cast<size_t>(input_image.rows));
  CHECK_NOTNULL(output_image);
  cv::remap(input_image, *output_image, map_x_, map_y_, interpolation_);
}

}  // namespace aslam
