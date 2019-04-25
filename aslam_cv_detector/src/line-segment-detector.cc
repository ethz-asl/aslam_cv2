#include "aslam/detectors/line-segment-detector.h"

#include <memory>

#include <Eigen/Core>
#include <glog/logging.h>
#include <lsd/lsd-opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

DEFINE_uint64(
    min_segment_length_px, 20u, "Minimum line segment length in pixels.");
DEFINE_double(
    image_scale, 0.8,
    "Scale of image that will be used to find the lines. Range (0..1].");
DEFINE_double(
    scaled_gaussian_filter_sigma, 0.6,
    "Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.");
DEFINE_double(
    gradient_norm_quantization_error_bound, 2.0,
    "Bound to the quantization error on the gradient norm.");
DEFINE_double(
    gradient_angle_threshold_deg, 16.5, "Gradient angle tolerance in degrees.");
DEFINE_double(
    detection_threshold_log_eps, 0.0,
    "Detection threshold: -log10(NFA) > _log_eps.");
DEFINE_double(
    aligned_region_points_in_rectangle_min_density_threshold, 0.65,
    "Minimal density of aligned region points in rectangle.");
DEFINE_uint64(
    gradient_modulus_num_bins, 1024u,
    "Number of bins in pseudo-ordering of gradient modulus.");

namespace aslam {

LineSegmentDetector::Options::Options()
    : min_segment_length_px(FLAGS_min_segment_length_px),
      image_scale(FLAGS_image_scale),
      scaled_gaussian_filter_sigma(FLAGS_scaled_gaussian_filter_sigma),
      gradient_norm_quantization_error_bound(
          FLAGS_gradient_norm_quantization_error_bound),
      gradient_angle_threshold_deg(FLAGS_gradient_angle_threshold_deg),
      detection_threshold_log_eps(FLAGS_detection_threshold_log_eps),
      aligned_region_points_in_rectangle_min_density_threshold(
          FLAGS_aligned_region_points_in_rectangle_min_density_threshold),
      gradient_modulus_num_bins(FLAGS_gradient_modulus_num_bins) {}

LineSegmentDetector::LineSegmentDetector(const Options& options)
    : options_(options) {
  line_detector_ = aslamcv::createLineSegmentDetectorPtr(
      cv::LSD_REFINE_STD, options.image_scale,
      options.scaled_gaussian_filter_sigma,
      options.gradient_norm_quantization_error_bound,
      options.gradient_angle_threshold_deg, options.detection_threshold_log_eps,
      options.aligned_region_points_in_rectangle_min_density_threshold,
      static_cast<int>(options.gradient_modulus_num_bins));
}

void LineSegmentDetector::detect(
    const cv::Mat& image, Lines2dWithAngle* lines) {
  CHECK_NOTNULL(lines)->clear();
  CHECK(!line_detector_.empty());

  std::vector<cv::Vec4i> raw_lines;
  line_detector_->detect(image, raw_lines);
  line_detector_->filterSize(
      raw_lines, raw_lines, options_.min_segment_length_px);

  lines->reserve(raw_lines.size());
  for (size_t line_idx = 0u; line_idx < raw_lines.size(); ++line_idx) {
    lines->emplace_back(
        Line2d::PointType(
            static_cast<double>(raw_lines[line_idx](0)),
            static_cast<double>(raw_lines[line_idx](1))),
        Line2d::PointType(
            static_cast<double>(raw_lines[line_idx](2)),
            static_cast<double>(raw_lines[line_idx](3))));
  }
}

}  // namespace aslam
