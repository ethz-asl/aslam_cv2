#ifndef ASLAM_DETECTORS_LINE_SEGMENT_DETECTOR_H_
#define ASLAM_DETECTORS_LINE_SEGMENT_DETECTOR_H_

#include <memory>

#include <aslam/common/macros.h>
#include <Eigen/Core>
#include <lsd/lsd-opencv.h>

#include <aslam/lines/line-2d-with-angle.h>

namespace aslam {

class LineSegmentDetector final {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options final {
    Options();
    ~Options() = default;
    // Minimum line segment length in pixels.
    size_t min_segment_length_px;

    // Scale of image that will be used to find the lines. Range (0..1].
    double image_scale;

    // Sigma for Gaussian filter is computed as sigma = _sigma_scale/_scale.
    double scaled_gaussian_filter_sigma;

    // Bound to the quantization error on the gradient norm.
    double gradient_norm_quantization_error_bound;

    // Gradient angle tolerance in degrees.
    double gradient_angle_threshold_deg;

    // Detection threshold: -log10(NFA) > _log_eps.
    double detection_threshold_log_eps;

    // Minimal density of aligned region points in rectangle.
    double aligned_region_points_in_rectangle_min_density_threshold;

    // Number of bins in pseudo-ordering of gradient modulus.
    size_t gradient_modulus_num_bins;
  };

  LineSegmentDetector() = default;
  explicit LineSegmentDetector(const Options& options);
  ~LineSegmentDetector() = default;

  void detect(const cv::Mat& image, Lines2dWithAngle* lines);

 private:
  cv::Ptr<aslamcv::LineSegmentDetector> line_detector_;
  const Options options_;
};

}  // namespace aslam

#endif  // ASLAM_DETECTORS_LINE_SEGMENT_DETECTOR_H_
