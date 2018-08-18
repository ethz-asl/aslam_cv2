#ifndef ASLAM_CV_DETECTORS_LSD
#define ASLAM_CV_DETECTORS_LSD

#include <memory>

#include <aslam/common/macros.h>
#include <Eigen/Core>
#include <lsd/lsd-opencv.h>

#include <aslam/common/line.h>

namespace aslam {

class LineSegmentDetector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct Options {
    size_t min_segment_length_px;
    double scale;
    double sigma_scale;
    double quant;
    double angle_threshold;
    double log_eps;
    double density_threshold;
    int num_bins;
    Options() :
      min_segment_length_px(20u), scale(0.8), sigma_scale(0.6), quant(2.0),
      angle_threshold(16.5), log_eps(0.0), density_threshold(0.65),
      num_bins(1024) {};
  };

  LineSegmentDetector(const Options& options);
  ~LineSegmentDetector();

  void detect(const cv::Mat& image, Lines* lines);

  /// Draw a list of lines onto a color(!) image.
  void drawLines(const Lines& lines, cv::Mat* image);

 private:
  cv::Ptr<aslamcv::LineSegmentDetector> line_detector_;
  const Options options_;
};

}  // namespace aslam

#endif  // ASLAM_CV_DETECTORS_LSD
