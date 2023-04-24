#ifndef ASLAM_CALIBRATION_TARGET_CHECKERBOARD_H
#define ASLAM_CALIBRATION_TARGET_CHECKERBOARD_H

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/core.hpp>

#include "aslam/calibration/target-base.h"
#include "aslam/calibration/target-observation.h"

namespace aslam {
namespace calibration {

/// \brief Construct a calibration target
///        rows:                      number of internal corners (8x8 chessboard would be 7x7 internal corners)
///        cols:                      number of internal corners
///        row_spacing_meters:        spacing in y-direction [m]
///        col_spacing_meters:        spacing in x-direction [m]
///
///   point ordering in _points: (e.g. 2x2 grid)
///          *-------*-------*-------*
///          | BLACK | WHITE | BLACK |
///          *------(2)-----(3)------*
///          | WHITE | BLACK | WHITE |
///          *------(0)-----(1)------*
///    y     | BLACK | WHITE | BLACK |
///   ^      *-------*-------*-------*
///   |-->x

class TargetCheckerboard : public TargetBase {
 public:
  ASLAM_POINTER_TYPEDEFS(TargetCheckerboard);

  //target extraction options
  struct TargetConfiguration {
    TargetConfiguration()
        : rows(6u),
          cols(6u),
          row_spacing_meters(0.088),
          col_spacing_meters(0.088){}
          
    /// number of internal corners in row direction.
    size_t rows;
    /// number of internal corners in col direction.
    size_t cols;

    /// spacing in y-direction [m]
    double row_spacing_meters;
    /// spacing in x-direction [m]
    double col_spacing_meters;

    static TargetConfiguration fromYaml(const std::string& yaml_file);
  };

  /// \brief initialize based on checkerboard geometry
  TargetCheckerboard(const TargetConfiguration& target_config);

  virtual ~TargetCheckerboard() {};

  const TargetConfiguration& getConfig() const {
    return target_config_;
  }

  private:
    const TargetConfiguration target_config_;
};

Eigen::Matrix3Xd createCheckerboardPoints(
    const TargetCheckerboard::TargetConfiguration& target_config); //do we need this

class DetectorCheckerboard : public DetectorBase {
 public:
  ASLAM_POINTER_TYPEDEFS(DetectorCheckerboard);

  struct DetectorConfiguration {
    DetectorConfiguration()
        : run_subpixel_refinement(true),
          max_subpixel_refine_displacement_px_sq(1.5),
          perform_fast_check(false),
          use_adaptive_threshold(true),
          normalize_image(true),
          filter_quads(true){};
    /// Perform subpixel refinement of extracted corners.
    bool run_subpixel_refinement;
    /// Max. displacement squared in subpixel refinement. [px^2]
    double max_subpixel_refine_displacement_px_sq;
    /// options for opencv flags
    bool perform_fast_check;
    bool use_adaptive_threshold;
    bool normalize_image;
    bool filter_quads;
  };
  DetectorCheckerboard(const TargetCheckerboard::Ptr& target,
                    const DetectorCheckerboard::DetectorConfiguration& detector_config);
  virtual ~DetectorCheckerboard() {};

  virtual TargetObservation::Ptr detectTargetInImage(const cv::Mat& image) const;

 private:
  const TargetCheckerboard::Ptr target_;
  const DetectorConfiguration detector_config_;
};


}  // namespace cameras
}  // namespace aslam

#endif /* ASLAM_CALIBRATION_TARGET_CHECKERBOARD_H */
