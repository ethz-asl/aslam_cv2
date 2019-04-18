#include <algorithm>
#include <memory>
#include <vector>

#include <aslam/common/yaml-serialization.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "aslam/calibration/target-checkerboard.h"
#include "aslam/calibration/target-observation.h"

namespace aslam {
namespace calibration {

TargetCheckerboard::TargetConfiguration
TargetCheckerboard::TargetConfiguration::fromYaml(const std::string& yaml_file) {
  TargetConfiguration target_config;
  try {
    const YAML::Node yaml_node = YAML::LoadFile(yaml_file.c_str());
    std::string target_type;
    YAML::safeGet(yaml_node, "target_type", &target_type);
    CHECK_EQ(target_type, "aprilgrid") << "Wrong target type.";
    YAML::safeGet(yaml_node, "tagCols", &target_config.num_tag_cols);
    CHECK_GT(target_config.num_tag_cols, 0u);
    YAML::safeGet(yaml_node, "tagRows", &target_config.num_tag_rows);
    CHECK_GT(target_config.num_tag_rows, 0u);
    YAML::safeGet(yaml_node, "tagSize", &target_config.tag_size_meter);
    CHECK_GT(target_config.tag_size_meter, 0.0);
    double relative_tag_spacing;
    YAML::safeGet(yaml_node, "tagSpacing", &relative_tag_spacing);
  } catch (const YAML::Exception& ex) {
    LOG(FATAL) << "Failed to load yaml file " << yaml_file
               << " with the error: \n " << ex.what() << ".";
  }
  return target_config;
}

TargetCheckerboard::TargetCheckerboard(const TargetCheckerboard::TargetConfiguration& target_config)
    : TargetBase(target_config.num_tag_rows, target_config.num_tag_cols, //maybe -1 here?
                  createCheckerboardPoints(target_config)), target_config_(target_config) {
  CHECK_GT(target_config.tag_size_meter, 0.0);
}

Eigen::Matrix3Xd createCheckerboardPoints( //not sure but might be correct as is
    const TargetCheckerboard::TargetConfiguration& target_config) {
  // point ordering: (e.g. 2x2 grid)
  //          *-------*-------*-------*
  //          | BLACK | WHITE | BLACK |
  //          *------(2)-----(3)------*
  //          | WHITE | BLACK | WHITE |
  //          *------(0)-----(1)------*
  //    y     | BLACK | WHITE | BLACK |
  //   ^      *-------*-------*-------*
  //   |-->x
  //|-->x
  const double tag_size = target_config.tag_size_meter;
  const size_t num_point_rows = target_config.num_tag_rows;
  const size_t num_point_cols = target_config.num_tag_cols;
  CHECK_GT(tag_size, 0.0);
  Eigen::Matrix3Xd grid_points_meters =
      Eigen::Matrix3Xd(3, num_point_rows * num_point_cols);
  for (size_t row_idx = 0u; row_idx < num_point_rows; ++row_idx) {
    for (size_t col_idx = 0u; col_idx < num_point_cols; ++col_idx) {
      Eigen::Vector3d point;
      point(0) = static_cast<double>(col_idx + 1) * tag_size;
      point(1) = static_cast<double>(row_idx + 1) * tag_size;
      point(2) = 0.0;

      grid_points_meters.col(row_idx * num_point_cols + col_idx) = point;
    }
  }
  return grid_points_meters;
}

DetectorCheckerboard::DetectorCheckerboard(
    const TargetCheckerboard::Ptr& target,
    const DetectorCheckerboard::DetectorConfiguration& detector_config)
    : target_(target),
      detector_config_(detector_config) {
  CHECK(target);
}

TargetObservation::Ptr DetectorCheckerboard::detectTargetInImage(const cv::Mat& image) const {
  // set the open cv flags
  int flags = 0;
  if (detector_config_.perform_fast_check)
    flags += cv::CALIB_CB_FAST_CHECK;
  if (detector_config_.use_adaptive_threshold)
    flags += cv::CALIB_CB_ADAPTIVE_THRESH;
  if ( detector_config_.normalize_image)
    flags += cv::CALIB_CB_NORMALIZE_IMAGE;
  if (detector_config_.filter_quads)
    flags += cv::CALIB_CB_FILTER_QUADS;

  // extract the checkerboard corners
  cv::Size patternSize(target_->cols(), target_->rows());
  cv::Mat corners(target_->size(), 2, CV_32F);
  bool success = cv::findChessboardCorners(image, patternSize, corners, flags);

  cv::Mat corners_raw = corners.clone();

  // do optional subpixel refinement
  if (detector_config_.run_subpixel_refinement && success) { 
    cv::cornerSubPix(
        image, corners, cv::Size(2, 2), cv::Size(-1, -1),
        cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
  }

  //if not successfull return empty observation
  if (!success) {
    return TargetObservation::Ptr();
  }

  Eigen::VectorXi corner_ids(target_->size());
  Eigen::Matrix2Xd image_corners(2, target_->size());
  size_t out_point_idx = 0u;

  //convert to eigen for output
  for (size_t i = 0; i < target_->size(); ++i)
  {
    const Eigen::Vector2d corner_refined(corners.row(i).at<float>(0), corners.row(i).at<float>(1));
    const Eigen::Vector2d corner_raw(corners_raw.row(i).at<float>(0), corners_raw.row(i).at<float>(1));
    
    const double subpix_displacement_squarred = (corner_refined.row(i) - corner_raw.row(i)).squaredNorm();
    if (subpix_displacement_squarred <= detector_config_.max_subpixel_refine_displacement_px_sq) {
      corner_ids(out_point_idx) = i;
      image_corners.col(out_point_idx) = corner_refined;
      ++out_point_idx;
    }
  }
      
  corner_ids.conservativeResize(out_point_idx);
  image_corners.conservativeResize(Eigen::NoChange, out_point_idx);

  return TargetObservation::Ptr(new TargetObservation(target_, image.rows, image.cols,
                                                      corner_ids, image_corners));

}

}  // namespace calibration
}  // namespace aslam
