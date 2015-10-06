#include <algorithm>
#include <memory>
#include <vector>

#include <apriltags/TagDetector.h>
#include <apriltags/Tag36h11.h>
#include <Eigen/Core>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "aslam/calibration/target-aprilgrid.h"
#include "aslam/calibration/target-observation.h"

namespace aslam {
namespace calibration {

TargetAprilGrid::TargetAprilGrid(const TargetAprilGrid::TargetConfiguration& target_config)
    : TargetBase(2 * target_config.num_tag_rows,
                 2 * target_config.num_tag_cols, // 4 points per tag.
                 createGridPoints(target_config)),
                 target_config_(target_config) {
  CHECK_GT(target_config.tag_size_meter, 0.0);
  CHECK_GT(target_config.tag_inbetween_space_meter, 0.0);
}

Eigen::Matrix3Xd TargetAprilGrid::createGridPoints(
    const TargetAprilGrid::TargetConfiguration& target_config) {
  /// \brief initialize an april grid
  ///   point ordering: (e.g. 2x2 grid)
  ///          12-----13  14-----15
  ///          | TAG 3 |  | TAG 4 |
  ///          8-------9  10-----11
  ///          4-------5  6-------7
  ///    y     | TAG 1 |  | TAG 2 |
  ///   ^      0-------1  2-------3
  ///   |-->x
  const double tag_size = target_config.tag_size_meter;
  const double tag_spacing = target_config.tag_inbetween_space_meter;
  const size_t num_point_rows = 2 * target_config.num_tag_rows;
  const size_t num_point_cols = 2 * target_config.num_tag_cols;
  CHECK_GT(tag_size, 0.0);
  CHECK_GT(tag_spacing, 0.0);

  Eigen::Matrix3Xd points_target_frame(3, num_point_rows * num_point_cols);
  for (size_t r = 0u; r < num_point_rows; r++) {
    for (size_t c = 0u; c < num_point_cols ; c++) {
      Eigen::Vector3d point;
      point(0) = (int) (c / 2) * (tag_size + tag_spacing) + (c % 2) * tag_size;
      point(1) = (int) (r / 2) * (tag_size + tag_spacing) + (r % 2) * tag_size;
      point(2) = 0.0;

      points_target_frame.col(r * num_point_cols + c) = point;
    }
  }
  return points_target_frame;
}

DetectorAprilGrid::DetectorAprilGrid(
    const TargetAprilGrid::Ptr& target,
    const DetectorAprilGrid::DetectorConfiguration& detector_config)
    : target_(target),
      detector_config_(detector_config),
      tag_codes_(AprilTags::tagCodes36h11) {
  CHECK(target);
  tag_detector_.reset(
      new AprilTags::TagDetector(tag_codes_, target_->getConfig().black_tag_border_bits));
}

TargetObservation::Ptr DetectorAprilGrid::detectTargetInImage(const cv::Mat& image) const {
  // Detect all Apriltags in the image.
  std::vector<AprilTags::TagDetection> detections = tag_detector_->extractTags(image);

  // Remove bad tags.
  std::vector<AprilTags::TagDetection>::iterator iter = detections.begin();
  for (iter = detections.begin(); iter != detections.end();) {
    bool remove = false;

    // Enforce min. distance of corners to the image border (tag removed if violated).
    for (int j = 0; j < 4; j++) {
      remove |= iter->p[j].first < detector_config_.min_border_distance_px;
      remove |= iter->p[j].first >
        static_cast<double>(image.cols) - detector_config_.min_border_distance_px;
      remove |= iter->p[j].second < detector_config_.min_border_distance_px;
      remove |= iter->p[j].second >
        static_cast<double>(image.rows) - detector_config_.min_border_distance_px;
    }

    // Flag for removal if tag deteftion is marked as bad.
    if (iter->good != 1) {
      remove |= true;
    }

    // Flag for removal if the tag ID is out-of-range for this grid (faulty detection or wild tag).
    if (iter->id >= static_cast<int>(target_->size() / 4.0)) {
      remove |= true;
    }

    // Remove tag from the observation list.
    if (remove) {
      VLOG(200) << "Tag with ID " << iter->id << " is only partially in image (corners outside) "
                << "and will be removed from the TargetObservation.\n";

      // delete the tag and advance in list
      iter = detections.erase(iter);
    } else {
      //advance in list
      ++iter;
    }
  }

  // Check if enough tags have been found.
  if (detections.size() < detector_config_.min_visible_tags_for_valid_obs) {
    // Detection failed; returnn nullptr.
    return TargetObservation::Ptr();
  }

  //sort detections by tagId
  std::sort(detections.begin(), detections.end(), AprilTags::TagDetection::sortByIdCompare);

  // Check for duplicate tag ids that would indicate Apriltags not belonging to calibration target.
  if (detections.size() > 1) {
    for (size_t i = 0; i < detections.size() - 1; i++)
      if (detections[i].id == detections[i + 1].id) {
        // Show image of duplicate Apriltag.
        cv::destroyAllWindows();
        cv::namedWindow("Wild Apriltag detected. Hide them!");
        cvStartWindowThread();

        cv::Mat imageCopy = image.clone();
        cv::cvtColor(imageCopy, imageCopy, CV_GRAY2RGB);

        // Mark duplicate tags in image.
        for (int j = 0; i < detections.size() - 1; i++) {
          if (detections[j].id == detections[j + 1].id) {
            detections[j].draw(imageCopy);
            detections[j + 1].draw(imageCopy);
          }
        }

        cv::putText(imageCopy, "Duplicate Apriltags detected. Hide them.", cv::Point(50, 50),
                    CV_FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2, 8, false);
        cv::putText(imageCopy, "Press enter to exit...", cv::Point(50, 80),
                    CV_FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(255, 0, 0), 2, 8, false);
        cv::imshow("Duplicate Apriltags detected. Hide them", imageCopy);
        cv::waitKey();

        LOG(WARNING) << "Found apriltag not belonging to calibration board. Check the image for "
                     << "the tag and hide it.\n";
        return TargetObservation::Ptr();
      }
  }

  // Convert corners to cv::Mat (4 consecutive corners form one tag).
  // point ordering here
  //          11-----10  15-----14
  //          | TAG 2 |  | TAG 3 |
  //          8-------9  12-----13
  //          3-------2  7-------6
  //    y     | TAG 0 |  | TAG 1 |
  //   ^      0-------1  4-------5
  //   |-->x
  cv::Mat tag_corners(4 * detections.size(), 2, CV_32F);
  for (unsigned i = 0; i < detections.size(); i++) {
    for (unsigned j = 0; j < 4; j++) {
      tag_corners.at<float>(4 * i + j, 0) = detections[i].p[j].first;
      tag_corners.at<float>(4 * i + j, 1) = detections[i].p[j].second;
    }
  }

  // Store a copy of the corner list before subpix refinement.
  cv::Mat tag_corners_raw = tag_corners.clone();

  // Perform optional subpixel refinement on all tag corners (four corners each tag).
  if (detector_config_.run_subpixel_refinement) {
    cv::cornerSubPix(image, tag_corners, cv::Size(2, 2), cv::Size(-1, -1),
                     cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
  }

  // Insert the observed points into the correct location of the grid point array.
  // point ordering
  //          12-----13  14-----15
  //          | TAG 2 |  | TAG 3 |
  //          8-------9  10-----11
  //          4-------5  6-------7
  //    y     | TAG 0 |  | TAG 1 |
  //   ^      0-------1  2-------3
  //   |-->x
  Eigen::VectorXi corner_ids(target_->size());
  Eigen::Matrix2Xd image_corners(2, target_->size());
  size_t out_point_idx = 0u;

  for (size_t i = 0; i < detections.size(); i++) {
    const unsigned int tag_id = detections[i].id;

    // Calculate the grid idx for all four tag corners given the tagId and cols.
    const size_t cols = target_->cols();
    unsigned int base_idx =
        static_cast<int>(tag_id / (cols / 2)) * cols * 2 + (tag_id % (cols / 2)) * 2;
    unsigned int point_indices_tag[] = {base_idx,
                                        base_idx + 1,
                                        base_idx + static_cast<unsigned int>(cols + 1),
                                        base_idx + static_cast<unsigned int>(cols)};

    // Add four points per tag
    for (int j = 0; j < 4; j++) {
      const Eigen::Vector2d corner_refined(tag_corners.row(4 * i + j).at<float>(0),
                                           tag_corners.row(4 * i + j).at<float>(1));
      const Eigen::Vector2d corner_raw(tag_corners_raw.row(4 * i + j).at<float>(0),
                                       tag_corners_raw.row(4 * i + j).at<float>(1));

      // Add corner points if it has not moved to far in the subpix refinement.
      const double subpix_displacement_squarred = (corner_refined - corner_raw).squaredNorm();
      if (subpix_displacement_squarred <= detector_config_.max_subpixel_refine_displacement_px_sq) {
        corner_ids(out_point_idx) = point_indices_tag[j];
        image_corners.col(out_point_idx) = corner_refined;
        ++out_point_idx;
      }
    }
  }
  corner_ids.conservativeResize(out_point_idx);
  image_corners.conservativeResize(Eigen::NoChange, out_point_idx);

  return TargetObservation::Ptr(new TargetObservation(target_, corner_ids, image_corners));
}

}  // namespace calibration
}  // namespace aslam
