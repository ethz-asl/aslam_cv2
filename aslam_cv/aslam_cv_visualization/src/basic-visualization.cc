#include "aslam/visualization/basic-visualization.h"

namespace aslam_cv_visualization {

void drawKeypoints(const aslam::VisualFrame& frame, cv::Mat* image) {
  CHECK_NOTNULL(image);
  const size_t num_keypoints = frame.getNumKeypointMeasurements();
  for (size_t keypoint_idx = 0u; keypoint_idx < num_keypoints; ++keypoint_idx) {
    Eigen::Vector2d keypoint;
    if (frame.getRawCameraGeometry()) {
      frame.getKeypointInRawImageCoordinates(keypoint_idx, &keypoint);
    } else {
      keypoint = frame.getKeypointMeasurement(keypoint_idx);
    }
    cv::circle(*image, cv::Point(keypoint[0], keypoint[1]), 1,
               cv::Scalar(0, 255, 255), 1, CV_AA);
  }
}

void visualizeKeypoints(const std::shared_ptr<aslam::VisualNFrame>& nframe, cv::Mat* image_ptr) {
  CHECK(nframe);
  CHECK_NOTNULL(image_ptr);

  const size_t num_frames = nframe->getNumFrames();
  CHECK_EQ(nframe->getNumCameras(), num_frames);

  Offsets offsets;
  assembleMultiImage(nframe, image_ptr, &offsets);
  CHECK_EQ(offsets.size(), num_frames);
  VLOG(4) << "Assembled full image.";

  cv::Mat& image = *image_ptr;

  for (size_t frame_idx = 0; frame_idx < num_frames; ++frame_idx) {
    const size_t image_width =
        nframe->getFrame(frame_idx).getCameraGeometry()->imageWidth();
    const size_t image_height =
        nframe->getFrame(frame_idx).getCameraGeometry()->imageHeight();
    cv::Mat slice = image(cv::Rect(offsets[frame_idx].width, offsets[frame_idx].height, image_width,
                                   image_height));
    aslam_cv_visualization::drawKeypoints(nframe->getFrame(frame_idx), &slice);
  }
}

void drawKeypointMatches(const aslam::VisualFrame& frame_kp1,
                         const aslam::VisualFrame& frame_k,
                         const aslam::Matches& matches_kp1_k,
                         cv::Scalar color_keypoint_kp1, cv::Scalar line_color,
                         cv::Mat* image) {
  CHECK_NOTNULL(image);

  for (const aslam::Match& match_kp1_k : matches_kp1_k) {
    Eigen::Vector2d keypoint_k;
    Eigen::Vector2d keypoint_kp1;
    CHECK_LT(match_kp1_k.second, frame_k.getNumKeypointMeasurements());
    if (frame_k.getRawCameraGeometry()) {
      frame_k.getKeypointInRawImageCoordinates(match_kp1_k.second, &keypoint_k);
      CHECK_LT(match_kp1_k.first, frame_kp1.getNumKeypointMeasurements());
      frame_kp1.getKeypointInRawImageCoordinates(match_kp1_k.first, &keypoint_kp1);
    } else {
      keypoint_k = frame_k.getKeypointMeasurement(match_kp1_k.second);
      CHECK_LT(match_kp1_k.first, frame_kp1.getNumKeypointMeasurements());
      keypoint_kp1 = frame_kp1.getKeypointMeasurement(match_kp1_k.first);
    }

    cv::circle(*image, cv::Point(keypoint_kp1[0], keypoint_kp1[1]), 4, color_keypoint_kp1);
    cv::line(*image, cv::Point(keypoint_k[0], keypoint_k[1]),
             cv::Point(keypoint_kp1[0], keypoint_kp1[1]), line_color);
  }
}

void assembleMultiImage(const std::shared_ptr<aslam::VisualNFrame>& nframe,
                        cv::Mat* full_image_ptr, Offsets* offsets_ptr) {
  CHECK(nframe);
  const size_t num_frames = nframe->getNumFrames();
  CHECK_GT(num_frames, 0u);
  CHECK_NOTNULL(full_image_ptr);
  CHECK_NOTNULL(offsets_ptr);

  cv::Mat& full_image = *full_image_ptr;

  VLOG(5) << "assembleMultiImage: num frames: " << num_frames;

  size_t num_rows = static_cast<size_t>(std::floor(std::sqrt(static_cast<double>(num_frames))));
  size_t num_images_per_row = static_cast<size_t>(std::ceil(static_cast<double>(num_frames) /
                                                            static_cast<double>(num_rows)));

  VLOG(5) << "assembleMultiImage: num rows: " << num_rows;
  VLOG(5) << "assembleMultiImage: num images per row: " << num_images_per_row;

  CHECK_GT(num_rows, 0u);
  CHECK_GT(num_images_per_row, 0u);

  size_t max_image_height_row = 0u;

  Offsets& offsets = *offsets_ptr;
  offsets.resize(num_frames);
  std::vector<cv::Mat> individual_images(num_frames);

  size_t row_index = 0u;
  size_t column_index = 0u;

  size_t max_column = 0;
  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    if ((frame_idx > 0u) && ((frame_idx % num_images_per_row) == 0u)) {
      // Time to switch rows.
      row_index += max_image_height_row;

      if (column_index > max_column) max_column = column_index;
      max_image_height_row = 0u;
      column_index = 0u;
    }

    cv::cvtColor(nframe->getFrame(frame_idx).getRawImage(), individual_images[frame_idx],
                 CV_GRAY2BGR);

    CHECK(nframe->getFrame(frame_idx).getCameraGeometry());
    const size_t image_width =
        nframe->getFrame(frame_idx).getCameraGeometry()->imageWidth();
    const size_t image_height =
        nframe->getFrame(frame_idx).getCameraGeometry()->imageHeight();

    VLOG(4) << "Adding image of dimension " << image_width << " x " << image_height;

    if (image_height > max_image_height_row) max_image_height_row = image_height;

    offsets[frame_idx].height = row_index;
    offsets[frame_idx].width = column_index;

    column_index += image_width;
  }

  row_index += max_image_height_row;

  full_image = cv::Mat(row_index, max_column, CV_8UC3);
  VLOG(3) << "Reshaped full image to the following dimensions: " << max_column << " x "
          << row_index;

  for (size_t frame_idx = 0u; frame_idx < num_frames; ++frame_idx) {
    const size_t image_width =
        nframe->getFrame(frame_idx).getCameraGeometry()->imageWidth();
    const size_t image_height =
        nframe->getFrame(frame_idx).getCameraGeometry()->imageHeight();

    VLOG(5) << "Accessing slice of full image starting at (" << offsets[frame_idx].width << " , "
            << offsets[frame_idx].height << ").";
    cv::Mat slice = full_image(cv::Rect(offsets[frame_idx].width, offsets[frame_idx].height,
                                        image_width, image_height));
    individual_images[frame_idx].copyTo(slice);
  }
}

void visualizeMatches(const aslam::VisualFrame& frame_kp1, const aslam::VisualFrame& frame_k,
                      const aslam::MatchesWithScore& matches_with_scores, cv::Mat* image) {
  CHECK_NOTNULL(image);

  cv::Mat& match_image = *image;
  cv::cvtColor(frame_kp1.getRawImage(), match_image, CV_GRAY2BGR);

  VLOG(4) << "Converted raw imagr from grayscale to color.";

  aslam::Matches matches;
  aslam::convertMatches(matches_with_scores, &matches);
  VLOG(4) << "Converted the matches.";

  CHECK_NOTNULL(match_image.data);

  drawKeypointMatches(frame_kp1, frame_k, matches, cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255),
                      &match_image);
}
  
}  // namespace aslam_cv_visualization
