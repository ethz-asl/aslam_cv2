#include <algorithm>

#include <aslam/matcher/match-helpers.h>

#include "aslam/frames/feature-track.h"
#include "aslam/frames/keypoint-identifier.h"
#include "aslam/visualization/basic-visualization.h"

namespace aslam_cv_visualization {

void drawKeypoints(const aslam::VisualFrame& frame, cv::Mat* image) {
  CHECK_NOTNULL(image);
  if (!frame.hasKeypointMeasurements()) {
    return;
  }
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

void visualizeKeypoints(const aslam::VisualNFrame::ConstPtr& nframe, cv::Mat* image_ptr) {
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
    cv::Mat frame_slice_of_multi_image =
        image(cv::Rect(
            offsets[frame_idx].width, offsets[frame_idx].height, image_width, image_height));
    aslam_cv_visualization::drawKeypoints(
        nframe->getFrame(frame_idx), &frame_slice_of_multi_image);
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

void assembleMultiImage(const aslam::VisualNFrame::ConstPtr& nframe,
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

      if (column_index > max_column) {
        max_column = column_index;
      }
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

    CHECK_EQ(individual_images[frame_idx].rows, static_cast<int>(image_height));
    CHECK_EQ(individual_images[frame_idx].cols, static_cast<int>(image_width));

    VLOG(4) << "Adding image of dimension " << image_width << " x " << image_height;

    if (image_height > max_image_height_row) {
      max_image_height_row = image_height;
    }

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

void drawFeatureTrackPatches(const aslam::FeatureTrack& track, size_t keypoint_neighborhood_px,
                             size_t num_cols, cv::Mat* image) {
  CHECK_NOTNULL(image);
  CHECK_GT(keypoint_neighborhood_px, 0u);
  CHECK_GT(num_cols, 0u);
  const size_t track_length = track.getTrackLength();
  CHECK_GT(track_length, 0u);

  const size_t subimage_edge_length_px = 2u * keypoint_neighborhood_px;
  const size_t num_subimage_rows =
      std::ceil(track_length / static_cast<double>(num_cols)) * subimage_edge_length_px;
  const size_t num_subimage_cols = num_cols * subimage_edge_length_px;

  image->create(num_subimage_rows, num_subimage_cols, CV_8UC3);
  image->setTo(0);

  size_t keypoint_index = 0u;
  for (const aslam::KeypointIdentifier& kip : track.getKeypointIdentifiers()) {
    const size_t subimage_row = std::floor(keypoint_index / num_cols);
    const size_t subimage_col = keypoint_index % num_cols;

    // roi denotes the region of interest wrt. to a larger image.
    cv::Rect roi_subimage(
        cv::Point(subimage_col * subimage_edge_length_px,
                  subimage_row * subimage_edge_length_px),
        cv::Point((subimage_col + 1) * subimage_edge_length_px,
                  (subimage_row + 1) * subimage_edge_length_px));

    // Extract the image patch around the keypoint and write it to the full image.
    const Eigen::Vector2d keypoint = kip.getKeypointMeasurement();
    const cv::Mat& keypoint_image = kip.getFrame().getRawImage();

    cv::Point roi_keypoint_topleft(std::max((keypoint(0) - keypoint_neighborhood_px), 0.0),
                                   std::max((keypoint(1) - keypoint_neighborhood_px), 0.0));
    cv::Point roi_keypoint_bottomright(
        std::min((keypoint(0) + keypoint_neighborhood_px),
                 static_cast<double>(keypoint_image.cols)),
        std::min((keypoint(1) + keypoint_neighborhood_px),
                 static_cast<double>(keypoint_image.rows)));
    cv::Rect roi_keypoint(roi_keypoint_topleft, roi_keypoint_bottomright);

    // Adjust the size in case of border truncation.
    roi_subimage.height = roi_keypoint.height;
    roi_subimage.width = roi_keypoint.width;
    CHECK_EQ(roi_keypoint.height, roi_subimage.height);
    CHECK_EQ(roi_keypoint.width, roi_subimage.width);

    // Draw the keypoint.
    cv::Mat keypoint_neighbourhood_image;
    cv::cvtColor(keypoint_image(roi_keypoint), keypoint_neighbourhood_image, CV_GRAY2BGR);
    cv::Point keypoint_coords_subimage(keypoint[0] - roi_keypoint.x, keypoint[1] - roi_keypoint.y);
    cv::circle(keypoint_neighbourhood_image, keypoint_coords_subimage, 1,
               cv::Scalar(0, 255, 255), 1, CV_AA);

    keypoint_neighbourhood_image.copyTo((*image)(roi_subimage));
    ++keypoint_index;
  }
}

bool drawFeatureTracksPatches(const aslam::FeatureTracks& tracks, size_t neighborhood_px,
                              size_t num_cols, cv::Mat* all_tracks_image) {
  CHECK_NOTNULL(all_tracks_image);
  CHECK_GT(num_cols, 0u);
  CHECK_GT(neighborhood_px, 0u);
  if (tracks.empty()) {
    return false;
  }

  const size_t num_tracks = tracks.size();
  const size_t edge_size = 2u * neighborhood_px;
  const size_t full_image_width = num_cols * edge_size;
  const size_t full_image_height = num_tracks * edge_size;
  all_tracks_image->create(full_image_height, full_image_width, CV_8UC3);
  all_tracks_image->setTo(0);

  size_t track_idx = 0u;
  for (const aslam::FeatureTrack& track : tracks) {
    cv::Mat track_image;
    drawFeatureTrackPatches(track, neighborhood_px, num_cols, &track_image);
    cv::Rect roi_track(cv::Point(0, track_idx * edge_size),
                       cv::Point(full_image_width, (track_idx + 1) * edge_size));

    track_image.copyTo((*all_tracks_image)(roi_track));
    ++track_idx;
  }
  return true;
}

bool drawFeatureTracks(const aslam::FeatureTracks& tracks, cv::Mat* image) {
  CHECK_NOTNULL(image);
  if (tracks.empty()) {
    return false;
  }

  // Draw the tracks in the image of the last keypoint of the first track.
  const cv::Mat& frame_image = tracks.front().getLastKeypointIdentifier().getFrame().getRawImage();
  cv::cvtColor(frame_image, *image, CV_GRAY2BGR);

  // Draw all keypoints on the tracks.
  for (const aslam::FeatureTrack& track : tracks) {
    cv::Point last_keypoint_cv;
    bool is_first_keypoint_drawn = false;
    for (const aslam::KeypointIdentifier& kip : track.getKeypointIdentifiers()) {
      const Eigen::Vector2d keypoint = kip.getKeypointMeasurement();
      const cv::Point keypoint_cv(keypoint(0), keypoint(1));
      const size_t kRadiusPx = 1u;
      const size_t kThicknessPx = 1u;
      cv::circle(*image, keypoint_cv, kRadiusPx, kYellow, kThicknessPx, CV_AA);
      if (is_first_keypoint_drawn) {
        cv::line(*image, keypoint_cv, last_keypoint_cv, kYellow);
      } else {
        is_first_keypoint_drawn = true;
      }
      last_keypoint_cv = keypoint_cv;
    }
  }
  return true;
}
  
}  // namespace aslam_cv_visualization
